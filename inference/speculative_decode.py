"""Custom speculative decoding loop (draft Qwen3-0.6B + verify Qwen3-4B)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


def _log_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits, dim=-1)


def _prime_cache(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
) -> tuple[Any, torch.Tensor]:
    outputs = model(input_ids, use_cache=True)
    return outputs.past_key_values, outputs.logits[:, -1, :]


def draft_generate(
    model: PreTrainedModel,
    K: int,
    past_key_values: Any,
    next_logits: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draft K tokens using KV cache. Returns (ids, log_probs)."""
    past = past_key_values
    logits = next_logits
    ids: list[int] = []
    log_probs: list[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(K):
            step_log_probs = _log_probs(logits)
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            ids.append(int(next_id.item()))
            log_probs.append(step_log_probs.gather(1, next_id).squeeze())
            outputs = model(next_id, past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

    return (
        torch.tensor(ids, device=device, dtype=dtype),
        torch.stack(log_probs),
    )


def _sample_from_adjusted(
    target_log_probs: torch.Tensor,
    draft_log_prob: torch.Tensor,
) -> torch.Tensor:
    """Sample a token from norm(max(0, p_target - p_draft))."""
    draft_probs = torch.zeros_like(target_log_probs)
    draft_probs.fill_(draft_log_prob.exp())
    adjusted = (target_log_probs.exp() - draft_probs).clamp_min(0.0)
    total = adjusted.sum()
    if total <= 0:
        return torch.argmax(target_log_probs, dim=-1)
    probs = adjusted / total
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _sync_caches(
    draft_model: PreTrainedModel,
    verify_model: PreTrainedModel,
    input_ids: torch.Tensor,
    generated: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, torch.Tensor, Any]:
    """Re-prime both models on prompt + generated tokens."""
    if generated:
        full_ids = torch.cat(
            [
                input_ids,
                torch.tensor([generated], device=device, dtype=dtype),
            ],
            dim=1,
        )
    else:
        full_ids = input_ids
    draft_past, draft_logits = _prime_cache(draft_model, full_ids)
    verify_past, _ = _prime_cache(verify_model, full_ids)
    return draft_past, draft_logits, verify_past


def speculative_decode(
    draft_model: PreTrainedModel,
    verify_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    K: int = 5,
    *,
    greedy: bool = True,
    stats: dict[str, int] | None = None,
) -> str:
    """Speculative decoding loop. Returns generated text (new tokens only)."""
    device = next(verify_model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    dtype = input_ids.dtype
    eos_id = tokenizer.eos_token_id

    generated: list[int] = []
    draft_proposed = 0
    draft_accepted = 0

    with torch.no_grad():
        draft_past, draft_logits, verify_past = _sync_caches(
            draft_model, verify_model, input_ids, generated, device, dtype
        )

        while len(generated) < max_new_tokens:
            gamma = min(K, max_new_tokens - len(generated))

            candidate_ids, candidate_log_probs = draft_generate(
                draft_model,
                gamma,
                draft_past,
                draft_logits,
                device=device,
                dtype=dtype,
            )
            draft_proposed += gamma

            verify_out = verify_model(
                candidate_ids.unsqueeze(0),
                past_key_values=verify_past,
                use_cache=True,
            )
            verify_logits = verify_out.logits[0]

            accepted: list[int] = []
            for i in range(gamma):
                step_logits = verify_logits[i]
                target_log_probs = _log_probs(step_logits.unsqueeze(0)).squeeze(0)
                draft_id = candidate_ids[i].item()
                draft_log_prob = candidate_log_probs[i]

                if greedy:
                    verify_id = torch.argmax(step_logits).item()
                    if verify_id == draft_id:
                        accepted.append(draft_id)
                        draft_accepted += 1
                    else:
                        accepted.append(verify_id)
                        break
                else:
                    target_prob = target_log_probs[draft_id].exp().item()
                    draft_prob = draft_log_prob.exp().item()
                    ratio = min(1.0, target_prob / max(draft_prob, 1e-12))
                    if torch.rand(1).item() <= ratio:
                        accepted.append(draft_id)
                        draft_accepted += 1
                    else:
                        resampled = _sample_from_adjusted(
                            target_log_probs, draft_log_prob
                        ).item()
                        accepted.append(resampled)
                        break
            else:
                if gamma == 1:
                    bonus_out = verify_model(
                        candidate_ids.unsqueeze(0),
                        past_key_values=verify_past,
                        use_cache=True,
                    )
                else:
                    partial_out = verify_model(
                        candidate_ids[:-1].unsqueeze(0),
                        past_key_values=verify_past,
                        use_cache=True,
                    )
                    bonus_out = verify_model(
                        candidate_ids[-1:].unsqueeze(0),
                        past_key_values=partial_out.past_key_values,
                        use_cache=True,
                    )
                bonus_logits = bonus_out.logits[0, -1, :]
                if greedy:
                    accepted.append(torch.argmax(bonus_logits).item())
                else:
                    bonus_log_probs = _log_probs(bonus_logits.unsqueeze(0)).squeeze(0)
                    accepted.append(
                        torch.multinomial(bonus_log_probs.exp(), 1).item()
                    )

            generated.extend(accepted[: max_new_tokens - len(generated)])
            if eos_id is not None and eos_id in accepted:
                break

            draft_past, draft_logits, verify_past = _sync_caches(
                draft_model, verify_model, input_ids, generated, device, dtype
            )

    if stats is not None:
        stats["draft_tokens_proposed"] = stats.get("draft_tokens_proposed", 0) + draft_proposed
        stats["draft_tokens_accepted"] = stats.get("draft_tokens_accepted", 0) + draft_accepted
        stats["generated_tokens"] = stats.get("generated_tokens", 0) + len(generated)

    return tokenizer.decode(generated, skip_special_tokens=True)
