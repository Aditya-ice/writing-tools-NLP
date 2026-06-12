"""Custom speculative decoding loop (draft Qwen3-0.6B + verify Qwen3-4B)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


def _log_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits, dim=-1)


def draft_generate(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autoregressively draft K tokens. Returns (candidate_ids, log_probs) of shape (K,)."""
    current = input_ids
    candidate_ids: list[torch.Tensor] = []
    candidate_log_probs: list[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(K):
            logits = model(current).logits[:, -1, :]
            log_probs = _log_probs(logits)
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            candidate_ids.append(next_id.squeeze(-1))
            candidate_log_probs.append(log_probs.gather(1, next_id).squeeze(-1))
            current = torch.cat([current, next_id], dim=-1)

    return torch.stack(candidate_ids), torch.stack(candidate_log_probs)


def _sample_from_adjusted(
    target_log_probs: torch.Tensor,
    draft_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Sample a token from norm(max(0, p_target - p_draft))."""
    adjusted = (target_log_probs.exp() - draft_log_probs.exp()).clamp_min(0.0)
    total = adjusted.sum()
    if total <= 0:
        return torch.argmax(target_log_probs, dim=-1)
    probs = adjusted / total
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


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
    eos_id = tokenizer.eos_token_id

    generated: list[int] = []
    draft_proposed = 0
    draft_accepted = 0

    with torch.no_grad():
        while len(generated) < max_new_tokens:
            gamma = min(K, max_new_tokens - len(generated))
            prefix = (
                torch.tensor([generated], device=device, dtype=torch.dtype(input_ids.dtype))
                if generated
                else torch.empty((1, 0), device=device, dtype=input_ids.dtype)
            )
            current = torch.cat([input_ids, prefix], dim=1)

            candidate_ids, candidate_log_probs = draft_generate(
                draft_model, current, gamma
            )
            draft_proposed += gamma

            verify_input = torch.cat([current, candidate_ids.unsqueeze(0)], dim=1)
            verify_logits = verify_model(verify_input).logits
            prefix_len = current.shape[1]

            accepted: list[int] = []
            for i in range(gamma):
                step_logits = verify_logits[0, prefix_len + i - 1, :]
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
                            target_log_probs, candidate_log_probs[i]
                        ).item()
                        accepted.append(resampled)
                        break
            else:
                bonus_logits = verify_logits[0, prefix_len + gamma - 1, :]
                if greedy:
                    accepted.append(torch.argmax(bonus_logits).item())
                else:
                    bonus_log_probs = _log_probs(bonus_logits.unsqueeze(0)).squeeze(0)
                    accepted.append(torch.multinomial(bonus_log_probs.exp(), 1).item())

            generated.extend(accepted[: max_new_tokens - len(generated)])
            if eos_id is not None and eos_id in accepted:
                break

    if stats is not None:
        stats["draft_tokens_proposed"] = stats.get("draft_tokens_proposed", 0) + draft_proposed
        stats["draft_tokens_accepted"] = stats.get("draft_tokens_accepted", 0) + draft_accepted
        stats["generated_tokens"] = stats.get("generated_tokens", 0) + len(generated)

    return tokenizer.decode(generated, skip_special_tokens=True)
