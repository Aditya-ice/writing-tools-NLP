"""Custom speculative decoding loop (draft Qwen3-0.6B + verify Qwen3-4B)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizer


def draft_generate(
    model: "PreTrainedModel",
    input_ids: "torch.Tensor",
    K: int,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Return (candidate_ids, candidate_log_probs) of shape (K,)."""
    raise NotImplementedError


def speculative_decode(
    draft_model: "PreTrainedModel",
    verify_model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    prompt: str,
    max_new_tokens: int = 100,
    K: int = 5,
) -> str:
    """Full speculative decoding loop. Returns generated text."""
    raise NotImplementedError
