#!/usr/bin/env python3
"""SFT training with QLoRA (run on Colab T4 — not on Mac)."""

from __future__ import annotations

import argparse

from train.config import TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-4B with QLoRA.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    config = TrainingConfig()
    # Implementation: load datasets, apply chat template, run SFTTrainer.
    # See notebooks/train_colab.ipynb for the full Colab workflow.
    raise NotImplementedError(
        "Training script stub — complete after data/prepare.py is verified. "
        f"Resume checkpoint: {args.resume_from_checkpoint!r}. "
        f"Config: {config.model_id}, max_seq_len={config.max_seq_len}."
    )


if __name__ == "__main__":
    main()
