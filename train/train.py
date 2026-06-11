#!/usr/bin/env python3
"""SFT training with QLoRA (run on Colab T4 — not on Mac)."""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

# Allow `python train/train.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from datasets import load_dataset
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from train.config import TrainingConfig


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "Training requires CUDA (e.g. Colab T4). "
            "bitsandbytes 4-bit QLoRA does not run on Apple Silicon."
        )


def load_tokenizer(model_id: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(config: TrainingConfig) -> AutoModelForCausalLM:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return get_peft_model(model, config.lora_config())


def load_datasets(config: TrainingConfig, tokenizer: AutoTokenizer):
    raw = load_dataset(
        "json",
        data_files={"train": config.train_data, "validation": config.val_data},
    )

    def to_text(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        return {"text": text}

    return raw.map(to_text, remove_columns=raw["train"].column_names)


def build_sft_config(config: TrainingConfig) -> SFTConfig:
    use_bf16 = config.bf16 and torch.cuda.is_bf16_supported()
    kwargs: dict = dict(
        output_dir=config.checkpoint_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        max_seq_length=config.max_seq_len,
        dataset_text_field="text",
        packing=False,
        report_to="none",
        save_total_limit=2,
    )
    # Train loss on assistant tokens only (trl >= 0.12).
    if "assistant_only_loss" in inspect.signature(SFTConfig).parameters:
        kwargs["assistant_only_loss"] = True
    return SFTConfig(**kwargs)


def train(config: TrainingConfig, resume_from_checkpoint: str | None) -> None:
    require_cuda()

    for path in (config.train_data, config.val_data):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `python data/prepare.py` first."
            )

    tokenizer = load_tokenizer(config.model_id)
    model = load_model(config)
    datasets = load_datasets(config, tokenizer)
    sft_config = build_sft_config(config)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
    )

    print(f"Training {config.model_id} with QLoRA")
    print(f"  train examples: {len(datasets['train'])}")
    print(f"  val examples:   {len(datasets['validation'])}")
    print(f"  max_seq_len:    {config.max_seq_len}")
    print(f"  checkpoints:    {config.checkpoint_dir}")
    print(f"  adapter output: {config.output_dir}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"Adapter saved to {config.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-4B with QLoRA.")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint dir (e.g. ./checkpoints/checkpoint-100)",
    )
    args = parser.parse_args()
    train(TrainingConfig(), resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
