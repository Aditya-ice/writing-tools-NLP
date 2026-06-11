#!/usr/bin/env python3
"""Unified inference pipeline: loads adapter and runs any writing task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from data.prepare import TASK_INSTRUCTIONS
from train.config import MODEL_ID


class WritingPipeline:
    """Load Qwen3 base (+ optional LoRA adapter) and run writing tasks."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        adapter_path: str | None = None,
        load_in_4bit: bool | None = None,
    ) -> None:
        if load_in_4bit is None:
            load_in_4bit = torch.cuda.is_available()

        self.model_id = model_id
        self.adapter_path = adapter_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict = {"trust_remote_code": True, "device_map": "auto"}
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            model_kwargs["torch_dtype"] = dtype

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if adapter_path is not None:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def _prompt_messages(
        self,
        task: str,
        user_input: str,
        *,
        tone: str | None = None,
    ) -> list[dict[str, str]]:
        if task == "rewrite":
            system = TASK_INSTRUCTIONS["rewrite"].format(tone=tone or "formal")
        else:
            system = TASK_INSTRUCTIONS[task]
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_input},
        ]

    def generate(
        self,
        task: str,
        user_input: str,
        *,
        tone: str | None = None,
        max_new_tokens: int = 256,
    ) -> str:
        messages = self._prompt_messages(task, user_input, tone=tone)
        return self.generate_from_messages(messages, max_new_tokens=max_new_tokens)

    def generate_from_messages(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 256,
    ) -> str:
        prompt_messages = messages
        if messages and messages[-1]["role"] == "assistant":
            prompt_messages = messages[:-1]

        prompt = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0, input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run writing-tools inference.")
    parser.add_argument(
        "--task",
        choices=["summarise", "rewrite", "smart_reply"],
        required=True,
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--tone", choices=["formal", "casual"], default="formal")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    pipeline = WritingPipeline(adapter_path=args.adapter_path)
    output = pipeline.generate(
        args.task,
        args.input,
        tone=args.tone if args.task == "rewrite" else None,
        max_new_tokens=args.max_new_tokens,
    )
    print(output)


if __name__ == "__main__":
    main()
