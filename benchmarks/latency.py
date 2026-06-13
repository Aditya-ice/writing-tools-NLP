#!/usr/bin/env python3
"""Latency benchmark: standard vs speculative vs HF assisted generation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from inference.speculative_decode import speculative_decode
from train.config import DRAFT_MODEL_ID, MODEL_ID, VAL_DATA


@dataclass
class BenchmarkResult:
    name: str
    mean_latency_ms: float
    tokens_per_sec: float
    mean_output_tokens: float
    acceptance_rate: float | None = None


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("Latency benchmark requires CUDA (Colab/Kaggle T4).")


def load_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_verify_model(adapter_path: str) -> AutoModelForCausalLM:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return PeftModel.from_pretrained(model, adapter_path)


def load_draft_model() -> AutoModelForCausalLM:
    dtype = torch.float16
    return AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL_ID,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )


def load_prompts(
    path: str, n_samples: int, tokenizer: AutoTokenizer
) -> list[str]:
    prompts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            if example.get("task") != "summarise":
                continue
            messages = example["messages"][:-1]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts.append(prompt)
            if len(prompts) >= n_samples:
                break
    if not prompts:
        raise ValueError(f"No summarise prompts found in {path}")
    return prompts


def count_new_tokens(tokenizer: AutoTokenizer, prompt: str, text: str) -> int:
    full = tokenizer(prompt + text, add_special_tokens=False)["input_ids"]
    prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    return max(0, len(full) - len(prefix))


def run_standard(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int,
) -> BenchmarkResult:
    latencies: list[float] = []
    token_counts: list[int] = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
        new_tokens = output_ids[0, input_len:]
        token_counts.append(len(new_tokens))

    mean_latency = sum(latencies) / len(latencies)
    total_tokens = sum(token_counts)
    total_sec = sum(latencies) / 1000
    return BenchmarkResult(
        name="Standard generation",
        mean_latency_ms=mean_latency,
        tokens_per_sec=total_tokens / total_sec,
        mean_output_tokens=total_tokens / len(token_counts),
    )


def run_speculative(
    draft_model: AutoModelForCausalLM,
    verify_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int,
    K: int,
) -> BenchmarkResult:
    latencies: list[float] = []
    token_counts: list[int] = []
    draft_proposed = 0
    draft_accepted = 0

    for prompt in prompts:
        stats: dict[str, int] = {}
        torch.cuda.synchronize()
        start = time.perf_counter()
        text = speculative_decode(
            draft_model,
            verify_model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            K=K,
            stats=stats,
        )
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
        token_counts.append(count_new_tokens(tokenizer, prompt, text))
        draft_proposed += stats.get("draft_tokens_proposed", 0)
        draft_accepted += stats.get("draft_tokens_accepted", 0)

    mean_latency = sum(latencies) / len(latencies)
    total_tokens = sum(token_counts)
    total_sec = sum(latencies) / 1000
    acceptance = draft_accepted / draft_proposed if draft_proposed else 0.0
    return BenchmarkResult(
        name="Custom speculative decoding",
        mean_latency_ms=mean_latency,
        tokens_per_sec=total_tokens / total_sec,
        mean_output_tokens=total_tokens / len(token_counts),
        acceptance_rate=acceptance,
    )


def run_hf_assisted(
    draft_model: AutoModelForCausalLM,
    verify_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int,
) -> BenchmarkResult:
    latencies: list[float] = []
    token_counts: list[int] = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(verify_model.device)
        input_len = inputs["input_ids"].shape[1]
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            output_ids = verify_model.generate(
                **inputs,
                assistant_model=draft_model,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
        new_tokens = output_ids[0, input_len:]
        token_counts.append(len(new_tokens))

    mean_latency = sum(latencies) / len(latencies)
    total_tokens = sum(token_counts)
    total_sec = sum(latencies) / 1000
    return BenchmarkResult(
        name="HF assisted generation",
        mean_latency_ms=mean_latency,
        tokens_per_sec=total_tokens / total_sec,
        mean_output_tokens=total_tokens / len(token_counts),
    )


def print_results(results: list[BenchmarkResult]) -> None:
    baseline = results[0].mean_latency_ms
    print("\nLatency benchmark (T4)")
    print(f"{'Method':<32} {'Latency (ms)':>14} {'Tokens/sec':>12} {'Speedup':>10}")
    print("-" * 72)
    for result in results:
        speedup = baseline / result.mean_latency_ms
        print(
            f"{result.name:<32} "
            f"{result.mean_latency_ms:>14.1f} "
            f"{result.tokens_per_sec:>12.1f} "
            f"{speedup:>9.2f}x"
        )
        if result.acceptance_rate is not None:
            print(f"  draft-token acceptance rate: {result.acceptance_rate:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark generation latency.")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--split_path", type=str, default=VAL_DATA)
    args = parser.parse_args()

    require_cuda()
    if not Path(args.split_path).exists():
        raise FileNotFoundError(f"Missing {args.split_path}. Run `python data/prepare.py` first.")
    if not Path(args.adapter_path).exists() and "/" not in args.adapter_path:
        raise FileNotFoundError(f"Missing adapter at {args.adapter_path}")

    tokenizer = load_tokenizer()
    prompts = load_prompts(args.split_path, args.n_samples, tokenizer)
    print(f"Loaded {len(prompts)} summarise prompts from {args.split_path}")

    print("Loading verify model (4B + adapter)...")
    verify_model = load_verify_model(args.adapter_path)
    print("Loading draft model (0.6B)...")
    draft_model = load_draft_model()

    # Warmup
    warmup_prompt = prompts[0]
    _ = speculative_decode(
        draft_model, verify_model, tokenizer, warmup_prompt, max_new_tokens=16, K=args.K
    )

    results = [
        run_standard(verify_model, tokenizer, prompts, args.max_new_tokens),
        run_speculative(
            draft_model, verify_model, tokenizer, prompts, args.max_new_tokens, args.K
        ),
        run_hf_assisted(draft_model, verify_model, tokenizer, prompts, args.max_new_tokens),
    ]
    print_results(results)


if __name__ == "__main__":
    main()
