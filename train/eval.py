#!/usr/bin/env python3
"""ROUGE + BERTScore evaluation: base vs fine-tuned."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate
import torch

from inference.pipeline import WritingPipeline
from train.config import TRAIN_DATA, VAL_DATA


def load_examples(path: str, max_samples: int | None) -> list[dict]:
    examples: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
            if max_samples is not None and len(examples) >= max_samples:
                break
    return examples


def generate_predictions(
    pipeline: WritingPipeline,
    examples: list[dict],
    *,
    max_new_tokens: int,
) -> tuple[list[str], list[str]]:
    predictions: list[str] = []
    references: list[str] = []
    for i, example in enumerate(examples, start=1):
        pred = pipeline.generate_from_messages(
            example["messages"],
            max_new_tokens=max_new_tokens,
        )
        ref = example["messages"][-1]["content"]
        predictions.append(pred)
        references.append(ref)
        if i % 25 == 0 or i == len(examples):
            print(f"  generated {i}/{len(examples)}")
    return predictions, references


def compute_metrics(
    predictions: list[str],
    references: list[str],
    *,
    bertscore_samples: int,
    seed: int,
) -> dict[str, float]:
    rouge = evaluate.load("rouge")
    rouge_result = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )

    n = len(predictions)
    sample_size = min(bertscore_samples, n)
    rng = random.Random(seed)
    indices = rng.sample(range(n), sample_size) if n > sample_size else list(range(n))
    sub_preds = [predictions[i] for i in indices]
    sub_refs = [references[i] for i in indices]

    print(f"  BERTScore on {sample_size} examples...")
    bertscore = evaluate.load("bertscore")
    bert_result = bertscore.compute(
        predictions=sub_preds,
        references=sub_refs,
        lang="en",
        batch_size=8,
    )
    bert_f1 = sum(bert_result["f1"]) / len(bert_result["f1"])

    return {
        "rouge1": float(rouge_result["rouge1"]),
        "rouge2": float(rouge_result["rouge2"]),
        "rougeL": float(rouge_result["rougeL"]),
        "bertscore_f1": float(bert_f1),
    }


def evaluate_model(
    label: str,
    examples: list[dict],
    *,
    adapter_path: str | None,
    max_new_tokens: int,
    bertscore_samples: int,
    seed: int,
) -> dict[str, float]:
    print(f"\n{label}")
    pipeline = WritingPipeline(adapter_path=adapter_path)
    predictions, references = generate_predictions(
        pipeline,
        examples,
        max_new_tokens=max_new_tokens,
    )
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("  computing metrics...")
    return compute_metrics(
        predictions,
        references,
        bertscore_samples=bertscore_samples,
        seed=seed,
    )


def print_table(base: dict[str, float], finetuned: dict[str, float]) -> None:
    headers = ("Metric", "Base Qwen3-4B", "Fine-tuned")
    rows = [
        ("ROUGE-1", base["rouge1"], finetuned["rouge1"]),
        ("ROUGE-2", base["rouge2"], finetuned["rouge2"]),
        ("ROUGE-L", base["rougeL"], finetuned["rougeL"]),
        ("BERTScore-F1", base["bertscore_f1"], finetuned["bertscore_f1"]),
    ]

    print("\nEvaluation results")
    print(f"{headers[0]:<14} {headers[1]:>16} {headers[2]:>16}")
    print("-" * 48)
    for name, base_val, ft_val in rows:
        print(f"{name:<14} {base_val:>16.4f} {ft_val:>16.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model quality.")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit examples for a quick smoke test",
    )
    parser.add_argument("--bertscore_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = VAL_DATA if args.split == "val" else TRAIN_DATA
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Missing {data_path}. Run `python data/prepare.py` first.")
    if not Path(args.adapter_path).exists():
        raise FileNotFoundError(f"Missing adapter at {args.adapter_path}")

    examples = load_examples(data_path, args.max_samples)
    print(f"Evaluating on {len(examples)} {args.split} examples from {data_path}")

    base_metrics = evaluate_model(
        "Base model",
        examples,
        adapter_path=None,
        max_new_tokens=args.max_new_tokens,
        bertscore_samples=args.bertscore_samples,
        seed=args.seed,
    )
    finetuned_metrics = evaluate_model(
        "Fine-tuned model",
        examples,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        bertscore_samples=args.bertscore_samples,
        seed=args.seed,
    )
    print_table(base_metrics, finetuned_metrics)


if __name__ == "__main__":
    main()
