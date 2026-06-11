#!/usr/bin/env python3
"""Download and format datasets for writing-tools SFT.

Sources:
  - CNN/DailyMail  → summarisation
  - PAWS (paraphrase) → rewrite (formal/casual tone labels)
  - Enron emails   → smart reply (parsed reply / quoted-context pairs)

Writes data/train.jsonl and data/val.jsonl (~4k examples total).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset
from transformers import AutoTokenizer

# Allow `python data/prepare.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.config import MODEL_ID  # noqa: E402

# Single source of truth — imported by app.py at inference time.
TASK_INSTRUCTIONS: dict[str, str] = {
    "summarise": (
        "You are a writing assistant. Summarise the following text in one "
        "concise paragraph."
    ),
    "rewrite": (
        "You are a writing assistant. Rewrite the following text in a {tone} "
        "tone while preserving the meaning."
    ),
    "smart_reply": (
        "You are a writing assistant. Draft a smart reply to the following "
        "email. Respond with valid JSON containing keys: subject, body, tone "
        "(one of: formal, casual, friendly)."
    ),
}

DATASET_COUNTS = {
    "summarise": 2000,
    "rewrite": 1000,
    "smart_reply": 1000,
}
VAL_RATIO = 0.1
RANDOM_SEED = 42

# Reserve tokens for system prompt, chat template overhead, and assistant output.
MAX_INPUT_TOKENS = 1024
MAX_OUTPUT_TOKENS = 384

QUOTE_PATTERNS = [
    re.compile(r'\n\s*"[^"]+"\s*<[^>]+>\s+on\s+\d', re.IGNORECASE),
    re.compile(r"\n-+\s*Forwarded by", re.IGNORECASE),
    re.compile(r"\n-+\s*Original Message\s*-+", re.IGNORECASE),
]


def truncate_to_tokens(text: str, tokenizer: AutoTokenizer, max_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def build_messages(
    task: str,
    user_content: str,
    assistant_content: str,
    *,
    tone: str | None = None,
) -> list[dict[str, str]]:
    if task == "rewrite":
        if tone is None:
            raise ValueError("rewrite examples require a tone")
        system = TASK_INSTRUCTIONS["rewrite"].format(tone=tone)
    else:
        system = TASK_INSTRUCTIONS[task]

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def example_record(
    task: str,
    user_content: str,
    assistant_content: str,
    *,
    tone: str | None = None,
) -> dict[str, Any]:
    return {
        "task": task,
        "messages": build_messages(task, user_content, assistant_content, tone=tone),
    }


def infer_tone(text: str) -> str:
    lower = text.lower()
    if any(w in lower for w in ("dear ", "sincerely", "regards", "respectfully")):
        return "formal"
    if any(w in lower for w in ("hey", "thanks!", "cheers", "cool", "awesome")):
        return "casual"
    return "friendly"


def extract_subject(block: str) -> str | None:
    for line in block.splitlines():
        if line.lower().startswith("subject:"):
            return line.split(":", 1)[1].strip()
    return None


def split_reply_email(text: str) -> tuple[str, str, str | None] | None:
    """Split Enron-style email into (reply_body, incoming_context, subject)."""
    text = text.strip()
    if len(text) < 120:
        return None

    split_at = None
    for pattern in QUOTE_PATTERNS:
        match = pattern.search(text)
        if match:
            split_at = match.start()
            break

    if split_at is None or split_at < 40:
        return None

    reply_body = text[:split_at].strip()
    incoming = text[split_at:].strip()
    subject = extract_subject(incoming)

    if len(reply_body) < 30 or len(incoming) < 80:
        return None

    return reply_body, incoming, subject


def load_summarise_examples(
    tokenizer: AutoTokenizer, n: int
) -> Iterator[dict[str, Any]]:
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train", streaming=True)
    count = 0
    for row in ds:
        article = truncate_to_tokens(row["article"], tokenizer, MAX_INPUT_TOKENS)
        summary = truncate_to_tokens(row["highlights"], tokenizer, MAX_OUTPUT_TOKENS)
        if len(article) < 200 or len(summary) < 40:
            continue
        yield example_record("summarise", article, summary)
        count += 1
        if count >= n:
            break


def load_rewrite_examples(tokenizer: AutoTokenizer, n: int) -> Iterator[dict[str, Any]]:
    ds = load_dataset(
        "google-research-datasets/paws",
        "labeled_final",
        split="train",
        streaming=True,
    )
    tones = ["formal", "casual"]
    count = 0
    tone_idx = 0
    for row in ds:
        if row["label"] != 1:
            continue
        source = truncate_to_tokens(row["sentence1"], tokenizer, MAX_INPUT_TOKENS)
        target = truncate_to_tokens(row["sentence2"], tokenizer, MAX_OUTPUT_TOKENS)
        if len(source) < 40 or len(target) < 40:
            continue
        tone = tones[tone_idx % len(tones)]
        tone_idx += 1
        yield example_record("rewrite", source, target, tone=tone)
        count += 1
        if count >= n:
            break


def load_smart_reply_examples(
    tokenizer: AutoTokenizer, n: int
) -> Iterator[dict[str, Any]]:
    ds = load_dataset("LLM-PBE/enron-email", split="train", streaming=True)
    count = 0
    for row in ds:
        parsed = split_reply_email(row["text"])
        if parsed is None:
            continue
        reply_body, incoming, subject = parsed
        incoming = truncate_to_tokens(incoming, tokenizer, MAX_INPUT_TOKENS)
        reply_body = truncate_to_tokens(reply_body, tokenizer, MAX_OUTPUT_TOKENS)
        tone = infer_tone(reply_body)
        subj = subject or "Re: your email"
        if not subj.lower().startswith("re:"):
            subj = f"Re: {subj}"
        assistant = json.dumps(
            {"subject": subj, "body": reply_body, "tone": tone},
            ensure_ascii=False,
        )
        yield example_record("smart_reply", incoming, assistant)
        count += 1
        if count >= n:
            break


def collect_examples(tokenizer: AutoTokenizer) -> list[dict[str, Any]]:
    loaders = {
        "summarise": load_summarise_examples,
        "rewrite": load_rewrite_examples,
        "smart_reply": load_smart_reply_examples,
    }
    examples: list[dict[str, Any]] = []
    for task, target_n in DATASET_COUNTS.items():
        print(f"Loading {target_n} {task} examples...")
        task_examples = list(loaders[task](tokenizer, target_n))
        print(f"  collected {len(task_examples)}")
        examples.extend(task_examples)
    return examples


def split_train_val(
    examples: list[dict[str, Any]], val_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_task: dict[str, list[dict[str, Any]]] = {}
    for ex in examples:
        by_task.setdefault(ex["task"], []).append(ex)

    train, val = [], []
    for task_examples in by_task.values():
        rng.shuffle(task_examples)
        n_val = max(1, int(len(task_examples) * val_ratio))
        val.extend(task_examples[:n_val])
        train.extend(task_examples[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT datasets.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data"),
        help="Directory for train.jsonl and val.jsonl",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    examples = collect_examples(tokenizer)
    train, val = split_train_val(examples, VAL_RATIO, args.seed)

    train_path = args.output_dir / "train.jsonl"
    val_path = args.output_dir / "val.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    print(f"\nWrote {len(train)} train + {len(val)} val examples")
    print(f"  {train_path}")
    print(f"  {val_path}")
    task_counts = {}
    for split_name, split in [("train", train), ("val", val)]:
        counts: dict[str, int] = {}
        for ex in split:
            counts[ex["task"]] = counts.get(ex["task"], 0) + 1
        task_counts[split_name] = counts
        print(f"  {split_name}: {counts}")


if __name__ == "__main__":
    main()
