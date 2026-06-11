#!/usr/bin/env python3
"""ROUGE + BERTScore evaluation: base vs fine-tuned."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model quality.")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--split", choices=["val", "train"], default="val")
    args = parser.parse_args()
    raise NotImplementedError(f"Eval stub — adapter={args.adapter_path}, split={args.split}")


if __name__ == "__main__":
    main()
