#!/usr/bin/env python3
"""Unified inference pipeline: loads adapter and runs any writing task."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run writing-tools inference.")
    parser.add_argument("--task", choices=["summarise", "rewrite", "smart_reply"], required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--tone", choices=["formal", "casual"], default="formal")
    args = parser.parse_args()
    raise NotImplementedError(f"Inference stub — task={args.task}")


if __name__ == "__main__":
    main()
