#!/usr/bin/env python3
"""Latency benchmark: standard vs speculative vs HF assisted generation."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark generation latency.")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=50)
    args = parser.parse_args()
    raise NotImplementedError(f"Latency benchmark stub — n_samples={args.n_samples}")


if __name__ == "__main__":
    main()
