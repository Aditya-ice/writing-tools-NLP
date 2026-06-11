#!/usr/bin/env python3
"""MLX on-device benchmark (Mac Apple Silicon only)."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MLX tokens/sec on M1.")
    parser.add_argument("--model_path", type=str, default="mlx_model")
    args = parser.parse_args()
    raise NotImplementedError(f"MLX benchmark stub — model_path={args.model_path}")


if __name__ == "__main__":
    main()
