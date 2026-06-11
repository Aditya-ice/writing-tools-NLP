#!/usr/bin/env python3
"""Fuse LoRA adapter and convert to MLX 4-bit (Mac only)."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert fused model to MLX 4-bit.")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="mlx_model")
    args = parser.parse_args()
    raise NotImplementedError(f"MLX convert stub — adapter={args.adapter_path}")


if __name__ == "__main__":
    main()
