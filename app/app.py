"""Gradio demo: Summarise / Rewrite / Smart Reply."""

from __future__ import annotations

import gradio as gr


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Writing Tools NLP") as demo:
        gr.Markdown("# Writing Tools NLP\nFine-tuned Qwen3-4B writing assistant.")
        gr.Markdown("_Demo stub — wire up inference/pipeline.py after training._")
    return demo


if __name__ == "__main__":
    build_app().launch()
