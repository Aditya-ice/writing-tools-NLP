# writing-tools-nlp

Fine-tuned **Qwen3-4B-Instruct-2507** with QLoRA adapters for Apple Intelligence–style writing tools: summarisation, rewriting, and smart reply. Implements **adapters**, **speculative decoding**, **guided generation**, and **on-device MLX inference** on Apple Silicon.

**Model:** [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) + LoRA (r=16, α=32)  
**Training:** QLoRA 4-bit NF4, ~3.6k SFT examples, Colab T4, 1 epoch  
**Repo:** [github.com/Aditya-ice/writing-tools-NLP](https://github.com/Aditya-ice/writing-tools-NLP)

## Evaluation (val split, n=400)

| Metric | Base Qwen3-4B | Fine-tuned |
|---|---|---|
| ROUGE-1 | 0.3877 | **0.5359** |
| ROUGE-2 | 0.1905 | **0.3788** |
| ROUGE-L | 0.3139 | **0.4838** |
| BERTScore-F1 | 0.8756 | **0.9136** |

Fine-tuned adapter beats base on all four metrics. BERTScore computed on a 200-example subset.

## Latency benchmark (T4)

| Method | Latency (ms) | Tokens/sec | Speedup |
|---|---|---|---|
| Standard generation | — | — | 1.0× |
| Custom speculative decoding | — | — | — |
| HF assisted generation (reference) | — | — | — |

## On-device (MacBook Air M1, MLX 4-bit)

| Model | Tokens/sec | Peak memory |
|---|---|---|
| Fine-tuned Qwen3-4B | — | — |

## Quick start

```bash
pip install -r requirements.txt
python data/prepare.py
python train/train.py          # Colab T4 only
python train/eval.py --adapter_path ./adapter --split val
python inference/pipeline.py --task summarise --input "..." --adapter_path ./adapter
```

## HuggingFace assets

| Asset | URL |
|---|---|
| Model adapter | [aditya-ice/writing-tools-qwen3](https://huggingface.co/aditya-ice/writing-tools-qwen3) |
| MLX 4-bit model | [aditya-ice/writing-tools-qwen3-mlx-4bit](https://huggingface.co/aditya-ice/writing-tools-qwen3-mlx-4bit) |
| Spaces demo | [aditya-ice/writing-tools-nlp](https://huggingface.co/spaces/aditya-ice/writing-tools-nlp) |

## Status

- [x] Data pipeline + QLoRA training
- [x] Eval: ROUGE + BERTScore (base vs fine-tuned)
- [ ] Speculative decoding benchmark
- [ ] Guided generation (Smart Reply JSON)
- [ ] MLX 4-bit on-device benchmark
- [ ] Gradio demo + HuggingFace Spaces
