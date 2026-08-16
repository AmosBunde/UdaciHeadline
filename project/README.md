# UdaciHeadline — LLM Inference Optimization Project

Accelerating a Llama‑3.2‑1B headline‑generation pipeline for a news portal, step by step:
baseline → KV cache → pruning → quantization → distributed inference → speculative decoding, with a
systematic latency / throughput / memory / ROUGE benchmark of every step.

* **Notebook:** [`UdaciHeadline_Project_Starter.ipynb`](UdaciHeadline_Project_Starter.ipynb) — all code, executed outputs included
* **Final report:** [`REPORT.md`](REPORT.md) (also `REPORT.pdf`)
* **Raw results:** [`results/`](results/) — one JSON per configuration (metrics + every generated headline), `summary.csv`/`summary.md`, plots
* **Multi‑GPU helper:** [`distributed_benchmark.py`](distributed_benchmark.py) — run on a ≥2‑GPU box (e.g. SageMaker); its JSON is picked up by Section 6 of the notebook

## Getting Started

### Dependencies

Python 3.10+ with PyTorch, transformers, datasets, evaluate, rouge_score, accelerate, bitsandbytes,
pandas, matplotlib, psutil, jupyter (see the repository [`requirements.txt`](../requirements.txt)).

### Installation

```bash
# from the repository root
./setup_env.sh              # creates .venv (CPU or CUDA torch auto-detected)
.venv/bin/python env_check.py
```

### Model and data

| | Default | Override |
|---|---|---|
| 1B model (all sections) | `/voc/shared/models/llama/Llama-3.2-1B` if present, else `unsloth/Llama-3.2-1B` (ungated mirror of Meta's weights) | `UDACI_MODEL=...` |
| 3B target (speculative decoding) | `/voc/shared/models/llama/Llama-3.2-3B` if present, else `unsloth/Llama-3.2-3B` | `UDACI_TARGET_MODEL=...` |
| Dataset | `../dataset/News_Category_Dataset.json` | `UDACI_DATASET=...` |
| Eval samples per configuration | 20 | `UDACI_N_EVAL=...` |
| Speculative-decoding samples | 8 on CPU / `N_EVAL` on GPU | `UDACI_N_SPEC=...` |
| Recompute cached `results/*.json` | off | `UDACI_FORCE_RERUN=1` |

### Running

```bash
cd project
../.venv/bin/jupyter lab UdaciHeadline_Project_Starter.ipynb        # interactive
# or headless (CPU: ~1.5 h, GPU: a few minutes)
UDACI_FORCE_RERUN=1 ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 UdaciHeadline_Project_Starter.ipynb
```

Every section writes `results/<label>.json`; re-running the notebook re-uses those files unless
`UDACI_FORCE_RERUN=1`, so individual sections can be re-executed cheaply.

## Testing

`env_check.py` validates the environment. The notebook itself contains functional checks:
`validate_quantized_model()` (quantized layers present, finite logits, non-empty generation),
sparsity assertions after pruning, tokenizer‑vocabulary equality between draft and target model, and
identical‑output checks between baseline/KV‑cache and target/speculative decoding.

## Project Instructions

See the top-level task description in the notebook and the rubric summary in [`REPORT.md`](REPORT.md).

## Built With

* [PyTorch](https://pytorch.org/) — tensors, `torch.profiler`, `torch.nn.utils.prune`
* [Transformers](https://huggingface.co/docs/transformers) — Llama‑3.2 models, `generate()` (KV cache, `assistant_model`, `stop_strings`)
* [Accelerate](https://huggingface.co/docs/accelerate) — `device_map` sharding for tensor / pipeline parallelism
* [Datasets](https://huggingface.co/docs/datasets) / [Evaluate](https://huggingface.co/docs/evaluate) + `rouge_score` — data & ROUGE
* [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) — 8‑bit / 4‑bit post‑training quantization
* pandas / matplotlib / psutil — tables, plots, memory measurement

## License

[License](../LICENSE.md)
