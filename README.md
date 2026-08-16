# UdaciHeadline — LLM Inference Optimization

Course repository for **LLM Inference Optimization** (Udacity `cd14455`). It contains the four
lesson exercise notebooks and the capstone project **UdaciHeadline**: optimising a
Llama‑3.2‑1B headline‑generation pipeline (baseline → KV‑cache → pruning → quantization →
distributed inference → speculative decoding) and reporting the latency / throughput /
memory / ROUGE trade‑offs.

## Repository layout

```
.
├── dataset/                              News Category Dataset (HuffPost, JSON lines)
├── lesson-1-Introduction_to_LLM_Inference_Optimization/
├── lesson-2-Transformer_Architecture_Optimizations /
├── lesson-3-Quantization_Pruning_and_Speculative_Decoding/
├── lesson-4-Model_Parallelism_Sharding_and_Deployment/
│   └── exercises/{starter,solution}/     starter notebooks (solved in-place) + reference solutions
├── project/
│   ├── UdaciHeadline_Project_Starter.ipynb   the project notebook (all steps + benchmarks)
│   └── README.md
├── requirements.txt                      Python dependencies (torch installed separately, see below)
├── setup_env.sh                          one-shot environment bootstrap (uv or venv)
└── env_check.py                          verifies packages, hardware, dataset and model cache
```

## Environment setup

Tested with Python 3.12, PyTorch 2.13, transformers 5.15, datasets 5.0, evaluate 0.4.6,
accelerate 1.14, bitsandbytes 0.50, optimum‑quanto 0.2.7, DeepSpeed 0.19 (CPU accelerator).

```bash
# 1. Create .venv and install everything (auto-detects CUDA; pass "cpu" or "cuda" to force)
./setup_env.sh

# 2. Verify
.venv/bin/python env_check.py

# 3. Launch notebooks with the registered kernel "Python (udaciheadline)"
.venv/bin/jupyter lab
```

Manual equivalent:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch   # or plain `torch` for CUDA
uv pip install --python .venv/bin/python -r requirements.txt
DS_BUILD_OPS=0 uv pip install --python .venv/bin/python deepspeed   # optional
```

### Model & data

* **Model:** `unsloth/Llama-3.2-1B` — an ungated mirror of Meta's `meta-llama/Llama-3.2-1B`
  weights (identical tensors) that downloads without an access token. If you have Meta access
  set `UDACI_MODEL=meta-llama/Llama-3.2-1B` (or a local path such as
  `/voc/shared/models/llama/Llama-3.2-1B` in the Udacity workspace).
* **Dataset:** `dataset/News_Category_Dataset.json` (≈210k HuffPost articles with
  `headline` and `short_description`), loaded with `datasets.load_dataset("json", ...)`.

### Hardware notes

Everything in this repo runs on CPU (that is how the results in this repo were produced —
see the hardware table in the project report) but a CUDA GPU is strongly recommended:

| Feature | CPU | CUDA GPU |
|---|---|---|
| Baseline / KV cache / pruning | ✅ (fp32, slow) | ✅ (bf16) |
| bitsandbytes 8‑bit / 4‑bit | experimental (falls back to `optimum-quanto` int8/int4) | ✅ |
| Tensor / pipeline parallelism | simulated with `accelerate` device maps | ✅ multi‑GPU |
| Speculative decoding | ✅ | ✅ |
| DeepSpeed pipeline (Lesson 4) | CPU accelerator, gloo backend | ✅ |

## Running the tests / notebooks headlessly

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    "lesson-1-Introduction_to_LLM_Inference_Optimization/exercises/starter/L1E1 Profiling Your First LLM Starter.ipynb"
```

## Project

See [`project/README.md`](project/README.md) for the project instructions, results and the
final report.

## Built With

* [PyTorch](https://pytorch.org/) — tensor library, profiler, `torch.nn.utils.prune`
* [Transformers](https://huggingface.co/docs/transformers) / [Accelerate](https://huggingface.co/docs/accelerate) — model loading, `generate()`, device maps
* [Datasets](https://huggingface.co/docs/datasets) / [Evaluate](https://huggingface.co/docs/evaluate) — data loading and ROUGE
* [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) / [optimum‑quanto](https://github.com/huggingface/optimum-quanto) — post‑training quantization
* [DeepSpeed](https://www.deepspeed.ai/) — pipeline parallelism (Lesson 4)

## License

[License](LICENSE.md)
