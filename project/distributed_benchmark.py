#!/usr/bin/env python
"""Multi-GPU benchmark for Section 6 of the UdaciHeadline notebook (tensor vs pipeline parallelism).

Run this on a machine with >= 2 GPUs (e.g. the Lesson-4 SageMaker notebook instance):

    python distributed_benchmark.py --n 20 --out results/distributed_multigpu.json

and copy the resulting JSON into project/results/.  The notebook's Section 6 loads it automatically and
prefers it over the single-device simulation.

Two placements are benchmarked with exactly the same prompt / decoding settings as the notebook:
  * tensor_parallel  : device_map="auto"  (accelerate shards weights across all visible GPUs)
  * pipeline_parallel: explicit layer map  (embeddings + first half of the decoder on GPU0, rest on GPU1 ...)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_LOCAL_1B = "/voc/shared/models/llama/Llama-3.2-1B"
MODEL_NAME = os.environ.get("UDACI_MODEL", _LOCAL_1B if os.path.isdir(_LOCAL_1B) else "unsloth/Llama-3.2-1B")
PROMPT = "You are a news editor. Write a short, catchy headline for each article summary.\n\n{examples}Summary: {summary}\nHeadline:"


def load_data(path, n, n_shot=2, seed=42):
    raw = load_dataset("json", data_files=path, split="train")
    ds = raw.filter(lambda ex: 15 <= len((ex.get("short_description") or "").split()) <= 80
                    and len((ex.get("headline") or "").split()) >= 4)
    ds = ds.rename_column("short_description", "summary").shuffle(seed=seed)
    fewshot = [ds[i] for i in range(n_shot)]
    block = "".join(f"Summary: {e['summary']}\nHeadline: {e['headline']}\n\n" for e in fewshot)
    return ds.select(range(n_shot, n_shot + n)), block


def pipeline_device_map(model_name, n_stages):
    cfg = AutoConfig.from_pretrained(model_name)
    L = cfg.num_hidden_layers
    per = math.ceil(L / n_stages)
    dm = {"model.embed_tokens": 0, "model.rotary_emb": 0}
    for i in range(L):
        dm[f"model.layers.{i}"] = min(i // per, n_stages - 1)
    dm["model.norm"] = n_stages - 1
    dm["lm_head"] = n_stages - 1
    return dm


def bench(model, tok, ds, block, max_new_tokens, label):
    rouge = load_metric("rouge")
    preds, refs, lats, toks = [], [], [], []
    torch.cuda.reset_peak_memory_stats()
    for i in range(len(ds)):
        prompt = PROMPT.format(examples=block, summary=ds[i]["summary"].strip())
        enc = tok(prompt, return_tensors="pt").to(model.device)
        args = dict(max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id,
                    stop_strings=["\n"], tokenizer=tok)
        if i == 0:
            model.generate(**enc, **args)  # warm-up
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = model.generate(**enc, **args)
        torch.cuda.synchronize(); lat = time.perf_counter() - t0
        new = out[0, enc["input_ids"].shape[1]:]
        text = tok.decode(new, skip_special_tokens=True).strip().split("\n")[0].strip().strip('"')
        preds.append(text); refs.append(ds[i]["headline"]); lats.append(lat)
        toks.append(int((new != tok.pad_token_id).sum().item()) or new.numel())
    lat = np.array(lats); r = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    peak = sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e9
    return {"label": label, "n_samples": len(ds), "max_new_tokens": max_new_tokens,
            "latency_mean_s": float(lat.mean()), "latency_std_s": float(lat.std()), "latency_p50_s": float(np.percentile(lat, 50)),
            "latency_p99_s": float(np.percentile(lat, 99)), "latency_min_s": float(lat.min()), "latency_max_s": float(lat.max()),
            "total_time_s": float(lat.sum()), "avg_new_tokens": float(np.mean(toks)),
            "throughput_tok_s": float(sum(toks) / lat.sum()), "throughput_samples_s": float(len(ds) / lat.sum()),
            "rouge1": float(r["rouge1"]), "rouge2": float(r["rouge2"]), "rougeL": float(r["rougeL"]), "rougeLsum": float(r["rougeLsum"]),
            "model_footprint_gb": model.get_memory_footprint() / 1e9, "peak_memory_gb": peak,
            "memory_kind": "sum of cuda max_memory_allocated over GPUs", "device": "cuda", "dtype": "torch.bfloat16",
            "device_map": {k: str(v) for k, v in model.hf_device_map.items()}, "n_gpus": torch.cuda.device_count(),
            "samples": [{"headline": p, "reference": rf, "new_tokens": t, "latency_s": l} for p, rf, t, l in zip(preds, refs, toks, lats)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--dataset", default=str(Path(__file__).resolve().parent.parent / "dataset" / "News_Category_Dataset.json"))
    ap.add_argument("--out", default="results/distributed_multigpu.json")
    a = ap.parse_args()
    assert torch.cuda.device_count() >= 2, "need at least 2 GPUs"

    ds, block = load_data(a.dataset, a.n)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME); tok.pad_token = tok.pad_token or tok.eos_token
    out = {"environment": {"gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                           "torch": torch.__version__, "model": MODEL_NAME}}

    print("== tensor parallel (device_map='auto') ==")
    m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="auto").eval()
    print(m.hf_device_map)
    out["tensor_parallel"] = bench(m, tok, ds, block, a.max_new_tokens, "tensor_parallel")
    print({k: out["tensor_parallel"][k] for k in ("latency_mean_s", "throughput_tok_s", "rouge1", "peak_memory_gb")})
    del m; torch.cuda.empty_cache()

    print("== pipeline parallel (layer map) ==")
    dm = pipeline_device_map(MODEL_NAME, torch.cuda.device_count())
    m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map=dm).eval()
    print(m.hf_device_map)
    out["pipeline_parallel"] = bench(m, tok, ds, block, a.max_new_tokens, "pipeline_parallel")
    print({k: out["pipeline_parallel"][k] for k in ("latency_mean_s", "throughput_tok_s", "rouge1", "peak_memory_gb")})

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    main()
