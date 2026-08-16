#!/usr/bin/env python
"""Quick environment sanity check for the UdaciHeadline project.

Run:  python env_check.py
Prints library versions, hardware details and verifies that the dataset and
model used by the project can be resolved.  Exit code 0 == everything needed
for the CPU/GPU baseline is present.
"""
from __future__ import annotations

import importlib
import os
import platform
import sys

REQUIRED = [
    "torch", "transformers", "datasets", "evaluate", "accelerate",
    "rouge_score", "pandas", "numpy", "matplotlib", "psutil",
]
OPTIONAL = ["bitsandbytes", "optimum.quanto", "deepspeed", "nbformat", "nbconvert"]

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "News_Category_Dataset.json")
MODEL_NAME = os.environ.get("UDACI_MODEL", "unsloth/Llama-3.2-1B")


def check_import(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
        return True, getattr(mod, "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ok = True
    print(f"Python      : {sys.version.split()[0]} ({platform.platform()})")
    print(f"CPU         : {os.cpu_count()} logical cores")

    print("\n== Required packages ==")
    for name in REQUIRED:
        present, info = check_import(name)
        ok &= present
        print(f"  {'OK ' if present else 'MISSING'}  {name:<14} {info}")

    print("\n== Optional packages ==")
    for name in OPTIONAL:
        present, info = check_import(name)
        print(f"  {'OK ' if present else '-- '}  {name:<14} {info}")

    import torch  # noqa: E402

    print("\n== Hardware ==")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  cuda:{i} {p.name}  {p.total_memory / 1e9:.1f} GB")
    else:
        try:
            import psutil

            print(f"  RAM available: {psutil.virtual_memory().available / 1e9:.1f} GB "
                  f"of {psutil.virtual_memory().total / 1e9:.1f} GB")
        except Exception:  # noqa: BLE001
            pass
        print("  No CUDA device: notebooks will run on CPU (slow but functional).")

    print("\n== Data / model ==")
    if os.path.exists(DATASET_PATH):
        print(f"  OK  dataset found: {DATASET_PATH}")
    else:
        ok = False
        print(f"  MISSING dataset: {DATASET_PATH}")

    try:
        from huggingface_hub import try_to_load_from_cache

        cfg = try_to_load_from_cache(MODEL_NAME, "config.json")
        if isinstance(cfg, str):
            print(f"  OK  model cached: {MODEL_NAME}")
        else:
            print(f"  --  model not cached yet: {MODEL_NAME} (will download on first use)")
    except Exception as exc:  # noqa: BLE001
        print(f"  --  could not query HF cache: {exc}")

    print("\nRESULT:", "environment OK" if ok else "environment INCOMPLETE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
