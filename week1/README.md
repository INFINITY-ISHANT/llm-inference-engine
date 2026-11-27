# Week 1: Foundations, Profiling, and First Kernels

This folder contains the three hands-on checkpoints for Week 1. Each script is a starter skeleton with `TODO` blocks and built-in checks. Finish them in order before moving on to the paged KV cache and scheduler in Week 2.

Default test model: `Qwen/Qwen3-0.6B` (28 layers, 16 query heads, 8 KV heads, ~1.5 GB in bfloat16).

## Setup

Create a virtual environment and install the dependencies from the repo root:

```bash
pip install -r requirements.txt
```

Make sure PyTorch sees your GPU before starting:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Checkpoints

### 1. Naive Baseline and Profiling (`01_baseline.py`)
Goal: Build an autoregressive greedy decoding loop **without** KV caching and measure how latency scales with prompt length.

What you will implement:
- Naive decode loop that feeds the entire sequence back into the model at every step (`use_cache=False`).
- Accurate GPU timing using `torch.cuda.synchronize()` and `torch.cuda.Event`.
- Sweep across prompt lengths (`64`, `256`, `1024`) to measure Time-to-First-Token (TTFT) and decode throughput (tokens/sec).

Run:
```bash
python week1/01_baseline.py
```

### 2. Manual KV Cache and GQA Attention (`02_kv_cache.py`)
Goal: Understand how KV caching cuts decode complexity from quadratic to linear, verify exact numerical equivalence, and implement a pre-allocated KV buffer from scratch.

What you will implement:
- **Part A (End-to-End Cached Decode):** Split generation into a prefill pass and a single-token decode loop using `past_key_values`. Assert that generated token IDs match Hugging Face `model.generate(do_sample=False)` bit-for-bit, and compare tokens/sec against Checkpoint 1.
- **Part B (Pre-Allocated Cache + GQA Attention):** Implement a standalone `StaticKVCache` tensor buffer with shape `(batch, num_kv_heads, max_seq_len, head_dim)` and a single-step Grouped Query Attention decode function. Pass the built-in `torch.allclose` test against full causal attention.

Run:
```bash
python week1/02_kv_cache.py
```

### 3. Triton Warmup Kernels (`03_triton_warmup.py`)
Goal: Write your first two GPU kernels in Triton, verify them against PyTorch, and measure achieved memory bandwidth (GB/s).

What you will implement:
- **Exercise 1 (Vector Add):** Write `add_kernel` and its launch wrapper, check exactness against `x + y`, and compute achieved GB/s.
- **Exercise 2 (Fused Row-wise Softmax):** Write a numerically stable row-wise softmax kernel that loads each row into SRAM once instead of multiple HBM passes. Pass the irregular-shape correctness test `(1823, 781)` and compare bandwidth against naive PyTorch softmax and `torch.softmax`.

Run:
```bash
python week1/03_triton_warmup.py
```

## Week 1 Sign-Off Checklist

- [ ] All assertions passing in `01_baseline.py`, `02_kv_cache.py`, and `03_triton_warmup.py`
- [ ] A short table in your PR description comparing naive vs. KV-cached TTFT and tokens/sec across prompt lengths `64`, `256`, `1024`
- [ ] Bandwidth numbers (GB/s) for your Triton Vector Add and Fused Softmax kernels vs. PyTorch
