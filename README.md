# High-Throughput LLM Inference Engine (From Scratch)

Building a fast, single-GPU LLM inference engine from scratch in PyTorch and Triton for `Qwen3-8B` (using `Qwen3-0.6B` for fast local debugging).

Instead of wrapping existing serving frameworks, we implement the core systems primitives ourselves:
- **Paged KV Cache & Continuous Batching** to eliminate memory fragmentation
- **Custom Triton Kernels** for PagedAttention and quantized (INT4/INT8) matmuls
- **EAGLE-style Speculative Decoding** to lower single-request latency
- **Benchmarking Suite** measuring numerical correctness against PyTorch and throughput/latency against `vLLM`

## Prerequisites

You do **not** need prior CUDA, Triton, or LLM serving experience. You will pick those up during the project.

Before Week 1 starts, you should be comfortable with:
- Writing clean Python and basic PyTorch (tensor indexing, broadcasting, shapes, running a forward pass)
- High-level Transformer mechanics (Q/K/V projections, scaled dot-product attention, causal masking)
- Basic Git (branches, PRs) and navigating a Linux terminal

## Quick Refresher (If You Are Rusty on Prereqs)

- **PyTorch & Transformers:** Make sure you can trace exact tensor shapes `(batch, heads, seq_len, head_dim)` through a single causal self-attention block by hand.
- **GPU Basics:** Understand the difference between compute-bound (prefill) and memory-bandwidth-bound (decode) operations.

## Week 1: Foundations, Profiling & First Kernels

The goal of Week 1 is to understand *why* naive LLM generation is slow, measure it ourselves, and write our first working Triton kernels.

### 1. Reading & Core Concepts (Days 1-2)
- **Why decoding is memory-bound:** Read [Transformer Inference Arithmetic](https://kipply.github.io/blog/transformer-inference-arithmetic/) (focus on KV cache sizing and FLOPs vs. memory bandwidth).
- **How production serving fixes memory waste:** Read the [vLLM PagedAttention blog post](https://vllm.ai/blog/2023-06-20-vllm).
- **How a minimal fast engine is structured:** Skim the codebase of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (~1,200 lines of Python) and the [gpt-fast blog post](https://pytorch.org/blog/accelerating-generative-ai-2/).
- **Intro to GPU programming in Python:** Watch [GPU MODE Lecture 14: Practitioner's Guide to Triton](https://github.com/gpu-mode/lectures) or walk through the first two [Official Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/) (Vector Addition and Fused Softmax).

### 2. Hands-On Checkpoints (Days 3-6)
All Week 1 tasks live in `week1/` and run on `Qwen/Qwen3-0.6B`:

1. **Checkpoint 1 (`01_baseline.py`):** Write a naive autoregressive generation loop without KV caching. Measure Time-to-First-Token (TTFT) and tokens/sec across prompt lengths (64, 256, 1024 tokens) and verify why latency blows up quadratically.
2. **Checkpoint 2 (`02_kv_cache.py`):** Implement a manual KV cache in pure PyTorch (pre-allocated buffer, no `transformers` magic) and assert that generated token IDs match Hugging Face `model.generate()` bit-for-bit on greedy decoding. Log the speedup.
3. **Checkpoint 3 (`03_triton_warmup.py`):** Write and benchmark a Triton Vector Addition kernel and a Fused Softmax kernel against PyTorch, verifying outputs with `torch.allclose` and calculating achieved GPU memory bandwidth (GB/s).

### Week 1 Deliverable
By the end of Week 1, every team member should have:
- A working dev environment
- Recorded baseline numbers (naive vs. manual KV cache vs. `vLLM` on `Qwen3-0.6B`)
- Working Triton warmup kernels with bandwidth numbers

## Roadmap (Weeks 2-6)
- **Week 2:** Model runner, block manager, continuous batching scheduler (MVP)
- **Week 3:** Custom Triton PagedAttention kernel + numerical verification
- **Week 4:** CUDA Graphs + INT8/INT4 quantized matmul kernels
- **Week 5:** Speculative decoding integration + KV block rollback
- **Week 6:** Full `Qwen3-8B` benchmarks against `vLLM` and final write-up