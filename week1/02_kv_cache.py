"""Week 1, Checkpoint 2: KV-Cached Decoding and Manual Pre-Allocated Cache.

Part A:
    Split autoregressive decoding into a prefill phase and a single-token
    decode phase using `past_key_values`. Verify exact token match against
    Hugging Face and compare throughput against Checkpoint 1.

Part B:
    Implement a standalone pre-allocated `StaticKVCache` and a single-step
    Grouped Query Attention (GQA) decode function in pure PyTorch. This builds
    the exact memory layout intuition needed for PagedAttention in Week 2.
"""

import argparse
import math
from typing import Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


# =============================================================================
# Part A: End-to-End KV-Cached Generation Loop
# =============================================================================

@torch.inference_mode()
def cached_greedy_generate(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> Tuple[torch.Tensor, float, float]:
    """Run greedy decoding with KV caching split into prefill + decode steps.

    Args:
        model: Loaded causal LM on GPU.
        input_ids: Tensor of shape (1, prompt_len).
        max_new_tokens: Number of new tokens to generate.

    Returns:
        generated_ids: Tensor of shape (1, prompt_len + max_new_tokens).
        ttft_ms: Time-to-first-token in milliseconds (prefill pass).
        decode_ms: Total decode time in milliseconds for the remaining tokens.
    """
    # TODO A1 (Prefill):
    # Run ONE forward pass on the full `input_ids` with `use_cache=True`.
    # Save `past_key_values = outputs.past_key_values` and grab the first
    # predicted token via argmax on `outputs.logits[:, -1, :]`.
    # Time this pass as `ttft_ms`.
    #
    # TODO A2 (Decode loop):
    # For the remaining `max_new_tokens - 1` steps:
    #   - Feed ONLY the single most recent token `next_token` (shape (1, 1))
    #     along with `past_key_values=past_key_values` and `use_cache=True`.
    #   - Update `past_key_values = outputs.past_key_values`.
    #   - Sample the next greedy token and append it.
    # Time the entire decode loop as `decode_ms`.
    raise NotImplementedError("Implement cached_greedy_generate")


# =============================================================================
# Part B: Pure-PyTorch Pre-Allocated KV Cache + GQA Attention
# =============================================================================

class StaticKVCache:
    """Pre-allocated contiguous KV buffer for a single transformer layer.

    Layout:
        k_cache, v_cache: (batch_size, num_kv_heads, max_seq_len, head_dim)
    """

    def __init__(
        self,
        batch_size: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.seq_len = 0
        # TODO B1:
        # Pre-allocate `self.k_cache` and `self.v_cache` as zero tensors of shape
        # (batch_size, num_kv_heads, max_seq_len, head_dim) on `device` with `dtype`.
        raise NotImplementedError("Allocate self.k_cache and self.v_cache")

    def update(
        self, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Write new K/V tokens into the pre-allocated buffer and return active slices.

        Args:
            k_new: (batch_size, num_kv_heads, new_tokens, head_dim)
            v_new: (batch_size, num_kv_heads, new_tokens, head_dim)

        Returns:
            k_all, v_all: Views covering positions [0 : self.seq_len] after insertion.
        """
        # TODO B2:
        # 1. Check that `self.seq_len + new_tokens <= self.max_seq_len`.
        # 2. Copy `k_new` and `v_new` in-place into the next free slice along dim=2.
        # 3. Advance `self.seq_len`.
        # 4. Return slices `self.k_cache[:, :, :self.seq_len, :]` and `self.v_cache[:, :, :self.seq_len, :]`.
        raise NotImplementedError("Implement StaticKVCache.update")


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match query heads for Grouped Query Attention (GQA).

    Args:
        x: Tensor of shape (batch, num_kv_heads, seq_len, head_dim).
        n_rep: Number of query heads per KV head (num_q_heads // num_kv_heads).

    Returns:
        Tensor of shape (batch, num_kv_heads * n_rep, seq_len, head_dim).
    """
    if n_rep == 1:
        return x
    # TODO B3:
    # Expand each KV head `n_rep` times without unnecessary copies where possible,
    # and reshape to (batch, num_kv_heads * n_rep, seq_len, head_dim).
    raise NotImplementedError("Implement repeat_kv")


def gqa_attention_step(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    cache: StaticKVCache,
) -> torch.Tensor:
    """Run one causal GQA attention step (prefill or single-token decode) using StaticKVCache.

    Args:
        q:     (batch, num_q_heads, q_len, head_dim)
        k_new: (batch, num_kv_heads, q_len, head_dim)
        v_new: (batch, num_kv_heads, q_len, head_dim)
        cache: StaticKVCache holding previous tokens.

    Returns:
        out:   (batch, num_q_heads, q_len, head_dim)
    """
    # TODO B4:
    # 1. Update `cache` with (k_new, v_new) to get (k_all, v_all).
    # 2. Expand KV heads with `repeat_kv` so K and V match `q.shape[1]`.
    # 3. Compute scaled dot-product attention scores: (q @ k^T) / sqrt(head_dim).
    # 4. If q_len > 1 (prefill), apply a causal mask so query i cannot attend to keys > i.
    #    (For single-token decode where q_len == 1, the new token can attend to all cached keys.)
    # 5. Softmax over the key dimension (in float32 for stability, then cast back) and multiply by V.
    raise NotImplementedError("Implement gqa_attention_step")


def test_part_b() -> None:
    """Verify StaticKVCache + GQA decode matches full causal attention numerically."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    batch_size = 2
    num_q_heads = 16   # Matches Qwen3-0.6B
    num_kv_heads = 8   # Matches Qwen3-0.6B (2:1 GQA ratio)
    head_dim = 64
    prompt_len = 12
    decode_steps = 8
    total_len = prompt_len + decode_steps

    q_full = torch.randn(batch_size, num_q_heads, total_len, head_dim, device=device, dtype=dtype)
    k_full = torch.randn(batch_size, num_kv_heads, total_len, head_dim, device=device, dtype=dtype)
    v_full = torch.randn(batch_size, num_kv_heads, total_len, head_dim, device=device, dtype=dtype)

    # Reference: full causal attention computed all at once
    n_rep = num_q_heads // num_kv_heads
    k_ref = k_full.repeat_interleave(n_rep, dim=1)
    v_ref = v_full.repeat_interleave(n_rep, dim=1)
    out_ref = F.scaled_dot_product_attention(q_full, k_ref, v_ref, is_causal=True)

    # Cached execution: prefill prompt_len tokens, then decode 1 token at a time
    cache = StaticKVCache(batch_size, num_kv_heads, max_seq_len=64, head_dim=head_dim, device=device, dtype=dtype)
    outputs = []

    out_prefill = gqa_attention_step(
        q_full[:, :, :prompt_len, :],
        k_full[:, :, :prompt_len, :],
        v_full[:, :, :prompt_len, :],
        cache,
    )
    outputs.append(out_prefill)

    for pos in range(prompt_len, total_len):
        out_step = gqa_attention_step(
            q_full[:, :, pos : pos + 1, :],
            k_full[:, :, pos : pos + 1, :],
            v_full[:, :, pos : pos + 1, :],
            cache,
        )
        outputs.append(out_step)

    out_cached = torch.cat(outputs, dim=2)
    max_diff = (out_cached - out_ref).abs().max().item()
    assert torch.allclose(out_cached, out_ref, atol=1e-5, rtol=1e-4), (
        f"Part B mismatch! Max abs diff: {max_diff:.6e}"
    )
    print(f"[PASS] Part B: StaticKVCache + GQA matches reference (max diff = {max_diff:.2e}).")


def make_prompt_ids(tokenizer, target_len: int, device: torch.device) -> torch.Tensor:
    seed_text = (
        "Explain how GPU memory bandwidth bottlenecks autoregressive LLM decoding. "
    )
    ids = tokenizer.encode(seed_text, add_special_tokens=False)
    repeats = (target_len // len(ids)) + 1
    return torch.tensor([(ids * repeats)[:target_len]], dtype=torch.long, device=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt-lens", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--only-part-b", action="store_true", help="Run only the Part B unit test.")
    args = parser.parse_args()

    # Always run Part B unit test first
    test_part_b()
    if args.only_part_b:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for Part A benchmarks.")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"\nLoading {args.model} on {torch.cuda.get_device_name(0)} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    # Correctness check against HF generate
    warmup_ids = make_prompt_ids(tokenizer, 32, device)
    hf_out = model.generate(warmup_ids, max_new_tokens=16, do_sample=False, use_cache=True)
    ours_out, _, _ = cached_greedy_generate(model, warmup_ids, max_new_tokens=16)
    assert torch.equal(ours_out, hf_out), "Part A output does not match Hugging Face!"
    print("[PASS] Part A: Cached loop matches Hugging Face greedy decoding bit-for-bit.")

    print("\n--- KV-Cached Baseline ---")
    print(f"{'Prompt Len':>10} | {'New Toks':>8} | {'TTFT (ms)':>10} | {'Decode (tok/s)':>14}")
    print("-" * 52)

    for plen in args.prompt_lens:
        input_ids = make_prompt_ids(tokenizer, plen, device)
        _, ttft_ms, decode_ms = cached_greedy_generate(
            model, input_ids, args.max_new_tokens
        )
        decode_tokens = max(args.max_new_tokens - 1, 1)
        tok_per_sec = decode_tokens / (decode_ms / 1000.0)
        print(f"{plen:>10} | {args.max_new_tokens:>8} | {ttft_ms:>10.2f} | {tok_per_sec:>14.2f}")


if __name__ == "__main__":
    main()
