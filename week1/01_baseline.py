"""Week 1, Checkpoint 1: Naive Autoregressive Decoding Baseline.

Goal:
    Implement a greedy decoding loop WITHOUT KV caching (use_cache=False),
    measure Time-to-First-Token (TTFT) and decode throughput (tokens/sec),
    and observe why naive decoding degrades as sequence length grows.

Note:
    We use greedy decoding (argmax) here strictly for deterministic testing
    and apples-to-apples latency measurement across checkpoints.
"""

import argparse
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def make_prompt_ids(tokenizer, target_len: int, device: torch.device) -> torch.Tensor:
    """Create a synthetic prompt of exact token length `target_len`."""
    seed_text = (
        "Explain how GPU memory bandwidth bottlenecks autoregressive LLM decoding. "
    )
    ids = tokenizer.encode(seed_text, add_special_tokens=False)
    repeats = (target_len // len(ids)) + 1
    prompt_ids = (ids * repeats)[:target_len]
    return torch.tensor([prompt_ids], dtype=torch.long, device=device)


@torch.inference_mode()
def naive_greedy_generate(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> Tuple[torch.Tensor, float, float]:
    """Run naive greedy decoding with NO KV cache.

    Args:
        model: Loaded causal LM on GPU.
        input_ids: Tensor of shape (1, prompt_len).
        max_new_tokens: Number of new tokens to generate.

    Returns:
        generated_ids: Tensor of shape (1, prompt_len + max_new_tokens).
        ttft_ms: Time-to-first-token in milliseconds (first forward pass).
        decode_ms: Total time in milliseconds for the remaining (max_new_tokens - 1) steps.
    """
    # TODO 1:
    # Implement a greedy decode loop for `max_new_tokens` steps.
    # On EVERY step:
    #   - Pass the entire current sequence into `model(input_ids=cur_ids, use_cache=False)`.
    #   - Take the logits of the last token: `logits[:, -1, :]`.
    #   - Pick the next token with `torch.argmax(..., dim=-1, keepdim=True)`.
    #   - Concatenate the new token onto `cur_ids`.
    #
    # TODO 2:
    # Measure timing accurately on GPU using `torch.cuda.Event(enable_timing=True)`
    # and `torch.cuda.synchronize()`:
    #   - Record `ttft_ms` for step 0 (prefill + first token).
    #   - Record `decode_ms` for steps 1 .. max_new_tokens - 1.
    raise NotImplementedError("Implement naive_greedy_generate")


@torch.inference_mode()
def verify_against_hf(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 16,
) -> None:
    """Check that naive_greedy_generate matches Hugging Face greedy output."""
    hf_out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    ours_out, _, _ = naive_greedy_generate(model, input_ids, max_new_tokens)

    assert torch.equal(ours_out, hf_out), (
        f"Mismatch with Hugging Face generate!\n"
        f"HF:   {hf_out[0, -max_new_tokens:].tolist()}\n"
        f"Ours: {ours_out[0, -max_new_tokens:].tolist()}"
    )
    print("[PASS] Output matches Hugging Face greedy decoding bit-for-bit.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--prompt-lens", type=int, nargs="+", default=[64, 256, 1024]
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for Week 1 benchmarks.")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"Loading {args.model} on {torch.cuda.get_device_name(0)} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    # Warmup + correctness check
    warmup_ids = make_prompt_ids(tokenizer, 32, device)
    verify_against_hf(model, warmup_ids, max_new_tokens=16)

    print("\n--- Naive Baseline (No KV Cache) ---")
    print(f"{'Prompt Len':>10} | {'New Toks':>8} | {'TTFT (ms)':>10} | {'Decode (tok/s)':>14}")
    print("-" * 52)

    for plen in args.prompt_lens:
        input_ids = make_prompt_ids(tokenizer, plen, device)
        _, ttft_ms, decode_ms = naive_greedy_generate(
            model, input_ids, args.max_new_tokens
        )
        decode_tokens = max(args.max_new_tokens - 1, 1)
        tok_per_sec = decode_tokens / (decode_ms / 1000.0)
        print(f"{plen:>10} | {args.max_new_tokens:>8} | {ttft_ms:>10.2f} | {tok_per_sec:>14.2f}")


if __name__ == "__main__":
    main()
