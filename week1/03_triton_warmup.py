"""Week 1, Checkpoint 3: Triton Warmup Kernels (Vector Add & Fused Softmax).

Goal:
    Learn the Triton SPMD programming model (program IDs, pointer arithmetic,
    block sizes, masking, and SRAM fusion) before writing PagedAttention in Week 3.

References:
    - https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
    - https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
"""

import torch
import triton
import triton.language as tl


# =============================================================================
# Exercise 1: Vector Addition
# =============================================================================

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Elementwise output = x + y."""
    # TODO 1:
    # 1. Get the 1D program ID with `pid = tl.program_id(axis=0)`.
    # 2. Compute `offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`.
    # 3. Create a bounds mask `mask = offsets < n_elements`.
    # 4. Load x and y with `tl.load(..., mask=mask)`.
    # 5. Compute `output = x + y` and write back with `tl.store(..., mask=mask)`.
    pass


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Host wrapper that allocates output and launches `add_kernel`."""
    assert x.is_cuda and y.is_cuda, "Inputs must be on CUDA."
    assert x.shape == y.shape, "Input shapes must match."

    output = torch.empty_like(x)
    n_elements = output.numel()

    # TODO 2:
    # Define `grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)`
    # and launch `add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)`.
    raise NotImplementedError("Launch add_kernel inside triton_add")


# =============================================================================
# Exercise 2: Fused Row-Wise Softmax
# =============================================================================

def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """Unfused PyTorch softmax (multiple HBM reads/writes per row)."""
    x_max = x.max(dim=1, keepdim=True)[0]
    z = x - x_max
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1, keepdim=True)
    return numerator / denominator


@triton.jit
def fused_softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute row-wise softmax with a single HBM read and single HBM write per row.

    Assume one Triton program handles one row, and BLOCK_SIZE >= n_cols (next power of 2).
    """
    # TODO 3:
    # 1. `row_idx = tl.program_id(0)`
    # 2. Compute pointers for this row: `row_start_ptr = input_ptr + row_idx * input_row_stride`
    # 3. Compute `col_offsets = tl.arange(0, BLOCK_SIZE)` and `mask = col_offsets < n_cols`.
    # 4. Load the row with `other=-float("inf")` so masked-out columns do not affect max or exp.
    # 5. Subtract `tl.max(row, axis=0)` for numerical stability.
    # 6. Compute `numerator = tl.exp(row_minus_max)` and `denominator = tl.sum(numerator, axis=0)`.
    # 7. Store `numerator / denominator` to `output_ptr + row_idx * output_row_stride + col_offsets`.
    pass


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Host wrapper for `fused_softmax_kernel` on 2D tensors."""
    assert x.is_cuda and x.ndim == 2, "Input must be a 2D CUDA tensor."
    n_rows, n_cols = x.shape

    # Next power of 2 >= n_cols so the whole row fits in one Triton block
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    output = torch.empty_like(x)

    # TODO 4:
    # Launch `fused_softmax_kernel[(n_rows,)](...)` with appropriate strides and BLOCK_SIZE.
    # Hint: tune `num_warps` (e.g., 4 or 8) if you want to explore occupancy.
    raise NotImplementedError("Launch fused_softmax_kernel inside triton_softmax")


# =============================================================================
# Verification & Bandwidth Benchmarks
# =============================================================================

def verify_and_benchmark() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required to run Triton kernels.")

    device = torch.device("cuda")
    print(f"Running Triton warmup on {torch.cuda.get_device_name(0)} (Triton {triton.__version__})\n")

    # --- Test 1: Vector Add ---
    torch.manual_seed(0)
    size = 98_432  # Non-power-of-2 to test boundary masking
    x = torch.randn(size, device=device, dtype=torch.float32)
    y = torch.randn(size, device=device, dtype=torch.float32)

    ref_add = x + y
    out_add = triton_add(x, y)
    assert torch.allclose(out_add, ref_add), (
        f"Vector add mismatch! Max diff: {(out_add - ref_add).abs().max().item()}"
    )
    print("[PASS] Exercise 1: Triton Vector Add matches PyTorch.")

    # Benchmark Vector Add on a large tensor (2^24 float32 elements = 64 MB per tensor)
    big_n = 2**24
    bx = torch.randn(big_n, device=device, dtype=torch.float32)
    by = torch.randn(big_n, device=device, dtype=torch.float32)

    ms_torch_add = triton.testing.do_bench(lambda: bx + by)
    ms_triton_add = triton.testing.do_bench(lambda: triton_add(bx, by))
    # 3 tensors touched: read x, read y, write output
    bytes_add = 3 * bx.numel() * bx.element_size()
    print(
        f"  Vector Add (N={big_n:,}) | "
        f"PyTorch: {bytes_add * 1e-9 / (ms_torch_add * 1e-3):7.1f} GB/s | "
        f"Triton: {bytes_add * 1e-9 / (ms_triton_add * 1e-3):7.1f} GB/s\n"
    )

    # --- Test 2: Fused Softmax ---
    # Irregular matrix dimensions to verify row strides and column masking
    s_in = torch.randn(1823, 781, device=device, dtype=torch.float32)
    ref_sm = torch.softmax(s_in, dim=1)
    out_sm = triton_softmax(s_in)
    assert torch.allclose(out_sm, ref_sm, atol=1e-5, rtol=1e-4), (
        f"Fused softmax mismatch! Max diff: {(out_sm - ref_sm).abs().max().item()}"
    )
    print("[PASS] Exercise 2: Triton Fused Softmax matches PyTorch.")

    # Benchmark Softmax across column widths
    M = 4096
    print(f"\n{'Cols (N)':>10} | {'Naive (GB/s)':>13} | {'Torch (GB/s)':>13} | {'Triton (GB/s)':>14}")
    print("-" * 58)
    for N in [256, 1024, 2048, 4096]:
        mat = torch.randn(M, N, device=device, dtype=torch.float32)
        bytes_sm = 2 * mat.numel() * mat.element_size()  # 1 read + 1 write ideal

        ms_naive = triton.testing.do_bench(lambda: naive_softmax(mat))
        ms_torch = triton.testing.do_bench(lambda: torch.softmax(mat, dim=1))
        ms_triton = triton.testing.do_bench(lambda: triton_softmax(mat))

        gbps_naive = bytes_sm * 1e-9 / (ms_naive * 1e-3)
        gbps_torch = bytes_sm * 1e-9 / (ms_torch * 1e-3)
        gbps_triton = bytes_sm * 1e-9 / (ms_triton * 1e-3)
        print(f"{N:>10} | {gbps_naive:>13.1f} | {gbps_torch:>13.1f} | {gbps_triton:>14.1f}")


if __name__ == "__main__":
    verify_and_benchmark()
