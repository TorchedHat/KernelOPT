# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused_convolution_min_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 66240 ns — **Best latency**: 62270 ns (**1.0638×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`vectorized_loads_with_contiguous_hints`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused_convolution_min_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_024/kernel_0_triton_per_fused_convolution_m/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (21600,1,1), Waves/SM: 10.23
- **Registers/thread**: 32
- **Duration**: 66240 ns
- **SM Throughput**: 33.24 % | **Memory**: 78.28 % | **DRAM**: 78.28 %
- **Achieved Occupancy**: 95.35 %
- **IPC Active**: 1.40 inst/cycle | **No Eligible**: 64.40 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.023× | 1.064× | ✓ | **✓ best** |
| B | autotune_xblock_and_pipeline_stages_for_memory_coalescing | 0.994× | 1.012× | ✓ |  |
| D | autotune_xblock_and_num_warps_for_cache_reuse | 0.990× | 0.998× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_xblock_a... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 66240 ns | 62270 ns | -3970 ns (-6.0%) | 65470 ns | -770 ns (-1.2%) | 66400 ns | +160 ns (+0.2%) |
| SM Throughput (%) | 33.24 % | 41.87 % | +8.63 pp | 37.56 % | +4.32 pp | 37.96 % | +4.72 pp |
| Memory Throughput (%) | 78.28 % | 83.29 % | +5.01 pp | 79.18 % | +0.90 pp | 78.03 % | -0.25 pp |
| DRAM Throughput (%) | 78.28 % | 83.29 % | +5.01 pp | 79.18 % | +0.90 pp | 78.03 % | -0.25 pp |
| Achieved Occupancy (%) | 95.35 % | 88.92 % | -6.43 pp | 95.39 % | +0.04 pp | 95.40 % | +0.05 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 10.23 | 10.23 | +0.00 | 10.23 | +0.00 | 10.23 | +0.00 |
| No Eligible (%) | 64.40 % | 62.24 % | -2.16 pp | 61.86 % | -2.54 pp | 62.35 % | -2.05 pp |
| IPC Active | 1.40 inst/cycle | 1.50 inst/cycle | +0.10 | 1.51 inst/cycle | +0.11 | 1.50 inst/cycle | +0.10 |


### Key Learnings

#### ✅ Plan A: `vectorized_loads_with_contiguous_hints` (1.064×)

When a Triton kernel is memory-bound with high L1TEX scoreboard stall cycles (CPIStall rule, ~17 avg stall cycles per warp) and low L1/TEX hit rate (~1%), adding tl.max_contiguous and tl.multiple_of hints to stride-1 index expressions enables the compiler to emit wider (LDG.128) vectorized memory instructions. This reduces warp cycles per issued instruction (44.2 → 37.6) and lowers kernel duration (~6%), while DRAM throughput increases as the bottleneck shifts from latency-bound scalar loads to bandwidth-bound vectorized loads. The L1/TEX cache throughput drops sharply (72% → 21%) because fewer, wider transactions replace many scalar ones.

> **Applicable when**: Triton kernel is memory-bound (DRAM SOL >> Compute SOL), CPIStall rule fires with high L1TEX stall cycles (>10 avg), L1/TEX hit rate is very low (<5%), and the load index has a stride-1 (contiguous) component across threads within a warp. Block size is a multiple of 4 (e.g., 128) ensuring block-level alignment.

> **Avoid when**: Index expression is not provably stride-1 contiguous across threads (e.g., strided or transposed access patterns). tl.multiple_of(expr, N) is only valid if expr is always a multiple of N — using too large an N (e.g., 16) when alignment is not guaranteed will produce incorrect results. Do not apply when the kernel is already compute-bound.

---

## Iteration 2

**Best so far**: 1.0638× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_for_latency_hiding | 1.019× | 1.069× | ✓ |  |
| B | evict_first_cache_policy_for_strided_reduction_loads | 1.010× | 1.032× | ✓ |  |
| C | evict_first_streaming_load_to_reduce_l2_pollution | 1.014× | 1.039× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | evict_first_cache... | Δ | evict_first_strea... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 66240 ns | 61950 ns | -4290 ns (-6.5%) | 64160 ns | -2080 ns (-3.1%) | 63780 ns | -2460 ns (-3.7%) |
| SM Throughput (%) | 33.24 % | 41.87 % | +8.63 pp | 38.51 % | +5.27 pp | 38.09 % | +4.85 pp |
| Memory Throughput (%) | 78.28 % | 83.84 % | +5.56 pp | 78.55 % | +0.27 pp | 79.02 % | +0.74 pp |
| DRAM Throughput (%) | 78.28 % | 83.84 % | +5.56 pp | 78.55 % | +0.27 pp | 79.02 % | +0.74 pp |
| Achieved Occupancy (%) | 95.35 % | 89.12 % | -6.23 pp | 95.17 % | -0.18 pp | 95.02 % | -0.33 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 10.23 | 10.23 | +0.00 | 10.23 | +0.00 | 10.23 | +0.00 |
| No Eligible (%) | 64.40 % | 61.52 % | -2.88 pp | 58.27 % | -6.13 pp | 59.62 % | -4.78 pp |
| IPC Active | 1.40 inst/cycle | 1.53 inst/cycle | +0.13 | 1.61 inst/cycle | +0.21 | 1.65 inst/cycle | +0.25 |


### Key Learnings

#### ✅ Plan A: `increase_num_warps_for_latency_hiding` (1.069×)

When a Triton kernel shows high no-eligible-warp cycles (>60%) driven by L1TEX scoreboard stalls (CPIStall rule, ~14 stall cycles per warp, ~39% of CPI), doubling XBLOCK and num_warps (e.g., 128→256 threads, 4→8 warps) gives the scheduler more warps to switch between while others await memory, reducing kernel duration. However, when the kernel is already deeply DRAM-bandwidth-bound (DRAM SOL >83%) and the no-eligible-warp fraction is only marginally reduced (62.24%→61.52%), the latency-hiding benefit is modest (~7%) because the bottleneck is bandwidth saturation rather than pure latency. The warp cycles per issued instruction remained essentially flat (37.56→37.59), confirming the bandwidth wall limits further gains from this approach alone.

> **Applicable when**: Triton kernel has high no-eligible-warp fraction (>50%) from L1TEX scoreboard stalls (CPIStall rule), DRAM throughput is not yet fully saturated (<80%), L1/TEX hit rate is low, and register budget allows doubling block size without spilling (registers_per_thread * 2 * block_size <= 65536 regs/SM).

> **Avoid when**: Kernel is already DRAM-bandwidth-saturated (DRAM SOL >83%) — in this regime, adding more warps cannot hide latency faster than the memory bus delivers data, so gains are marginal. Also avoid when doubling block size would exceed the register file budget per SM, causing register spilling that degrades performance.

---

## Iteration 3

**Best so far**: 1.0638× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_strided_reduction_loads_plus_larger_xblock | 0.997× | 1.071× | ✓ |  |
| B | reduce_r0block_to_improve_l1_reuse_and_eliminate_masked_waste | 0.805× | 0.750× | ✓ |  |
| C | unroll_fixed_reduction_to_eliminate_mask_waste_and_enable_fma | 1.008× | 1.074× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_strid... | Δ | reduce_r0block_to... | Δ | unroll_fixed_redu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 66240 ns | 61860 ns | -4380 ns (-6.6%) | 88350 ns | +22110 ns (+33.4%) | 61700 ns | -4540 ns (-6.9%) |
| SM Throughput (%) | 33.24 % | 41.97 % | +8.73 pp | 27.87 % | -5.37 pp | 42.26 % | +9.02 pp |
| Memory Throughput (%) | 78.28 % | 81.42 % | +3.14 pp | 90.99 % | +12.71 pp | 81.63 % | +3.35 pp |
| DRAM Throughput (%) | 78.28 % | 81.42 % | +3.14 pp | 28.84 % | -49.44 pp | 81.63 % | +3.35 pp |
| Achieved Occupancy (%) | 95.35 % | 88.66 % | -6.69 pp | 94.36 % | -0.99 pp | 90.55 % | -4.80 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 28.00 register/thread | -4 | 32.00 register/thread | +0 |
| Waves/SM | 10.23 | 10.23 | +0.00 | 10.23 | +0.00 | 10.23 | +0.00 |
| No Eligible (%) | 64.40 % | 60.18 % | -4.22 pp | 71.18 % | +6.78 pp | 55.90 % | -8.50 pp |
| IPC Active | 1.40 inst/cycle | 1.59 inst/cycle | +0.19 | 1.15 inst/cycle | -0.25 | 1.73 inst/cycle | +0.33 |

---

## Cumulative Progress

```
Baseline:  66240 ns  (1.000×)
Iter 1     62270 ns  (1.064×)  — vectorized_loads_with_contiguous_hints

Final best: 62270 ns  (1.0638×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_per_fused_convolution_min_0(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
@@ -25,15 +26,17 @@

     x4 = xindex // 900
     x1 = ((xindex // 900) % 24)
     x5 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0 + 900*r0_3 + 19800*x4), r0_mask, other=0.0)
+    # Hint: x0 is stride-1 contiguous across threads; 4-float (16-byte) alignment is safe
+    # x0 and x5 are [XBLOCK, 1] shaped, so values must be a list of length 2
+    _x0 = tl.max_contiguous(tl.multiple_of(x0, [4, 1]), [4, 1])
+    _x5 = tl.max_contiguous(tl.multiple_of(x5, [4, 1]), [4, 1])
+    tmp0 = tl.load(in_ptr0 + (_x0 + 900*r0_3 + 19800*x4), xmask & r0_mask, other=0.0)
     tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
     tmp5 = tl.where(r0_mask, tmp3, float("inf"))
     tmp6 = triton_helpers.min2(tmp5, 1)[:, None].to(tl.float32)
-    tl.store(out_ptr0 + (x5), tmp6, None)
-
-
+    tl.store(out_ptr0 + (_x5), tmp6, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
