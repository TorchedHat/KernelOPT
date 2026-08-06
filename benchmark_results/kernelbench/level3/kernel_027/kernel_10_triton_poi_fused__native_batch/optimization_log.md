# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_10` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 33730 ns — **Best latency**: 33250 ns (**1.0144×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`eliminate_runtime_mask_predication_with_static_true_mask`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_10`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_027/kernel_10_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (50176,1,1), Waves/SM: 23.76
- **Registers/thread**: 19
- **Duration**: 33730 ns
- **SM Throughput**: 59.56 % | **Memory**: 27.85 % | **DRAM**: 16.19 %
- **Achieved Occupancy**: 56.02 %
- **IPC Active**: 2.55 inst/cycle | **No Eligible**: 35.81 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune | 0.347× | 0.334× | ✓ |  |
| B | autotune_xblock_increase_occupancy | 0.651× | 0.641× | ✓ |  |
| C | add_autotune_with_larger_xblock_for_vectorized_loads | 0.996× | 1.006× | ✓ |  |
| D | autotune_xblock_and_num_warps_for_occupancy | 1.005× | 1.001× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_i... | Δ | add_autotune_with... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 33730 ns | 101020 ns | +67290 ns (+199.5%) | 52610 ns | +18880 ns (+56.0%) | 33540 ns | -190 ns (-0.6%) | 33700 ns | -30 ns (-0.1%) |
| SM Throughput (%) | 59.56 % | 82.70 % | +23.14 pp | 83.95 % | +24.39 pp | 72.52 % | +12.96 pp | 51.87 % | -7.69 pp |
| Memory Throughput (%) | 27.85 % | 30.26 % | +2.41 pp | 26.51 % | -1.34 pp | 27.96 % | +0.11 pp | 24.97 % | -2.88 pp |
| DRAM Throughput (%) | 16.19 % | 5.40 % | -10.79 pp | 10.38 % | -5.81 pp | 16.25 % | +0.06 pp | 16.23 % | +0.04 pp |
| Achieved Occupancy (%) | 56.02 % | 80.07 % | +24.05 pp | 88.46 % | +32.44 pp | 61.61 % | +5.59 pp | 37.22 % | -18.80 pp |
| Registers/Thread | 19.00 register/thread | 20.00 register/thread | +1 | 32.00 register/thread | +13 | 20.00 register/thread | +1 | 28.00 register/thread | +9 |
| Waves/SM | 23.76 | 95.03 | +71.27 | 23.76 | +0.00 | 23.76 | +0.00 | 11.88 | -11.88 |
| No Eligible (%) | 35.81 % | 15.60 % | -20.21 pp | 13.38 % | -22.43 pp | 22.03 % | -13.78 pp | 42.13 % | +6.32 pp |
| IPC Active | 2.55 inst/cycle | 3.37 inst/cycle | +0.82 | 3.47 inst/cycle | +0.92 | 3.11 inst/cycle | +0.56 | 2.33 inst/cycle | -0.22 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune` (0.334×)

Increasing XBLOCK from 128 to 512 (via autotune) caused a 3x slowdown despite improving occupancy (56% → 80%) and eligible warps (1.31 → 6.78). The root cause is that the grid was not updated to match the new XBLOCK: the launch code kept `grid = (triton.cdiv(xnumel, 128),)` as a fixed tuple instead of a lambda, so each CTA processed 4x more elements than the grid covered, causing massive redundant work — executed instructions ballooned from 15.6M to 66M (4.2x increase). Additionally, changing `xmask = tl.full([XBLOCK], True, tl.int1)` to `xmask = xindex < xnumel` introduced unnecessary predication overhead. When using autotune to vary XBLOCK, the grid MUST be a lambda `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)` — a fixed grid is an implementation flaw that invalidates the strategy.

> **Applicable when**: Never apply this pattern (fixed grid + autotune XBLOCK). Always use lambda grid when autotuning XBLOCK.

> **Avoid when**: Any kernel where the grid is a fixed tuple and XBLOCK is varied via autotune — this combination causes the kernel to process far more elements than intended, multiplying instruction count by the XBLOCK ratio.

#### ❌ Plan B: `autotune_xblock_increase_occupancy` (0.641×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) increased achieved occupancy from 56% to 88% and eligible warps per scheduler from 1.31 to 7.48, but the kernel duration increased by 56% (33.73 µs → 52.61 µs). The root cause was a 2.2× increase in total executed instructions (15.6M → 34.8M) and a jump in registers per thread from 19 to 32, indicating the autotune selected a config that introduced significant instruction overhead (likely from the xmask guard on every load/store that was previously elided via tl.full([XBLOCK], True)). When the original kernel uses tl.full([XBLOCK], True, tl.int1) as its mask (meaning xnumel is always a perfect multiple of XBLOCK), replacing it with a runtime bounds check adds unnecessary predication overhead and prevents compiler optimizations that assume no masking is needed.

> **Applicable when**: Never: this specific combination of adding runtime xmask + broken grid (still using hardcoded 128 instead of meta['XBLOCK']) causes both instruction bloat and incorrect tiling.

> **Avoid when**: When the original kernel uses tl.full([XBLOCK], True, tl.int1) as the mask (indicating xnumel is always divisible by the original XBLOCK), do NOT replace it with a runtime xmask = xindex < xnumel — this adds predication overhead to every memory operation. Also, do NOT add @triton.autotune with larger XBLOCK configs while keeping the launch grid hardcoded to the original XBLOCK — the grid must be updated to lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),) or the kernel will launch with too few CTAs and produce wrong results or idle SMs.

---

## Iteration 2

**Best so far**: 1.0009× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_to_enable_larger_xblock | 1.002× | 1.007× | ✓ |  |
| B | fix_autotune_grid_lambda_to_enable_xblock_scaling | 0.999× | 0.997× | ✓ |  |
| C | reduce_non_fused_fp32_instructions_via_algebraic_simplification | 0.905× | 0.935× | ✓ |  |
| D | fix_grid_lambda_and_replace_reciprocal_sqrt_with_rsqrt | 0.425× | 0.422× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | reduce_non_fused_... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 33730 ns | 33500 ns | -230 ns (-0.7%) | 33820 ns | +90 ns (+0.3%) | 36060 ns | +2330 ns (+6.9%) | 79840 ns | +46110 ns (+136.7%) |
| SM Throughput (%) | 59.56 % | 72.38 % | +12.82 pp | 52.10 % | -7.46 pp | 63.88 % | +4.32 pp | 75.88 % | +16.32 pp |
| Memory Throughput (%) | 27.85 % | 27.93 % | +0.08 pp | 25.18 % | -2.67 pp | 40.28 % | +12.43 pp | 38.82 % | +10.97 pp |
| DRAM Throughput (%) | 16.19 % | 16.28 % | +0.09 pp | 16.13 % | -0.06 pp | 15.09 % | -1.10 pp | 6.84 % | -9.35 pp |
| Achieved Occupancy (%) | 56.02 % | 61.49 % | +5.47 pp | 37.20 % | -18.82 pp | 40.27 % | -15.75 pp | 73.45 % | +17.43 pp |
| Registers/Thread | 19.00 register/thread | 20.00 register/thread | +1 | 28.00 register/thread | +9 | 32.00 register/thread | +13 | 20.00 register/thread | +1 |
| Waves/SM | 23.76 | 23.76 | +0.00 | 11.88 | -11.88 | 23.76 | +0.00 | 95.03 | +71.27 |
| No Eligible (%) | 35.81 % | 22.61 % | -13.20 pp | 41.84 % | +6.03 pp | 30.38 % | -5.43 pp | 22.27 % | -13.54 pp |
| IPC Active | 2.55 inst/cycle | 3.11 inst/cycle | +0.56 | 2.32 inst/cycle | -0.23 | 2.77 inst/cycle | +0.22 | 3.11 inst/cycle | +0.56 |


### Key Learnings

#### ❌ Plan D: `fix_grid_lambda_and_replace_reciprocal_sqrt_with_rsqrt` (0.422×)

Replacing the fixed grid tuple with a lambda and substituting tl.math.rsqrt for the multi-step reciprocal-sqrt sequence still caused a 2.4x slowdown (33.5 µs → 79.8 µs). The autotune configs (XBLOCK 512/1024/2048, num_warps 16/32/32, num_stages=1) selected XBLOCK=512 with block_size=512, causing waves_per_sm to jump from 23.76 to 95.03 and executed instructions to balloon from 19M to 47.8M (2.5x). Even with a correct lambda grid, choosing autotune configs with much larger XBLOCK than the original 128 multiplies total work when the kernel's compute-to-memory ratio does not benefit from the extra occupancy — the dominant stall shifted from L1TEX dependency (CPIStall, 41.2%) to warp-not-selected (42.1%), indicating the GPU is now over-subscribed with warps rather than latency-bound. The rsqrt change alone is insufficient to overcome the instruction count explosion from the oversized tile configs.

> **Applicable when**: Never: even with a correct lambda grid, adding autotune configs with XBLOCK >> original block size causes instruction count explosion when the kernel is already compute-bound (Compute SM Throughput ~72%) and the dominant stall is L1TEX dependency rather than warp scheduling. The rsqrt optimization is too small (~5 instructions per element) to compensate.

> **Avoid when**: When the original kernel runs at XBLOCK=128 with ~24 waves/SM and is compute-bound (SM Throughput >70%), do NOT introduce autotune configs with XBLOCK 4-16x larger (512-2048). Even with a correct lambda grid, the autotune will select a larger XBLOCK, multiplying total executed instructions proportionally and increasing kernel duration. The warp-not-selected stall (replacing L1TEX stall) confirms over-subscription, not latency hiding.

---

## Iteration 3

**Best so far**: 1.0009× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_dead_instructions_via_rsqrt_algebraic_simplification | 1.009× | 1.017× | ✓ |  |
| B | streaming_eviction_policy_for_large_tensor_with_correct_grid_lambda | 1.011× | 1.009× | ✓ | **✓ best** |
| C | fix_grid_lambda_to_enable_correct_autotune_selection | 1.003× | 1.009× | ✓ |  |
| D | fuse_fp32_multiply_add_pairs_into_fma_with_restored_xblock128 | 1.006× | 1.014× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_dead_in... | Δ | streaming_evictio... | Δ | fix_grid_lambda_t... | Δ | fuse_fp32_multipl... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 33730 ns | 33180 ns | -550 ns (-1.6%) | 33440 ns | -290 ns (-0.9%) | 33440 ns | -290 ns (-0.9%) | 33250 ns | -480 ns (-1.4%) |
| SM Throughput (%) | 59.56 % | 46.73 % | -12.83 pp | 37.69 % | -21.87 pp | 46.10 % | -13.46 pp | 46.53 % | -13.03 pp |
| Memory Throughput (%) | 27.85 % | 27.51 % | -0.34 pp | 13.09 % | -14.76 pp | 27.11 % | -0.74 pp | 27.38 % | -0.47 pp |
| DRAM Throughput (%) | 16.19 % | 16.40 % | +0.21 pp | 7.83 % | -8.36 pp | 16.33 % | +0.14 pp | 16.59 % | +0.40 pp |
| Achieved Occupancy (%) | 56.02 % | 41.81 % | -14.21 pp | 31.01 % | -25.01 pp | 42.02 % | -14.00 pp | 42.09 % | -13.93 pp |
| Registers/Thread | 19.00 register/thread | 20.00 register/thread | +1 | 20.00 register/thread | +1 | 20.00 register/thread | +1 | 20.00 register/thread | +1 |
| Waves/SM | 23.76 | 23.76 | +0.00 | 11.88 | -11.88 | 23.76 | +0.00 | 23.76 | +0.00 |
| No Eligible (%) | 35.81 % | 49.41 % | +13.60 pp | 57.34 % | +21.53 pp | 49.75 % | +13.94 pp | 50.05 % | +14.24 pp |
| IPC Active | 2.55 inst/cycle | 2.02 inst/cycle | -0.53 | 1.70 inst/cycle | -0.85 | 2.02 inst/cycle | -0.53 | 2.02 inst/cycle | -0.53 |

---

## Iteration 4

**Best so far**: 1.0087× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_latency | 0.761× | 0.770× | ✓ |  |
| B | increase_num_warps_to_hide_l1tex_latency | 1.004× | 1.008× | ✓ |  |
| C | fix_grid_lambda_enable_moderate_xblock_num_warps_sweep | 1.006× | 1.002× | ✓ |  |
| D | fix_grid_lambda_and_tune_num_warps_at_fixed_xblock128 | 1.010× | 1.010× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | increase_num_warp... | Δ | fix_grid_lambda_e... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 33730 ns | 43780 ns | +10050 ns (+29.8%) | 33470 ns | -260 ns (-0.8%) | 33660 ns | -70 ns (-0.2%) | 33410 ns | -320 ns (-0.9%) |
| SM Throughput (%) | 59.56 % | 72.26 % | +12.70 pp | 46.64 % | -12.92 pp | 37.72 % | -21.84 pp | 46.67 % | -12.89 pp |
| Memory Throughput (%) | 27.85 % | 39.90 % | +12.05 pp | 27.50 % | -0.35 pp | 13.11 % | -14.74 pp | 27.48 % | -0.37 pp |
| DRAM Throughput (%) | 16.19 % | 12.61 % | -3.58 pp | 16.48 % | +0.29 pp | 7.78 % | -8.41 pp | 16.49 % | +0.30 pp |
| Achieved Occupancy (%) | 56.02 % | 77.34 % | +21.32 pp | 41.75 % | -14.27 pp | 30.95 % | -25.07 pp | 41.91 % | -14.11 pp |
| Registers/Thread | 19.00 register/thread | 22.00 register/thread | +3 | 20.00 register/thread | +1 | 20.00 register/thread | +1 | 20.00 register/thread | +1 |
| Waves/SM | 23.76 | 47.52 | +23.76 | 23.76 | +0.00 | 11.88 | -11.88 | 23.76 | +0.00 |
| No Eligible (%) | 35.81 % | 23.54 % | -12.27 pp | 49.55 % | +13.74 pp | 57.45 % | +21.64 pp | 49.99 % | +14.18 pp |
| IPC Active | 2.55 inst/cycle | 3.05 inst/cycle | +0.50 | 2.00 inst/cycle | -0.55 | 1.70 inst/cycle | -0.85 | 2.02 inst/cycle | -0.53 |


### Key Learnings

#### ❌ Plan A: `increase_num_warps_to_hide_l1tex_latency` (0.770×)

Doubling num_warps from 4 to 8 (block_size 128→256) while keeping XBLOCK=128 fixed caused a 23% slowdown (33.18 µs → 43.78 µs) despite improving achieved occupancy (41.8% → 77.3%), eligible warps per scheduler (0.83 → 3.11), and compute SM throughput (46.7% → 72.3%). The root cause is that doubling the block size with a fixed XBLOCK=128 doubled the total executed instructions from 12.2M to 24.9M (2.03x), because each block now launches twice as many threads all executing the same tile of work — the extra warps are redundant, not latency-hiding. The waves_per_sm also doubled (23.76 → 47.52), confirming that total GPU work doubled. When XBLOCK is fixed and block_size increases, the grid stays the same but each CTA does 2x the work via redundant threads, negating any latency-hiding benefit.

> **Applicable when**: Never: increasing num_warps beyond XBLOCK/32 (the number of warps needed to cover XBLOCK elements) causes redundant thread execution. In Triton, XBLOCK determines the tile size and must equal block_size (num_warps * 32) for correct 1:1 thread-to-element mapping.

> **Avoid when**: When XBLOCK is fixed and equals the original block_size (num_warps * 32), do NOT increase num_warps without also increasing XBLOCK proportionally. In Triton point-wise kernels, XBLOCK = num_warps * 32 is the invariant: each thread handles exactly one element in the tile. Breaking this invariant by adding more warps without expanding XBLOCK causes each CTA to launch redundant threads that re-execute the same tile, doubling instructions and waves_per_sm.

#### ❌ Plan B: `increase_num_warps_to_hide_l1tex_latency` (0.770×)

Doubling num_warps from 4 to 8 (block_size 128→256) while keeping XBLOCK=128 fixed caused a 23% slowdown (33.18 µs → 43.78 µs) despite improving achieved occupancy (41.8% → 77.3%), eligible warps per scheduler (0.83 → 3.11), and compute SM throughput (46.7% → 72.3%). The root cause is that doubling the block size with a fixed XBLOCK=128 doubled the total executed instructions from 12.2M to 24.9M (2.03x), because each block now launches twice as many threads all executing the same tile of work — the extra warps are redundant, not latency-hiding. The waves_per_sm also doubled (23.76 → 47.52), confirming that total GPU work doubled. When XBLOCK is fixed and block_size increases, the grid stays the same but each CTA does 2x the work via redundant threads, negating any latency-hiding benefit.

> **Applicable when**: Never: increasing num_warps beyond XBLOCK/32 (the number of warps needed to cover XBLOCK elements) causes redundant thread execution. In Triton, XBLOCK determines the tile size and must equal block_size (num_warps * 32) for correct 1:1 thread-to-element mapping.

> **Avoid when**: When XBLOCK is fixed and equals the original block_size (num_warps * 32), do NOT increase num_warps without also increasing XBLOCK proportionally. In Triton point-wise kernels, XBLOCK = num_warps * 32 is the invariant: each thread handles exactly one element in the tile. Breaking this invariant by adding more warps without expanding XBLOCK causes each CTA to launch redundant threads that re-execute the same tile, doubling instructions and waves_per_sm.

---

## Iteration 5

**Best so far**: 1.0087× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_num_stages_and_fix_grid_lambda_for_broadcast_load_kernel | 1.005× | 1.008× | ✓ |  |
| B | eliminate_runtime_mask_predication_with_static_true_mask | 1.012× | 1.014× | ✓ | **✓ best** |
| C | fix_grid_lambda_to_match_autotuned_xblock_and_add_num_stages_sweep | 1.010× | 1.005× | ✓ |  |
| D | fuse_fp32_multiply_add_into_explicit_fma_instructions | 0.760× | 0.772× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_num_stages... | Δ | eliminate_runtime... | Δ | fix_grid_lambda_t... | Δ | fuse_fp32_multipl... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 33730 ns | 33470 ns | -260 ns (-0.8%) | 33250 ns | -480 ns (-1.4%) | 33570 ns | -160 ns (-0.5%) | 43710 ns | +9980 ns (+29.6%) |
| SM Throughput (%) | 59.56 % | 46.63 % | -12.93 pp | 42.39 % | -17.17 pp | 37.45 % | -22.11 pp | 72.61 % | +13.05 pp |
| Memory Throughput (%) | 27.85 % | 27.47 % | -0.38 pp | 26.88 % | -0.97 pp | 13.00 % | -14.85 pp | 40.12 % | +12.27 pp |
| DRAM Throughput (%) | 16.19 % | 16.49 % | +0.30 pp | 16.65 % | +0.46 pp | 7.81 % | -8.38 pp | 12.64 % | -3.55 pp |
| Achieved Occupancy (%) | 56.02 % | 41.62 % | -14.40 pp | 39.43 % | -16.59 pp | 30.72 % | -25.30 pp | 77.34 % | +21.32 pp |
| Registers/Thread | 19.00 register/thread | 20.00 register/thread | +1 | 18.00 register/thread | -1 | 20.00 register/thread | +1 | 22.00 register/thread | +3 |
| Waves/SM | 23.76 | 23.76 | +0.00 | 23.76 | +0.00 | 11.88 | -11.88 | 47.52 | +23.76 |
| No Eligible (%) | 35.81 % | 49.77 % | +13.96 pp | 62.99 % | +27.18 pp | 57.44 % | +21.63 pp | 23.59 % | -12.22 pp |
| IPC Active | 2.55 inst/cycle | 2.01 inst/cycle | -0.54 | 1.49 inst/cycle | -1.06 | 1.70 inst/cycle | -0.85 | 3.05 inst/cycle | +0.50 |

---

## Cumulative Progress

```
Baseline:  33730 ns  (1.000×)
Iter 1     33700 ns  (1.001×)  — autotune_xblock_and_num_warps_for_occupancy
Iter 3     33440 ns  (1.009×)  — streaming_eviction_policy_for_large_tensor_with_correct_grid_lambda
Iter 5     33250 ns  (1.014×)  — eliminate_runtime_mask_predication_with_static_true_mask

Final best: 33250 ns  (1.0144×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,12 +5,13 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
     xnumel = 6422528
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    # xmask removed -- xnumel=6422528 is exactly divisible by XBLOCK=128 (50176 x 128 = 6422528)
     x2 = xindex
     x0 = (xindex % 256)
     tmp0 = tl.load(in_out_ptr0 + (x2), None)
@@ -21,21 +22,12 @@

     tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp4 = tmp2 - tmp3
-    tmp6 = tl.full([1], 1e-05, tl.float32)
-    tmp7 = tmp5 + tmp6
-    tmp8 = tl.sqrt_rn(tmp7)
-    tmp9 = tl.full([1], 1, tl.int32)
-    tmp10 = (tmp9 / tmp8)
-    tmp11 = tl.full([1], 1.0, tl.float32)
-    tmp12 = tmp10 * tmp11
-    tmp13 = tmp4 * tmp12
-    tmp15 = tmp13 * tmp14
-    tmp17 = tmp15 + tmp16
-    tmp18 = tl.full([1], 0, tl.int32)
-    tmp19 = tl.maximum(tmp18, tmp17)
+    tmp7 = tmp5 + 1e-05
+    tmp8 = libdevice.rsqrt(tmp7)
+    tmp13 = tmp4 * tmp8
+    tmp17 = libdevice.fma(tmp13, tmp14, tmp16)
+    tmp19 = tl.maximum(tl.zeros([XBLOCK], dtype=tl.float32), tmp17)
     tl.store(in_out_ptr0 + (x2), tmp19, None)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
