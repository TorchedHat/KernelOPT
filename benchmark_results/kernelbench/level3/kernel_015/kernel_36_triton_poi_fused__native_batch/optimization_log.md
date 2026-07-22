# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_36` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 19580 ns — **Best latency**: 19420 ns (**1.0082×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`restore_xblock128_low_register_pressure_arithmetic`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_36`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_36_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (27440,1,1), Waves/SM: 12.99
- **Registers/thread**: 22
- **Duration**: 19580 ns
- **SM Throughput**: 61.38 % | **Memory**: 23.02 % | **DRAM**: 14.78 %
- **Achieved Occupancy**: 61.74 %
- **IPC Active**: 2.78 inst/cycle | **No Eligible**: 29.95 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune | 1.007× | 1.005× | ✓ | **✓ best** |
| B | autotune_xblock_num_warps_num_stages | 1.000× | 0.378× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_stores | 0.866× | 0.208× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_n... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 19580 ns | 19490 ns | -90 ns (-0.5%) | 51780 ns | +32200 ns (+164.5%) | 94270 ns | +74690 ns (+381.5%) |
| SM Throughput (%) | 61.38 % | 61.48 % | +0.10 pp | 82.97 % | +21.59 pp | 84.75 % | +23.37 pp |
| Memory Throughput (%) | 23.02 % | 23.05 % | +0.03 pp | 18.93 % | -4.09 pp | 17.15 % | -5.87 pp |
| DRAM Throughput (%) | 14.78 % | 14.86 % | +0.08 pp | 5.58 % | -9.20 pp | 3.07 % | -11.71 pp |
| Achieved Occupancy (%) | 61.74 % | 62.41 % | +0.67 pp | 87.19 % | +25.45 pp | 55.21 % | -6.53 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 32.00 register/thread | +10 | 48.00 register/thread | +26 |
| Waves/SM | 12.99 | 12.99 | +0.00 | 25.98 | +12.99 | 41.58 | +28.59 |
| No Eligible (%) | 29.95 % | 30.87 % | +0.92 pp | 13.42 % | -16.53 pp | 13.13 % | -16.82 pp |
| IPC Active | 2.78 inst/cycle | 2.78 inst/cycle | +0.00 | 3.46 inst/cycle | +0.68 | 3.47 inst/cycle | +0.69 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_num_warps_num_stages` (0.518×)

Replacing the autotune config list (swapping in num_stages and reordering configs) while also rewriting the batch norm arithmetic to FMA form caused the autotuner to select XBLOCK=256 with 32 registers/thread instead of the original XBLOCK=128 with 22 registers/thread. Despite the grid lambda being correctly set (or the same grid mismatch persisting), executed instructions nearly tripled from 9.3M to 22.9M and duration nearly doubled from 19.5us to 37.8us. The register increase from 22 to 32 per thread reduced Block_Limit_Registers from 21 to 8 blocks/SM, and while achieved occupancy improved from 62% to 84%, the instruction count explosion dominated. The FMA rewrite alone cannot compensate for the overhead introduced by a larger XBLOCK config with higher register pressure on a short-running pointwise kernel.

> **Applicable when**: Never apply this pattern. Adding autotune with num_stages to a short-running pointwise kernel (<25us) that already has low register usage (<=22 regs/thread) and 100% theoretical occupancy will cause the compiler to allocate more registers for larger XBLOCK configs, collapsing Block_Limit_Registers and inflating instruction count.

> **Avoid when**: Do not add @triton.autotune with num_stages > 1 to pointwise kernels with no shared memory and no software pipelining benefit. num_stages is meaningful for kernels with tiled reductions or matmul-style loops; for simple elementwise kernels it only increases register pressure. Also do not swap autotune config lists when the original kernel already achieves near-optimal duration — the autotuner may select a worse config than the original fixed XBLOCK.

#### ❌ Plan C: `increase_xblock_for_vectorized_loads_and_stores` (0.208×)

Adding @triton.autotune with large XBLOCK configs (512–2048) caused the compiler to select a configuration that dramatically increased registers per thread from 22 to 48, collapsing Block_Limit_Registers from 21 to 5 blocks/SM and reducing theoretical occupancy from 100% to 62.5%. This register pressure increase caused a 6.7x explosion in executed instructions (9.3M → 63M) and a 4.8x increase in kernel duration (19.6 µs → 94.3 µs), despite higher SM compute throughput. When autotune selects a large XBLOCK that causes the compiler to allocate significantly more registers per thread, the occupancy loss and instruction count increase far outweigh any vectorization benefit.

> **Applicable when**: Never apply this exact pattern. The anti-pattern is: adding autotune with large XBLOCK configs when the kernel already has low register usage (≤24 regs/thread) and high theoretical occupancy (≥100%), because the autotune overhead and compiler register allocation for larger tiles can cause severe regressions.

> **Avoid when**: Do not add @triton.autotune with large XBLOCK configs (≥512) to pointwise kernels that already achieve 100% theoretical occupancy with small block sizes and low register counts (≤24 regs/thread). The autotune may select a config that causes register pressure to spike (e.g., 22→48 regs/thread), collapsing Block_Limit_Registers and theoretical occupancy, and massively increasing instruction count. Also note: the grid was not updated to use meta['XBLOCK'], which is an implementation flaw that may have forced a suboptimal config selection.

---

## Iteration 2

**Best so far**: 1.0046× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_bug | 1.006× | 1.000× | ✓ |  |
| B | fix_autotune_grid_lambda_mismatch | 1.013× | 0.518× | ✓ | **✓ best** |
| C | fix_grid_lambda_and_reduce_register_pressure_with_rsqrt | 1.012× | 1.011× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 19580 ns | 19580 ns | +0 ns (+0.0%) | 37760 ns | +18180 ns (+92.8%) | 19360 ns | -220 ns (-1.1%) |
| SM Throughput (%) | 61.38 % | 61.24 % | -0.14 pp | 77.15 % | +15.77 pp | 46.35 % | -15.03 pp |
| Memory Throughput (%) | 23.02 % | 22.94 % | -0.08 pp | 26.89 % | +3.87 pp | 22.75 % | -0.27 pp |
| DRAM Throughput (%) | 14.78 % | 14.84 % | +0.06 pp | 7.65 % | -7.13 pp | 14.98 % | +0.20 pp |
| Achieved Occupancy (%) | 61.74 % | 61.92 % | +0.18 pp | 84.35 % | +22.61 pp | 43.70 % | -18.04 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 32.00 register/thread | +10 | 20.00 register/thread | -2 |
| Waves/SM | 12.99 | 12.99 | +0.00 | 25.98 | +12.99 | 12.99 | +0.00 |
| No Eligible (%) | 29.95 % | 30.10 % | +0.15 pp | 18.54 % | -11.41 pp | 55.97 % | +26.02 pp |
| IPC Active | 2.78 inst/cycle | 2.77 inst/cycle | -0.01 | 3.24 inst/cycle | +0.46 | 1.73 inst/cycle | -1.05 |

---

## Iteration 3

**Best so far**: 1.0046× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_for_l1tex_latency_hiding | 1.006× | 0.798× | ✓ |  |
| B | reduce_nonfused_fp32_instructions_via_rsqrt_arithmetic_simplification | 1.005× | 1.014× | ✓ |  |
| C | restore_xblock128_low_register_pressure_arithmetic | 1.015× | 1.008× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | reduce_nonfused_f... | Δ | restore_xblock128... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 19580 ns | 24540 ns | +4960 ns (+25.3%) | 19300 ns | -280 ns (-1.4%) | 19420 ns | -160 ns (-0.8%) |
| SM Throughput (%) | 61.38 % | 74.59 % | +13.21 pp | 46.55 % | -14.83 pp | 46.37 % | -15.01 pp |
| Memory Throughput (%) | 23.02 % | 34.58 % | +11.56 pp | 22.79 % | -0.23 pp | 22.82 % | -0.20 pp |
| DRAM Throughput (%) | 14.78 % | 11.79 % | -2.99 pp | 15.00 % | +0.22 pp | 14.89 % | +0.11 pp |
| Achieved Occupancy (%) | 61.74 % | 76.56 % | +14.82 pp | 43.37 % | -18.37 pp | 44.03 % | -17.71 pp |
| Registers/Thread | 22.00 register/thread | 20.00 register/thread | -2 | 20.00 register/thread | -2 | 20.00 register/thread | -2 |
| Waves/SM | 12.99 | 25.98 | +12.99 | 12.99 | +0.00 | 12.99 | +0.00 |
| No Eligible (%) | 29.95 % | 33.23 % | +3.28 pp | 56.36 % | +26.41 pp | 51.10 % | +21.15 pp |
| IPC Active | 2.78 inst/cycle | 2.71 inst/cycle | -0.07 | 1.75 inst/cycle | -1.03 | 1.97 inst/cycle | -0.81 |


### Key Learnings

#### ❌ Plan A: `increase_num_warps_for_l1tex_latency_hiding` (0.798×)

Increasing num_warps from 4 to 8 (block size 128→256) while keeping XBLOCK=128 caused the autotuner to select the 256-thread config, which doubled the total thread count (3.5M→7.0M) and doubled executed instructions (5.8M→11.6M) without reducing per-thread work. Although eligible warps/scheduler improved from 0.77 to 2.49 and achieved occupancy rose from 43.7% to 76.6%, the instruction count inflation dominated and duration increased from 19.4µs to 24.5µs. The key insight is that for a pointwise kernel where each thread processes a fixed number of elements (XBLOCK/num_warps elements per warp), doubling num_warps at fixed XBLOCK halves the work per warp, requiring twice as many blocks to cover the same data — but since grid_size was fixed at 27440, the actual effect was each block doing the same work with twice the threads, wasting half the thread slots and doubling instruction overhead. The CPIStall L1TEX stall fraction did decrease (61.6%→49.1%), confirming latency hiding improved, but the instruction overhead outweighed the stall reduction.

> **Applicable when**: Never apply this pattern to pointwise kernels where XBLOCK equals block_size (1 element per thread). Increasing num_warps without increasing XBLOCK creates excess threads that do no additional work, doubling instruction count.

> **Avoid when**: Do not increase num_warps independently of XBLOCK in pointwise Triton kernels where each thread processes exactly one element (XBLOCK == block_size). The extra threads do not reduce per-block work; they only inflate total instruction count. This pattern is only valid when XBLOCK > block_size (i.e., each thread processes multiple elements), so that more warps can overlap memory latency while processing the same data. Also: even when L1TEX stall fraction is high (>60%) and eligible warps are low (<1.0/scheduler), if the kernel is already near-minimal in per-element instruction count, latency hiding via more warps may not overcome the instruction overhead of a larger block.

---

## Iteration 4

**Best so far**: 1.0082× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_eviction_policy_for_main_tensor_loads | 1.014× | 1.017× | ✓ |  |
| B | reduce_nonfused_fp32_via_rsqrt_multiply_isolated | 1.007× | 1.014× | ✓ |  |
| C | reduce_thread_waste_and_fuse_batchnorm_arithmetic_to_fma | 1.008× | 1.011× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_evictio... | Δ | reduce_nonfused_f... | Δ | reduce_thread_was... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 19580 ns | 19260 ns | -320 ns (-1.6%) | 19300 ns | -280 ns (-1.4%) | 19360 ns | -220 ns (-1.1%) |
| SM Throughput (%) | 61.38 % | 46.50 % | -14.88 pp | 46.14 % | -15.24 pp | 46.12 % | -15.26 pp |
| Memory Throughput (%) | 23.02 % | 22.85 % | -0.17 pp | 22.66 % | -0.36 pp | 22.64 % | -0.38 pp |
| DRAM Throughput (%) | 14.78 % | 15.03 % | +0.25 pp | 15.01 % | +0.23 pp | 14.94 % | +0.16 pp |
| Achieved Occupancy (%) | 61.74 % | 43.58 % | -18.16 pp | 44.16 % | -17.58 pp | 43.22 % | -18.52 pp |
| Registers/Thread | 22.00 register/thread | 20.00 register/thread | -2 | 20.00 register/thread | -2 | 20.00 register/thread | -2 |
| Waves/SM | 12.99 | 12.99 | +0.00 | 12.99 | +0.00 | 12.99 | +0.00 |
| No Eligible (%) | 29.95 % | 52.33 % | +22.38 pp | 56.13 % | +26.18 pp | 55.71 % | +25.76 pp |
| IPC Active | 2.78 inst/cycle | 1.89 inst/cycle | -0.89 | 1.73 inst/cycle | -1.05 | 1.77 inst/cycle | -1.01 |

---

## Iteration 5

**Best so far**: 1.0082× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | multi_element_per_thread_via_xblock_gt_blocksize | 1.013× | 0.995× | ✓ |  |
| C | proportional_xblock_numwarps_scale_for_l1tex_latency_hiding | 1.006× | 1.017× | ✓ |  |


### Metric Comparison

| Metric | Baseline | multi_element_per... | Δ | proportional_xblo... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 19580 ns | 19680 ns | +100 ns (+0.5%) | 19260 ns | -320 ns (-1.6%) |
| SM Throughput (%) | 61.38 % | 46.00 % | -15.38 pp | 46.65 % | -14.73 pp |
| Memory Throughput (%) | 23.02 % | 30.59 % | +7.57 pp | 22.80 % | -0.22 pp |
| DRAM Throughput (%) | 14.78 % | 14.73 % | -0.05 pp | 15.04 % | +0.26 pp |
| Achieved Occupancy (%) | 61.74 % | 32.41 % | -29.33 pp | 43.31 % | -18.43 pp |
| Registers/Thread | 22.00 register/thread | 24.00 register/thread | +2 | 20.00 register/thread | -2 |
| Waves/SM | 12.99 | 12.99 | +0.00 | 12.99 | +0.00 |
| No Eligible (%) | 29.95 % | 48.53 % | +18.58 pp | 55.78 % | +25.83 pp |
| IPC Active | 2.78 inst/cycle | 2.05 inst/cycle | -0.73 | 1.76 inst/cycle | -1.02 |

---

## Cumulative Progress

```
Baseline:  19580 ns  (1.000×)
Iter 1     19490 ns  (1.005×)  — increase_xblock_with_autotune
Iter 3     19420 ns  (1.008×)  — restore_xblock128_low_register_pressure_arithmetic

Final best: 19420 ns  (1.0082×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,16 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 256}, num_warps=4),
+        triton.Config({'XBLOCK': 256}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 3512320
@@ -13,27 +23,19 @@

     xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 448)
-    tmp0 = tl.load(in_ptr0 + (x2), xmask)
+    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_first')
     tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
     tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
     tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1], 1e-05, tl.float32)
-    tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
+    tmp5 = tmp3 + 1e-05
+    tmp6 = libdevice.rsqrt(tmp5)
+    tmp11 = tmp2 * tmp6
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
-    tmp16 = tl.full([1], 0, tl.int32)
-    tmp17 = tl.maximum(tmp16, tmp15)
-    tl.store(out_ptr0 + (x2), tmp17, xmask)
-
-
+    tmp17 = tl.maximum(0.0, tmp15)
+    tl.store(out_ptr0 + (x2), tmp17, xmask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -44,7 +46,7 @@

     arg_5 = inputs[5]
     xnumel = 3512320
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_relu_36[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 3512320, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_relu_36[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 3512320)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
