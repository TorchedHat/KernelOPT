# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_addmm_mul_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 42940 ns — **Best latency**: 42940 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_addmm_mul_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_033/kernel_1_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (65536,1,1), Waves/SM: 31.03
- **Registers/thread**: 22
- **Duration**: 42940 ns
- **SM Throughput**: 62.10 % | **Memory**: 31.31 % | **DRAM**: 19.41 %
- **Achieved Occupancy**: 57.24 %
- **IPC Active**: 2.60 inst/cycle | **No Eligible**: 33.89 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_via_autotune_larger_xblock | 0.611× | 0.597× | ✓ |  |
| B | increase_xblock_tile_size_with_autotune | 0.613× | 0.599× | ✓ |  |
| C | autotune_xblock_num_warps_for_occupancy_and_latency_hiding | 0.614× | 0.601× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | increase_xblock_t... | Δ | autotune_xblock_n... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 42940 ns | 71900 ns | +28960 ns (+67.4%) | 71710 ns | +28770 ns (+67.0%) | 71490 ns | +28550 ns (+66.5%) |
| SM Throughput (%) | 62.10 % | 83.24 % | +21.14 pp | 83.09 % | +20.99 pp | 83.30 % | +21.20 pp |
| Memory Throughput (%) | 31.31 % | 28.45 % | -2.86 pp | 28.44 % | -2.87 pp | 28.50 % | -2.81 pp |
| DRAM Throughput (%) | 19.41 % | 11.62 % | -7.79 pp | 11.68 % | -7.73 pp | 11.63 % | -7.78 pp |
| Achieved Occupancy (%) | 57.24 % | 89.02 % | +31.78 pp | 89.01 % | +31.77 pp | 89.01 % | +31.77 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 33.89 % | 14.50 % | -19.39 pp | 14.28 % | -19.61 pp | 14.32 % | -19.57 pp |
| IPC Active | 2.60 inst/cycle | 3.42 inst/cycle | +0.82 | 3.42 inst/cycle | +0.82 | 3.42 inst/cycle | +0.82 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling_via_autotune_larger_xblock` (0.597×)

Adding @triton.autotune with larger XBLOCK configs (512–4096) improved occupancy from 57% to 89% and eligible warps/scheduler from 1.37 to 7.46, but the kernel regressed by 1.67x because the grid lambda was not updated to use meta['XBLOCK'] — the grid remained fixed at cdiv(xnumel, 128), so the autotuner selected a larger XBLOCK while the grid was still sized for XBLOCK=128, causing each element to be processed ~2.24x more times (executed instructions jumped from 20.9M to 46.9M). Additionally, the introduction of a non-trivial xmask (xindex < xnumel) on a power-of-2 problem size added unnecessary predication overhead. When the grid is not updated to match the autotuned XBLOCK, instruction count explodes and wall time increases despite better occupancy.

> **Applicable when**: Kernel has low achieved occupancy (<65%) and low eligible warps/scheduler (<2.0) due to small XBLOCK/block size, with a large total element count (xnumel >> SM_count * XBLOCK).

> **Avoid when**: The grid launch site is not simultaneously updated to use a lambda reading meta['XBLOCK']. Also avoid adding xmask = xindex < xnumel when xnumel is already a power-of-2 and the original kernel used tl.full([XBLOCK], True, tl.int1) — this adds unnecessary predication overhead with no correctness benefit.

#### ❌ Plan B: `increase_xblock_tile_size_with_autotune` (0.599×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) to a pointwise kernel that was bottlenecked by L1TEX stalls caused a 67% runtime regression. Although achieved occupancy improved from 57% to 89% and compute SM throughput rose from 62% to 83%, the total executed instruction count more than doubled (20.9M → 46.9M) and registers per thread increased from 22 to 32, which reduced the block limit from 21 to 16 blocks/SM. The grid size remained unchanged at 65536 (the autotune did not select a larger tile), indicating the autotuner selected a config that issued far more instructions per element without reducing grid overhead. The net effect was a 67% slowdown despite better occupancy, because the instruction explosion outweighed any latency-hiding benefit.

> **Applicable when**: Kernel has very low achieved occupancy (<50%) AND L1TEX stall is the dominant bottleneck AND instruction count is low AND registers per thread are well below the block-limit threshold

> **Avoid when**: The kernel already has moderate occupancy (>55%) and the primary bottleneck is L1TEX stall caused by small tiles; adding autotune with larger XBLOCK configs can cause the autotuner to select a configuration that dramatically increases instruction count (2x+) and register pressure without actually reducing grid size, resulting in severe regression. Also avoid when xnumel is not cleanly divisible by all candidate XBLOCK values or when the grid lambda is not updated to use meta['XBLOCK'].

#### ❌ Plan C: `autotune_xblock_num_warps_for_occupancy_and_latency_hiding` (0.601×)

Adding @triton.autotune with larger XBLOCK configs (512–4096) increased achieved occupancy from 57% to 89% and reduced the CPIStall bottleneck, but total executed instructions ballooned from ~21M to ~47M (2.24x) and registers per thread jumped from 22 to 32, causing the kernel duration to increase from 43µs to 71µs (0.6x speedup). The instruction explosion likely came from the autotune selecting a config that unrolled more work per thread or introduced additional predicated mask computations (xmask = xindex < xnumel replacing tl.full([XBLOCK], True)), combined with the restore_value=['in_out_ptr0'] overhead. When a pointwise kernel is already compute-bound (62% SM throughput) and the instruction count is the real limiter, increasing XBLOCK inflates per-thread work without proportional throughput gains, resulting in a net regression despite better occupancy.

> **Applicable when**: Kernel is memory-latency-bound (CPIStall >40% from L1TEX), achieved occupancy is far below theoretical (e.g. <60%), and the kernel has very few instructions per thread — increasing XBLOCK would genuinely hide latency without inflating instruction count.

> **Avoid when**: Kernel is already partially compute-bound (SM throughput >60%), instruction count is the primary limiter, or the autotune adds restore_value overhead for in-place buffers. Also avoid when the grid is not updated to use META['XBLOCK'] — a static grid with removed XBLOCK kwarg will cause incorrect tiling. Do not introduce explicit xmask when xnumel is a power-of-2 and all XBLOCK configs divide it evenly, as the mask adds unnecessary predication instructions.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fma_fusion_via_rsqrt_and_arithmetic_restructuring | 0.892× | 0.907× | ✓ |  |
| B | fused_fp32_fma_arithmetic_with_clean_fixed_grid | 1.000× | 1.000× | ✓ |  |
| C | replace_sqrt_div_with_rsqrt_to_enable_fma_fusion | 0.888× | 0.897× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fma_fusion_via_rs... | Δ | fused_fp32_fma_ar... | Δ | replace_sqrt_div_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 42940 ns | 47360 ns | +4420 ns (+10.3%) | 42940 ns | +0 ns (+0.0%) | 47870 ns | +4930 ns (+11.5%) |
| SM Throughput (%) | 62.10 % | 66.25 % | +4.15 pp | 60.26 % | -1.84 pp | 66.34 % | +4.24 pp |
| Memory Throughput (%) | 31.31 % | 44.30 % | +12.99 pp | 31.22 % | -0.09 pp | 44.26 % | +12.95 pp |
| DRAM Throughput (%) | 19.41 % | 17.50 % | -1.91 pp | 19.48 % | +0.07 pp | 17.43 % | -1.98 pp |
| Achieved Occupancy (%) | 57.24 % | 45.77 % | -11.47 pp | 56.51 % | -0.73 pp | 46.25 % | -10.99 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 22.00 register/thread | +0 | 32.00 register/thread | +10 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 33.89 % | 29.51 % | -4.38 pp | 36.07 % | +2.18 pp | 29.53 % | -4.36 pp |
| IPC Active | 2.60 inst/cycle | 2.82 inst/cycle | +0.22 | 2.54 inst/cycle | -0.06 | 2.82 inst/cycle | +0.22 |

---

## Cumulative Progress

```
Baseline:  42940 ns  (1.000×)

Final best: 42940 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,21 +5,34 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 4096}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=4,  num_stages=2),
+    ],
+    key=['xnumel'],
+    restore_value=['in_out_ptr0'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_addmm_mul_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
     xnumel = 8388608
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 8192)
     x2 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_out_ptr0 + (x2), None)
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
-    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
+    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
+    tmp18 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp4 = tmp2 * tmp3
     tmp6 = tmp4 - tmp5
@@ -33,9 +46,7 @@

     tmp15 = tmp6 * tmp14
     tmp17 = tmp15 * tmp16
     tmp19 = tmp17 + tmp18
-    tl.store(in_out_ptr0 + (x2), tmp19, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -47,7 +58,7 @@

     arg_6 = inputs[6]
     xnumel = 8388608
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_addmm_mul_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 8388608, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_addmm_mul_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 8388608)
     return arg_6
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
