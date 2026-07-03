# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 176000 ns — **Best latency**: 174820 ns (**1.0067×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`safe_autotune_minimal_configs`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_065/kernel_2_triton_poi_fused_convolution_2/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (73728,1,1), Waves/SM: 34.91
- **Registers/thread**: 32
- **Duration**: 176000 ns
- **SM Throughput**: 28.46 % | **Memory**: 81.02 % | **DRAM**: 66.73 %
- **Achieved Occupancy**: 94.22 %
- **IPC Active**: 1.13 inst/cycle | **No Eligible**: 71.56 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | safe_autotune_minimal_configs | 1.815× | 1.007× | ✓ | **✓ best** |
| B | shared_memory_transpose_tile_for_coalesced_nhwc_to_nchw | 0.997× | 0.997× | ✓ |  |
| D | shared_memory_transpose_for_nhwc_to_nchw_coalescing | 1.000× | 1.009× | ✓ |  |


### Metric Comparison

| Metric | Baseline | safe_autotune_min... | Δ | shared_memory_tra... | Δ | shared_memory_tra... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 176000 ns | 174820 ns | -1180 ns (-0.7%) | 176480 ns | +480 ns (+0.3%) | 174500 ns | -1500 ns (-0.9%) |
| SM Throughput (%) | 28.46 % | 26.85 % | -1.61 pp | 25.72 % | -2.74 pp | 27.66 % | -0.80 pp |
| Memory Throughput (%) | 81.02 % | 80.98 % | -0.04 pp | 80.97 % | -0.05 pp | 81.58 % | +0.56 pp |
| DRAM Throughput (%) | 66.73 % | 67.20 % | +0.47 pp | 66.58 % | -0.15 pp | 67.31 % | +0.58 pp |
| Achieved Occupancy (%) | 94.22 % | 94.79 % | +0.57 pp | 94.11 % | -0.11 pp | 94.23 % | +0.01 pp |
| Registers/Thread | 32.00 register/thread | 31.00 register/thread | -1 | 28.00 register/thread | -4 | 32.00 register/thread | +0 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.56 % | 73.21 % | +1.65 pp | 73.95 % | +2.39 pp | 70.94 % | -0.62 pp |
| IPC Active | 1.13 inst/cycle | 1.08 inst/cycle | -0.05 | 1.03 inst/cycle | -0.10 | 1.17 inst/cycle | +0.04 |

---

## Iteration 3

**Best so far**: 1.0067× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_static_contiguous_hints | 1.000× | 1.007× | ✓ |  |
| B | vectorized_loads_store_hints_fixed_config | 1.006× | 1.010× | ✓ |  |
| C | vectorized_store_hints_for_coalesced_nchw_output | 0.997× | 0.998× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ | vectorized_store_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 176000 ns | 174750 ns | -1250 ns (-0.7%) | 174240 ns | -1760 ns (-1.0%) | 176420 ns | +420 ns (+0.2%) |
| SM Throughput (%) | 28.46 % | 28.13 % | -0.33 pp | 26.64 % | -1.82 pp | 26.07 % | -2.39 pp |
| Memory Throughput (%) | 81.02 % | 81.25 % | +0.23 pp | 81.10 % | +0.08 pp | 81.01 % | -0.01 pp |
| DRAM Throughput (%) | 66.73 % | 67.18 % | +0.45 pp | 67.38 % | +0.65 pp | 66.58 % | -0.15 pp |
| Achieved Occupancy (%) | 94.22 % | 94.20 % | -0.02 pp | 94.75 % | +0.53 pp | 94.16 % | -0.06 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 31.00 register/thread | -1 | 28.00 register/thread | -4 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.56 % | 71.66 % | +0.10 pp | 72.44 % | +0.88 pp | 73.71 % | +2.15 pp |
| IPC Active | 1.13 inst/cycle | 1.16 inst/cycle | +0.03 | 1.07 inst/cycle | -0.06 | 1.04 inst/cycle | -0.09 |

---

## Iteration 4

**Best so far**: 1.0067× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | swap_tile_dims_for_channel_coalesced_loads | 1.008× | 1.011× | ✓ |  |
| B | swap_fast_thread_dim_to_channel_for_coalesced_nhwc_load | 0.999× | 1.008× | ✓ |  |
| C | evict_first_strided_load_plus_correct_vectorized_store_hints | 0.767× | 0.798× | ✓ |  |


### Metric Comparison

| Metric | Baseline | swap_tile_dims_fo... | Δ | swap_fast_thread_... | Δ | evict_first_strid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 176000 ns | 174080 ns | -1920 ns (-1.1%) | 174590 ns | -1410 ns (-0.8%) | 220670 ns | +44670 ns (+25.4%) |
| SM Throughput (%) | 28.46 % | 26.24 % | -2.22 pp | 27.80 % | -0.66 pp | 19.85 % | -8.61 pp |
| Memory Throughput (%) | 81.02 % | 81.01 % | -0.01 pp | 81.63 % | +0.61 pp | 67.47 % | -13.55 pp |
| DRAM Throughput (%) | 66.73 % | 67.44 % | +0.71 pp | 67.25 % | +0.52 pp | 58.74 % | -7.99 pp |
| Achieved Occupancy (%) | 94.22 % | 94.78 % | +0.56 pp | 94.20 % | -0.02 pp | 94.73 % | +0.51 pp |
| Registers/Thread | 32.00 register/thread | 31.00 register/thread | -1 | 32.00 register/thread | +0 | 28.00 register/thread | -4 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.56 % | 73.11 % | +1.55 pp | 70.94 % | -0.62 pp | 79.94 % | +8.38 pp |
| IPC Active | 1.13 inst/cycle | 1.09 inst/cycle | -0.04 | 1.17 inst/cycle | +0.04 | 0.80 inst/cycle | -0.33 |


### Key Learnings

#### ❌ Plan C: `evict_first_strided_load_plus_correct_vectorized_store_hints` (0.798×)

Changing the eviction policy from 'evict_last' to 'evict_first' on a stride-96 load in a memory-bound layout-transpose kernel caused a 20% regression (174 us → 221 us). Despite the theoretical appeal of streaming eviction for non-reusable data, the change worsened all memory metrics: L2 hit rate dropped from 69% to 64%, warp cycles per issued instruction increased from 56 to 76, and eligible warps per scheduler fell from 0.66 to 0.46. The L1 hit rate was essentially unchanged (18.9% → 18.0%), indicating evict_first did not free up useful L1 capacity. The store hint fix (scalar vs 2D-list arguments to tl.multiple_of/tl.max_contiguous) was also applied but the diff shows the 'after' code reverted to using 2D-list arguments [1,16] and [1,XBLOCK] for the store, meaning neither fix was cleanly applied. The fundamental bottleneck — stride-96 loads — cannot be resolved by eviction policy alone.

> **Applicable when**: Never: this specific combination of evict_first on a stride-N load in a layout-transpose kernel caused regression even when L2 was saturated and L1 hit rate was low.

> **Avoid when**: Do not apply evict_first to strided loads in layout-transpose kernels when the kernel is already L2-bound (L2 throughput >80%) and L1 hit rate is low (<20%). The eviction policy change does not improve L1 hit rate for stride-N patterns and increases warp stall cycles. The fundamental issue (stride-N access) requires algorithmic changes (shared memory transpose, dimension reordering) not cache policy hints.

---

## Cumulative Progress

```
Baseline:  176000 ns  (1.000×)
Iter 2     174820 ns  (1.007×)  — safe_autotune_minimal_configs

Final best: 174820 ns  (1.0067×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,26 +5,25 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 24576
     xnumel = 3025
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
-    x2 = xindex
+    x2 = tl.max_contiguous(tl.multiple_of(xindex, [1, XBLOCK]), [1, XBLOCK])
     y0 = (yindex % 96)
     y1 = yindex // 96
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 290400*y1), xmask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 290400*y1), xmask & ymask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr1 + (y0), xmask & ymask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
-    tl.store(out_ptr0 + (x2 + 3025*y3), tmp2, xmask)
-
-
+    tl.store(out_ptr0 + (x2 + 3025*y3), tmp2, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
