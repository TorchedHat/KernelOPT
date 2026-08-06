# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5920 ns — **Best latency**: 5570 ns (**1.0628×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`vectorized_loads_via_max_contiguous_multiple_of_hints`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_073/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5920 ns
- **SM Throughput**: 15.61 % | **Memory**: 41.69 % | **DRAM**: 20.88 %
- **Achieved Occupancy**: 55.00 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 70.49 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_wider_xblock | 1.001× | 1.005× | ✓ |  |
| B | autotune_tile_sizes_for_coalesced_access_and_full_wave_utilization | 1.292× | 1.028× | ✓ | **✓ best** |
| C | memory_coalescing_via_autotune_tile_sizes | 1.137× | 0.607× | ✓ |  |
| D | add_autotune_increase_tile_sizes | 1.036× | 0.656× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | memory_coalescing... | Δ | add_autotune_incr... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5920 ns | 5890 ns | -30 ns (-0.5%) | 5760 ns | -160 ns (-2.7%) | 9760 ns | +3840 ns (+64.9%) | 9020 ns | +3100 ns (+52.4%) |
| SM Throughput (%) | 15.61 % | 19.63 % | +4.02 pp | 18.76 % | +3.15 pp | 13.80 % | -1.81 pp | 27.79 % | +12.18 pp |
| Memory Throughput (%) | 41.69 % | 42.04 % | +0.35 pp | 30.64 % | -11.05 pp | 32.00 % | -9.69 pp | 49.79 % | +8.10 pp |
| DRAM Throughput (%) | 20.88 % | 20.95 % | +0.07 pp | 10.72 % | -10.16 pp | 6.33 % | -14.55 pp | 13.65 % | -7.23 pp |
| Achieved Occupancy (%) | 55.00 % | 55.82 % | +0.82 pp | 47.32 % | -7.68 pp | 53.96 % | -1.04 pp | 53.13 % | -1.87 pp |
| Registers/Thread | 26.00 register/thread | 28.00 register/thread | +2 | 26.00 register/thread | +0 | 32.00 register/thread | +6 | 40.00 register/thread | +14 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.99 | +0.25 |
| No Eligible (%) | 70.49 % | 65.88 % | -4.61 pp | 65.20 % | -5.29 pp | 73.44 % | +2.95 pp | 56.74 % | -13.75 pp |
| IPC Active | 1.15 inst/cycle | 1.24 inst/cycle | +0.09 | 1.47 inst/cycle | +0.32 | 1.04 inst/cycle | -0.11 | 1.72 inst/cycle | +0.57 |


### Key Learnings

#### ❌ Plan D: `add_autotune_increase_tile_sizes` (0.656×)

Adding @triton.autotune with larger XBLOCK/YBLOCK configs (up to 1024x4) caused registers per thread to jump from 26 to 40, reducing theoretical occupancy from 100% to 75% and increasing dynamic shared memory from 4 KB to 16 KB per block. Despite reducing warp cycles per issued instruction (30.2→19.61) and increasing instruction throughput, the total executed instruction count more than doubled (771K→1847K) and elapsed cycles grew by 53% (8788→13422), resulting in a 0.66x slowdown. The autotune overhead combined with register pressure from larger tiles negated any cache-reuse benefit for this small, latency-bound kernel.

> **Applicable when**: Never — this specific pattern caused regression. In general, autotune with larger tiles is worth trying when registers_per_thread is low (<20), theoretical occupancy is already limited by other factors, and the kernel is compute-bound rather than latency-bound.

> **Avoid when**: Kernel already has low register pressure (26 registers/thread) and 100% theoretical occupancy. Adding larger tile configs via autotune will increase register usage, reduce theoretical occupancy, and increase instruction count without proportional cache benefit. Avoid when the kernel is small and latency-bound (duration <10us, grid <2000 blocks) with uncoalesced stores that cannot be fixed by tiling alone.

---

## Iteration 2

**Best so far**: 1.0278× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_max_contiguous_multiple_of_hints | 1.346× | 1.063× | ✓ | **✓ best** |
| B | increase_num_warps_for_l1tex_latency_hiding | 1.030× | 1.028× | ✓ |  |
| C | vectorized_loads_with_contiguous_hints_and_remove_autotune_regression | 1.024× | 1.000× | ✓ |  |
| D | vectorized_loads_via_contiguous_hints | 0.993× | 0.964× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | increase_num_warp... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5920 ns | 5570 ns | -350 ns (-5.9%) | 5760 ns | -160 ns (-2.7%) | 5920 ns | +0 ns (+0.0%) | 6140 ns | +220 ns (+3.7%) |
| SM Throughput (%) | 15.61 % | 18.03 % | +2.42 pp | 19.00 % | +3.39 pp | 16.86 % | +1.25 pp | 16.95 % | +1.34 pp |
| Memory Throughput (%) | 41.69 % | 32.16 % | -9.53 pp | 42.46 % | +0.77 pp | 42.53 % | +0.84 pp | 40.29 % | -1.40 pp |
| DRAM Throughput (%) | 20.88 % | 11.08 % | -9.80 pp | 21.35 % | +0.47 pp | 20.81 % | -0.07 pp | 20.00 % | -0.88 pp |
| Achieved Occupancy (%) | 55.00 % | 47.37 % | -7.63 pp | 55.23 % | +0.23 pp | 55.43 % | +0.43 pp | 55.13 % | +0.13 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 28.00 register/thread | +2 | 26.00 register/thread | +0 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.49 % | 65.14 % | -5.35 pp | 66.34 % | -4.15 pp | 70.83 % | +0.34 pp | 71.00 % | +0.51 pp |
| IPC Active | 1.15 inst/cycle | 1.37 inst/cycle | +0.22 | 1.31 inst/cycle | +0.16 | 1.13 inst/cycle | -0.02 | 1.17 inst/cycle | +0.02 |


### Key Learnings

#### ✅ Plan A: `vectorized_loads_via_max_contiguous_multiple_of_hints` (1.063×)

Adding `tl.multiple_of(pid * BLOCK, BLOCK)` and `tl.max_contiguous(arange(BLOCK), BLOCK)` as compiler hints for the load index tells the Triton compiler that the access is BLOCK-aligned and contiguous, enabling 128-bit vectorized LDG instructions instead of scalar 32-bit loads. This reduced elapsed cycles from 8562 to 8268 (~3.4%) and improved memory throughput from 523 to 541 GB/s and L2 hit rate from 65.5% to 67.0%, with zero register overhead (registers per thread unchanged at 26) and no occupancy impact. This is a zero-cost compiler hint applicable whenever the load address is structurally stride-1 in the innermost dimension but the compiler cannot statically prove alignment.

> **Applicable when**: Load address has stride-1 in the innermost (x) dimension; XBLOCK is a power of 2 and >= 64; kernel is flagged by UncoalescedGlobalAccess or MemoryCacheAccessPattern NCU rules on the load side; registers per thread is already low (no room for tile-size increases); kernel duration is short (< 10 µs) making autotune overhead undesirable.

> **Avoid when**: The load address has a non-unit stride (e.g., stride-3 channel-interleaved layout) — hints will be semantically incorrect and may cause wrong results or no benefit. Do not apply to the store address if it has a structural non-unit stride. Do not apply if the bottleneck is purely on the store side (uncoalesced stores with non-unit stride cannot be fixed by load hints).

---

## Iteration 3

**Best so far**: 1.0628× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_eviction_policy_for_single_pass_load | 1.236× | 0.983× | ✓ |  |
| C | streaming_load_eviction_policy_evict_first | 0.881× | 0.983× | ✓ |  |
| D | store_evict_first_to_free_l1_for_loads | 0.998× | 0.983× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_evictio... | Δ | streaming_load_ev... | Δ | store_evict_first... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5920 ns | 6020 ns | +100 ns (+1.7%) | 6020 ns | +100 ns (+1.7%) | 6020 ns | +100 ns (+1.7%) |
| SM Throughput (%) | 15.61 % | 17.87 % | +2.26 pp | 15.90 % | +0.29 pp | 17.79 % | +2.18 pp |
| Memory Throughput (%) | 41.69 % | 29.20 % | -12.49 pp | 41.72 % | +0.03 pp | 40.16 % | -1.53 pp |
| DRAM Throughput (%) | 20.88 % | 10.28 % | -10.60 pp | 20.48 % | -0.40 pp | 20.60 % | -0.28 pp |
| Achieved Occupancy (%) | 55.00 % | 46.37 % | -8.63 pp | 55.60 % | +0.60 pp | 56.26 % | +1.26 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 26.00 register/thread | +0 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.49 % | 63.82 % | -6.67 pp | 70.66 % | +0.17 pp | 68.77 % | -1.72 pp |
| IPC Active | 1.15 inst/cycle | 1.42 inst/cycle | +0.27 | 1.13 inst/cycle | -0.02 | 1.21 inst/cycle | +0.06 |

---

## Iteration 4

**Best so far**: 1.0628× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_yblock_to_increase_grid_waves_and_sm_utilization | 0.881× | 0.990× | ✓ |  |
| C | cache_streaming_store_to_bypass_l1_scatter_pollution | 1.026× | 0.969× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_yblock_to_... | Δ | cache_streaming_s... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5920 ns | 5980 ns | +60 ns (+1.0%) | 6110 ns | +190 ns (+3.2%) |
| SM Throughput (%) | 15.61 % | 16.71 % | +1.10 pp | 16.74 % | +1.13 pp |
| Memory Throughput (%) | 41.69 % | 41.65 % | -0.04 pp | 41.47 % | -0.22 pp |
| DRAM Throughput (%) | 20.88 % | 20.52 % | -0.36 pp | 20.30 % | -0.58 pp |
| Achieved Occupancy (%) | 55.00 % | 55.66 % | +0.66 pp | 55.59 % | +0.59 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.49 % | 70.76 % | +0.27 pp | 70.99 % | +0.50 pp |
| IPC Active | 1.15 inst/cycle | 1.14 inst/cycle | -0.01 | 1.15 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  5920 ns  (1.000×)
Iter 1     5760 ns  (1.028×)  — autotune_tile_sizes_for_coalesced_access_and_full_wave_utilization
Iter 2     5570 ns  (1.063×)  — vectorized_loads_via_max_contiguous_multiple_of_hints

Final best: 5570 ns  (1.0628×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=8),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=2),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 32}, num_warps=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -12,17 +26,15 @@

     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xoffset = tl.multiple_of(tl.program_id(0) * XBLOCK, XBLOCK)
+    xindex = xoffset + tl.max_contiguous(tl.arange(0, XBLOCK), XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
     y0 = (yindex % 3)
     y1 = yindex // 3
-    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_first')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +42,7 @@

     ynumel = 30
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
