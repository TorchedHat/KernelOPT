# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_convolution_mul_native_group_norm_sigmoid_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 704670 ns — **Best latency**: 746750 ns (**0.9436×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_convolution_mul_native_group_norm_sigmoid_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_021/kernel_3_triton_poi_fused_add_convoluti/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (258560,1,1), Waves/SM: 217.64
- **Registers/thread**: 56
- **Duration**: 704670 ns
- **SM Throughput**: 72.93 % | **Memory**: 92.07 % | **DRAM**: 60.61 %
- **Achieved Occupancy**: 53.04 %
- **IPC Active**: 2.92 inst/cycle | **No Eligible**: 26.92 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_autotune_reduce_register_pressure_improve_occupancy | 3.168× | 2.836× | ✓ |  |
| B | reduce_register_pressure_via_autotune_tile_sizes | 3.621× | 0.944× | ✓ | **✓ best** |
| C | vectorized_loads_via_autotune_tile_reshape | 1.887× | 0.820× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_autotune_redu... | Δ | reduce_register_p... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 704670 ns | 248480 ns | -456190 ns (-64.7%) | 746750 ns | +42080 ns (+6.0%) | 859230 ns | +154560 ns (+21.9%) |
| SM Throughput (%) | 72.93 % | 87.82 % | +14.89 pp | 79.68 % | +6.75 pp | 87.04 % | +14.11 pp |
| Memory Throughput (%) | 92.07 % | 67.69 % | -24.38 pp | 88.02 % | -4.05 pp | 69.56 % | -22.51 pp |
| DRAM Throughput (%) | 60.61 % | 42.15 % | -18.46 pp | 57.18 % | -3.43 pp | 5.97 % | -54.64 pp |
| Achieved Occupancy (%) | 53.04 % | 90.89 % | +37.85 pp | 93.73 % | +40.69 pp | 41.58 % | -11.46 pp |
| Registers/Thread | 56.00 register/thread | 29.00 register/thread | -27 | 32.00 register/thread | -24 | 72.00 register/thread | +16 |
| Waves/SM | 217.64 | 122.42 | -95.22 | 244.85 | +27.21 | 279.83 | +62.19 |
| No Eligible (%) | 26.92 % | 11.45 % | -15.47 pp | 19.97 % | -6.95 pp | 12.69 % | -14.23 pp |
| IPC Active | 2.92 inst/cycle | 3.54 inst/cycle | +0.62 | 3.20 inst/cycle | +0.28 | 3.49 inst/cycle | +0.57 |


### Key Learnings

#### ✅ Plan A: `add_autotune_reduce_register_pressure_improve_occupancy` (2.858×)

Adding larger tile configs (XBLOCK=256, YBLOCK=64) to an already-autotuned Triton kernel that is compute-bound at ~88% SM SOL and 100% theoretical occupancy produced essentially no kernel-level improvement: duration dropped only 0.4% (247.49 → 246.53 µs), all memory and compute metrics were statistically unchanged. When a kernel is already at full theoretical occupancy with high compute utilization, expanding the autotune config space beyond what is already explored does not yield meaningful gains because the bottleneck is not tile shape but rather the compute/memory balance itself. The 2.86x reported speedup appears to originate from system-level effects outside the profiled kernel, not from the config expansion.

> **Applicable when**: Kernel is already autotuned with a lambda grid and at 100% theoretical occupancy; MemoryCacheAccessPattern or FPInstructions rules suggest potential gains from larger tiles; kernel has not yet explored XBLOCK=256 or YBLOCK=64.

> **Avoid when**: Kernel is already compute-bound at >85% SM SOL with 100% theoretical occupancy and high achieved occupancy (>90%) — in this regime, expanding the autotune config space beyond what is already explored yields negligible kernel-level improvement (<1%). Do not expect this change alone to produce wall-clock speedup when all key metrics (duration, throughput, occupancy) are already near their ceiling.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_streaming_input_free_l2_for_broadcast_tensors | 3.066× | 2.745× | ✓ |  |
| B | streaming_eviction_policy_and_native_fma_fusion | 2.852× | 0.871× | ✓ |  |
| C | fix_autotune_grid_lambda_and_reduce_register_pressure_via_small_xblock | 3.170× | 2.847× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_strea... | Δ | streaming_evictio... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 704670 ns | 256740 ns | -447930 ns (-63.6%) | 808930 ns | +104260 ns (+14.8%) | 247490 ns | -457180 ns (-64.9%) |
| SM Throughput (%) | 72.93 % | 87.53 % | +14.60 pp | 74.08 % | +1.15 pp | 87.83 % | +14.90 pp |
| Memory Throughput (%) | 92.07 % | 65.39 % | -26.68 pp | 81.71 % | -10.36 pp | 67.70 % | -24.37 pp |
| DRAM Throughput (%) | 60.61 % | 42.71 % | -17.90 pp | 70.55 % | +9.94 pp | 42.32 % | -18.29 pp |
| Achieved Occupancy (%) | 53.04 % | 91.43 % | +38.39 pp | 94.34 % | +41.30 pp | 90.90 % | +37.86 pp |
| Registers/Thread | 56.00 register/thread | 28.00 register/thread | -28 | 32.00 register/thread | -24 | 29.00 register/thread | -27 |
| Waves/SM | 217.64 | 122.42 | -95.22 | 244.85 | +27.21 | 122.42 | -95.22 |
| No Eligible (%) | 26.92 % | 11.60 % | -15.32 pp | 25.80 % | -1.12 pp | 11.39 % | -15.53 pp |
| IPC Active | 2.92 inst/cycle | 3.54 inst/cycle | +0.62 | 2.97 inst/cycle | +0.05 | 3.54 inst/cycle | +0.62 |


### Key Learnings

#### ✅ Plan A: `evict_first_streaming_input_free_l2_for_broadcast_tensors` (2.745×)

When a kernel loads a large non-reused streaming tensor alongside small broadcast tensors that are reused across all grid tiles, marking the streaming tensor's load with eviction_policy='evict_first' is intended to free L2 capacity for the broadcast tensors. However, in this case the NCU profiling metrics show no meaningful improvement: L2 hit rate actually dropped slightly (68.34% → 65.86%), L1/TEX hit rate was unchanged (34.46% → 34.44%), and kernel duration increased marginally (248.48 → 256.74 µs) at the per-kernel level. The reported 2.74x speedup appears to come from system-level effects outside the profiled kernel, not from the eviction policy change itself — the kernel-level metrics show no benefit and a slight regression.

> **Applicable when**: Kernel loads a large streaming tensor (non-reused across CTA tiles) alongside small broadcast tensors (reused across many tiles); MemoryL2Compression rule fires with 0% success rate; L2 hit rate is moderate (60-75%); kernel is memory-bound with high L2 cache throughput.

> **Avoid when**: Kernel is already compute-bound at >85% SM SOL with high occupancy (>90%) — in this regime, L2 eviction policy changes do not translate to wall-clock speedup because the bottleneck is compute, not memory latency. Also avoid when the streaming tensor has partial reuse across tiles or when L2 hit rate is already low (<40%), indicating the broadcast tensors are not benefiting from L2 caching regardless.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_and_swap_tile_dims_for_coalesced_stores | 3.192× | 2.858× | ✓ |  |
| B | fma_chain_fusion_eliminate_non_fused_fp32_intermediates | 3.082× | 2.771× | ✓ |  |
| C | software_pipelining_via_num_stages_to_hide_l1tex_latency | 1.970× | 0.871× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fma_chain_fusion_... | Δ | software_pipelini... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 704670 ns | 246530 ns | -458140 ns (-65.0%) | 254270 ns | -450400 ns (-63.9%) | 809540 ns | +104870 ns (+14.9%) |
| SM Throughput (%) | 72.93 % | 87.82 % | +14.89 pp | 88.06 % | +15.13 pp | 73.92 % | +0.99 pp |
| Memory Throughput (%) | 92.07 % | 67.69 % | -24.38 pp | 65.88 % | -26.19 pp | 81.51 % | -10.56 pp |
| DRAM Throughput (%) | 60.61 % | 42.49 % | -18.12 pp | 43.04 % | -17.57 pp | 70.43 % | +9.82 pp |
| Achieved Occupancy (%) | 53.04 % | 90.92 % | +37.88 pp | 91.80 % | +38.76 pp | 94.34 % | +41.30 pp |
| Registers/Thread | 56.00 register/thread | 29.00 register/thread | -27 | 28.00 register/thread | -28 | 32.00 register/thread | -24 |
| Waves/SM | 217.64 | 122.42 | -95.22 | 122.42 | -95.22 | 244.85 | +27.21 |
| No Eligible (%) | 26.92 % | 11.36 % | -15.56 pp | 11.25 % | -15.67 pp | 25.65 % | -1.27 pp |
| IPC Active | 2.92 inst/cycle | 3.54 inst/cycle | +0.62 | 3.55 inst/cycle | +0.63 | 2.97 inst/cycle | +0.05 |


### Key Learnings

#### ❌ Plan B: `fma_chain_fusion_eliminate_non_fused_fp32_intermediates` (2.771×)

Replacing a FDIV+FADD variance normalization sequence with a single libdevice.fma() call (using a precomputed reciprocal constant) and nesting a second fma() to eliminate a standalone FMUL did not reduce kernel duration — it increased from 247.49 µs to 254.27 µs (~2.7% regression at the kernel level). Despite the NCU FPInstructions rule reporting 71% non-fused FP32 instructions with up to 36% relative speedup potential, the actual instruction count increased (167.5M → 172.7M executed instructions) and the non-fused count only dropped modestly (20.7M → 18.6M). The kernel was already compute-bound at ~88% SM SOL with high occupancy (90.9%), meaning the FMA fusion did not reduce the true bottleneck and the additional libdevice.fma() call overhead slightly increased total instruction count.

> **Applicable when**: Kernel has a high fraction of non-fused FP32 instructions (>60%) per NCU FPInstructions rule AND is NOT already compute-bound (SM SOL <80%) AND has headroom in instruction throughput.

> **Avoid when**: Kernel is already compute-bound at >85% SM SOL with high occupancy (>90%): in this regime, FMA fusion does not reduce the bottleneck and may increase total instruction count due to libdevice call overhead. Also avoid when Triton's compiler already performs automatic FMA fusion — manual libdevice.fma() calls may interfere with compiler optimizations and produce more instructions than the original code.

---

## Cumulative Progress

```
Baseline:  704670 ns  (1.000×)

Final best: 746750 ns  (0.9436×)
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
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 8},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=1),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_add_convolution_mul_native_group_norm_sigmoid_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 4096
@@ -47,7 +61,6 @@

     tmp20 = libdevice.fma(tmp16, tmp17, tmp19)
     tl.store(out_ptr0 + (x2 + 64516*y3), tmp20, xmask & ymask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -61,7 +74,7 @@

     ynumel = 4096
     xnumel = 64516
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_add_convolution_mul_native_group_norm_sigmoid_3[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, 4096, 64516, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_add_convolution_mul_native_group_norm_sigmoid_3[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, 4096, 64516)
     return arg_8
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
