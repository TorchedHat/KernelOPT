# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_230` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7260 ns — **Best latency**: 9120 ns (**0.7961×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_230`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_230_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7105,1,1), Waves/SM: 3.36
- **Registers/thread**: 22
- **Duration**: 7260 ns
- **SM Throughput**: 43.57 % | **Memory**: 16.65 % | **DRAM**: 10.34 %
- **Achieved Occupancy**: 62.42 %
- **IPC Active**: 2.58 inst/cycle | **No Eligible**: 35.80 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_to_eliminate_partial_wave | 0.953× | 0.793× | ✓ |  |
| B | autotune_xblock_and_num_warps_to_eliminate_partial_wave_and_hide_l1tex_latency | 1.028× | 0.796× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7260 ns | 9150 ns | +1890 ns (+26.0%) | 9120 ns | +1860 ns (+25.6%) |
| SM Throughput (%) | 43.57 % | 63.38 % | +19.81 pp | 64.31 % | +20.74 pp |
| Memory Throughput (%) | 16.65 % | 17.12 % | +0.47 pp | 17.37 % | +0.72 pp |
| DRAM Throughput (%) | 10.34 % | 8.19 % | -2.15 pp | 8.21 % | -2.13 pp |
| Achieved Occupancy (%) | 62.42 % | 80.98 % | +18.56 pp | 80.91 % | +18.49 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 3.36 | 3.36 | +0.00 | 3.36 | +0.00 |
| No Eligible (%) | 35.80 % | 17.73 % | -18.07 pp | 17.80 % | -18.00 pp |
| IPC Active | 2.58 inst/cycle | 3.27 inst/cycle | +0.69 | 3.27 inst/cycle | +0.69 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_to_eliminate_partial_wave` (0.793×)

Adding @triton.autotune with larger XBLOCK configs (512/1024/2048) to eliminate a partial wave caused a 26% regression. The autotune selected a config that nearly doubled register usage (22→32 registers/thread), which nearly doubled executed instructions (2.4M→4.6M) and increased elapsed cycles by 26% (10770→13597). Although achieved occupancy improved (62%→81%) and compute throughput rose (43%→63%), the instruction count explosion dominated, indicating the selected XBLOCK caused redundant computation or unrolling overhead that outweighed the partial-wave savings. The grid was not updated to use a lambda, so the fixed grid=(cdiv(xnumel,128),) was passed to an autotuned kernel, likely causing a grid/tile mismatch that forced extra iterations.

> **Applicable when**: NCU LaunchConfiguration rule fires with partial wave penalty >20%; kernel is pure pointwise with no shared memory; grid size is not a multiple of SM count.

> **Avoid when**: The grid launch expression is not updated to a lambda that reads meta['XBLOCK']; autotune is added without fixing the grid, causing a tile/grid mismatch that multiplies instruction count. Also avoid when register pressure is already near the block limit (registers_per_thread close to 32), as larger XBLOCK configs may push register usage higher and cause spilling or instruction bloat.

#### ❌ Plan B: `autotune_xblock_and_num_warps_to_eliminate_partial_wave_and_hide_l1tex_latency` (0.796×)

Adding @triton.autotune with larger XBLOCK values (512–2048) and more warps increased registers per thread from 22 to 32 and nearly doubled the executed instruction count (2.4M → 4.6M), causing a 25% wall-time regression despite improving achieved occupancy (62% → 81%) and compute throughput (44% → 64%). The extra instructions outweighed the scheduling gains: warp cycles per issued instruction remained flat (~15.3 → 15.7), meaning latency hiding did not improve, while the kernel simply did more work per element. When a pointwise kernel already has low register pressure and the instruction count is sensitive to XBLOCK (e.g., due to unrolling or boundary-check overhead scaling with tile size), increasing XBLOCK via autotune can inflate instruction count faster than it reduces grid overhead.

> **Applicable when**: Never apply this specific pattern when the kernel has low register pressure (≤22 regs/thread) and the instruction count is already near-minimal for the workload; increasing XBLOCK via autotune in such cases tends to inflate instruction count without proportional latency hiding benefit.

> **Avoid when**: Pointwise kernels where: (1) registers per thread are already low (≤22), (2) executed instruction count is tightly coupled to XBLOCK (boundary checks, loop unrolling), and (3) warp cycles per issued instruction does not decrease after the change — all three conditions were met here and caused regression. Also avoid when the grid is already producing 3+ full waves, as the partial-wave tail benefit is limited.

---

## Cumulative Progress

```
Baseline:  7260 ns  (1.000×)

Final best: 9120 ns  (0.7961×)
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
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=3),
+        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=2),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_230(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 909440
@@ -33,8 +47,6 @@

     tmp17 = tl.maximum(tmp16, tmp15)
     tl.store(out_ptr0 + (x2), tmp17, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -44,7 +56,7 @@

     arg_5 = inputs[5]
     xnumel = 909440
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_relu_230[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 909440, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_relu_230[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 909440)
     return arg_5
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
