# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5890 ns — **Best latency**: 3330 ns (**1.7688×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_coalescing_and_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_025/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5890 ns
- **SM Throughput**: 17.12 % | **Memory**: 41.75 % | **DRAM**: 20.93 %
- **Achieved Occupancy**: 54.94 %
- **IPC Active**: 1.16 inst/cycle | **No Eligible**: 71.37 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.202× | 1.462× | ✓ |  |
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.407× | 1.769× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5890 ns | 4030 ns | -1860 ns (-31.6%) | 3330 ns | -2560 ns (-43.5%) |
| SM Throughput (%) | 17.12 % | 17.14 % | +0.02 pp | 17.40 % | +0.28 pp |
| Memory Throughput (%) | 41.75 % | 34.25 % | -7.50 pp | 20.78 % | -20.97 pp |
| DRAM Throughput (%) | 20.93 % | 16.39 % | -4.54 pp | 4.99 % | -15.94 pp |
| Achieved Occupancy (%) | 54.94 % | 53.31 % | -1.63 pp | 43.74 % | -11.20 pp |
| Registers/Thread | 26.00 register/thread | 21.00 register/thread | -5 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.37 % | 64.81 % | -6.56 pp | 53.98 % | -17.39 pp |
| IPC Active | 1.16 inst/cycle | 1.33 inst/cycle | +0.17 | 1.70 inst/cycle | +0.54 |


### Key Learnings

#### ✅ Plan A: `vectorized_loads_with_autotune` (1.462×)

When a Triton kernel shows high L1TEX stall cycles (CPIStall ~50%) and uncoalesced global accesses due to a stride-1 load pattern, applying tl.max_contiguous + tl.multiple_of hints to the contiguous load index enables the compiler to emit wider (128-bit) vectorized loads, reducing stall cycles and instruction count. Adding @triton.autotune over XBLOCK/YBLOCK configurations allows the runtime to select the tile size that best balances vectorization width against grid coverage, reducing warp cycles per issued instruction from ~30.5 to ~24.1 and elapsed cycles by ~32%. Register pressure also drops (26→21 registers/thread) as a side effect of the restructured index computation.

> **Applicable when**: Triton pointwise/permute kernel with: (1) CPIStall rule triggered with L1TEX stall >40% of warp cycles, (2) UncoalescedGlobalAccess rule with >10% excessive sectors, (3) load index is stride-1 contiguous in the x-dimension (xindex = xoffset + arange(XBLOCK)), (4) fixed block size launch without autotune, (5) low waves-per-SM (<1.5) indicating under-utilization

> **Avoid when**: Load index is not stride-1 contiguous (e.g., has gather pattern or non-unit stride in x); store index is also stride-1 contiguous (apply hints there instead or in addition); kernel is already autotuned; XBLOCK is not a power-of-2 or not aligned to the actual memory access stride

#### ✅ Plan B: `autotune_tile_sizes_for_coalescing_and_occupancy` (1.769×)

When a Triton kernel performs a layout permutation with a strided store pattern (e.g., stride-3 scatter in the Y dimension), fixing YBLOCK=1 makes the strided index component constant per tile, collapsing the scatter into coalesced writes. Adding @triton.autotune over XBLOCK and YBLOCK configurations lets the runtime discover this optimal tile shape, reducing Warp Cycles Per Issued Instruction from 30.5 to 15.5 and cutting elapsed cycles by 44%. The autotune also reduces register pressure (26→18 registers/thread) and shared memory usage, which together drove the 1.77x speedup despite achieved occupancy slightly decreasing.

> **Applicable when**: Triton 2D-tiled kernel with a strided store pattern where one tile dimension indexes into a small modular value (e.g., y0 = yindex % K for small K like 3), causing stride-K scatter across threads. CPIStall rule fires with L1TEX stall >40% of cycles, UncoalescedGlobalAccess reports >15% excess sectors, and Warp Cycles Per Issued Instruction >25. Kernel has fixed XBLOCK/YBLOCK hardcoded at launch.

> **Avoid when**: Kernel is already autotuned or tile sizes are already optimal. Grid is already large enough to saturate all SMs (waves_per_sm >> 1). The strided access is on loads rather than stores (different fix needed). YBLOCK reduction would create too many blocks and exceed grid dimension limits.

---

## Cumulative Progress

```
Baseline:  5890 ns  (1.000×)
Iter 1     3330 ns  (1.769×)  — autotune_tile_sizes_for_coalescing_and_occupancy

Final best: 3330 ns  (1.7688×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 4}, num_warps=2, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +38,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 30
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
