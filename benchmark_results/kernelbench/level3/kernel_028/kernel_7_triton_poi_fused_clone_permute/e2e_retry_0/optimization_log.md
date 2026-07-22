# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_clone_permute_7` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3330 ns — **Best latency**: 3200 ns (**1.0406×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_xblock_and_num_warps_for_persistent_tile_efficiency`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_clone_permute_7`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_028/kernel_7_triton_poi_fused_clone_permute/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1576,1,1), Waves/SM: 0.75
- **Registers/thread**: 16
- **Duration**: 3330 ns
- **SM Throughput**: 8.95 % | **Memory**: 7.20 % | **DRAM**: 5.05 %
- **Achieved Occupancy**: 55.76 %
- **IPC Active**: 0.83 inst/cycle | **No Eligible**: 72.89 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_and_num_warps_for_persistent_tile_efficiency | 1.043× | 1.041× | ✓ | **✓ best** |
| B | autotune_xblock_and_num_warps_for_permute_kernel | 1.010× | 1.031× | ✓ |  |
| C | add_triton_autotune | 0.993× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_a... | Δ | autotune_xblock_a... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3330 ns | 3200 ns | -130 ns (-3.9%) | 3230 ns | -100 ns (-3.0%) | 3330 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 8.95 % | 4.47 % | -4.48 pp | 4.35 % | -4.60 pp | 8.93 % | -0.02 pp |
| Memory Throughput (%) | 7.20 % | 7.33 % | +0.13 pp | 7.36 % | +0.16 pp | 7.31 % | +0.11 pp |
| DRAM Throughput (%) | 5.05 % | 5.16 % | +0.11 pp | 5.17 % | +0.12 pp | 5.01 % | -0.04 pp |
| Achieved Occupancy (%) | 55.76 % | 27.96 % | -27.80 pp | 26.74 % | -29.02 pp | 53.82 % | -1.94 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 0.75 | 0.37 | -0.38 | 0.37 | -0.38 | 0.75 | +0.00 |
| No Eligible (%) | 72.89 % | 83.45 % | +10.56 pp | 84.14 % | +11.25 pp | 73.07 % | +0.18 pp |
| IPC Active | 0.83 inst/cycle | 0.51 inst/cycle | -0.32 | 0.55 inst/cycle | -0.28 | 0.96 inst/cycle | +0.13 |

---

## Cumulative Progress

```
Baseline:  3330 ns  (1.000×)
Iter 1     3200 ns  (1.041×)  — autotune_xblock_and_num_warps_for_persistent_tile_efficiency

Final best: 3200 ns  (1.0406×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,9 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=2),
+        triton.Config({'XBLOCK': 128},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 2048}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_clone_permute_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
-    xnumel = 201728
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
@@ -18,14 +31,12 @@

     tmp0 = tl.load(in_ptr0 + (x0 + 512*x2 + 1024*x1), xmask)
     tl.store(out_ptr0 + (x3), tmp0, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 201728
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_clone_permute_7[grid](arg_0, arg_1, 201728, XBLOCK=128)
+    triton_poi_fused_clone_permute_7[grid](arg_0, arg_1, 201728)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
