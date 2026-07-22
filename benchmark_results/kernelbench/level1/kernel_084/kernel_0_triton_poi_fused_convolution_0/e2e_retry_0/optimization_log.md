# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5890 ns — **Best latency**: 3680 ns (**1.6005×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_and_coalesced_stores_via_block_reshape`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_084/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5890 ns
- **SM Throughput**: 15.96 % | **Memory**: 42.96 % | **DRAM**: 20.93 %
- **Achieved Occupancy**: 55.79 %
- **IPC Active**: 1.09 inst/cycle | **No Eligible**: 70.82 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_and_coalesced_stores_via_block_reshape | 1.403× | 1.601× | ✓ | **✓ best** |
| B | autotune_tile_sizes_to_reduce_l1tex_stalls | 1.118× | 0.990× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5890 ns | 3680 ns | -2210 ns (-37.5%) | 5950 ns | +60 ns (+1.0%) |
| SM Throughput (%) | 15.96 % | 26.29 % | +10.33 pp | 16.08 % | +0.12 pp |
| Memory Throughput (%) | 42.96 % | 18.47 % | -24.49 pp | 41.64 % | -1.32 pp |
| DRAM Throughput (%) | 20.93 % | 4.48 % | -16.45 pp | 20.72 % | -0.21 pp |
| Achieved Occupancy (%) | 55.79 % | 65.61 % | +9.82 pp | 55.20 % | -0.59 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 |
| No Eligible (%) | 70.82 % | 42.96 % | -27.86 pp | 70.64 % | -0.18 pp |
| IPC Active | 1.09 inst/cycle | 2.21 inst/cycle | +1.12 | 1.13 inst/cycle | +0.04 |

---

## Cumulative Progress

```
Baseline:  5890 ns  (1.000×)
Iter 1     3680 ns  (1.601×)  — vectorized_loads_and_coalesced_stores_via_block_reshape

Final best: 3680 ns  (1.6005×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,16 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=4),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=4),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=8),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=8),
+], key=['ynumel', 'xnumel'])
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +23,7 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +32,13 @@

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

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
