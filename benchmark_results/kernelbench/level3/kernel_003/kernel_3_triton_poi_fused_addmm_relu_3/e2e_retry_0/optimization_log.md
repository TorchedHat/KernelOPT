# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_relu_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7580 ns — **Best latency**: 7520 ns (**1.0080×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_xblock_and_num_warps_for_pointwise_kernel`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_addmm_relu_3`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_003/kernel_3_triton_poi_fused_addmm_relu_3/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (8192,1,1), Waves/SM: 3.88
- **Registers/thread**: 16
- **Duration**: 7580 ns
- **SM Throughput**: 21.16 % | **Memory**: 14.90 % | **DRAM**: 11.31 %
- **Achieved Occupancy**: 42.76 %
- **IPC Active**: 1.26 inst/cycle | **No Eligible**: 67.89 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.992× | 1.004× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_pointwise_kernel | 1.031× | 1.008× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7580 ns | 7550 ns | -30 ns (-0.4%) | 7520 ns | -60 ns (-0.8%) |
| SM Throughput (%) | 21.16 % | 24.62 % | +3.46 pp | 37.73 % | +16.57 pp |
| Memory Throughput (%) | 14.90 % | 14.48 % | -0.42 pp | 20.57 % | +5.67 pp |
| DRAM Throughput (%) | 11.31 % | 11.40 % | +0.09 pp | 11.45 % | +0.14 pp |
| Achieved Occupancy (%) | 42.76 % | 37.09 % | -5.67 pp | 38.45 % | -4.31 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 24.00 register/thread | +8 |
| Waves/SM | 3.88 | 3.88 | +0.00 | 3.88 | +0.00 |
| No Eligible (%) | 67.89 % | 57.85 % | -10.04 pp | 36.51 % | -31.38 pp |
| IPC Active | 1.26 inst/cycle | 1.63 inst/cycle | +0.37 | 2.44 inst/cycle | +1.18 |

---

## Cumulative Progress

```
Baseline:  7580 ns  (1.000×)
Iter 1     7520 ns  (1.008×)  — autotune_xblock_and_num_warps_for_pointwise_kernel

Final best: 7520 ns  (1.0080×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,9 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1024}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 4096}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 4096}, num_warps=16, num_stages=3),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_addmm_relu_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
-    xnumel = 1048576
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
@@ -20,13 +31,12 @@

     tmp4 = tl.maximum(tmp3, tmp2)
     tl.store(in_out_ptr0 + (x2), tmp4, xmask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 1048576
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_addmm_relu_3[grid](arg_0, arg_1, 1048576, XBLOCK=128)
+    triton_poi_fused_addmm_relu_3[grid](arg_0, arg_1, 1048576)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
