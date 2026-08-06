# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_relu_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 244510 ns — **Best latency**: 244160 ns (**1.0014×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`eviction_policies_streaming_stores`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_relu_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_2_triton_poi_fused_convolution_r/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (393216,1,1), Waves/SM: 186.18
- **Registers/thread**: 16
- **Duration**: 244510 ns
- **SM Throughput**: 32.70 % | **Memory**: 48.15 % | **DRAM**: 48.15 %
- **Achieved Occupancy**: 45.52 %
- **IPC Active**: 0.56 inst/cycle | **No Eligible**: 85.96 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_stores | 1.000× | 1.001× | ✓ | **✓ best** |
| B | vectorized_loads_larger_xblock | 1.000× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 244510 ns | 244160 ns | -350 ns (-0.1%) | 244350 ns | -160 ns (-0.1%) |
| SM Throughput (%) | 32.70 % | 32.55 % | -0.15 pp | 32.65 % | -0.05 pp |
| Memory Throughput (%) | 48.15 % | 48.12 % | -0.03 pp | 48.15 % | +0.00 pp |
| DRAM Throughput (%) | 48.15 % | 48.12 % | -0.03 pp | 48.15 % | +0.00 pp |
| Achieved Occupancy (%) | 45.52 % | 45.64 % | +0.12 pp | 45.71 % | +0.19 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 186.18 | +0.00 |
| No Eligible (%) | 85.96 % | 81.03 % | -4.93 pp | 86.04 % | +0.08 pp |
| IPC Active | 0.56 inst/cycle | 0.76 inst/cycle | +0.20 | 0.56 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  244510 ns  (1.000×)
Iter 1     244160 ns  (1.001×)  — eviction_policies_streaming_stores

Final best: 244160 ns  (1.0014×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_poi_fused_convolution_relu_2(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
@@ -15,10 +16,8 @@

     tmp0 = tl.load(in_ptr0 + (x0), None)
     tmp1 = tl.full([1], 0, tl.int32)
     tmp2 = tl.maximum(tmp1, tmp0)
-    tl.store(out_ptr0 + (x0), tmp2, None)
-    tl.store(out_ptr1 + (x0), tmp2, None)
-
-
+    tl.store(out_ptr0 + (x0), tmp2, None, eviction_policy='evict_first')
+    tl.store(out_ptr1 + (x0), tmp2, None, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
