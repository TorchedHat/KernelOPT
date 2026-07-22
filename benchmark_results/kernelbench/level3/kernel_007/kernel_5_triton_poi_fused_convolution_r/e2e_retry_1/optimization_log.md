# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_relu_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 918880 ns — **Best latency**: 918500 ns (**1.0004×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_with_xblock_increase`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_relu_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_5_triton_poi_fused_convolution_r/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1492992,1,1), Waves/SM: 706.91
- **Registers/thread**: 16
- **Duration**: 918880 ns
- **SM Throughput**: 23.80 % | **Memory**: 33.26 % | **DRAM**: 33.26 %
- **Achieved Occupancy**: 41.18 %
- **IPC Active**: 0.95 inst/cycle | **No Eligible**: 76.04 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_xblock_increase | 1.000× | 1.000× | ✓ | **✓ best** |
| C | add_triton_autotune | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 918880 ns | 918500 ns | -380 ns (-0.0%) | 918780 ns | -100 ns (-0.0%) |
| SM Throughput (%) | 23.80 % | 23.80 % | +0.00 pp | 27.08 % | +3.28 pp |
| Memory Throughput (%) | 33.26 % | 33.28 % | +0.02 pp | 33.27 % | +0.01 pp |
| DRAM Throughput (%) | 33.26 % | 33.28 % | +0.02 pp | 33.27 % | +0.01 pp |
| Achieved Occupancy (%) | 41.18 % | 41.21 % | +0.03 pp | 41.54 % | +0.36 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 706.91 | 706.91 | +0.00 | 706.91 | +0.00 |
| No Eligible (%) | 76.04 % | 76.05 % | +0.01 pp | 72.76 % | -3.28 pp |
| IPC Active | 0.95 inst/cycle | 0.96 inst/cycle | +0.01 | 1.09 inst/cycle | +0.14 |

---

## Cumulative Progress

```
Baseline:  918880 ns  (1.000×)
Iter 1     918500 ns  (1.000×)  — vectorized_loads_with_xblock_increase

Final best: 918500 ns  (1.0004×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,22 +5,21 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused_convolution_relu_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 191102976
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
+    xindex = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK)[:], XBLOCK), XBLOCK)
     xmask = tl.full([XBLOCK], True, tl.int1)[:]
     x2 = xindex
-    x0 = (xindex % 256)
+    x0 = tl.max_contiguous(tl.multiple_of(xindex % 256, 256), 256)
     tmp0 = tl.load(in_out_ptr0 + (x2), None)
     tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp3 = tl.full([1], 0, tl.int32)
     tmp4 = tl.maximum(tmp3, tmp2)
     tl.store(in_out_ptr0 + (x2), tmp4, None)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
