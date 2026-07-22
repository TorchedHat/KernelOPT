# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_224` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6880 ns — **Best latency**: 25600 ns (**0.2687×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_224`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_224_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (6860,1,1), Waves/SM: 3.25
- **Registers/thread**: 22
- **Duration**: 6880 ns
- **SM Throughput**: 42.25 % | **Memory**: 16.10 % | **DRAM**: 10.53 %
- **Achieved Occupancy**: 62.91 %
- **IPC Active**: 2.58 inst/cycle | **No Eligible**: 35.31 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_to_eliminate_partial_wave_and_improve_occupancy | 0.965× | 0.757× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_pointwise_batchnorm_relu | 1.006× | 0.269× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_xblock_t... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6880 ns | 9090 ns | +2210 ns (+32.1%) | 25600 ns | +18720 ns (+272.1%) |
| SM Throughput (%) | 42.25 % | 63.42 % | +21.17 pp | 78.74 % | +36.49 pp |
| Memory Throughput (%) | 16.10 % | 17.12 % | +1.02 pp | 15.67 % | -0.43 pp |
| DRAM Throughput (%) | 10.53 % | 7.97 % | -2.56 pp | 2.83 % | -7.70 pp |
| Achieved Occupancy (%) | 62.91 % | 80.88 % | +17.97 pp | 54.71 % | -8.20 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 48.00 register/thread | +26 |
| Waves/SM | 3.25 | 3.25 | +0.00 | 10.39 | +7.14 |
| No Eligible (%) | 35.31 % | 17.64 % | -17.67 pp | 13.46 % | -21.85 pp |
| IPC Active | 2.58 inst/cycle | 3.24 inst/cycle | +0.66 | 3.45 inst/cycle | +0.87 |

---

## Cumulative Progress

```
Baseline:  6880 ns  (1.000×)

Final best: 25600 ns  (0.2687×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,19 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 2048}, num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 128},  num_warps=4),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_224(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 878080
@@ -33,8 +46,6 @@

     tmp17 = tl.maximum(tmp16, tmp15)
     tl.store(out_ptr0 + (x2), tmp17, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -44,7 +55,7 @@

     arg_5 = inputs[5]
     xnumel = 878080
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_relu_224[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 878080, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_relu_224[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 878080)
     return arg_5
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
