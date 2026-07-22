# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 55940 ns — **Best latency**: 61950 ns (**0.9030×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_35_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (80,1,1), Waves/SM: 0.20
- **Registers/thread**: 94
- **Duration**: 55940 ns
- **SM Throughput**: 7.95 % | **Memory**: 53.78 % | **DRAM**: 0.92 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.55 inst/cycle | **No Eligible**: 86.33 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.996× | 1.000× | ✓ |  |
| B | autotune_tile_sizes_to_eliminate_smem_bank_conflicts_and_improve_occupancy | 1.026× | 0.903× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 55940 ns | 55940 ns | +0 ns (+0.0%) | 61950 ns | +6010 ns (+10.7%) |
| SM Throughput (%) | 7.95 % | 7.97 % | +0.02 pp | 6.31 % | -1.64 pp |
| Memory Throughput (%) | 53.78 % | 53.91 % | +0.13 pp | 45.43 % | -8.35 pp |
| DRAM Throughput (%) | 0.92 % | 0.92 % | +0.00 pp | 0.83 % | -0.09 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 94.00 register/thread | 94.00 register/thread | +0 | 96.00 register/thread | +2 |
| Waves/SM | 0.20 | 0.20 | +0.00 | 0.06 | -0.14 |
| No Eligible (%) | 86.33 % | 86.33 % | +0.00 pp | 78.57 % | -7.76 pp |
| IPC Active | 0.55 inst/cycle | 0.55 inst/cycle | +0.00 | 0.43 inst/cycle | -0.12 |

---

## Cumulative Progress

```
Baseline:  55940 ns  (1.000×)

Final best: 61950 ns  (0.9030×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,21 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=1),
+        triton.Config({}, num_warps=8, num_stages=2),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = False
@@ -12,7 +27,7 @@

     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 32
     BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 128
+    BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -90,8 +105,6 @@

     xindex = idx_n + 160*idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
