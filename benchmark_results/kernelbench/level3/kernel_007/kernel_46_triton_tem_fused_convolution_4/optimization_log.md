# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_46` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 30400 ns — **Best latency**: 30400 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_46`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_46_triton_tem_fused_convolution_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (62,1,1), Waves/SM: 0.06
- **Registers/thread**: 64
- **Duration**: 30400 ns
- **SM Throughput**: 5.91 % | **Memory**: 38.50 % | **DRAM**: 2.73 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.54 inst/cycle | **No Eligible**: 86.55 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.474× | 0.450× | ✓ |  |
| B | tma_loads_and_autotune_tile_sizes | 0.984× | 0.593× | ✓ |  |
| C | memory_coalescing_via_autotune_tile_sizes | 0.802× | 0.786× | ✓ |  |
| D | autotune_tile_sizes_to_fix_underutilization_and_bank_conflicts | 0.695× | 0.526× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_and_aut... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 30400 ns | 67490 ns | +37090 ns (+122.0%) | 51300 ns | +20900 ns (+68.8%) | 38690 ns | +8290 ns (+27.3%) | 57820 ns | +27420 ns (+90.2%) |
| SM Throughput (%) | 5.91 % | 9.02 % | +3.11 pp | 4.16 % | -1.75 pp | 8.48 % | +2.57 pp | 5.45 % | -0.46 pp |
| Memory Throughput (%) | 38.50 % | 34.99 % | -3.51 pp | 43.24 % | +4.74 pp | 40.41 % | +1.91 pp | 40.29 % | +1.79 pp |
| DRAM Throughput (%) | 2.73 % | 1.24 % | -1.49 pp | 1.62 % | -1.11 pp | 2.16 % | -0.57 pp | 1.44 % | -1.29 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 12.50 % | +6.25 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 64.00 register/thread | 128.00 register/thread | +64 | 54.00 register/thread | -10 | 116.00 register/thread | +52 | 96.00 register/thread | +32 |
| Waves/SM | 0.06 | 0.16 | +0.10 | 0.12 | +0.06 | 0.16 | +0.10 | 0.09 | +0.03 |
| No Eligible (%) | 86.55 % | 79.70 % | -6.85 pp | 90.79 % | +4.24 pp | 80.93 % | -5.62 pp | 87.90 % | +1.35 pp |
| IPC Active | 0.54 inst/cycle | 0.82 inst/cycle | +0.28 | 0.37 inst/cycle | -0.17 | 0.76 inst/cycle | +0.22 | 0.48 inst/cycle | -0.06 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | shared_memory_padding_to_eliminate_bank_conflicts | 0.612× | 0.587× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_pad... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 30400 ns | 51780 ns | +21380 ns (+70.3%) |
| SM Throughput (%) | 5.91 % | 6.29 % | +0.38 pp |
| Memory Throughput (%) | 38.50 % | 42.71 % | +4.21 pp |
| DRAM Throughput (%) | 2.73 % | 1.61 % | -1.12 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 64.00 register/thread | 96.00 register/thread | +32 |
| Waves/SM | 0.06 | 0.12 | +0.06 |
| No Eligible (%) | 86.55 % | 86.12 % | -0.43 pp |
| IPC Active | 0.54 inst/cycle | 0.56 inst/cycle | +0.02 |

---

## Cumulative Progress

```
Baseline:  30400 ns  (1.000×)

Final best: 30400 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,14 +5,27 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_convolution_46(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 32
-    BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 32
+    BLOCK_M : tl.constexpr = 128
+    BLOCK_N : tl.constexpr = 16
+    BLOCK_K : tl.constexpr = 64
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -89,8 +102,6 @@

     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
