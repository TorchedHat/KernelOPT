# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_relu_t_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1380000 ns — **Best latency**: 1380000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_relu_t_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_003/kernel_0_triton_tem_fused_addmm_relu_t_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (128,1,1), Waves/SM: 0.32
- **Registers/thread**: 168
- **Duration**: 1380000 ns
- **SM Throughput**: 27.51 % | **Memory**: 62.00 % | **DRAM**: 1.03 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 1.23 inst/cycle | **No Eligible**: 69.28 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.001× | 1.000× | ✓ | **✓ best** |
| B | autotune_tile_sizes_and_pipeline_stages | 0.033× | 0.039× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 1380000 ns | 1380000 ns | +0 ns (+0.0%) | 35060000 ns | +33680000 ns (+2440.6%) |
| SM Throughput (%) | 27.51 % | 27.50 % | -0.01 pp | 12.06 % | -15.45 pp |
| Memory Throughput (%) | 62.00 % | 61.98 % | -0.02 pp | 23.58 % | -38.42 pp |
| DRAM Throughput (%) | 1.03 % | 1.04 % | +0.01 pp | 13.63 % | +12.60 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 32.00 register/thread | -136 |
| Waves/SM | 0.32 | 0.32 | +0.00 | 0.97 | +0.65 |
| No Eligible (%) | 69.28 % | 69.29 % | +0.01 pp | 94.60 % | +25.32 pp |
| IPC Active | 1.23 inst/cycle | 1.23 inst/cycle | +0.00 | 0.22 inst/cycle | -1.01 |

---

## Cumulative Progress

```
Baseline:  1380000 ns  (1.000×)

Final best: 1380000 ns  (1.0000×)
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
 def triton_tem_fused_addmm_relu_t_0(arg_A, arg_B, in_ptr2, out_ptr1):
@@ -72,9 +73,7 @@

         xindex = idx_n + 1024*idx_m
         b = tl.load(B + ((tl.broadcast_to(idx_m + 8192*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
 
-
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -90,8 +89,6 @@

     tmp2 = tl.full([1], 0, tl.int32)
     tmp3 = tl.maximum(tmp2, tmp1)
     tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 1024*idx_m, [BLOCK_M, BLOCK_N])), tmp3, mask)
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
