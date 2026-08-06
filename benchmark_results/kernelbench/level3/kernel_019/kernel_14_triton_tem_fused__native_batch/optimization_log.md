# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_14` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 47550 ns — **Best latency**: 47550 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_14`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_019/kernel_14_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (124,1,1), Waves/SM: 0.31
- **Registers/thread**: 168
- **Duration**: 47550 ns
- **SM Throughput**: 24.76 % | **Memory**: 55.93 % | **DRAM**: 1.10 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 1.19 inst/cycle | **No Eligible**: 70.19 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 1.001× | ✓ |  |
| C | add_cache_eviction_policies_for_streaming_loads_and_stores | 1.000× | 1.003× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | add_cache_evictio... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 47550 ns | 47490 ns | -60 ns (-0.1%) | 47420 ns | -130 ns (-0.3%) |
| SM Throughput (%) | 24.76 % | 24.70 % | -0.06 pp | 24.74 % | -0.02 pp |
| Memory Throughput (%) | 55.93 % | 55.77 % | -0.16 pp | 55.88 % | -0.05 pp |
| DRAM Throughput (%) | 1.10 % | 1.11 % | +0.01 pp | 1.11 % | +0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 0.31 | 0.31 | +0.00 | 0.31 | +0.00 |
| No Eligible (%) | 70.19 % | 70.26 % | +0.07 pp | 70.22 % | +0.03 pp |
| IPC Active | 1.19 inst/cycle | 1.19 inst/cycle | +0.00 | 1.19 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_transposed_b_load_to_eliminate_bank_conflicts_and_enable_safe_autotuning | 0.990× | 0.999× | ✓ |  |
| B | tma_loads | 0.973× | 0.981× | ✓ |  |
| D | tma_loads_fix_b_matrix_strided_access_pattern | 0.980× | 0.980× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_transposed_b_... | Δ | tma_loads | Δ | tma_loads_fix_b_m... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 47550 ns | 47580 ns | +30 ns (+0.1%) | 48480 ns | +930 ns (+2.0%) | 48540 ns | +990 ns (+2.1%) |
| SM Throughput (%) | 24.76 % | 24.65 % | -0.11 pp | 24.51 % | -0.25 pp | 24.55 % | -0.21 pp |
| Memory Throughput (%) | 55.93 % | 55.66 % | -0.27 pp | 54.87 % | -1.06 pp | 54.96 % | -0.97 pp |
| DRAM Throughput (%) | 1.10 % | 1.10 % | +0.00 pp | 1.08 % | -0.02 pp | 1.08 % | -0.02 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 0.31 | 0.31 | +0.00 | 0.31 | +0.00 | 0.31 | +0.00 |
| No Eligible (%) | 70.19 % | 70.34 % | +0.15 pp | 70.36 % | +0.17 pp | 70.37 % | +0.18 pp |
| IPC Active | 1.19 inst/cycle | 1.19 inst/cycle | +0.00 | 1.19 inst/cycle | +0.00 | 1.19 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  47550 ns  (1.000×)

Final best: 47550 ns  (1.0000×)
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
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_14(arg_A, arg_B, out_ptr0):
@@ -65,12 +66,12 @@

         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 256*idx_m
-        a = tl.load(A + (xindex))
+        a = tl.load(A + (xindex), eviction_policy='evict_last')
 
         idx_m = b_k_idx_vals
         idx_n = offs_b_n[None, :]
         xindex = idx_n + 512*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 256*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        b = tl.load(B + ((tl.broadcast_to(idx_m + 256*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), eviction_policy='evict_first')
 
         
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
@@ -85,9 +86,7 @@

 
     # inductor generates a suffix
     xindex = idx_n + 512*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
