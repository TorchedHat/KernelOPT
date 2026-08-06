# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_27` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 29820 ns — **Best latency**: 29820 ns (**1.0000×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_27`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_27_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (144,1,1), Waves/SM: 0.55
- **Registers/thread**: 244
- **Duration**: 29820 ns
- **SM Throughput**: 17.98 % | **Memory**: 40.49 % | **DRAM**: 0.70 %
- **Achieved Occupancy**: 7.07 %
- **IPC Active**: 1.13 inst/cycle | **No Eligible**: 71.42 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.008× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 29820 ns | 29820 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 17.98 % | 18.12 % | +0.14 pp |
| Memory Throughput (%) | 40.49 % | 40.81 % | +0.32 pp |
| DRAM Throughput (%) | 0.70 % | 0.70 % | +0.00 pp |
| Achieved Occupancy (%) | 7.07 % | 7.07 % | +0.00 pp |
| Registers/Thread | 244.00 register/thread | 244.00 register/thread | +0 |
| Waves/SM | 0.55 | 0.55 | +0.00 |
| No Eligible (%) | 71.42 % | 71.30 % | -0.12 pp |
| IPC Active | 1.13 inst/cycle | 1.15 inst/cycle | +0.02 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_load_indexing_to_eliminate_bank_conflicts_and_correctness_errors | 0.998× | 0.987× | ✓ |  |
| B | tma_loads_via_block_ptr_with_corrected_b_indexing | 1.000× | 0.995× | ✓ |  |
| D | fix_b_matrix_load_and_increase_block_k | 1.007× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | tma_loads_via_blo... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 29820 ns | 30210 ns | +390 ns (+1.3%) | 29980 ns | +160 ns (+0.5%) | 29790 ns | -30 ns (-0.1%) |
| SM Throughput (%) | 17.98 % | 18.05 % | +0.07 pp | 18.11 % | +0.13 pp | 17.82 % | -0.16 pp |
| Memory Throughput (%) | 40.49 % | 40.64 % | +0.15 pp | 40.66 % | +0.17 pp | 40.12 % | -0.37 pp |
| DRAM Throughput (%) | 0.70 % | 0.69 % | -0.01 pp | 0.69 % | -0.01 pp | 0.70 % | +0.00 pp |
| Achieved Occupancy (%) | 7.07 % | 7.06 % | -0.01 pp | 7.07 % | +0.00 pp | 7.07 % | +0.00 pp |
| Registers/Thread | 244.00 register/thread | 244.00 register/thread | +0 | 234.00 register/thread | -10 | 244.00 register/thread | +0 |
| Waves/SM | 0.55 | 0.55 | +0.00 | 0.55 | +0.00 | 0.55 | +0.00 |
| No Eligible (%) | 71.42 % | 71.24 % | -0.18 pp | 71.33 % | -0.09 pp | 71.92 % | +0.50 pp |
| IPC Active | 1.13 inst/cycle | 1.15 inst/cycle | +0.02 | 1.14 inst/cycle | +0.01 | 1.13 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cap_register_usage_with_maxnreg_to_increase_occupancy | 1.007× | 0.996× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_register_usag... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 29820 ns | 29950 ns | +130 ns (+0.4%) |
| SM Throughput (%) | 17.98 % | 17.16 % | -0.82 pp |
| Memory Throughput (%) | 40.49 % | 41.24 % | +0.75 pp |
| DRAM Throughput (%) | 0.70 % | 0.69 % | -0.01 pp |
| Achieved Occupancy (%) | 7.07 % | 7.06 % | -0.01 pp |
| Registers/Thread | 244.00 register/thread | 232.00 register/thread | -12 |
| Waves/SM | 0.55 | 0.55 | +0.00 |
| No Eligible (%) | 71.42 % | 72.52 % | +1.10 pp |
| IPC Active | 1.13 inst/cycle | 1.10 inst/cycle | -0.03 |

---

## Cumulative Progress

```
Baseline:  29820 ns  (1.000×)

Final best: 29820 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,10 +5,11 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_27(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
-    USE_FAST_ACCUM : tl.constexpr = False
+    USE_FAST_ACCUM : tl.constexpr = True
     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 128
     BLOCK_N : tl.constexpr = 64
@@ -67,11 +68,8 @@

         xindex = idx_n + 96*idx_m
         a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 576*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 96*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
+        b_index = b_k_idx_vals + 96 * offs_b_n[None, :]  # [BLOCK_K, BLOCK_N], stride_bk=1 stride_bn=96
+        b = tl.load(B + b_index)
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
@@ -86,8 +84,6 @@

     # inductor generates a suffix
     xindex = idx_n + 576*idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
