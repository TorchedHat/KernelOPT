# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_16` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 21250 ns — **Best latency**: 25790 ns (**0.8240×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_16`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_16_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (123,1,1), Waves/SM: 0.23
- **Registers/thread**: 128
- **Duration**: 21250 ns
- **SM Throughput**: 18.74 % | **Memory**: 70.72 % | **DRAM**: 4.39 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.90 inst/cycle | **No Eligible**: 77.52 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.998× | 1.005× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 21250 ns | 21150 ns | -100 ns (-0.5%) |
| SM Throughput (%) | 18.74 % | 18.62 % | -0.12 pp |
| Memory Throughput (%) | 70.72 % | 70.27 % | -0.45 pp |
| DRAM Throughput (%) | 4.39 % | 4.40 % | +0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.23 | +0.00 |
| No Eligible (%) | 77.52 % | 78.27 % | +0.75 pp |
| IPC Active | 0.90 inst/cycle | 0.90 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_shared_memory_bank_conflicts_via_safe_autotune | 1.163× | 0.824× | ✓ | **✓ best** |
| C | eviction_policies_streaming_loads | 1.010× | 1.005× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_shared_memory... | Δ | eviction_policies... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 21250 ns | 25790 ns | +4540 ns (+21.4%) | 21150 ns | -100 ns (-0.5%) |
| SM Throughput (%) | 18.74 % | 12.97 % | -5.77 pp | 18.66 % | -0.08 pp |
| Memory Throughput (%) | 70.72 % | 53.74 % | -16.98 pp | 70.37 % | -0.35 pp |
| DRAM Throughput (%) | 4.39 % | 3.60 % | -0.79 pp | 4.40 % | +0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 96.00 register/thread | -32 | 128.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.19 | -0.04 | 0.23 | +0.00 |
| No Eligible (%) | 77.52 % | 84.30 % | +6.78 pp | 77.60 % | +0.08 pp |
| IPC Active | 0.90 inst/cycle | 0.63 inst/cycle | -0.27 | 0.90 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_load_direct_2d_indexing_no_broadcast | 0.963× | 0.960× | ✓ |  |
| B | fix_b_matrix_load_index_arithmetic_eliminate_double_broadcast | 0.998× | 1.001× | ✓ |  |
| D | fix_b_load_index_and_safe_warp_stage_autotune | 1.146× | 0.880× | ✓ |  |
| E | add_triton_autotune | 1.005× | 0.866× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | fix_b_matrix_load... | Δ | fix_b_load_index_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21250 ns | 22140 ns | +890 ns (+4.2%) | 21220 ns | -30 ns (-0.1%) | 24160 ns | +2910 ns (+13.7%) | 24540 ns | +3290 ns (+15.5%) |
| SM Throughput (%) | 18.74 % | 17.96 % | -0.78 pp | 18.60 % | -0.14 pp | 13.81 % | -4.93 pp | 15.69 % | -3.05 pp |
| Memory Throughput (%) | 70.72 % | 67.56 % | -3.16 pp | 70.15 % | -0.57 pp | 34.83 % | -35.89 pp | 60.44 % | -10.28 pp |
| DRAM Throughput (%) | 4.39 % | 4.20 % | -0.19 pp | 4.39 % | +0.00 pp | 3.85 % | -0.54 pp | 3.79 % | -0.60 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 254.00 register/thread | +126 | 128.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.23 | +0.00 | 0.23 | +0.00 | 0.23 | +0.00 | 0.23 | +0.00 |
| No Eligible (%) | 77.52 % | 78.57 % | +1.05 pp | 77.53 % | +0.01 pp | 66.94 % | -10.58 pp | 81.30 % | +3.78 pp |
| IPC Active | 0.90 inst/cycle | 0.86 inst/cycle | -0.04 | 0.88 inst/cycle | -0.02 | 0.66 inst/cycle | -0.24 | 0.75 inst/cycle | -0.15 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | cap_registers_with_maxnreg_to_improve_occupancy | 1.149× | 0.872× | ✓ |  |
| D | fix_b_matrix_load_coalesced_pointer_arithmetic | 1.010× | 0.857× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_registers_wit... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 21250 ns | 24380 ns | +3130 ns (+14.7%) | 24800 ns | +3550 ns (+16.7%) |
| SM Throughput (%) | 18.74 % | 13.47 % | -5.27 pp | 15.71 % | -3.03 pp |
| Memory Throughput (%) | 70.72 % | 33.96 % | -36.76 pp | 60.47 % | -10.25 pp |
| DRAM Throughput (%) | 4.39 % | 3.82 % | -0.57 pp | 3.76 % | -0.63 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 254.00 register/thread | +126 | 128.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.23 | +0.00 | 0.23 | +0.00 |
| No Eligible (%) | 77.52 % | 66.88 % | -10.64 pp | 81.29 % | +3.77 pp |
| IPC Active | 0.90 inst/cycle | 0.66 inst/cycle | -0.24 | 0.75 inst/cycle | -0.15 |

---

## Cumulative Progress

```
Baseline:  21250 ns  (1.000×)

Final best: 25790 ns  (0.8240×)
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
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_16(arg_A, arg_B, out_ptr0):
@@ -68,13 +69,10 @@

         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 144*idx_m
-        a = tl.load(A + (xindex), mask=a_mask, other=0.0)
+        a = tl.load(A + (xindex), mask=a_mask, other=0.0, eviction_policy='evict_first')
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 40*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 144*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
-
+        b_ptrs = B + (b_k_idx_vals + 144 * offs_b_n[None, :])
+        b = tl.load(b_ptrs, mask=b_mask, other=0.0, eviction_policy='evict_first')
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
@@ -88,9 +86,7 @@

 
     # inductor generates a suffix
     xindex = idx_n + 40*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask=mask, eviction_policy='evict_last')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
