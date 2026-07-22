# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 45980 ns — **Best latency**: 45820 ns (**1.0035×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`coalesce_b_matrix_global_load_by_transposing_load_order`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_023/kernel_5_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1960,1,1), Waves/SM: 4.95
- **Registers/thread**: 168
- **Duration**: 45980 ns
- **SM Throughput**: 30.62 % | **Memory**: 77.27 % | **DRAM**: 13.01 %
- **Achieved Occupancy**: 17.92 %
- **IPC Active**: 1.33 inst/cycle | **No Eligible**: 66.88 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.004× | 1.010× | ✓ |  |
| D | add_autotune_reduce_register_pressure_fix_bank_conflicts | 1.047× | 1.038× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | add_autotune_redu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 45980 ns | 45500 ns | -480 ns (-1.0%) | 44320 ns | -1660 ns (-3.6%) |
| SM Throughput (%) | 30.62 % | 30.43 % | -0.19 pp | 30.78 % | +0.16 pp |
| Memory Throughput (%) | 77.27 % | 76.80 % | -0.47 pp | 56.52 % | -20.75 pp |
| DRAM Throughput (%) | 13.01 % | 13.14 % | +0.13 pp | 13.52 % | +0.51 pp |
| Achieved Occupancy (%) | 17.92 % | 17.92 % | +0.00 pp | 11.78 % | -6.14 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 254.00 register/thread | +86 |
| Waves/SM | 4.95 | 4.95 | +0.00 | 3.71 | -1.24 |
| No Eligible (%) | 66.88 % | 66.87 % | -0.01 pp | 64.63 % | -2.25 pp |
| IPC Active | 1.33 inst/cycle | 1.33 inst/cycle | +0.00 | 1.39 inst/cycle | +0.06 |

---

## Iteration 2

**Best so far**: 1.0375× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_hopper_async_bulk_copy | 1.048× | 1.050× | ✓ |  |
| C | autotune_num_warps_and_cap_registers_to_improve_occupancy | 1.005× | 1.013× | ✓ |  |
| D | tma_loads | 1.014× | 1.024× | ✓ |  |
| E | add_triton_autotune | 1.049× | 1.034× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tma_loads_hopper_... | Δ | autotune_num_warp... | Δ | tma_loads | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 45980 ns | 43810 ns | -2170 ns (-4.7%) | 45380 ns | -600 ns (-1.3%) | 44900 ns | -1080 ns (-2.3%) | 44450 ns | -1530 ns (-3.3%) |
| SM Throughput (%) | 30.62 % | 30.42 % | -0.20 pp | 30.57 % | -0.05 pp | 29.92 % | -0.70 pp | 30.70 % | +0.08 pp |
| Memory Throughput (%) | 77.27 % | 55.89 % | -21.38 pp | 77.13 % | -0.14 pp | 77.20 % | -0.07 pp | 56.39 % | -20.88 pp |
| DRAM Throughput (%) | 13.01 % | 13.85 % | +0.84 pp | 13.28 % | +0.27 pp | 13.30 % | +0.29 pp | 13.54 % | +0.53 pp |
| Achieved Occupancy (%) | 17.92 % | 11.84 % | -6.08 pp | 17.93 % | +0.01 pp | 17.90 % | -0.02 pp | 11.80 % | -6.12 pp |
| Registers/Thread | 168.00 register/thread | 254.00 register/thread | +86 | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 254.00 register/thread | +86 |
| Waves/SM | 4.95 | 3.71 | -1.24 | 4.95 | +0.00 | 4.95 | +0.00 | 3.71 | -1.24 |
| No Eligible (%) | 66.88 % | 64.40 % | -2.48 pp | 66.78 % | -0.10 pp | 67.13 % | +0.25 pp | 64.72 % | -2.16 pp |
| IPC Active | 1.33 inst/cycle | 1.39 inst/cycle | +0.06 | 1.32 inst/cycle | -0.01 | 1.31 inst/cycle | -0.02 | 1.37 inst/cycle | +0.04 |

---

## Iteration 3

**Best so far**: 1.0375× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_maxnreg_cap_to_improve_occupancy_without_index_changes | 1.052× | 1.050× | ✓ | **✓ best** |
| B | fix_b_block_ptr_order_to_eliminate_shared_memory_bank_conflicts | 1.009× | 1.025× | ✓ |  |
| C | fix_block_n_to_match_n_dimension_and_eliminate_partial_tile_mask | 1.036× | 1.028× | ✓ |  |
| D | fix_b_matrix_load_to_eliminate_shared_memory_bank_conflicts | 0.998× | 1.008× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_maxnreg_... | Δ | fix_b_block_ptr_o... | Δ | fix_block_n_to_ma... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 45980 ns | 43810 ns | -2170 ns (-4.7%) | 44860 ns | -1120 ns (-2.4%) | 44740 ns | -1240 ns (-2.7%) | 45600 ns | -380 ns (-0.8%) |
| SM Throughput (%) | 30.62 % | 30.94 % | +0.32 pp | 29.95 % | -0.67 pp | 30.54 % | -0.08 pp | 30.42 % | -0.20 pp |
| Memory Throughput (%) | 77.27 % | 56.84 % | -20.43 pp | 77.91 % | +0.64 pp | 55.91 % | -21.36 pp | 76.78 % | -0.49 pp |
| DRAM Throughput (%) | 13.01 % | 13.66 % | +0.65 pp | 13.36 % | +0.35 pp | 13.49 % | +0.48 pp | 13.11 % | +0.10 pp |
| Achieved Occupancy (%) | 17.92 % | 11.84 % | -6.08 pp | 17.89 % | -0.03 pp | 11.91 % | -6.01 pp | 17.94 % | +0.02 pp |
| Registers/Thread | 168.00 register/thread | 254.00 register/thread | +86 | 168.00 register/thread | +0 | 254.00 register/thread | +86 | 168.00 register/thread | +0 |
| Waves/SM | 4.95 | 3.71 | -1.24 | 4.95 | +0.00 | 3.71 | -1.24 | 4.95 | +0.00 |
| No Eligible (%) | 66.88 % | 64.76 % | -2.12 pp | 67.26 % | +0.38 pp | 64.34 % | -2.54 pp | 66.85 % | -0.03 pp |
| IPC Active | 1.33 inst/cycle | 1.37 inst/cycle | +0.04 | 1.31 inst/cycle | -0.02 | 1.41 inst/cycle | +0.08 | 1.32 inst/cycle | -0.01 |


### Key Learnings

#### ✅ Plan A: `autotune_maxnreg_cap_to_improve_occupancy_without_index_changes` (1.050×)

Adding maxnreg=128 and maxnreg=64 variants to the Triton autotune config set was intended to cap register usage via PTX compiler spilling, raising Block Limit Registers from 4 to 8 or 16 blocks/SM and theoretical occupancy from 12.5% to 25-50%. However, the profiling data shows no meaningful change: registers_per_thread remained at 254, Block Limit Registers stayed at 4, theoretical and achieved occupancy were unchanged at 12.5% and 11.84%, and duration was identical at 43.81 us. The autotuner selected a config without maxnreg (or maxnreg had no effect on the compiled register count), yielding only a negligible ~0.5% speedup from noise. The strategy did not achieve its intended occupancy improvement.

> **Applicable when**: Never clearly applicable: when a kernel has extreme register pressure (>200 registers/thread) limiting Block Limit Registers to <=4, and prior tile-size reductions have failed correctness, maxnreg in autotune configs is a candidate. However, evidence from this iteration shows the autotuner does not select maxnreg configs when the kernel body itself demands high register counts.

> **Avoid when**: When the kernel's register usage is driven by the accumulator tile size (e.g., BLOCK_M x BLOCK_N FP32 accumulators) rather than compiler inefficiency. In such cases, maxnreg causes register spilling without reducing the actual working set, and the autotuner will prefer the non-maxnreg configs. The occupancy improvement is illusory — the autotuner selects the baseline config, yielding no benefit. Do not apply when registers_per_thread > 200 and the high register count is structurally necessary (large tile accumulator).

---

## Iteration 4

**Best so far**: 1.0495× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_transposed_index_to_eliminate_bank_conflicts_and_uncoalesced_access | 1.029× | 1.031× | ✓ |  |
| B | convert_a_matrix_load_to_block_ptr_to_eliminate_shared_memory_bank_conflicts | 0.985× | 0.990× | ✓ |  |
| D | fix_b_load_transposed_to_eliminate_shared_memory_bank_conflicts | 1.002× | 1.014× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_tran... | Δ | convert_a_matrix_... | Δ | fix_b_load_transp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 45980 ns | 44610 ns | -1370 ns (-3.0%) | 46430 ns | +450 ns (+1.0%) | 45340 ns | -640 ns (-1.4%) |
| SM Throughput (%) | 30.62 % | 30.40 % | -0.22 pp | 29.76 % | -0.86 pp | 30.70 % | +0.08 pp |
| Memory Throughput (%) | 77.27 % | 55.71 % | -21.56 pp | 75.53 % | -1.74 pp | 77.32 % | +0.05 pp |
| DRAM Throughput (%) | 13.01 % | 13.55 % | +0.54 pp | 12.88 % | -0.13 pp | 13.16 % | +0.15 pp |
| Achieved Occupancy (%) | 17.92 % | 11.92 % | -6.00 pp | 18.01 % | +0.09 pp | 17.90 % | -0.02 pp |
| Registers/Thread | 168.00 register/thread | 254.00 register/thread | +86 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 4.95 | 3.71 | -1.24 | 4.95 | +0.00 | 4.95 | +0.00 |
| No Eligible (%) | 66.88 % | 65.33 % | -1.55 pp | 67.73 % | +0.85 pp | 66.75 % | -0.13 pp |
| IPC Active | 1.33 inst/cycle | 1.39 inst/cycle | +0.06 | 1.28 inst/cycle | -0.05 | 1.33 inst/cycle | +0.00 |

---

## Iteration 5

**Best so far**: 1.0495× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesce_b_matrix_global_load_by_transposing_load_order | 1.235× | 1.004× | ✓ | **✓ best** |
| B | eliminate_tl_trans_b_to_fix_shared_memory_bank_conflicts | 0.988× | 0.999× | ✓ |  |
| C | convert_b_matrix_load_to_block_ptr_with_n_dim_fastest_order_to_eliminate_bank_conflicts | 1.016× | 1.018× | ✓ |  |
| D | add_triton_autotune | 0.997× | 0.994× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesce_b_matrix... | Δ | eliminate_tl_tran... | Δ | convert_b_matrix_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 45980 ns | 45820 ns | -160 ns (-0.3%) | 46050 ns | +70 ns (+0.2%) | 45180 ns | -800 ns (-1.7%) | 46270 ns | +290 ns (+0.6%) |
| SM Throughput (%) | 30.62 % | 29.08 % | -1.54 pp | 30.07 % | -0.55 pp | 29.86 % | -0.76 pp | 30.42 % | -0.20 pp |
| Memory Throughput (%) | 77.27 % | 56.20 % | -21.07 pp | 76.35 % | -0.92 pp | 77.39 % | +0.12 pp | 76.64 % | -0.63 pp |
| DRAM Throughput (%) | 13.01 % | 13.15 % | +0.14 pp | 12.97 % | -0.04 pp | 13.23 % | +0.22 pp | 12.99 % | -0.02 pp |
| Achieved Occupancy (%) | 17.92 % | 12.04 % | -5.88 pp | 18.02 % | +0.10 pp | 17.90 % | -0.02 pp | 17.90 % | -0.02 pp |
| Registers/Thread | 168.00 register/thread | 240.00 register/thread | +72 | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 4.95 | 3.71 | -1.24 | 4.95 | +0.00 | 4.95 | +0.00 | 4.95 | +0.00 |
| No Eligible (%) | 66.88 % | 66.56 % | -0.32 pp | 68.16 % | +1.28 pp | 67.37 % | +0.49 pp | 67.15 % | +0.27 pp |
| IPC Active | 1.33 inst/cycle | 1.32 inst/cycle | -0.01 | 1.28 inst/cycle | -0.05 | 1.30 inst/cycle | -0.03 | 1.32 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  45980 ns  (1.000×)
Iter 1     44320 ns  (1.038×)  — add_autotune_reduce_register_pressure_fix_bank_conflicts
Iter 3     43810 ns  (1.050×)  — autotune_maxnreg_cap_to_improve_occupancy_without_index_changes

Final best: 45820 ns  (1.0035×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,26 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=2, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=1),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=1),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=16, num_stages=1),
+        triton.Config({}, num_warps=16, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=16, num_stages=3),
+        triton.Config({}, num_warps=16, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -65,12 +85,14 @@

         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 16*idx_m
-        a = tl.load(A + (xindex))
+        a = tl.load(A + (xindex), eviction_policy='evict_first')
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 96*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 16*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        # Coalesced B load: load as (BLOCK_N, BLOCK_K) with k in fast dim (stride 1),
+        # then transpose to (BLOCK_K, BLOCK_N) for tl.dot
+        idx_m_b = offs_b_n[:, None]   # (BLOCK_N, 1) -- n is the slow dim
+        idx_k_b = b_k_idx_vals[None, :].reshape(1, BLOCK_K)  # (1, BLOCK_K) -- k is the fast dim
+        b_T = tl.load(B + 16 * idx_m_b + idx_k_b, eviction_policy='evict_first')
+        b = tl.trans(b_T)  # (BLOCK_K, BLOCK_N) for tl.dot
 
         
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
@@ -86,8 +108,6 @@

     # inductor generates a suffix
     xindex = idx_n + 96*idx_m
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
