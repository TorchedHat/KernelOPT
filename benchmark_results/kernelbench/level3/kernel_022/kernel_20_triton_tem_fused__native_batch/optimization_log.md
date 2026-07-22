# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 31360 ns — **Best latency**: 30780 ns (**1.0188×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`enable_even_k_and_software_pipelining_without_touching_load_logic`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_20_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (123,1,1), Waves/SM: 0.23
- **Registers/thread**: 128
- **Duration**: 31360 ns
- **SM Throughput**: 19.57 % | **Memory**: 74.84 % | **DRAM**: 4.93 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.91 inst/cycle | **No Eligible**: 77.44 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.992× | 0.996× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 31360 ns | 31490 ns | +130 ns (+0.4%) |
| SM Throughput (%) | 19.57 % | 19.50 % | -0.07 pp |
| Memory Throughput (%) | 74.84 % | 74.58 % | -0.26 pp |
| DRAM Throughput (%) | 4.93 % | 4.91 % | -0.02 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.23 | +0.00 |
| No Eligible (%) | 77.44 % | 77.40 % | -0.04 pp |
| IPC Active | 0.91 inst/cycle | 0.90 inst/cycle | -0.01 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_load_pointer_arithmetic | 0.998× | 0.991× | ✓ |  |
| B | tma_loads | 1.001× | 1.008× | ✓ | **✓ best** |
| C | evict_first_streaming_loads_to_reduce_l2_pollution | 0.991× | 0.993× | ✓ |  |
| D | fix_b_matrix_load_pattern_and_enable_software_pipelining | 0.954× | 0.952× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | tma_loads | Δ | evict_first_strea... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 31360 ns | 31650 ns | +290 ns (+0.9%) | 31100 ns | -260 ns (-0.8%) | 31580 ns | +220 ns (+0.7%) | 32930 ns | +1570 ns (+5.0%) |
| SM Throughput (%) | 19.57 % | 19.51 % | -0.06 pp | 19.61 % | +0.04 pp | 19.57 % | +0.00 pp | 18.75 % | -0.82 pp |
| Memory Throughput (%) | 74.84 % | 74.73 % | -0.11 pp | 75.76 % | +0.92 pp | 74.82 % | -0.02 pp | 71.48 % | -3.36 pp |
| DRAM Throughput (%) | 4.93 % | 4.88 % | -0.05 pp | 4.97 % | +0.04 pp | 4.90 % | -0.03 pp | 4.70 % | -0.23 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.23 | +0.00 | 0.23 | +0.00 | 0.23 | +0.00 | 0.23 | +0.00 |
| No Eligible (%) | 77.44 % | 77.47 % | +0.03 pp | 77.26 % | -0.18 pp | 77.32 % | -0.12 pp | 78.21 % | +0.77 pp |
| IPC Active | 0.91 inst/cycle | 0.90 inst/cycle | -0.01 | 0.91 inst/cycle | +0.00 | 0.91 inst/cycle | +0.00 | 0.87 inst/cycle | -0.04 |

---

## Iteration 3

**Best so far**: 1.0084× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_block_ptr_order_and_transpose_for_coalesced_loads | 1.009× | 1.008× | ✓ | **✓ best** |
| B | fix_b_matrix_load_pointer_arithmetic_eliminate_broadcast | 0.998× | 0.996× | ✓ |  |
| C | fix_b_load_coalescing_via_transpose_then_dot | 0.951× | 0.948× | ✓ |  |
| D | fix_b_matrix_load_layout_to_eliminate_bank_conflicts | 0.952× | 0.948× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_bloc... | Δ | fix_b_matrix_load... | Δ | fix_b_load_coales... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 31360 ns | 31100 ns | -260 ns (-0.8%) | 31490 ns | +130 ns (+0.4%) | 33090 ns | +1730 ns (+5.5%) | 33090 ns | +1730 ns (+5.5%) |
| SM Throughput (%) | 19.57 % | 19.63 % | +0.06 pp | 19.42 % | -0.15 pp | 18.81 % | -0.76 pp | 18.76 % | -0.81 pp |
| Memory Throughput (%) | 74.84 % | 75.91 % | +1.07 pp | 74.33 % | -0.51 pp | 71.70 % | -3.14 pp | 71.50 % | -3.34 pp |
| DRAM Throughput (%) | 4.93 % | 4.97 % | +0.04 pp | 4.91 % | -0.02 pp | 4.67 % | -0.26 pp | 4.67 % | -0.26 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.23 | +0.00 | 0.23 | +0.00 | 0.23 | +0.00 | 0.23 | +0.00 |
| No Eligible (%) | 77.44 % | 77.25 % | -0.19 pp | 77.38 % | -0.06 pp | 78.27 % | +0.83 pp | 78.22 % | +0.78 pp |
| IPC Active | 0.91 inst/cycle | 0.90 inst/cycle | -0.01 | 0.90 inst/cycle | -0.01 | 0.87 inst/cycle | -0.04 | 0.87 inst/cycle | -0.04 |

---

## Iteration 4

**Best so far**: 1.0084× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_b_matrix_n_dimension_mask_correctness_bug | 0.997× | 0.995× | ✓ |  |
| C | fix_b_matrix_load_eliminate_tl_trans_use_direct_kn_layout | 0.953× | 0.951× | ✓ |  |
| D | eliminate_tl_trans_by_loading_b_as_k_n_layout_directly | 0.976× | 0.975× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_n_di... | Δ | fix_b_matrix_load... | Δ | eliminate_tl_tran... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 31360 ns | 31520 ns | +160 ns (+0.5%) | 32990 ns | +1630 ns (+5.2%) | 32160 ns | +800 ns (+2.6%) |
| SM Throughput (%) | 19.57 % | 19.57 % | +0.00 pp | 18.82 % | -0.75 pp | 19.09 % | -0.48 pp |
| Memory Throughput (%) | 74.84 % | 74.91 % | +0.07 pp | 71.73 % | -3.11 pp | 73.72 % | -1.12 pp |
| DRAM Throughput (%) | 4.93 % | 4.91 % | -0.02 pp | 4.69 % | -0.24 pp | 4.81 % | -0.12 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.23 | +0.00 | 0.23 | +0.00 | 0.23 | +0.00 |
| No Eligible (%) | 77.44 % | 77.50 % | +0.06 pp | 78.19 % | +0.75 pp | 77.74 % | +0.30 pp |
| IPC Active | 0.91 inst/cycle | 0.90 inst/cycle | -0.01 | 0.86 inst/cycle | -0.05 | 0.89 inst/cycle | -0.02 |

---

## Iteration 5

**Best so far**: 1.0084× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | enable_even_k_and_software_pipelining_without_touching_load_logic | 1.021× | 1.019× | ✓ | **✓ best** |
| C | add_autotune_num_stages_num_warps_keep_b_load_intact | 0.958× | 0.818× | ✓ |  |


### Metric Comparison

| Metric | Baseline | enable_even_k_and... | Δ | add_autotune_num_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 31360 ns | 30780 ns | -580 ns (-1.8%) | 38340 ns | +6980 ns (+22.3%) |
| SM Throughput (%) | 19.57 % | 18.73 % | -0.84 pp | 16.17 % | -3.40 pp |
| Memory Throughput (%) | 74.84 % | 71.65 % | -3.19 pp | 65.51 % | -9.33 pp |
| DRAM Throughput (%) | 4.93 % | 5.01 % | +0.08 pp | 4.03 % | -0.90 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.23 | +0.00 | 0.23 | +0.00 |
| No Eligible (%) | 77.44 % | 78.32 % | +0.88 pp | 80.63 % | +3.19 pp |
| IPC Active | 0.91 inst/cycle | 0.86 inst/cycle | -0.05 | 0.77 inst/cycle | -0.14 |


### Key Learnings

#### ❌ Plan C: `add_autotune_num_stages_num_warps_keep_b_load_intact` (0.818×)

Adding @triton.autotune varying only num_warps and num_stages (without changing tile sizes) caused a 22% regression (31.5us → 38.3us). Despite halving dynamic shared memory per block (32.77 KB → 16.38 KB), elapsed cycles increased from 47,229 to 57,481 and warp cycles per issued instruction worsened from 4.42 to 5.17. The bank conflict count remained essentially unchanged (~11.9-way, ~77.5% of wavefronts), indicating that num_stages/num_warps tuning alone cannot resolve bank conflicts that are inherent to the tile shape and access pattern. The regression likely stems from the autotuner selecting a suboptimal config (e.g., num_warps=8 with the same register-heavy kernel causing spilling or scheduling overhead) while the root bank-conflict bottleneck persists.

> **Applicable when**: Never: this strategy is a dead end for kernels with severe shared memory bank conflicts caused by tile shape. Varying num_stages/num_warps alone cannot fix structural bank conflicts.

> **Avoid when**: When the primary bottleneck is SharedMemoryConflicts (11.9-way bank conflicts, >75% of shared load wavefronts conflicted). Autotune of num_stages/num_warps without changing tile sizes does not reduce bank conflicts and may introduce scheduling overhead that worsens latency. The bank conflict count before and after is nearly identical (~3.53M conflicts), confirming the strategy has no effect on the root cause.

---

## Cumulative Progress

```
Baseline:  31360 ns  (1.000×)
Iter 2     31100 ns  (1.008×)  — tma_loads
Iter 5     30780 ns  (1.019×)  — enable_even_k_and_software_pipelining_without_touching_load_logic

Final best: 30780 ns  (1.0188×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,14 +5,15 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20(arg_A, arg_B, out_ptr0):
-    EVEN_K : tl.constexpr = False
+    EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 64
     BLOCK_N : tl.constexpr = 64
-    BLOCK_K : tl.constexpr = 32
+    BLOCK_K : tl.constexpr = 16
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -59,23 +60,20 @@

 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
 
-        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
-        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
-
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
+        k_offs = offs_k + k_idx * BLOCK_K   # shape [BLOCK_K], absolute k indices
 
         idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
+        idx_n = k_offs[None, :]
         xindex = idx_n + 240*idx_m
-        a = tl.load(A + (xindex), mask=a_mask, other=0.0)
+        a = tl.load(A + (xindex))   # no K mask needed: EVEN_K=True
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 40*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 240*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
-
-
+        # Load B directly as [BLOCK_K, BLOCK_N]: k outer, n inner
+        # B stored as B[n, k] = B_ptr + n*stride_bn + k*stride_bk = B_ptr + n*240 + k*1
+        # Access pattern: b[k_local, n_local] = B_ptr + k_offs[k_local]*1 + offs_b_n[n_local]*240
+        b_ptrs = B + k_offs[:, None] * stride_bk + offs_b_n[None, :] * stride_bn  # [BLOCK_K, BLOCK_N]
+        b_mask2d = offs_b_n[None, :] < N   # only N-dim mask; K-dim dropped (EVEN_K=True)
+        b = tl.load(b_ptrs, mask=b_mask2d, other=0.0)  # shape [BLOCK_K, BLOCK_N]
+        # tl.dot(a, b): a=[BLOCK_M, BLOCK_K], b=[BLOCK_K, BLOCK_N] -> acc=[BLOCK_M, BLOCK_N]
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
 
@@ -89,8 +87,6 @@

     # inductor generates a suffix
     xindex = idx_n + 40*idx_m
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
