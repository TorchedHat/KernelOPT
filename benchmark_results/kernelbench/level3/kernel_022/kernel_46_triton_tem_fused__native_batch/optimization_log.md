# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_hardtanh_mean_relu_t_view_46` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 63490 ns — **Best latency**: 63330 ns (**1.0025×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_matrix_b_load_coalesced_index_via_correct_stride_formula`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_hardtanh_mean_relu_t_view_46`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_46_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (32,1,1), Waves/SM: 0.06
- **Registers/thread**: 66
- **Duration**: 63490 ns
- **SM Throughput**: 1.93 % | **Memory**: 22.31 % | **DRAM**: 1.67 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.33 inst/cycle | **No Eligible**: 91.78 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | — | fail | ✓ |  |
| C | fix_matrix_b_load_index_for_memory_coalescing | — | fail | ✓ |  |
| D | autotune_tile_sizes_to_increase_parallelism_and_reduce_bank_conflicts | — | fail | ✓ |  |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_shared_memory_bank_conflicts_via_transposed_b_load | — | fail | ✓ |  |
| B | fix_b_matrix_shared_memory_bank_conflicts_via_padded_staging | — | fail | ✓ |  |
| C | fix_matrix_b_load_coalesced_index_via_correct_stride_formula | 1.001× | 1.002× | ✓ | **✓ best** |
| D | software_pipeline_to_hide_shared_memory_mio_stalls | 0.882× | 0.865× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_matrix_b_load... | Δ | software_pipeline... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 63490 ns | 63330 ns | -160 ns (-0.3%) | 73380 ns | +9890 ns (+15.6%) |
| SM Throughput (%) | 1.93 % | 1.93 % | +0.00 pp | 3.13 % | +1.20 pp |
| Memory Throughput (%) | 22.31 % | 22.31 % | +0.00 pp | 20.56 % | -1.75 pp |
| DRAM Throughput (%) | 1.67 % | 1.75 % | +0.08 pp | 1.44 % | -0.23 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 6.25 % | +0.01 pp |
| Registers/Thread | 66.00 register/thread | 66.00 register/thread | +0 | 72.00 register/thread | +6 |
| Waves/SM | 0.06 | 0.06 | +0.00 | 0.03 | -0.03 |
| No Eligible (%) | 91.78 % | 91.76 % | -0.02 pp | 86.62 % | -5.16 pp |
| IPC Active | 0.33 inst/cycle | 0.33 inst/cycle | +0.00 | 0.53 inst/cycle | +0.20 |


### Key Learnings

#### ❌ Plan D: `software_pipeline_to_hide_shared_memory_mio_stalls` (0.865×)

Reducing BLOCK_K (128→32) and adding num_stages=3 to tl.range successfully reduced shared memory per block (49→20 KB) and improved theoretical occupancy (25%→43.75%), and even reduced warp cycles per instruction (12.1→7.5). However, the total executed instruction count nearly doubled (973K→1819K) because the smaller BLOCK_K tile requires 4x more k-loop iterations, and the register count increased (66→72), causing the kernel duration to increase from 63.49 µs to 73.38 µs (0.87x speedup). When the grid is already severely undersized (32 blocks on 132 SMs, achieved occupancy ~6%), the extra instruction overhead from more loop iterations dominates over the latency-hiding benefit of software pipelining, and the achieved occupancy does not improve despite better theoretical occupancy.

> **Applicable when**: Never: this specific combination caused regression. In general, reducing BLOCK_K with num_stages pipelining is beneficial when the grid is large enough to fill SMs and the instruction overhead from more iterations is amortized by higher occupancy.

> **Avoid when**: Grid is severely undersized (waves_per_sm < 0.1, grid_size << sm_count). In this regime, shrinking BLOCK_K multiplies loop iteration count and total instructions without improving achieved occupancy (which stays near 6% regardless of theoretical occupancy improvement). The extra register pressure from num_stages further limits blocks per SM. The net effect is more work per SM with no latency hiding benefit, causing regression.

---

## Iteration 3

**Best so far**: 1.0025× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_via_block_ptr_no_autotuner | 0.999× | 1.007× | ✓ |  |
| C | load_b_transposed_in_registers_to_eliminate_bank_conflicts | 0.882× | 0.868× | ✓ |  |
| D | tma_loads_via_block_ptr_without_autotuner | 0.870× | 0.856× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_via_blo... | Δ | load_b_transposed... | Δ | tma_loads_via_blo... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 63490 ns | 63070 ns | -420 ns (-0.7%) | 73180 ns | +9690 ns (+15.3%) | 74210 ns | +10720 ns (+16.9%) |
| SM Throughput (%) | 1.93 % | 1.96 % | +0.03 pp | 3.14 % | +1.21 pp | 3.22 % | +1.29 pp |
| Memory Throughput (%) | 22.31 % | 22.42 % | +0.11 pp | 20.60 % | -1.71 pp | 20.62 % | -1.69 pp |
| DRAM Throughput (%) | 1.67 % | 1.68 % | +0.01 pp | 2.20 % | +0.53 pp | 1.42 % | -0.25 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 6.25 % | +0.01 pp | 6.25 % | +0.01 pp |
| Registers/Thread | 66.00 register/thread | 64.00 register/thread | -2 | 72.00 register/thread | +6 | 72.00 register/thread | +6 |
| Waves/SM | 0.06 | 0.06 | +0.00 | 0.03 | -0.03 | 0.03 | -0.03 |
| No Eligible (%) | 91.78 % | 91.60 % | -0.18 pp | 86.67 % | -5.11 pp | 86.25 % | -5.53 pp |
| IPC Active | 0.33 inst/cycle | 0.34 inst/cycle | +0.01 | 0.53 inst/cycle | +0.20 | 0.55 inst/cycle | +0.22 |


### Key Learnings

#### ❌ Plan C: `load_b_transposed_in_registers_to_eliminate_bank_conflicts` (0.868×)

Loading the B matrix as [BLOCK_N, BLOCK_K] (K in fast dim) followed by tl.trans() to get [BLOCK_K, BLOCK_N] for tl.dot was intended to eliminate 12-way shared memory bank conflicts. While warp cycles per instruction improved (12.1→7.5) and shared memory usage dropped (49→20 KB), the change was bundled with a BLOCK_K reduction (128→32) and num_stages=3 pipelining, which nearly doubled the executed instruction count (973K→1820K) and increased register pressure (66→72 registers/thread). The achieved occupancy remained stuck at ~6.25% (grid of 32 blocks on 132 SMs), so the extra instruction overhead dominated and duration increased from 63.33 µs to 73.18 µs (0.87x). When the grid is severely undersized, reducing BLOCK_K to enable register-transpose loads multiplies loop iterations without improving SM utilization, causing net regression regardless of bank-conflict reduction.

> **Applicable when**: Never effective in this form. In general, register-transpose (load [N,K] then tl.trans) is beneficial when: (1) BLOCK_K is already small (so loop iteration count is not multiplied), (2) the grid is large enough to fill SMs (waves_per_sm >= 1), and (3) bank conflicts are the dominant bottleneck without a co-dependent BLOCK_K change.

> **Avoid when**: Grid is severely undersized (waves_per_sm < 0.1, grid_size << sm_count). Combining register-transpose with BLOCK_K reduction multiplies loop iterations (e.g., 128→32 means 4x more k-loop iterations) and total instruction count without improving achieved occupancy. The register pressure increase from num_stages pipelining further compounds the regression. Even if bank conflicts are reduced, the instruction-count overhead dominates when SMs are starved of work.

---

## Iteration 4

**Best so far**: 1.0025× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | restore_block_k_128_and_remove_software_pipeline_to_reduce_instruction_count | 0.921× | 0.925× | ✓ |  |


### Metric Comparison

| Metric | Baseline | restore_block_k_1... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 63490 ns | 68640 ns | +5150 ns (+8.1%) |
| SM Throughput (%) | 1.93 % | 3.32 % | +1.39 pp |
| Memory Throughput (%) | 22.31 % | 22.38 % | +0.07 pp |
| DRAM Throughput (%) | 1.67 % | 1.54 % | -0.13 pp |
| Achieved Occupancy (%) | 6.24 % | 6.25 % | +0.01 pp |
| Registers/Thread | 66.00 register/thread | 116.00 register/thread | +50 |
| Waves/SM | 0.06 | 0.12 | +0.06 |
| No Eligible (%) | 91.78 % | 85.90 % | -5.88 pp |
| IPC Active | 0.33 inst/cycle | 0.56 inst/cycle | +0.23 |

---

## Cumulative Progress

```
Baseline:  63490 ns  (1.000×)
Iter 2     63330 ns  (1.002×)  — fix_matrix_b_load_coalesced_index_via_correct_stride_formula

Final best: 63330 ns  (1.0025×)
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
 def triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_hardtanh_mean_relu_t_view_46(in_ptr0, arg_A, arg_B, out_ptr0):
@@ -44,37 +45,33 @@

     tl.assume(pid_m >= 0)
     tl.assume(pid_n >= 0)
 
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
-        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
-    else:
-        offs_a_m = rm % M
-    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
-        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
-    else:
-        offs_b_n = rn % N
-    offs_k = tl.arange(0, BLOCK_K)
+    # A shape (M=10, K=1280), row-major strides (1280, 1)
+    a_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),  # (1280, 1)
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),  # K is contiguous (inner dim)
+    )
+    # B stored as (N=1000, K=1280) row-major; logically accessed as (K, N)
+    # stride_bk=1, stride_bn=1280 -> B[k,n] at k*1 + n*1280
+    b_block_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),  # (1, 1280)
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(0, 1),  # K is contiguous for B (stride_bk=1)
+    )
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 1280*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 1000*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 1280*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-        
+        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option='zero')
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
+        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
+        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -88,8 +85,6 @@

     tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
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
