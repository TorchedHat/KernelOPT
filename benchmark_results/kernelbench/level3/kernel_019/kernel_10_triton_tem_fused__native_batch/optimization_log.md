# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_10` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 37630 ns — **Best latency**: 37250 ns (**1.0102×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`eliminate_shared_memory_bank_conflicts_via_explicit_smem_staging_with_swizzle`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_10`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_019/kernel_10_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (246,1,1), Waves/SM: 0.62
- **Registers/thread**: 168
- **Duration**: 37630 ns
- **SM Throughput**: 31.50 % | **Memory**: 71.61 % | **DRAM**: 2.27 %
- **Achieved Occupancy**: 11.89 %
- **IPC Active**: 1.43 inst/cycle | **No Eligible**: 64.07 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.001× | 0.988× | ✓ |  |
| C | add_autotune_with_evict_first_cache_policy | 1.004× | 0.987× | ✓ | **✓ best** |
| D | autotune_tile_sizes_reduce_register_pressure | 1.001× | 0.717× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | add_autotune_with... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 37630 ns | 38080 ns | +450 ns (+1.2%) | 38140 ns | +510 ns (+1.4%) | 52510 ns | +14880 ns (+39.5%) |
| SM Throughput (%) | 31.50 % | 31.60 % | +0.10 pp | 31.47 % | -0.03 pp | 22.69 % | -8.81 pp |
| Memory Throughput (%) | 71.61 % | 71.79 % | +0.18 pp | 71.55 % | -0.06 pp | 34.30 % | -37.31 pp |
| DRAM Throughput (%) | 2.27 % | 2.24 % | -0.03 pp | 2.24 % | -0.03 pp | 1.66 % | -0.61 pp |
| Achieved Occupancy (%) | 11.89 % | 11.88 % | -0.01 pp | 11.89 % | +0.00 pp | 5.87 % | -6.02 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 255.00 register/thread | +87 |
| Waves/SM | 0.62 | 0.62 | +0.00 | 0.62 | +0.00 | 0.47 | -0.15 |
| No Eligible (%) | 64.07 % | 64.14 % | +0.07 pp | 64.10 % | +0.03 pp | 72.32 % | +8.25 pp |
| IPC Active | 1.43 inst/cycle | 1.44 inst/cycle | +0.01 | 1.43 inst/cycle | +0.00 | 1.05 inst/cycle | -0.38 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_load_indexing_to_eliminate_bank_conflicts | 0.998× | 0.992× | ✓ |  |
| B | fix_noncoalesced_b_matrix_load_to_eliminate_bank_conflicts | 1.003× | 0.993× | ✓ |  |
| C | cap_registers_with_maxnreg_to_improve_occupancy | 0.879× | 0.108× | ✓ |  |
| D | add_triton_autotune | 1.003× | 0.930× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | fix_noncoalesced_... | Δ | cap_registers_wit... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 37630 ns | 37920 ns | +290 ns (+0.8%) | 37890 ns | +260 ns (+0.7%) | 347710 ns | +310080 ns (+824.0%) | 40480 ns | +2850 ns (+7.6%) |
| SM Throughput (%) | 31.50 % | 31.74 % | +0.24 pp | 31.50 % | +0.00 pp | 6.18 % | -25.32 pp | 29.36 % | -2.14 pp |
| Memory Throughput (%) | 71.61 % | 72.11 % | +0.50 pp | 71.60 % | -0.01 pp | 11.03 % | -60.58 pp | 66.84 % | -4.77 pp |
| DRAM Throughput (%) | 2.27 % | 2.25 % | -0.02 pp | 2.25 % | -0.02 pp | 5.91 % | +3.64 pp | 2.11 % | -0.16 pp |
| Achieved Occupancy (%) | 11.89 % | 11.87 % | -0.02 pp | 11.89 % | +0.00 pp | 5.74 % | -6.15 pp | 11.93 % | +0.04 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 128.00 register/thread | -40 | 168.00 register/thread | +0 |
| Waves/SM | 0.62 | 0.62 | +0.00 | 0.62 | +0.00 | 0.23 | -0.39 | 0.62 | +0.00 |
| No Eligible (%) | 64.07 % | 64.34 % | +0.27 pp | 64.24 % | +0.17 pp | 93.60 % | +29.53 pp | 67.05 % | +2.98 pp |
| IPC Active | 1.43 inst/cycle | 1.43 inst/cycle | +0.00 | 1.43 inst/cycle | +0.00 | 0.23 inst/cycle | -1.20 | 1.32 inst/cycle | -0.11 |


### Key Learnings

#### ❌ Plan C: `cap_registers_with_maxnreg_to_improve_occupancy` (0.108×)

Adding maxnreg=128 to Triton autotune configs reduced registers/thread from 168 to 128 and raised theoretical occupancy from 18.75% to 25%, but achieved occupancy collapsed from 11.88% to 5.74% and duration increased 9x (38us → 348us). The root cause is that forced register spilling to local memory introduced massive L1TEX stalls: warp cycles per issued instruction jumped from 5.3 to 15.65, idle scheduler cycles rose from 64% to 93.6%, and the CPIStall rule flagged L1TEX scoreboard stalls consuming 78.6% of cycles. The anti-pattern is applying maxnreg when the kernel already has low wave count (0.62 waves): halving the block size from 128 to 64 threads also halved the grid's effective parallelism (waves dropped from 0.62 to 0.23), compounding the spill penalty with even less occupancy to hide latency.

> **Applicable when**: Never: this strategy caused a 9x regression. Do not apply maxnreg capping when the kernel already has low wave count (<1.0 waves per SM) and the block size reduction would further reduce parallelism. Register spill latency to L1TEX/L2 is far more costly than the occupancy gain when there are insufficient waves to hide it.

> **Avoid when**: Any kernel with waves_per_sm < 1.0 where the grid size is already small relative to SM count. Also avoid when the kernel's working set per thread is large (e.g., large accumulator tiles in GEMM), as forced spilling of accumulator registers to local memory creates uncoalesced L1TEX accesses (1 of 32 bytes utilized per sector, per NCU MemoryCacheAccessPattern rule). The combination of low occupancy + register spill is catastrophic: schedulers have no other warps to switch to while waiting on spill loads.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_shared_memory_bank_conflicts_via_explicit_smem_staging_with_swizzle | 1.024× | 1.010× | ✓ | **✓ best** |
| B | transpose_b_tile_load_to_eliminate_bank_conflicts | 0.998× | 0.931× | ✓ |  |
| C | remove_maxnreg_spill_and_fix_b_matrix_coalesced_load | 1.003× | 0.712× | ✓ |  |
| D | fix_b_matrix_load_to_eliminate_shared_memory_bank_conflicts | 1.001× | 0.933× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_shared_... | Δ | transpose_b_tile_... | Δ | remove_maxnreg_sp... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 37630 ns | 37250 ns | -380 ns (-1.0%) | 40420 ns | +2790 ns (+7.4%) | 52830 ns | +15200 ns (+40.4%) | 40350 ns | +2720 ns (+7.2%) |
| SM Throughput (%) | 31.50 % | 32.02 % | +0.52 pp | 29.39 % | -2.11 pp | 22.64 % | -8.86 pp | 29.25 % | -2.25 pp |
| Memory Throughput (%) | 71.61 % | 72.70 % | +1.09 pp | 66.87 % | -4.74 pp | 34.22 % | -37.39 pp | 66.59 % | -5.02 pp |
| DRAM Throughput (%) | 2.27 % | 2.29 % | +0.02 pp | 2.11 % | -0.16 pp | 1.65 % | -0.62 pp | 2.12 % | -0.15 pp |
| Achieved Occupancy (%) | 11.89 % | 11.90 % | +0.01 pp | 11.93 % | +0.04 pp | 5.87 % | -6.02 pp | 11.92 % | +0.03 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 255.00 register/thread | +87 | 168.00 register/thread | +0 |
| Waves/SM | 0.62 | 0.62 | +0.00 | 0.62 | +0.00 | 0.47 | -0.15 | 0.62 | +0.00 |
| No Eligible (%) | 64.07 % | 63.51 % | -0.56 pp | 67.12 % | +3.05 pp | 72.01 % | +7.94 pp | 67.08 % | +3.01 pp |
| IPC Active | 1.43 inst/cycle | 1.46 inst/cycle | +0.03 | 1.32 inst/cycle | -0.11 | 1.05 inst/cycle | -0.38 | 1.32 inst/cycle | -0.11 |

---

## Iteration 4

**Best so far**: 1.0102× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_shared_memory_bank_conflicts_via_smem_padding_on_b_tile | 1.014× | 1.006× | ✓ |  |
| B | stage_b_matrix_in_transposed_padded_shared_memory_to_eliminate_bank_conflicts | 0.998× | 0.932× | ✓ |  |
| C | restore_block_size_128_and_add_b_load_contiguity_hints | 0.999× | 0.933× | ✓ |  |
| D | eliminate_tl_trans_b_to_fix_shared_memory_bank_conflicts | 0.997× | 0.930× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_shared_... | Δ | stage_b_matrix_in... | Δ | restore_block_siz... | Δ | eliminate_tl_tran... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 37630 ns | 37410 ns | -220 ns (-0.6%) | 40380 ns | +2750 ns (+7.3%) | 40320 ns | +2690 ns (+7.1%) | 40480 ns | +2850 ns (+7.6%) |
| SM Throughput (%) | 31.50 % | 32.04 % | +0.54 pp | 29.19 % | -2.31 pp | 29.37 % | -2.13 pp | 29.24 % | -2.26 pp |
| Memory Throughput (%) | 71.61 % | 72.67 % | +1.06 pp | 66.40 % | -5.21 pp | 66.90 % | -4.71 pp | 66.58 % | -5.03 pp |
| DRAM Throughput (%) | 2.27 % | 2.28 % | +0.01 pp | 2.11 % | -0.16 pp | 2.12 % | -0.15 pp | 2.11 % | -0.16 pp |
| Achieved Occupancy (%) | 11.89 % | 11.89 % | +0.00 pp | 11.92 % | +0.03 pp | 11.92 % | +0.03 pp | 11.92 % | +0.03 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 0.62 | 0.62 | +0.00 | 0.62 | +0.00 | 0.62 | +0.00 | 0.62 | +0.00 |
| No Eligible (%) | 64.07 % | 63.53 % | -0.54 pp | 66.97 % | +2.90 pp | 67.10 % | +3.03 pp | 67.05 % | +2.98 pp |
| IPC Active | 1.43 inst/cycle | 1.46 inst/cycle | +0.03 | 1.31 inst/cycle | -0.12 | 1.31 inst/cycle | -0.12 | 1.32 inst/cycle | -0.11 |

---

## Iteration 5

**Best so far**: 1.0102× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_manual_b_transpose_with_block_ptr_column_major_order_to_eliminate_smem_bank_conflicts | 1.024× | 1.004× | ✓ |  |
| B | cache_modifier_differentiation_for_a_and_b_loads | 0.997× | 0.930× | ✓ |  |
| C | eliminate_tl_trans_by_loading_b_directly_as_block_k_block_n | 1.001× | 0.930× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_manual_b_... | Δ | cache_modifier_di... | Δ | eliminate_tl_tran... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 37630 ns | 37500 ns | -130 ns (-0.3%) | 40450 ns | +2820 ns (+7.5%) | 40450 ns | +2820 ns (+7.5%) |
| SM Throughput (%) | 31.50 % | 31.81 % | +0.31 pp | 29.30 % | -2.20 pp | 29.26 % | -2.24 pp |
| Memory Throughput (%) | 71.61 % | 72.22 % | +0.61 pp | 66.74 % | -4.87 pp | 66.62 % | -4.99 pp |
| DRAM Throughput (%) | 2.27 % | 2.28 % | +0.01 pp | 2.11 % | -0.16 pp | 2.11 % | -0.16 pp |
| Achieved Occupancy (%) | 11.89 % | 11.91 % | +0.02 pp | 11.92 % | +0.03 pp | 11.92 % | +0.03 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 0.62 | 0.62 | +0.00 | 0.62 | +0.00 | 0.62 | +0.00 |
| No Eligible (%) | 64.07 % | 63.74 % | -0.33 pp | 66.95 % | +2.88 pp | 67.09 % | +3.02 pp |
| IPC Active | 1.43 inst/cycle | 1.46 inst/cycle | +0.03 | 1.32 inst/cycle | -0.11 | 1.31 inst/cycle | -0.12 |

---

## Cumulative Progress

```
Baseline:  37630 ns  (1.000×)
Iter 3     37250 ns  (1.010×)  — eliminate_shared_memory_bank_conflicts_via_explicit_smem_staging_with_swizzle

Final best: 37250 ns  (1.0102×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_10(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -46,35 +60,32 @@

 
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
     rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
-        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
-    else:
-        offs_a_m = rm % M
-    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
-        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
-    else:
-        offs_b_n = rn % N
-    offs_k = tl.arange(0, BLOCK_K)
+
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
+    # Create block pointer for A (unchanged)
+    a_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),  # (128, 1)
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),  # (64, 32)
+        order=(1, 0)  # row-major: forces conflict-free smem layout for A
+    )
+    # B is loaded transposed inside the loop to eliminate smem bank conflicts
+
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 128*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 256*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 128*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-
+        a = tl.load(a_ptr, boundary_check=(0, 1))  # Triton generates swizzled smem staging
+        # Load B transposed as (BLOCK_N, BLOCK_K) to get conflict-free smem inner dim = BLOCK_K=32
+        rk = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)        # shape (BLOCK_K,)
+        rn_local = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # shape (BLOCK_N,)
+        b_T_ptrs = B + rn_local[:, None] * stride_bn + rk[None, :] * stride_bk  # (BLOCK_N, BLOCK_K)
+        b_T_mask = (rn_local[:, None] < N) & (rk[None, :] < K)
+        b_T = tl.load(b_T_ptrs, mask=b_T_mask, other=0.0)  # (BLOCK_N, BLOCK_K) - inner dim=32, conflict-free
+        b = tl.trans(b_T)  # (BLOCK_K, BLOCK_N) - smem staged as (BLOCK_N, BLOCK_K) row-major
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
+        a_ptr = tl.advance(a_ptr, (0, BLOCK_K))   # advance K dimension
+        # No b_ptr advance needed - k_idx drives rk directly
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -86,8 +97,6 @@

     # inductor generates a suffix
     xindex = idx_n + 256*idx_m
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
