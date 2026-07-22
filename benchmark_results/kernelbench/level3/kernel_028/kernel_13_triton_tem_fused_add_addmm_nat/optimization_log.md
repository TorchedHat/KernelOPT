# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_add_addmm_native_layer_norm_select_t_view_13` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 27550 ns — **Best latency**: 15680 ns (**1.7570×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`reduce_block_n_to_cut_smem_pressure_and_fix_grid`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_add_addmm_native_layer_norm_select_t_view_13`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_028/kernel_13_triton_tem_fused_add_addmm_nat/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (64,1,1), Waves/SM: 0.05
- **Registers/thread**: 54
- **Duration**: 27550 ns
- **SM Throughput**: 3.91 % | **Memory**: 41.18 % | **DRAM**: 3.11 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.35 inst/cycle | **No Eligible**: 91.28 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.567× | 0.535× | ✓ |  |
| C | add_eviction_policies_and_increase_tile_sizes | 0.997× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | add_eviction_poli... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 27550 ns | 51490 ns | +23940 ns (+86.9%) | 27580 ns | +30 ns (+0.1%) |
| SM Throughput (%) | 3.91 % | 6.38 % | +2.47 pp | 3.86 % | -0.05 pp |
| Memory Throughput (%) | 41.18 % | 44.33 % | +3.15 pp | 41.19 % | +0.01 pp |
| DRAM Throughput (%) | 3.11 % | 1.67 % | -1.44 pp | 3.11 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 54.00 register/thread | 80.00 register/thread | +26 | 64.00 register/thread | +10 |
| Waves/SM | 0.05 | 0.08 | +0.03 | 0.10 | +0.05 |
| No Eligible (%) | 91.28 % | 86.28 % | -5.00 pp | 91.38 % | +0.10 pp |
| IPC Active | 0.35 inst/cycle | 0.55 inst/cycle | +0.20 | 0.34 inst/cycle | -0.01 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads | 1.011× | 1.008× | ✓ | **✓ best** |
| C | fix_grid_and_reduce_block_m_for_tiny_m_gemm | 1.006× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads | Δ | fix_grid_and_redu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 27550 ns | 27330 ns | -220 ns (-0.8%) | 27520 ns | -30 ns (-0.1%) |
| SM Throughput (%) | 3.91 % | 3.98 % | +0.07 pp | 3.92 % | +0.01 pp |
| Memory Throughput (%) | 41.18 % | 42.70 % | +1.52 pp | 41.19 % | +0.01 pp |
| DRAM Throughput (%) | 3.11 % | 3.14 % | +0.03 pp | 3.12 % | +0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 54.00 register/thread | 66.00 register/thread | +12 | 54.00 register/thread | +0 |
| Waves/SM | 0.05 | 0.10 | +0.05 | 0.05 | +0.00 |
| No Eligible (%) | 91.28 % | 91.19 % | -0.09 pp | 91.29 % | +0.01 pp |
| IPC Active | 0.35 inst/cycle | 0.36 inst/cycle | +0.01 | 0.35 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.008× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_with_correct_grid_lambda_for_tiny_m_gemm | 0.953× | 0.711× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_with_cor... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 27550 ns | 38750 ns | +11200 ns (+40.7%) |
| SM Throughput (%) | 3.91 % | 3.05 % | -0.86 pp |
| Memory Throughput (%) | 41.18 % | 31.53 % | -9.65 pp |
| DRAM Throughput (%) | 3.11 % | 2.22 % | -0.89 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp |
| Registers/Thread | 54.00 register/thread | 130.00 register/thread | +76 |
| Waves/SM | 0.05 | 0.08 | +0.03 |
| No Eligible (%) | 91.28 % | 84.51 % | -6.77 pp |
| IPC Active | 0.35 inst/cycle | 0.31 inst/cycle | -0.04 |


### Key Learnings

#### ❌ Plan A: `autotune_with_correct_grid_lambda_for_tiny_m_gemm` (0.711×)

Reducing block size from 128 to 64 threads (while keeping the same tile shape BLOCK_M=2, BLOCK_N=256, BLOCK_K=32) caused registers per thread to nearly double from 66 to 130, cutting theoretical occupancy from 31.25% to 18.75% and achieved occupancy from 6.25% to 3.12%. Although Warp Cycles Per Issued Instruction improved from 11.24 to 6.54 (bank conflict stalls reduced), the overall kernel duration increased by 42% (27.33→38.75 µs) because the occupancy collapse outweighed the stall reduction. The autotune configs in this iteration did not vary BLOCK_M/BLOCK_N/BLOCK_K (they were hardcoded as constexprs inside the kernel body), so the autotuner only swept num_warps/num_stages without changing tile geometry — the grid remained fixed at 64 blocks, providing no additional SM coverage.

> **Applicable when**: Never — this specific pattern (autotune only sweeps num_warps/num_stages while tile sizes remain hardcoded constexprs, and block size is halved) is always harmful.

> **Avoid when**: Do not reduce block_size (num_warps) without also reducing tile sizes proportionally. When registers per thread are already moderate (66/thread), halving the block size forces the compiler to spill or allocate more registers per thread (130/thread), collapsing theoretical occupancy. Do not add @triton.autotune with an empty config dict (no BLOCK_* keys) when the goal is to autotune tile geometry — the autotuner will only vary num_warps/num_stages and leave tile shapes unchanged. Do not keep the grid hardcoded at (64,1,1) when changing BLOCK_M from 16 to 2, as this leaves the grid coverage unchanged despite the tile size change.

---

## Iteration 4

**Best so far**: 1.008× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_block_ptr_memory_order_to_eliminate_bank_conflicts | 0.953× | 0.709× | ✓ |  |
| B | fix_b_block_ptr_order_to_eliminate_smem_bank_conflicts | 0.950× | 0.938× | ✓ |  |
| C | evict_first_streaming_loads_and_store | 0.953× | 0.712× | ✓ |  |
| D | autotune_tile_geometry_via_kernel_args_with_dynamic_grid | 1.882× | 1.367× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_b_block_ptr_m... | Δ | fix_b_block_ptr_o... | Δ | evict_first_strea... | Δ | autotune_tile_geo... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 27550 ns | 38850 ns | +11300 ns (+41.0%) | 29380 ns | +1830 ns (+6.6%) | 38690 ns | +11140 ns (+40.4%) | 20160 ns | -7390 ns (-26.8%) |
| SM Throughput (%) | 3.91 % | 3.07 % | -0.84 pp | 4.81 % | +0.90 pp | 3.01 % | -0.90 pp | 2.02 % | -1.89 pp |
| Memory Throughput (%) | 41.18 % | 31.73 % | -9.45 pp | 43.22 % | +2.04 pp | 31.17 % | -10.01 pp | 16.19 % | -24.99 pp |
| DRAM Throughput (%) | 3.11 % | 2.21 % | -0.90 pp | 2.92 % | -0.19 pp | 2.22 % | -0.89 pp | 4.25 % | +1.14 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 54.00 register/thread | 130.00 register/thread | +76 | 80.00 register/thread | +26 | 130.00 register/thread | +76 | 56.00 register/thread | +2 |
| Waves/SM | 0.05 | 0.08 | +0.03 | 0.16 | +0.11 | 0.08 | +0.03 | 0.03 | -0.02 |
| No Eligible (%) | 91.28 % | 84.20 % | -7.08 pp | 89.44 % | -1.84 pp | 84.85 % | -6.43 pp | 89.55 % | -1.73 pp |
| IPC Active | 0.35 inst/cycle | 0.31 inst/cycle | -0.04 | 0.42 inst/cycle | +0.07 | 0.30 inst/cycle | -0.05 | 0.21 inst/cycle | -0.14 |


### Key Learnings

#### ❌ Plan C: `evict_first_streaming_loads_and_store` (0.712×)

Switching tl.load calls for A/B matrix tiles and the epilogue tl.store to eviction_policy='evict_first' produced essentially no change in any profiling metric: duration moved from 38.75 to 38.69 µs (within noise), L2 hit rate actually dropped slightly from 6.46% to 6.31%, and L1/TEX throughput fell marginally from 79.06% to 78.22%. The dominant bottleneck — 22-way shared memory bank conflicts causing 84.81% excess wavefronts and 60%+ MIO stall cycles — was completely unaffected by eviction policy changes, since bank conflicts occur in shared memory (L1), not in L2 or DRAM. When the kernel is already DRAM-underutilized (DRAM throughput only 2.22%) and L2 hit rate is already very low (6.46%), eviction policy hints have no meaningful lever to pull: there is no reusable L2 data being evicted, and the bottleneck is entirely in shared memory access patterns.

> **Applicable when**: Never applicable as a primary optimization when shared memory bank conflicts are the dominant bottleneck (SharedMemoryConflicts rule with >65% speedup_pct, MIO stall >60% of CPI, DRAM throughput <5%). Eviction policy changes only help when L2 thrashing from large streaming tensors is the actual bottleneck (high DRAM throughput, high L2 miss rate on reused data).

> **Avoid when**: Do not apply evict_first to streaming loads/stores when: (1) DRAM throughput is already very low (<5%), indicating the kernel is not memory-bandwidth bound at the DRAM level; (2) L2 hit rate is already very low (<10%), meaning there is no reusable L2 data being evicted; (3) the dominant bottleneck is shared memory bank conflicts (SharedMemoryConflicts rule, MIO stall >50% of CPI) — eviction policies have zero effect on shared memory access patterns. The speedup in this case was 0.71x (regression within noise), confirming the strategy is a dead end for this bottleneck profile.

---

## Iteration 5

**Best so far**: 1.3666× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_to_match_actual_tile_count | 1.885× | 1.380× | ✓ |  |
| B | reduce_block_n_to_cut_smem_pressure_and_fix_grid | 1.982× | 1.757× | ✓ | **✓ best** |
| C | reduce_block_n_constexpr_to_cut_register_pressure | 1.012× | 0.723× | ✓ |  |
| D | reduce_block_n_constexpr_to_cut_register_pressure | 1.875× | 1.384× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | reduce_block_n_to... | Δ | reduce_block_n_co... | Δ | reduce_block_n_co... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 27550 ns | 19970 ns | -7580 ns (-27.5%) | 15680 ns | -11870 ns (-43.1%) | 38110 ns | +10560 ns (+38.3%) | 19900 ns | -7650 ns (-27.8%) |
| SM Throughput (%) | 3.91 % | 2.01 % | -1.90 pp | 3.69 % | -0.22 pp | 4.34 % | +0.43 pp | 2.02 % | -1.89 pp |
| Memory Throughput (%) | 41.18 % | 16.12 % | -25.06 pp | 38.18 % | -3.00 pp | 30.26 % | -10.92 pp | 16.23 % | -24.95 pp |
| DRAM Throughput (%) | 3.11 % | 4.29 % | +1.18 pp | 5.47 % | +2.36 pp | 2.25 % | -0.86 pp | 4.31 % | +1.20 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 54.00 register/thread | 56.00 register/thread | +2 | 48.00 register/thread | -6 | 96.00 register/thread | +42 | 56.00 register/thread | +2 |
| Waves/SM | 0.05 | 0.03 | -0.02 | 0.05 | +0.00 | 0.05 | +0.00 | 0.03 | -0.02 |
| No Eligible (%) | 91.28 % | 89.36 % | -1.92 pp | 91.32 % | +0.04 pp | 79.77 % | -11.51 pp | 89.45 % | -1.83 pp |
| IPC Active | 0.35 inst/cycle | 0.21 inst/cycle | -0.14 | 0.35 inst/cycle | +0.00 | 0.41 inst/cycle | +0.06 | 0.21 inst/cycle | -0.14 |


### Key Learnings

#### ✅ Plan A: `fix_hardcoded_grid_to_match_actual_tile_count` (1.380×)

When BLOCK_N is reduced (e.g., from 256 to 64), the number of required output tiles increases proportionally (e.g., from 8 to 32 for N=2048), but a hardcoded grid value in the kernel launch function becomes stale and launches too many or too few blocks. Correcting the grid from the stale value to ceil(M/BLOCK_M)*ceil(N/BLOCK_N) eliminates wasted blocks that do no useful work, reducing kernel duration. The profiling signature for this bug is: LaunchConfiguration rule triggered, achieved occupancy far below theoretical (3.12% vs 56.25%), and waves_per_sm << 1 with a grid size that is an integer multiple of the correct tile count.

> **Applicable when**: LaunchConfiguration NCU rule is triggered; achieved occupancy is far below theoretical occupancy (e.g., <10% vs >50%); waves_per_sm << 1; the grid size is a round number that does not match ceil(M/BLOCK_M)*ceil(N/BLOCK_N) for the current tile parameters. Most likely to occur after a prior optimization changed BLOCK_N or BLOCK_M without updating the hardcoded grid value in the kernel launch wrapper.

> **Avoid when**: When the grid is already dynamically computed as a lambda or expression using META['BLOCK_M'] and META['BLOCK_N']. When the grid size correctly matches the tile count. When the kernel uses split-K or other parallelism strategies that intentionally launch more blocks than output tiles.

#### ✅ Plan C: `reduce_block_n_constexpr_to_cut_register_pressure` (1.757×)

In a small Triton GEMM kernel (M=2, N=2048, K=512) with BLOCK_N=64 and 64 threads (2 warps), doubling the block size to 128 threads (4 warps) while expanding the autotune config space (adding num_stages=3/4 variants) reduced registers per thread from 56 to 48, improved achieved occupancy from 3.12% to 6.25%, and drove L1/TEX cache throughput from 43.44% to 90.68%. Memory throughput increased from 208.67 to 268.21 GB/s and kernel duration dropped from 20.16 to 15.68 µs (1.76x speedup). The dominant causal mechanism was better warp-level parallelism within each block: more warps per block allowed the scheduler to hide latency more effectively, even though Warp Cycles Per Issued Instruction slightly increased (9.44→11.37) due to the larger shared memory footprint per block.

> **Applicable when**: Triton GEMM kernel where achieved occupancy is very low (<5%) and L1/TEX cache throughput is well below 80%, indicating the scheduler cannot hide latency with the current number of warps per block. Registers per thread are moderate (40-60) and not the binding occupancy limiter. The autotune config space only covers num_stages=2 without exploring num_stages=3/4, which may enable better software pipelining. Block size is small (64 threads / 2 warps) relative to what the SM can support.

> **Avoid when**: When registers per thread are already high (>80) and Block Limit Registers is the binding occupancy constraint — increasing block size will further inflate register pressure. When shared memory per block is already near the SM limit and Block Limit Shared Mem is the binding constraint. When the kernel is already compute-bound (Compute SM Throughput >50%) and adding more warps per block would not help hide latency.

#### ✅ Plan D: `reduce_block_n_constexpr_to_cut_register_pressure` (1.757×)

In a small Triton GEMM kernel (M=2, N=2048, K=512) with BLOCK_N=64 and 64 threads (2 warps), doubling the block size to 128 threads (4 warps) while expanding the autotune config space (adding num_stages=3/4 variants) reduced registers per thread from 56 to 48, improved achieved occupancy from 3.12% to 6.25%, and drove L1/TEX cache throughput from 43.44% to 90.68%. Memory throughput increased from 208.67 to 268.21 GB/s and kernel duration dropped from 20.16 to 15.68 µs (1.76x speedup). The dominant causal mechanism was better warp-level parallelism within each block: more warps per block allowed the scheduler to hide latency more effectively, even though Warp Cycles Per Issued Instruction slightly increased (9.44→11.37) due to the larger shared memory footprint per block.

> **Applicable when**: Triton GEMM kernel where achieved occupancy is very low (<5%) and L1/TEX cache throughput is well below 80%, indicating the scheduler cannot hide latency with the current number of warps per block. Registers per thread are moderate (40-60) and not the binding occupancy limiter. The autotune config space only covers num_stages=2 without exploring num_stages=3/4, which may enable better software pipelining. Block size is small (64 threads / 2 warps) relative to what the SM can support.

> **Avoid when**: When registers per thread are already high (>80) and Block Limit Registers is the binding occupancy constraint — increasing block size will further inflate register pressure. When shared memory per block is already near the SM limit and Block Limit Shared Mem is the binding constraint. When the kernel is already compute-bound (Compute SM Throughput >50%) and adding more warps per block would not help hide latency.

---

## Cumulative Progress

```
Baseline:  27550 ns  (1.000×)
Iter 2     27330 ns  (1.008×)  — tma_loads
Iter 4     20160 ns  (1.367×)  — autotune_tile_geometry_via_kernel_args_with_dynamic_grid
Iter 5     15680 ns  (1.757×)  — reduce_block_n_to_cut_smem_pressure_and_fix_grid

Final best: 15680 ns  (1.7570×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,27 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_add_addmm_native_layer_norm_select_t_view_13(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 16
-    BLOCK_N : tl.constexpr = 32
+    BLOCK_M : tl.constexpr = 2
+    BLOCK_N : tl.constexpr = 64
     BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
@@ -57,24 +71,33 @@

     offs_k = tl.arange(0, BLOCK_K)
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
+    # Create TMA block pointers before the k_idx loop
+    # A has shape (M=2, K=512) with strides (stride_am=100864, stride_ak=1)
+    a_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    # B is accessed as B[k,n] = B + k*1 + n*512, so shape=(K,N), strides=(stride_bk=1, stride_bn=512)
+    # order=(0,1): K-dim (dim 0) is fastest in smem, matching stride_bk=1 physical layout -> conflict-free
+    b_block_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(0, 1),
+    )
+
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 512*idx_m
-        a = tl.load(A + ((tl.broadcast_to(idx_n + 100864*idx_m, [BLOCK_M, BLOCK_K])).broadcast_to(xindex.shape)))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 2048*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
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
@@ -85,9 +108,7 @@

 
     # inductor generates a suffix
     xindex = idx_n + 2048*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask, eviction_policy='evict_last')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
