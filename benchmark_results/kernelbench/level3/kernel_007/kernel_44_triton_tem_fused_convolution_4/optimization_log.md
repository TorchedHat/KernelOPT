# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_44` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 57340 ns — **Best latency**: 57500 ns (**0.9972×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_44`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_44_triton_tem_fused_convolution_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (124,1,1), Waves/SM: 0.19
- **Registers/thread**: 92
- **Duration**: 57340 ns
- **SM Throughput**: 11.23 % | **Memory**: 81.47 % | **DRAM**: 1.51 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.50 inst/cycle | **No Eligible**: 87.49 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.995× | 1.003× | ✓ |  |
| B | autotune_tile_sizes_to_eliminate_smem_bank_conflicts_and_improve_occupancy | 0.985× | 0.983× | ✓ |  |
| C | fix_matrix_b_load_coalescing_and_autotune_tile_sizes | 1.413× | 0.997× | ✓ | **✓ best** |
| D | autotune_tile_sizes_to_fix_bank_conflicts_and_occupancy | 0.858× | 0.859× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | autotune_tile_siz... | Δ | fix_matrix_b_load... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 57340 ns | 57150 ns | -190 ns (-0.3%) | 58340 ns | +1000 ns (+1.7%) | 57500 ns | +160 ns (+0.3%) | 66780 ns | +9440 ns (+16.5%) |
| SM Throughput (%) | 11.23 % | 11.23 % | +0.00 pp | 21.25 % | +10.02 pp | 11.21 % | -0.02 pp | 17.90 % | +6.67 pp |
| Memory Throughput (%) | 81.47 % | 81.47 % | +0.00 pp | 81.33 % | -0.14 pp | 81.32 % | -0.15 pp | 69.66 % | -11.81 pp |
| DRAM Throughput (%) | 1.51 % | 1.52 % | +0.01 pp | 1.49 % | -0.02 pp | 1.51 % | +0.00 pp | 1.31 % | -0.20 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 92.00 register/thread | 92.00 register/thread | +0 | 128.00 register/thread | +36 | 92.00 register/thread | +0 | 128.00 register/thread | +36 |
| Waves/SM | 0.19 | 0.19 | +0.00 | 0.23 | +0.04 | 0.19 | +0.00 | 0.31 | +0.12 |
| No Eligible (%) | 87.49 % | 87.49 % | +0.00 pp | 76.24 % | -11.25 pp | 87.50 % | +0.01 pp | 79.61 % | -7.88 pp |
| IPC Active | 0.50 inst/cycle | 0.50 inst/cycle | +0.00 | 0.95 inst/cycle | +0.45 | 0.50 inst/cycle | +0.00 | 0.81 inst/cycle | +0.31 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_matrix_b_broadcast_load_to_eliminate_smem_bank_conflicts | 0.995× | 0.999× | ✓ |  |
| B | eliminate_tl_trans_to_remove_shared_memory_bank_conflicts | 1.412× | 1.000× | ✓ |  |
| C | fix_matrix_b_smem_layout_order_to_eliminate_bank_conflicts | 0.980× | 0.983× | ✓ |  |
| D | fix_b_matrix_load_with_transposed_coalesced_load_to_eliminate_bank_conflicts | 0.850× | 0.854× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_matrix_b_broa... | Δ | eliminate_tl_tran... | Δ | fix_matrix_b_smem... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 57340 ns | 57410 ns | +70 ns (+0.1%) | 57340 ns | +0 ns (+0.0%) | 58340 ns | +1000 ns (+1.7%) | 67140 ns | +9800 ns (+17.1%) |
| SM Throughput (%) | 11.23 % | 11.23 % | +0.00 pp | 11.25 % | +0.02 pp | 21.22 % | +9.99 pp | 17.97 % | +6.74 pp |
| Memory Throughput (%) | 81.47 % | 81.43 % | -0.04 pp | 81.64 % | +0.17 pp | 81.25 % | -0.22 pp | 69.91 % | -11.56 pp |
| DRAM Throughput (%) | 1.51 % | 1.51 % | +0.00 pp | 1.51 % | +0.00 pp | 1.49 % | -0.02 pp | 1.30 % | -0.21 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 92.00 register/thread | 92.00 register/thread | +0 | 92.00 register/thread | +0 | 128.00 register/thread | +36 | 128.00 register/thread | +36 |
| Waves/SM | 0.19 | 0.19 | +0.00 | 0.19 | +0.00 | 0.23 | +0.04 | 0.31 | +0.12 |
| No Eligible (%) | 87.49 % | 87.53 % | +0.04 pp | 87.50 % | +0.01 pp | 76.27 % | -11.22 pp | 79.53 % | -7.96 pp |
| IPC Active | 0.50 inst/cycle | 0.50 inst/cycle | +0.00 | 0.50 inst/cycle | +0.00 | 0.95 inst/cycle | +0.45 | 0.82 inst/cycle | +0.32 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | transposed_b_load_at_original_tile_sizes_to_eliminate_bank_conflicts | 0.996× | 1.000× | ✓ |  |
| C | reduce_register_pressure_via_maxnreg_to_improve_occupancy | 0.957× | 0.461× | ✓ |  |
| D | reduce_tile_sizes_to_recover_occupancy_and_eliminate_tl_trans_bank_conflicts | 1.410× | 1.207× | ✓ |  |


### Metric Comparison

| Metric | Baseline | transposed_b_load... | Δ | reduce_register_p... | Δ | reduce_tile_sizes... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 57340 ns | 57310 ns | -30 ns (-0.1%) | 124450 ns | +67110 ns (+117.0%) | 47520 ns | -9820 ns (-17.1%) |
| SM Throughput (%) | 11.23 % | 11.24 % | +0.01 pp | 17.32 % | +6.09 pp | 12.77 % | +1.54 pp |
| Memory Throughput (%) | 81.47 % | 81.55 % | +0.08 pp | 55.14 % | -26.33 pp | 55.17 % | -26.30 pp |
| DRAM Throughput (%) | 1.51 % | 1.51 % | +0.00 pp | 0.70 % | -0.81 pp | 1.83 % | +0.32 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 92.00 register/thread | 92.00 register/thread | +0 | 64.00 register/thread | -28 | 128.00 register/thread | +36 |
| Waves/SM | 0.19 | 0.19 | +0.00 | 0.16 | -0.03 | 0.12 | -0.07 |
| No Eligible (%) | 87.49 % | 87.51 % | +0.02 pp | 84.54 % | -2.95 pp | 71.34 % | -16.15 pp |
| IPC Active | 0.50 inst/cycle | 0.50 inst/cycle | +0.00 | 0.62 inst/cycle | +0.12 | 0.57 inst/cycle | +0.07 |


### Key Learnings

#### ❌ Plan C: `reduce_register_pressure_via_maxnreg_to_improve_occupancy` (0.461×)

Capping registers per thread via maxnreg=64 (from 92) successfully reduced Block Limit Registers from 5 to 8 blocks/SM and raised theoretical occupancy from 31.25% to 37.5%, but achieved occupancy remained stuck at 6.25% because the grid (124 blocks, 132 SMs) is still less than one full wave — occupancy is wave-count-limited, not register-limited. The forced register spills to local memory caused instruction count to nearly triple (5.1M → 13.6M) and introduced severe uncoalesced local load/store traffic (MemoryCacheAccessPattern rule triggered, 53% speedup potential flagged), with only 1.0 of 32 bytes per sector utilized. The net effect was a 2.17x slowdown (duration 57 μs → 124 μs) because the spill overhead vastly outweighed the negligible occupancy gain in a wave-count-bottlenecked kernel.

> **Applicable when**: Never when the kernel is wave-count-limited (grid_size < sm_count, waves_per_sm < 1.0): in this regime, achieved occupancy is bounded by the number of blocks launched, not by register or shared memory limits. maxnreg only helps when Block Limit Registers is the binding occupancy constraint AND the grid has enough blocks to fill multiple waves.

> **Avoid when**: When grid_size ≤ sm_count (waves_per_sm < 1.0): the kernel cannot achieve higher occupancy regardless of register count because there are not enough blocks to assign to idle SMs. In this case, maxnreg causes register spills to local memory (uncoalesced, 1/32 bytes/sector utilization) that multiply instruction count without any occupancy benefit. Also do not apply when the kernel simultaneously changes tile sizes (BLOCK_N 32→64 in this case) — the confounded change makes it impossible to isolate the maxnreg effect and may introduce additional instruction overhead.

#### ✅ Plan D: `reduce_tile_sizes_to_recover_occupancy_and_eliminate_tl_trans_bank_conflicts` (1.207×)

When a Triton GEMM-style kernel suffers from severe shared memory bank conflicts (17.9-way, 83% conflict wavefronts) and high MIO stall cycles (8.0 cycles/instruction), reducing BLOCK_N and BLOCK_K (e.g. 64→32) cuts shared memory per block (65.5 KB→12.3 KB) and eliminates the need for tl.trans() by loading B directly in [BLOCK_K, BLOCK_N] layout. This reduced Warp Cycles Per Issued Instruction from 8.0 to 3.5 and L1/TEX Cache Throughput from 90% to 62%, indicating the L1 bottleneck was relieved, yielding a 1.21x speedup despite a slight drop in theoretical occupancy (31.25%→25%) and achieved occupancy (6.25%→3.12%). The key insight is that eliminating the tl.trans() shared-memory transpose — which is the root cause of bank conflicts — combined with smaller tiles that reduce smem pressure, produces a net win even when occupancy does not improve.

> **Applicable when**: Triton kernel with tl.trans() inside tl.dot() causing SharedMemoryConflicts (>10-way bank conflicts, >75% conflict wavefronts) and high MIO stall fraction (>50% of warp cycles). The B matrix has a memory layout where one dimension is stride-1 (fast) and the other is stride-N (slow), allowing a direct [BLOCK_K, BLOCK_N] load to be coalesced along the fast dimension without needing a transpose.

> **Avoid when**: When the kernel is already occupancy-limited and reducing tile sizes would drop theoretical occupancy below the point where latency hiding is possible. Also avoid when BLOCK_K reduction would cause the K-loop trip count to increase enough to dominate instruction overhead. Do not apply when the B matrix layout does not permit a coalesced [BLOCK_K, BLOCK_N] load (i.e., when neither K nor N is the stride-1 dimension).

---

## Cumulative Progress

```
Baseline:  57340 ns  (1.000×)

Final best: 57500 ns  (0.9972×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,19 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_convolution_44(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -58,23 +71,21 @@

     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
+
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 512*idx_m
         a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 112*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        # Direct B load: shape [BLOCK_K, BLOCK_N] — eliminates tl.trans() and shared memory bank conflicts
+        # B has stride_bk=1, stride_bn=512 (column-major: B[n, k] stored as B + n*512 + k)
+        idx_k_b = offs_k[:, None] + (k_idx * BLOCK_K)   # [BLOCK_K, 1]
+        idx_n_b = offs_b_n[None, :]                       # [1, BLOCK_N]
+        b = tl.load(B + (idx_n_b * 512 + idx_k_b))        # [BLOCK_K, BLOCK_N]
 
-        
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -88,8 +99,6 @@

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
