# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 52610 ns — **Best latency**: 45060 ns (**1.1676×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`fix_uncoalesced_b_matrix_load_and_shared_memory_bank_conflicts`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_019/kernel_3_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (980,1,1), Waves/SM: 3.71
- **Registers/thread**: 214
- **Duration**: 52610 ns
- **SM Throughput**: 23.66 % | **Memory**: 59.83 % | **DRAM**: 8.81 %
- **Achieved Occupancy**: 12.41 %
- **IPC Active**: 1.08 inst/cycle | **No Eligible**: 72.93 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | add_autotune_reduce_register_pressure_improve_occupancy | 1.177× | 1.168× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | add_autotune_redu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 52610 ns | 45060 ns | -7550 ns (-14.4%) |
| SM Throughput (%) | 23.66 % | 27.67 % | +4.01 pp |
| Memory Throughput (%) | 59.83 % | 48.31 % | -11.52 pp |
| DRAM Throughput (%) | 8.81 % | 11.11 % | +2.30 pp |
| Achieved Occupancy (%) | 12.41 % | 11.74 % | -0.67 pp |
| Registers/Thread | 214.00 register/thread | 255.00 register/thread | +41 |
| Waves/SM | 3.71 | 1.86 | -1.85 |
| No Eligible (%) | 72.93 % | 67.95 % | -4.98 pp |
| IPC Active | 1.08 inst/cycle | 1.26 inst/cycle | +0.18 |

---

## Iteration 2

**Best so far**: 1.1676× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_uncoalesced_b_matrix_load_and_shared_memory_bank_conflicts | 1.167× | 1.168× | ✓ |  |
| B | tma_loads_via_make_block_ptr | 1.108× | 1.113× | ✓ |  |
| C | add_cache_eviction_policies_streaming_A_evict_first_reused_B_evict_last | 1.175× | 1.200× | ✓ |  |
| D | fix_b_matrix_load_coalescing_and_reduce_block_m_for_occupancy | 1.169× | 1.183× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_uncoalesced_b... | Δ | tma_loads_via_mak... | Δ | add_cache_evictio... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 52610 ns | 45060 ns | -7550 ns (-14.4%) | 47260 ns | -5350 ns (-10.2%) | 43840 ns | -8770 ns (-16.7%) | 44480 ns | -8130 ns (-15.5%) |
| SM Throughput (%) | 23.66 % | 28.08 % | +4.42 pp | 26.25 % | +2.59 pp | 27.76 % | +4.10 pp | 27.97 % | +4.31 pp |
| Memory Throughput (%) | 59.83 % | 49.02 % | -10.81 pp | 48.59 % | -11.24 pp | 48.39 % | -11.44 pp | 48.84 % | -10.99 pp |
| DRAM Throughput (%) | 8.81 % | 11.14 % | +2.33 pp | 16.28 % | +7.47 pp | 11.29 % | +2.48 pp | 11.25 % | +2.44 pp |
| Achieved Occupancy (%) | 12.41 % | 11.76 % | -0.65 pp | 11.63 % | -0.78 pp | 11.87 % | -0.54 pp | 11.82 % | -0.59 pp |
| Registers/Thread | 214.00 register/thread | 255.00 register/thread | +41 | 255.00 register/thread | +41 | 255.00 register/thread | +41 | 255.00 register/thread | +41 |
| Waves/SM | 3.71 | 1.86 | -1.85 | 1.86 | -1.85 | 1.86 | -1.85 | 1.86 | -1.85 |
| No Eligible (%) | 72.93 % | 68.18 % | -4.75 pp | 69.14 % | -3.79 pp | 68.50 % | -4.43 pp | 68.07 % | -4.86 pp |
| IPC Active | 1.08 inst/cycle | 1.25 inst/cycle | +0.17 | 1.23 inst/cycle | +0.15 | 1.24 inst/cycle | +0.16 | 1.27 inst/cycle | +0.19 |


### Key Learnings

#### ❌ Plan A: `fix_uncoalesced_b_matrix_load_and_shared_memory_bank_conflicts` (1.168×)

Replacing a broadcast-based B-matrix load (tl.broadcast_to with strided index) with a direct coalesced load (rk + rn*stride) produced no measurable improvement: duration remained at 45.06 us, L1 hit rate was unchanged (17.56% vs 17.54%), bank conflict severity stayed at 5.2-way, and No Eligible scheduler cycles were essentially flat (67.95% vs 68.18%). The NCU MemoryCacheAccessPattern rule (1.0/32 bytes per sector for local loads/stores) persisted after the change, indicating the uncoalesced access pattern is rooted in register spill traffic to local memory — not in the B-matrix global load itself. When the root cause of strided local memory access is register spilling (255 registers/thread, Block Limit Registers=4), rewriting the global load index does not address the actual bottleneck.

> **Applicable when**: Never: this rewrite did not improve any profiling metric. The MemoryCacheAccessPattern rule (1/32 bytes per sector for local loads/stores) is caused by register spill traffic, not by the global B-matrix load pattern.

> **Avoid when**: When MemoryCacheAccessPattern rule fires on local (not global) loads/stores — this indicates register spilling, not a fixable global memory access pattern. When registers_per_thread > 200 and Block Limit Registers is the binding occupancy constraint, rewriting global load indices will not reduce spill traffic or improve scheduler eligibility.

#### ✅ Plan B: `tma_loads_via_make_block_ptr` (1.113×)

Replacing manual scalar pointer arithmetic (index computation via xindex = idx_n + stride*idx_m, broadcast_to, etc.) with tl.make_block_ptr() + tl.load(boundary_check=()) in Triton did NOT reduce register pressure (registers/thread remained at 255) or improve occupancy (12.5% theoretical, Block Limit Registers still 4). Despite the 1.11x wall-clock speedup, the NCU profiling shows the kernel actually became slightly slower per-cycle (Duration increased from 45.06 to 47.26 us, Warp Cycles Per Issued Instruction rose from 5.99 to 6.21, No Eligible increased from 67.95% to 69.14%). The apparent speedup likely comes from improved L1/TEX hit rate (17.56% → 25.54%) and L2 hit rate (70.7% → 77.9%) due to more cache-friendly access patterns from block pointer loads, but the primary bottleneck (register-limited occupancy) was not addressed. The strategy of using make_block_ptr to reduce register pressure via TMA address offloading did not materialize on this kernel configuration.

> **Applicable when**: Triton GEMM kernel with manual scalar pointer arithmetic (xindex = idx_n + stride*idx_m patterns, broadcast_to calls) where MemoryCacheAccessPattern rule is triggered (uncoalesced local loads/stores with low bytes-per-sector utilization). L1/TEX hit rate is low (<20%) and L2 hit rate is moderate (60-75%). The kernel does NOT already hit the register ceiling from accumulator tiles alone.

> **Avoid when**: Register pressure is already at the hardware maximum (255 registers/thread) due to accumulator tile size (BLOCK_M x BLOCK_N fp32 values), not from pointer arithmetic. In this regime, make_block_ptr does not reduce register count because the accumulator dominates. Also avoid when the kernel has non-standard strided B-matrix layouts that require careful stride mapping. Do not expect occupancy improvements when Block Limit Registers is already constrained by tile size, not index computation.

#### ✅ Plan C: `add_cache_eviction_policies_streaming_A_evict_first_reused_B_evict_last` (1.200×)

When a GEMM kernel has a large streaming input (A-matrix, accessed once per tile, never reused across tiles) and a small weight matrix (B-matrix, reused across all M-dimension tiles), annotating A loads with eviction_policy='evict_first' and B loads with eviction_policy='evict_last' in Triton improves effective L2 residency for the reused data. In this case, elapsed cycles dropped from 67181 to 65341 (~2.7%) and L1/TEX hit rate improved slightly (17.56% → 18.74%), yielding a 1.2x speedup. Write-once output stores should also use evict_first to avoid polluting L2 with data that will never be re-read by the kernel.

> **Applicable when**: Triton GEMM kernel where: (1) one input matrix is large and streaming (each tile reads a unique slice, no cross-tile reuse), (2) another input is small and reused across many tiles (fits in L2), (3) MemoryCacheAccessPattern NCU rule is triggered, (4) L2 hit rate is moderate (60-80%) suggesting cache pressure from competing access streams, (5) output is write-once. Particularly effective when the reused matrix is much smaller than L2 capacity.

> **Avoid when**: Both matrices are large and streaming (no reuse benefit from evict_last). Kernel is already compute-bound (SM Throughput > 70%) — cache hints have negligible effect. The reused matrix is larger than L2 capacity (evict_last cannot pin it). Register pressure is the dominant bottleneck and cache efficiency is not the limiting factor (speedup will be marginal, as seen here where occupancy remained at 12.5%).

---

## Iteration 3

**Best so far**: 1.1676× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_double_broadcast_load_eliminate_local_memory_spill | 1.175× | 1.177× | ✓ |  |
| D | cap_registers_with_maxnreg_to_improve_occupancy | 0.985× | 0.552× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_doub... | Δ | cap_registers_wit... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 52610 ns | 44700 ns | -7910 ns (-15.0%) | 95300 ns | +42690 ns (+81.1%) |
| SM Throughput (%) | 23.66 % | 27.45 % | +3.79 pp | 20.91 % | -2.75 pp |
| Memory Throughput (%) | 59.83 % | 47.83 % | -12.00 pp | 47.84 % | -11.99 pp |
| DRAM Throughput (%) | 8.81 % | 11.21 % | +2.40 pp | 37.79 % | +28.98 pp |
| Achieved Occupancy (%) | 12.41 % | 11.87 % | -0.54 pp | 22.49 % | +10.08 pp |
| Registers/Thread | 214.00 register/thread | 255.00 register/thread | +41 | 128.00 register/thread | -86 |
| Waves/SM | 3.71 | 1.86 | -1.85 | 0.93 | -2.78 |
| No Eligible (%) | 72.93 % | 68.34 % | -4.59 pp | 81.60 % | +8.67 pp |
| IPC Active | 1.08 inst/cycle | 1.23 inst/cycle | +0.15 | 0.74 inst/cycle | -0.34 |


### Key Learnings

#### ❌ Plan D: `cap_registers_with_maxnreg_to_improve_occupancy` (0.552×)

Capping registers per thread via maxnreg in triton.Config (255→128) successfully doubled theoretical occupancy (12.5%→25%) and Block Limit Registers (4→8 blocks/SM), but caused a catastrophic 1.81x slowdown (43.84 us→95.3 us). The CPIStall rule appeared post-change with 69.1% of stall cycles attributed to L1TEX scoreboard waits (13.5 cycles/warp average), and Warp Cycles Per Issued Instruction tripled from 6.11 to 19.58. The L1/TEX hit rate collapsed from 18.74% to 2.59% while DRAM throughput jumped from 11.29% to 37.79%, indicating that forced register spills to local memory created massive uncoalesced L1TEX traffic that overwhelmed any latency-hiding benefit from the higher occupancy. When the accumulator tile (BLOCK_M=128 × BLOCK_N=64 × 4 bytes = 32KB of fp32) dominates the register file, capping registers forces the entire accumulator to spill to local memory on every loop iteration, creating a catastrophic memory traffic amplification that far outweighs the occupancy gain.

> **Applicable when**: Never apply maxnreg when the accumulator tile alone (BLOCK_M × BLOCK_N × dtype_bytes) would exceed the capped register budget. Specifically: if BLOCK_M × BLOCK_N × 4 bytes > maxnreg × 32 threads_per_warp × 4 bytes, the accumulator must spill to local memory on every dot-product iteration.

> **Avoid when**: The kernel has a large accumulator tile (BLOCK_M × BLOCK_N fp32 values) that fills most of the register file. In this case (BLOCK_M=128, BLOCK_N=64, 8192 fp32 accumulators), capping to maxnreg=128 forces the entire accumulator to spill to local memory, causing Warp Cycles Per Issued Instruction to triple (6→19), L1/TEX hit rate to collapse (18%→2.6%), DRAM throughput to spike (11%→38%), and No Eligible scheduler cycles to worsen (68%→81%) despite doubled occupancy. The spill traffic cost dominates any latency-hiding benefit from higher occupancy.

---

## Cumulative Progress

```
Baseline:  52610 ns  (1.000×)
Iter 1     45060 ns  (1.168×)  — add_autotune_reduce_register_pressure_improve_occupancy

Final best: 45060 ns  (1.1676×)
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
+        triton.Config({}, num_warps=2, num_stages=1),
+        triton.Config({}, num_warps=2, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=1),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=1),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=3),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_3(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -58,23 +72,21 @@

     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
+
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
         b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 32*idx_m
-        a = tl.load(A + (xindex))
+        a = tl.load(A + (xindex), eviction_policy='evict_first')
 
         idx_m = b_k_idx_vals
         idx_n = offs_b_n[None, :]
         xindex = idx_n + 64*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 32*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        b = tl.load(B + ((tl.broadcast_to(idx_m + 32*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), eviction_policy='evict_last')
 
-        
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -85,9 +97,7 @@

 
     # inductor generates a suffix
     xindex = idx_n + 64*idx_m
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
