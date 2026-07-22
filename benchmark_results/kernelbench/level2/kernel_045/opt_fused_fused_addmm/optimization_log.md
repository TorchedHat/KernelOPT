# Kernel Optimization Log

## Summary

- **Kernel**: `gemm_sigmoid_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 14940000 ns — **Best latency**: 14970000 ns (**0.9980×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `gemm_sigmoid_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_045/synthesis/kernel_0_fused_fused_addmm.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4096,1,1), Waves/SM: 15.52
- **Registers/thread**: 255
- **Duration**: 14940000 ns
- **SM Throughput**: 44.17 % | **Memory**: 60.67 % | **DRAM**: 1.45 %
- **Achieved Occupancy**: 12.48 %
- **IPC Active**: 1.83 inst/cycle | **No Eligible**: 54.32 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | enable_tf32_tensor_cores_and_expand_autotune_configs | 0.998× | 1.002× | ✓ |  |
| B | enable_tf32_and_expand_autotune_configs | 1.001× | 0.998× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | enable_tf32_tenso... | Δ | enable_tf32_and_e... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 14940000 ns | 14910000 ns | -30000 ns (-0.2%) | 14970000 ns | +30000 ns (+0.2%) |
| SM Throughput (%) | 44.17 % | 44.59 % | +0.42 pp | 44.40 % | +0.23 pp |
| Memory Throughput (%) | 60.67 % | 61.25 % | +0.58 pp | 61.02 % | +0.35 pp |
| DRAM Throughput (%) | 1.45 % | 1.47 % | +0.02 pp | 1.47 % | +0.02 pp |
| Achieved Occupancy (%) | 12.48 % | 12.47 % | -0.01 pp | 12.48 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 54.32 % | 54.24 % | -0.08 pp | 54.31 % | -0.01 pp |
| IPC Active | 1.83 inst/cycle | 1.82 inst/cycle | -0.01 | 1.83 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | cap_registers_with_maxnreg_to_improve_occupancy | 0.869× | 0.369× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_registers_wit... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 14940000 ns | 40480000 ns | +25540000 ns (+171.0%) |
| SM Throughput (%) | 44.17 % | 28.67 % | -15.50 pp |
| Memory Throughput (%) | 60.67 % | 96.24 % | +35.57 pp |
| DRAM Throughput (%) | 1.45 % | 2.87 % | +1.42 pp |
| Achieved Occupancy (%) | 12.48 % | 24.88 % | +12.40 pp |
| Registers/Thread | 255.00 register/thread | 64.00 register/thread | -191 |
| Waves/SM | 15.52 | 31.03 | +15.51 |
| No Eligible (%) | 54.32 % | 77.53 % | +23.21 pp |
| IPC Active | 1.83 inst/cycle | 0.90 inst/cycle | -0.93 |


### Key Learnings

#### ❌ Plan C: `cap_registers_with_maxnreg_to_improve_occupancy` (0.369×)

Adding maxnreg to Triton autotune configs reduced registers/thread from 255 to 64, doubling theoretical occupancy (12.5%→25%) and active warps/scheduler (1.99→3.98), but the kernel slowed by 2.7x. The root cause is that register capping forced the compiler to spill more aggressively to shared memory, causing shared memory bank conflicts to worsen from 5.4-way to 12.5-way (bank conflict speedup potential grew from 36.6% to 76.3%), and warp cycles per issued instruction exploded from 4.36 to 17.72 cycles — indicating severe L1TEX stall from shared memory contention. Higher occupancy did not help because the bottleneck shifted from occupancy to shared memory bank conflicts: more concurrent warps amplified the conflict rate, and the no-eligible-warp percentage actually increased from 54% to 77.5%.

> **Applicable when**: Never beneficial in this pattern: when a GEMM kernel already has severe shared memory bank conflicts (>5x average conflict, >36% speedup potential flagged by NCU SharedMemoryConflicts), capping registers via maxnreg will worsen bank conflicts further by forcing more spill into shared memory and increasing concurrent warp pressure on the same shared memory banks.

> **Avoid when**: Do NOT apply maxnreg register capping to Triton GEMM kernels when: (1) shared memory bank conflicts are already present (NCU SharedMemoryConflicts rule triggered with >5x average conflict), because more occupancy amplifies bank conflict rate; (2) the kernel's primary bottleneck is shared memory access pattern rather than pure occupancy — warp_cycles_per_issued_instruction will increase dramatically (4.36→17.72 observed); (3) the register pressure is caused by large tile accumulation buffers that cannot be spilled without creating strided shared memory access patterns.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | enable_tf32_and_remove_maxnreg_from_autotune_configs | 1.000× | 0.585× | ✓ |  |
| B | enable_tf32_and_remove_maxnreg_to_fix_bank_conflicts | 0.999× | 0.585× | ✓ |  |
| C | fix_b_matrix_load_layout_for_coalesced_access_and_bank_conflict_elimination | 0.870× | 0.368× | ✓ |  |
| D | fix_b_matrix_load_layout_to_eliminate_shared_memory_bank_conflicts | 0.871× | 0.369× | ✓ |  |


### Metric Comparison

| Metric | Baseline | enable_tf32_and_r... | Δ | enable_tf32_and_r... | Δ | fix_b_matrix_load... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 14940000 ns | 25530000 ns | +10590000 ns (+70.9%) | 25530000 ns | +10590000 ns (+70.9%) | 40570000 ns | +25630000 ns (+171.6%) | 40490000 ns | +25550000 ns (+171.0%) |
| SM Throughput (%) | 44.17 % | 25.19 % | -18.98 pp | 25.18 % | -18.99 pp | 28.61 % | -15.56 pp | 28.65 % | -15.52 pp |
| Memory Throughput (%) | 60.67 % | 96.42 % | +35.75 pp | 96.38 % | +35.71 pp | 96.03 % | +35.36 pp | 96.18 % | +35.51 pp |
| DRAM Throughput (%) | 1.45 % | 1.17 % | -0.28 pp | 1.17 % | -0.28 pp | 2.99 % | +1.54 pp | 3.00 % | +1.55 pp |
| Achieved Occupancy (%) | 12.48 % | 18.67 % | +6.19 pp | 18.67 % | +6.19 pp | 24.88 % | +12.40 pp | 24.88 % | +12.40 pp |
| Registers/Thread | 255.00 register/thread | 162.00 register/thread | -93 | 162.00 register/thread | -93 | 64.00 register/thread | -191 | 64.00 register/thread | -191 |
| Waves/SM | 15.52 | 41.37 | +25.85 | 41.37 | +25.85 | 31.03 | +15.51 | 31.03 | +15.51 |
| No Eligible (%) | 54.32 % | 74.63 % | +20.31 pp | 74.63 % | +20.31 pp | 77.53 % | +23.21 pp | 77.53 % | +23.21 pp |
| IPC Active | 1.83 inst/cycle | 1.01 inst/cycle | -0.82 | 1.01 inst/cycle | -0.82 | 0.90 inst/cycle | -0.93 | 0.90 inst/cycle | -0.93 |


### Key Learnings

#### ❌ Plan C: `fix_b_matrix_load_layout_for_coalesced_access_and_bank_conflict_elimination` (0.368×)

Transposing the B matrix tile load from (BLOCK_K, BLOCK_N) to (BLOCK_N, BLOCK_K) with tl.trans() in tl.dot produced zero measurable improvement: all metrics (L1/TEX throughput 97%, shared memory bank conflicts 12.5-way, warp cycles/instruction 17.72, duration 40.48ms vs 40.57ms) remained statistically identical before and after. The NCU-reported bank conflict count (3,961,569,478 vs 3,961,580,355) and uncoalesced wavefronts (3,758,096,384 both) are unchanged, indicating the layout swap did not affect the actual shared memory access pattern driving the bottleneck. When the kernel is already in a regime where shared memory bank conflicts are the dominant bottleneck (12.5-way average, 76% of wavefronts excessive) and the root cause is the tile accumulator spill pattern rather than the B-load indexing order, reordering the B load dimensions has no effect on the conflict source.

> **Applicable when**: Never: this direction produced no improvement despite strong NCU evidence of bank conflicts and uncoalesced accesses.

> **Avoid when**: Do NOT apply B-matrix load layout transposition (offs_n[:, None] + k_offs[None, :] with tl.trans) when: (1) the kernel is already in a shared-memory-bank-conflict-dominated regime (12.5-way average, 76%+ excessive wavefronts) caused by accumulator spill rather than B-load indexing; (2) all profiling metrics remain identical after the change, indicating the bank conflict source is elsewhere (e.g., register spill to shared memory from large tile accumulators); (3) the kernel uses allow_tf32=False with large tile sizes that force accumulator spill — the spill pattern, not the B load order, is the conflict source.

#### ❌ Plan D: `fix_b_matrix_load_layout_to_eliminate_shared_memory_bank_conflicts` (0.369×)

Swapping the B matrix tile load from (BLOCK_K, BLOCK_N) to (BLOCK_N, BLOCK_K) and using tl.trans(b) in tl.dot produced zero measurable improvement: duration stayed at ~40.49ms, L1/TEX throughput remained at 97%, warp cycles per issued instruction stayed at 17.72, and the SharedMemoryConflicts rule still reported 12.5-way average bank conflicts with 78.64% conflict wavefronts. The bank conflict root cause in this kernel is not the B-tile load layout but something else — likely the A-tile access pattern or the shared memory staging done internally by Triton's tl.dot implementation, which the user cannot directly control via pointer arithmetic changes. Changing the pointer indexing order for B and adding tl.trans() is a no-op from the profiler's perspective when Triton's compiler already handles tile transposition internally.

> **Applicable when**: Never: this specific change (swap B tile indexing + tl.trans) does not reduce shared memory bank conflicts in Triton GEMM kernels when the conflicts are already present at 12.5-way severity.

> **Avoid when**: Do NOT apply B-matrix load layout transposition (offs_n[:, None] + k_offs[None, :] with tl.trans) as a fix for SharedMemoryConflicts in Triton GEMM kernels. The profiling data shows zero change in any metric after this transformation: duration, L1/TEX throughput, warp cycles per instruction, bank conflict rate, and occupancy are all identical before and after. Triton's compiler appears to handle the tile layout internally regardless of the pointer arithmetic order, making this a no-op optimization. The 12.5-way bank conflicts and 78.64% conflict wavefronts persist unchanged.

---

## Cumulative Progress

```
Baseline:  14940000 ns  (1.000×)

Final best: 14970000 ns  (0.9980×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,14 +3,20 @@

 import triton.language as tl
 from triton.language.extra import libdevice
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        # Small tiles with aggressive register cap for high occupancy
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4, maxnreg=64),
+        # Medium tiles with moderate register cap
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4, maxnreg=128),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4, maxnreg=128),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4, maxnreg=128),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8, maxnreg=128),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8, maxnreg=128),
+        # Larger tiles with moderate register cap
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8, maxnreg=160),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8, maxnreg=160),
     ],
     key=['M', 'N', 'K'],
 )
@@ -47,12 +53,12 @@

         
         a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
         a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
-        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
+        a = tl.load(a_ptrs, mask=a_mask, other=0.0, eviction_policy='evict_first')
         
         # B is weight matrix (N, K), we access B[n, k] = B_ptr + n*stride_bn + k*stride_bk
         b_ptrs = B_ptr + offs_n[None, :] * stride_bn + k_offs[:, None] * stride_bk
         b_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
-        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
+        b = tl.load(b_ptrs, mask=b_mask, other=0.0, eviction_policy='evict_first')
         
         acc = tl.dot(a, b, acc, allow_tf32=False)
     
@@ -81,6 +87,7 @@

     key=['N'],
 )
 @triton.jit
+
 def logsumexp_kernel(
     input_ptr, output_ptr,
     M, N,
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
