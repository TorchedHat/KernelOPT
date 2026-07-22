# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 37120 ns — **Best latency**: 37630 ns (**0.9864×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_11`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_11_triton_tem_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (123,1,1), Waves/SM: 0.31
- **Registers/thread**: 168
- **Duration**: 37120 ns
- **SM Throughput**: 24.19 % | **Memory**: 53.75 % | **DRAM**: 3.37 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 1.18 inst/cycle | **No Eligible**: 70.37 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.005× | 1.000× | ✓ |  |
| B | autotune_tile_sizes_and_pipeline_stages_to_reduce_register_pressure | 1.010× | 0.992× | ✓ | **✓ best** |
| C | fix_b_matrix_load_coalescing_eliminate_bank_conflicts | 1.005× | 0.994× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | autotune_tile_siz... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 37120 ns | 37120 ns | +0 ns (+0.0%) | 37440 ns | +320 ns (+0.9%) | 37340 ns | +220 ns (+0.6%) |
| SM Throughput (%) | 24.19 % | 24.24 % | +0.05 pp | 24.11 % | -0.08 pp | 24.07 % | -0.12 pp |
| Memory Throughput (%) | 53.75 % | 53.86 % | +0.11 pp | 53.86 % | +0.11 pp | 53.56 % | -0.19 pp |
| DRAM Throughput (%) | 3.37 % | 3.37 % | +0.00 pp | 3.34 % | -0.03 pp | 3.35 % | -0.02 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 0.31 | 0.31 | +0.00 | 0.31 | +0.00 | 0.31 | +0.00 |
| No Eligible (%) | 70.37 % | 70.25 % | -0.12 pp | 70.45 % | +0.08 pp | 70.49 % | +0.12 pp |
| IPC Active | 1.18 inst/cycle | 1.17 inst/cycle | -0.01 | 1.18 inst/cycle | +0.00 | 1.18 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_shared_memory_bank_conflicts_via_block_ptr_coalesced_b_load | 1.008× | 1.004× | ✓ |  |
| B | cap_registers_with_maxnreg_to_improve_occupancy_and_hide_mio_latency | 0.825× | 0.797× | ✓ |  |
| C | fix_b_matrix_load_indexing_to_eliminate_shared_memory_bank_conflicts | 1.011× | 0.986× | ✓ | **✓ best** |
| D | add_triton_autotune | 1.009× | 0.929× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_shared_... | Δ | cap_registers_wit... | Δ | fix_b_matrix_load... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 37120 ns | 36960 ns | -160 ns (-0.4%) | 46560 ns | +9440 ns (+25.4%) | 37630 ns | +510 ns (+1.4%) | 39940 ns | +2820 ns (+7.6%) |
| SM Throughput (%) | 24.19 % | 23.94 % | -0.25 pp | 19.55 % | -4.64 pp | 23.95 % | -0.24 pp | 22.51 % | -1.68 pp |
| Memory Throughput (%) | 53.75 % | 54.17 % | +0.42 pp | 75.36 % | +21.61 pp | 53.51 % | -0.24 pp | 50.33 % | -3.42 pp |
| DRAM Throughput (%) | 3.37 % | 3.41 % | +0.04 pp | 2.68 % | -0.69 pp | 3.33 % | -0.04 pp | 3.13 % | -0.24 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 12.48 % | +6.23 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 128.00 register/thread | -40 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 0.31 | 0.31 | +0.00 | 0.47 | +0.16 | 0.31 | +0.00 | 0.31 | +0.00 |
| No Eligible (%) | 70.37 % | 70.64 % | +0.27 pp | 77.94 % | +7.57 pp | 70.48 % | +0.11 pp | 72.82 % | +2.45 pp |
| IPC Active | 1.18 inst/cycle | 1.17 inst/cycle | -0.01 | 0.88 inst/cycle | -0.30 | 1.18 inst/cycle | +0.00 | 1.08 inst/cycle | -0.10 |


### Key Learnings

#### ❌ Plan B: `cap_registers_with_maxnreg_to_improve_occupancy_and_hide_mio_latency` (0.797×)

Capping registers with maxnreg (168→128 regs/thread) doubled achieved occupancy (6.25%→12.48%) and increased waves/SM (0.31→0.47), but caused a 25% regression in wall time (37.12→46.56 µs). The root cause is that the larger block size (128→256 threads) needed to exploit the freed register budget amplified shared memory bank conflicts from 7.9-way to 11.9-way, increasing total bank conflict wavefronts from 2.6M to 5.3M. Warp cycles per issued instruction nearly tripled (3.38→9.01), and the scheduler idle fraction rose from 70.25% to 77.94%, meaning the extra warps from higher occupancy could not hide the worsened MIO stalls — the bank conflict bottleneck dominated over the occupancy gain.

> **Applicable when**: Never: this specific combination of maxnreg capping + larger num_warps on a kernel with severe shared memory bank conflicts (>7-way) causes regression. The bank conflict penalty scales with thread count, overwhelming any occupancy benefit.

> **Avoid when**: When the kernel already has significant shared memory bank conflicts (>5-way). Increasing block size (num_warps) to exploit freed registers amplifies bank conflict wavefronts proportionally to the thread count increase, causing MIO stall cycles to grow faster than the occupancy gain can hide them. Warp cycles per issued instruction is the key signal: if it rises after the change, the strategy is counterproductive.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | eliminate_tl_trans_by_loading_b_in_k_n_layout_to_remove_bank_conflicts | 1.002× | 0.988× | ✓ |  |
| C | transpose_b_tile_load_to_eliminate_shared_memory_bank_conflicts | 0.816× | 0.795× | ✓ |  |
| D | fix_b_matrix_load_via_direct_2d_pointer_arithmetic_to_eliminate_bank_conflicts | 1.000× | 0.925× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_tl_tran... | Δ | transpose_b_tile_... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 37120 ns | 37570 ns | +450 ns (+1.2%) | 46660 ns | +9540 ns (+25.7%) | 40130 ns | +3010 ns (+8.1%) |
| SM Throughput (%) | 24.19 % | 24.07 % | -0.12 pp | 19.58 % | -4.61 pp | 22.55 % | -1.64 pp |
| Memory Throughput (%) | 53.75 % | 53.77 % | +0.02 pp | 75.52 % | +21.77 pp | 50.40 % | -3.35 pp |
| DRAM Throughput (%) | 3.37 % | 3.33 % | -0.04 pp | 2.67 % | -0.70 pp | 3.12 % | -0.25 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 12.48 % | +6.23 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 128.00 register/thread | -40 | 168.00 register/thread | +0 |
| Waves/SM | 0.31 | 0.31 | +0.00 | 0.47 | +0.16 | 0.31 | +0.00 |
| No Eligible (%) | 70.37 % | 70.32 % | -0.05 pp | 77.91 % | +7.54 pp | 72.87 % | +2.50 pp |
| IPC Active | 1.18 inst/cycle | 1.19 inst/cycle | +0.01 | 0.88 inst/cycle | -0.30 | 1.08 inst/cycle | -0.10 |


### Key Learnings

#### ❌ Plan C: `transpose_b_tile_load_to_eliminate_shared_memory_bank_conflicts` (0.795×)

Switching from a [BLOCK_K, BLOCK_N] B-tile load to a [BLOCK_N, BLOCK_K] load with tl.trans(b) in tl.dot was intended to eliminate 11.9-way shared memory bank conflicts by making threads access consecutive K-stride-1 elements instead of N-stride-192 elements. However, the profiling metrics after the change are identical to before (bank conflicts: 5.29M, conflict-way: 11.9, duration: 46.66 µs), indicating the transposed load pattern did not change the shared memory access pattern as expected — likely because the data still passes through shared memory in the same conflicting layout when Triton stages the tl.trans operand. The speedup of 0.7955x reflects the regression introduced by the accompanying autotune overhead and USE_FAST_ACCUM=False change relative to the original baseline (37 µs), not a change from the immediately prior iteration.

> **Applicable when**: Never: transposing the B-tile load pointer arithmetic while using tl.trans() in tl.dot does not eliminate shared memory bank conflicts in Triton when the underlying data layout causes the same conflict pattern after staging through shared memory.

> **Avoid when**: When shared memory bank conflicts are caused by the layout of data staged through Triton's shared memory pipeline. Changing the pointer arithmetic for the global load and using tl.trans() does not change how Triton lays out the tile in shared memory — the conflict pattern is determined by how the compiler stages the tile, not by the global pointer stride. This approach failed to reduce bank conflicts from 11.9-way even though the global memory access pattern was corrected.

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | pad_b_smem_tile_to_eliminate_bank_conflicts | 0.989× | 0.997× | ✓ |  |
| C | eliminate_smem_bank_conflicts_via_explicit_block_ptr_with_padding | 0.844× | 0.783× | ✓ |  |
| D | add_triton_autotune | 1.009× | 0.939× | ✓ |  |


### Metric Comparison

| Metric | Baseline | pad_b_smem_tile_t... | Δ | eliminate_smem_ba... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 37120 ns | 37220 ns | +100 ns (+0.3%) | 47420 ns | +10300 ns (+27.7%) | 39550 ns | +2430 ns (+6.5%) |
| SM Throughput (%) | 24.19 % | 23.82 % | -0.37 pp | 19.40 % | -4.79 pp | 22.35 % | -1.84 pp |
| Memory Throughput (%) | 53.75 % | 54.06 % | +0.31 pp | 74.67 % | +20.92 pp | 49.97 % | -3.78 pp |
| DRAM Throughput (%) | 3.37 % | 3.36 % | -0.01 pp | 2.63 % | -0.74 pp | 3.16 % | -0.21 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 12.47 % | +6.22 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 167.00 register/thread | -1 | 128.00 register/thread | -40 | 168.00 register/thread | +0 |
| Waves/SM | 0.31 | 0.31 | +0.00 | 0.47 | +0.16 | 0.31 | +0.00 |
| No Eligible (%) | 70.37 % | 70.43 % | +0.06 pp | 77.97 % | +7.60 pp | 73.00 % | +2.63 pp |
| IPC Active | 1.18 inst/cycle | 1.17 inst/cycle | -0.01 | 0.87 inst/cycle | -0.31 | 1.08 inst/cycle | -0.10 |

---

## Cumulative Progress

```
Baseline:  37120 ns  (1.000×)

Final best: 37630 ns  (0.9864×)
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
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_convolution_11(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -58,23 +72,22 @@

     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
+
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 192*idx_m
         a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 96*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 192*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        # Load B directly as [BLOCK_K, BLOCK_N] using stride_bk=1, stride_bn=192
+        # B[n,k] = B_ptr + n*192 + k*1; load as [BLOCK_K, BLOCK_N] to avoid tl.trans
+        offs_k_cur = tl.arange(0, BLOCK_K) + (k_idx * BLOCK_K)
+        b = tl.load(B + (offs_k_cur[:, None] * 1 + offs_b_n[None, :] * 192))
 
-        
+        # tl.dot(a=[BLOCK_M,BLOCK_K], b=[BLOCK_K,BLOCK_N]) -> [BLOCK_M,BLOCK_N]
+        # No tl.trans needed — eliminates shared memory bank conflicts
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -88,8 +101,6 @@

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
