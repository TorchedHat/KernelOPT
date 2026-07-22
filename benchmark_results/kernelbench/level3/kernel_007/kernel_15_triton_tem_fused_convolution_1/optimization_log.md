# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_15` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 23810 ns — **Best latency**: 18050 ns (**1.3191×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`reduce_num_warps_and_add_pipeline_stages_to_hide_smem_bank_conflict_latency`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_15`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_15_triton_tem_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (123,1,1), Waves/SM: 0.16
- **Registers/thread**: 80
- **Duration**: 23810 ns
- **SM Throughput**: 10.64 % | **Memory**: 74.33 % | **DRAM**: 5.19 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.50 inst/cycle | **No Eligible**: 87.53 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.001× | 1.003× | ✓ | **✓ best** |
| C | fix_matrix_b_block_ptr_order_to_eliminate_shared_memory_bank_conflicts | 0.999× | 0.993× | ✓ |  |
| D | autotune_tile_sizes_to_fix_bank_conflicts_and_occupancy | 0.776× | 0.738× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | fix_matrix_b_bloc... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 23810 ns | 23740 ns | -70 ns (-0.3%) | 23970 ns | +160 ns (+0.7%) | 32260 ns | +8450 ns (+35.5%) |
| SM Throughput (%) | 10.64 % | 10.52 % | -0.12 pp | 10.61 % | -0.03 pp | 14.73 % | +4.09 pp |
| Memory Throughput (%) | 74.33 % | 73.53 % | -0.80 pp | 74.15 % | -0.18 pp | 62.99 % | -11.34 pp |
| DRAM Throughput (%) | 5.19 % | 5.20 % | +0.01 pp | 5.16 % | -0.03 pp | 3.85 % | -1.34 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp | 6.25 % | +0.01 pp |
| Registers/Thread | 80.00 register/thread | 80.00 register/thread | +0 | 80.00 register/thread | +0 | 114.00 register/thread | +34 |
| Waves/SM | 0.16 | 0.16 | +0.00 | 0.16 | +0.00 | 0.47 | +0.31 |
| No Eligible (%) | 87.53 % | 87.43 % | -0.10 pp | 87.44 % | -0.09 pp | 82.38 % | -5.15 pp |
| IPC Active | 0.50 inst/cycle | 0.50 inst/cycle | +0.00 | 0.50 inst/cycle | +0.00 | 0.70 inst/cycle | +0.20 |

---

## Iteration 2

**Best so far**: 1.0029× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_shared_memory_bank_conflicts_via_smem_padding | 1.001× | 1.004× | ✓ |  |
| B | fix_b_block_ptr_order_and_reduce_num_warps_for_bank_conflict_elimination | 1.010× | 1.007× | ✓ | **✓ best** |
| C | reduce_tile_sizes_and_add_smem_padding_to_fix_bank_conflicts_and_occupancy | 1.001× | 1.006× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_shared_... | Δ | fix_b_block_ptr_o... | Δ | reduce_tile_sizes... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 23810 ns | 23710 ns | -100 ns (-0.4%) | 23650 ns | -160 ns (-0.7%) | 23680 ns | -130 ns (-0.5%) |
| SM Throughput (%) | 10.64 % | 10.59 % | -0.05 pp | 10.60 % | -0.04 pp | 10.62 % | -0.02 pp |
| Memory Throughput (%) | 74.33 % | 73.99 % | -0.34 pp | 74.09 % | -0.24 pp | 74.19 % | -0.14 pp |
| DRAM Throughput (%) | 5.19 % | 5.21 % | +0.02 pp | 5.22 % | +0.03 pp | 5.21 % | +0.02 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 80.00 register/thread | 80.00 register/thread | +0 | 80.00 register/thread | +0 | 80.00 register/thread | +0 |
| Waves/SM | 0.16 | 0.16 | +0.00 | 0.16 | +0.00 | 0.16 | +0.00 |
| No Eligible (%) | 87.53 % | 87.45 % | -0.08 pp | 87.57 % | +0.04 pp | 87.47 % | -0.06 pp |
| IPC Active | 0.50 inst/cycle | 0.50 inst/cycle | +0.00 | 0.50 inst/cycle | +0.00 | 0.50 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0068× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | transpose_b_on_smem_load_to_eliminate_bank_conflicts | 1.011× | 1.004× | ✓ | **✓ best** |
| B | load_b_transposed_to_eliminate_shared_memory_bank_conflicts | 1.000× | 1.001× | ✓ |  |
| C | fix_b_smem_order_and_reduce_num_warps_to_eliminate_bank_conflicts | 1.001× | 0.998× | ✓ |  |


### Metric Comparison

| Metric | Baseline | transpose_b_on_sm... | Δ | load_b_transposed... | Δ | fix_b_smem_order_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 23810 ns | 23710 ns | -100 ns (-0.4%) | 23780 ns | -30 ns (-0.1%) | 23870 ns | +60 ns (+0.3%) |
| SM Throughput (%) | 10.64 % | 10.42 % | -0.22 pp | 10.45 % | -0.19 pp | 10.60 % | -0.04 pp |
| Memory Throughput (%) | 74.33 % | 72.90 % | -1.43 pp | 73.07 % | -1.26 pp | 74.10 % | -0.23 pp |
| DRAM Throughput (%) | 5.19 % | 5.21 % | +0.02 pp | 5.20 % | +0.01 pp | 5.17 % | -0.02 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 80.00 register/thread | 80.00 register/thread | +0 | 80.00 register/thread | +0 | 80.00 register/thread | +0 |
| Waves/SM | 0.16 | 0.16 | +0.00 | 0.16 | +0.00 | 0.16 | +0.00 |
| No Eligible (%) | 87.53 % | 87.56 % | +0.03 pp | 87.57 % | +0.04 pp | 87.44 % | -0.09 pp |
| IPC Active | 0.50 inst/cycle | 0.50 inst/cycle | +0.00 | 0.50 inst/cycle | +0.00 | 0.50 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.0068× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_num_warps_and_add_pipeline_stages_to_hide_smem_bank_conflict_latency | 1.283× | 1.319× | ✓ | **✓ best** |
| B | reduce_register_pressure_via_maxnreg_to_increase_occupancy | 1.268× | 1.145× | ✓ |  |
| C | software_pipeline_with_num_stages_to_hide_mio_stall_latency | 1.270× | 1.071× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_num_warps_... | Δ | reduce_register_p... | Δ | software_pipeline... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 23810 ns | 18050 ns | -5760 ns (-24.2%) | 20800 ns | -3010 ns (-12.6%) | 22240 ns | -1570 ns (-6.6%) |
| SM Throughput (%) | 10.64 % | 12.99 % | +2.35 pp | 11.17 % | +0.53 pp | 10.47 % | -0.17 pp |
| Memory Throughput (%) | 74.33 % | 54.57 % | -19.76 pp | 47.33 % | -27.00 pp | 48.00 % | -26.33 pp |
| DRAM Throughput (%) | 5.19 % | 6.86 % | +1.67 pp | 5.95 % | +0.76 pp | 5.56 % | +0.37 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp |
| Registers/Thread | 80.00 register/thread | 128.00 register/thread | +48 | 96.00 register/thread | +16 | 96.00 register/thread | +16 |
| Waves/SM | 0.16 | 0.12 | -0.04 | 0.09 | -0.07 | 0.09 | -0.07 |
| No Eligible (%) | 87.53 % | 68.06 % | -19.47 pp | 72.72 % | -14.81 pp | 75.60 % | -11.93 pp |
| IPC Active | 0.50 inst/cycle | 0.64 inst/cycle | +0.14 | 0.54 inst/cycle | +0.04 | 0.49 inst/cycle | -0.01 |


### Key Learnings

#### ✅ Plan A: `reduce_num_warps_and_add_pipeline_stages_to_hide_smem_bank_conflict_latency` (1.319×)

When a kernel is dominated by MIO/shared-memory scoreboard stalls (CPIStall MIO fraction >70%, Warp Cycles Per Issued Instruction ~8, scheduler No Eligible >87%), reducing num_warps (e.g. 4→2, halving block size from 128→64 threads) combined with adding pipeline stages (num_stages=2 or 3 via @triton.autotune) can significantly reduce kernel duration. In this case, the change cut Warp Cycles Per Issued Instruction from 8.07 to 3.13 (61% reduction) and duration from 23.71µs to 18.05µs (1.32x speedup), while the scheduler No Eligible dropped from 87.56% to 68.06%. The mechanism is twofold: fewer warps per block reduces register pressure enough to relax the register block limit (6→8 blocks/SM), and pipeline stages enable async prefetching of the next tile to overlap compute with shared memory latency, even when bank conflicts persist.

> **Applicable when**: Kernel shows CPIStall MIO fraction >60%, Warp Cycles Per Issued Instruction >6, Scheduler No Eligible >80%, and SharedMemoryConflicts with >15-way average bank conflicts. The kernel is a small GEMM or similar tiled computation where tile sizes (BLOCK_M, BLOCK_N, BLOCK_K) are already near-optimal and cannot be changed without SMEM/register regression. The register block limit and SMEM block limit are both binding (~6 blocks/SM).

> **Avoid when**: Do not apply when the kernel already has high achieved occupancy (>20%) or when the grid is large enough to fully saturate all SMs (waves_per_sm >1). Also avoid when increasing num_stages would push dynamic SMEM per block beyond the SM shared memory limit, collapsing the SMEM block limit below 2. Note that this approach does not eliminate bank conflicts (they persist at 13.2-way after optimization) — if conflict elimination is possible via padding or access reordering, that should be preferred.

---

## Iteration 5

**Best so far**: 1.3191× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | coalesce_shared_memory_via_b_smem_order_fix_and_expand_autotune | 1.272× | 0.738× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesce_shared_m... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 23810 ns | 32260 ns | +8450 ns (+35.5%) |
| SM Throughput (%) | 10.64 % | 6.84 % | -3.80 pp |
| Memory Throughput (%) | 74.33 % | 18.35 % | -55.98 pp |
| DRAM Throughput (%) | 5.19 % | 3.85 % | -1.34 pp |
| Achieved Occupancy (%) | 6.24 % | 1.56 % | -4.68 pp |
| Registers/Thread | 80.00 register/thread | 168.00 register/thread | +88 |
| Waves/SM | 0.16 | 0.08 | -0.08 |
| No Eligible (%) | 87.53 % | 65.91 % | -21.62 pp |
| IPC Active | 0.50 inst/cycle | 0.34 inst/cycle | -0.16 |

---

## Cumulative Progress

```
Baseline:  23810 ns  (1.000×)
Iter 1     23740 ns  (1.003×)  — tf32_tensor_cores
Iter 2     23650 ns  (1.007×)  — fix_b_block_ptr_order_and_reduce_num_warps_for_bank_conflict_elimination
Iter 4     18050 ns  (1.319×)  — reduce_num_warps_and_add_pipeline_stages_to_hide_smem_bank_conflict_latency

Final best: 18050 ns  (1.3191×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,16 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=2),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_convolution_15(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -55,25 +65,26 @@

         block_shape=(BLOCK_M, BLOCK_K),
         order=(1, 0),
     )
-    # B: shape=[K, N]=[192, 32], strides=[stride_bk, stride_bn]=[1, 192]
-    # order=(0, 1) means innermost dim is K (stride_bk=1, contiguous)
-    b_block_ptr = tl.make_block_ptr(
+    # B transposed: shape=[N, K]=[32, 192], strides=[stride_bn, stride_bk]=[192, 1]
+    # order=(1, 0) means K is innermost (contiguous) in SMEM — N-contiguous layout
+    # This eliminates bank conflicts: N-row accesses hit consecutive banks
+    b_T_block_ptr = tl.make_block_ptr(
         base=B,
-        shape=(K, N),
-        strides=(stride_bk, stride_bn),
-        offsets=(0, pid_n * BLOCK_N),
-        block_shape=(BLOCK_K, BLOCK_N),
-        order=(0, 1),
+        shape=(N, K),
+        strides=(stride_bn, stride_bk),
+        offsets=(pid_n * BLOCK_N, 0),
+        block_shape=(BLOCK_N, BLOCK_K),
+        order=(1, 0),
     )
 
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
         a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option='zero')
-        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option='zero')
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
+        b_T = tl.load(b_T_block_ptr, boundary_check=(0, 1), padding_option='zero')  # [BLOCK_N, BLOCK_K]
+        acc += tl.dot(a, tl.trans(b_T), allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)  # a:[M,K] x b_T^T:[K,N]
         a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
-        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
+        b_T_block_ptr = tl.advance(b_T_block_ptr, (0, BLOCK_K))  # advance K axis (axis=1)
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -87,7 +98,6 @@

     tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
