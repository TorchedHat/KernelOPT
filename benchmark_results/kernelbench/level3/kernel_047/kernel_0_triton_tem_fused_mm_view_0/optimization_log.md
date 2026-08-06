# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_mm_view_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 247550 ns — **Best latency**: 249570 ns (**0.9919×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_mm_view_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_047/kernel_0_triton_tem_fused_mm_view_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1600,1,1), Waves/SM: 3.03
- **Registers/thread**: 128
- **Duration**: 247550 ns
- **SM Throughput**: 64.72 % | **Memory**: 86.15 % | **DRAM**: 36.33 %
- **Achieved Occupancy**: 23.45 %
- **IPC Active**: 2.80 inst/cycle | **No Eligible**: 30.00 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | add_autotune_with_vectorized_loads_and_reduced_register_pressure | 1.004× | 0.992× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | add_autotune_with... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 247550 ns | 249570 ns | +2020 ns (+0.8%) |
| SM Throughput (%) | 64.72 % | 64.88 % | +0.16 pp |
| Memory Throughput (%) | 86.15 % | 86.34 % | +0.19 pp |
| DRAM Throughput (%) | 36.33 % | 36.01 % | -0.32 pp |
| Achieved Occupancy (%) | 23.45 % | 23.45 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 |
| Waves/SM | 3.03 | 3.03 | +0.00 |
| No Eligible (%) | 30.00 % | 29.92 % | -0.08 pp |
| IPC Active | 2.80 inst/cycle | 2.80 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_shared_memory_bank_conflicts_via_block_k_increase | 0.988× | 0.869× | ✓ |  |
| B | cap_registers_with_maxnreg_to_improve_occupancy_without_tile_reshape | 0.783× | 0.328× | ✓ |  |
| C | increase_block_k_to_reduce_shared_memory_bank_conflicts | 0.990× | 0.865× | ✓ |  |
| D | fix_grid_and_reduce_register_pressure_via_dynamic_block_m_autotune | 0.988× | 0.871× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_shared_... | Δ | cap_registers_wit... | Δ | increase_block_k_... | Δ | fix_grid_and_redu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 247550 ns | 284740 ns | +37190 ns (+15.0%) | 755520 ns | +507970 ns (+205.2%) | 286080 ns | +38530 ns (+15.6%) | 284350 ns | +36800 ns (+14.9%) |
| SM Throughput (%) | 64.72 % | 55.30 % | -9.42 pp | 32.96 % | -31.76 pp | 56.17 % | -8.55 pp | 55.55 % | -9.17 pp |
| Memory Throughput (%) | 86.15 % | 74.59 % | -11.56 pp | 70.69 % | -15.46 pp | 75.75 % | -10.40 pp | 74.94 % | -11.21 pp |
| DRAM Throughput (%) | 36.33 % | 31.74 % | -4.59 pp | 37.07 % | +0.74 pp | 31.61 % | -4.72 pp | 31.82 % | -4.51 pp |
| Achieved Occupancy (%) | 23.45 % | 12.41 % | -11.04 pp | 28.73 % | +5.28 pp | 12.44 % | -11.01 pp | 12.42 % | -11.03 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 64.00 register/thread | -64 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 3.03 | 6.06 | +3.03 | 2.42 | -0.61 | 6.06 | +3.03 | 6.06 | +3.03 |
| No Eligible (%) | 30.00 % | 37.78 % | +7.78 pp | 64.88 % | +34.88 pp | 37.72 % | +7.72 pp | 37.61 % | +7.61 pp |
| IPC Active | 2.80 inst/cycle | 2.49 inst/cycle | -0.31 | 1.41 inst/cycle | -1.39 | 2.49 inst/cycle | -0.31 | 2.49 inst/cycle | -0.31 |


### Key Learnings

#### ❌ Plan B: `cap_registers_with_maxnreg_to_improve_occupancy_without_tile_reshape` (0.328×)

Capping registers via maxnreg=64 in Triton autotune configs reduced registers/thread from 128 to 64 and raised Block Limit Registers from 4 to 8, but theoretical occupancy only improved from 25% to 31.25% (still limited by shared memory). The forced register spills caused massive L1TEX local load/store traffic: executed instructions grew from 124M to 175M, Warp Cycles Per Issued Instruction jumped from 5.36 to 12.99, eligible warps per scheduler collapsed from 1.78 to 0.53, and kernel duration tripled from 249 µs to 755 µs. The MemoryCacheAccessPattern rule flagged 1.0/32 bytes utilized per sector for local loads/stores, confirming catastrophic spill traffic that overwhelmed any occupancy benefit.

> **Applicable when**: Never — this strategy caused a 3x regression. Do not apply maxnreg capping when the kernel already uses a large accumulator tile (e.g., BLOCK_M=128 × BLOCK_N=32 FP32 values) that genuinely requires high register counts. The spill traffic from forced register eviction will always dominate any occupancy gain in such cases.

> **Avoid when**: When the kernel has a large accumulator tile (BLOCK_M × BLOCK_N FP32 values requiring ~128 regs/thread), applying maxnreg to force register spills causes catastrophic L1TEX local load/store traffic. Warp Cycles Per Issued Instruction increased from 5.36 to 12.99, eligible warps collapsed from 1.78 to 0.53, and instruction count grew by 41%. The occupancy gain (25%→31.25%) was insufficient to offset the spill penalty. Do not use maxnreg when shared memory is also a binding occupancy limiter (Block Limit Shared Mem=4), as the register cap cannot break the shared memory ceiling.

#### ❌ Plan C: `increase_block_k_to_reduce_shared_memory_bank_conflicts` (0.869×)

Increasing BLOCK_K from 32 to 64 in a Triton GEMM kernel doubled the dynamic shared memory per block (40 KB → 82 KB), which caused Block Limit Shared Mem to drop from 4 to 2 and theoretical occupancy to halve from 25% to 12.5%. Despite slightly reducing L1/TEX throughput (93.25% → 83.93%) and Mem Busy (86.34% → 74.59%), the occupancy collapse caused a net 14% regression in kernel duration (249.57 µs → 284.74 µs). When a kernel is already register- and shared-memory-limited (Block Limit Registers = Block Limit Shared Mem = 4), increasing BLOCK_K trades away the remaining occupancy headroom without sufficient arithmetic intensity gain to compensate.

> **Applicable when**: Never beneficial in this configuration. Avoid when Block Limit Shared Mem is already at or below 4 and theoretical occupancy is at or below 25% — increasing BLOCK_K will further reduce Block Limit Shared Mem and occupancy.

> **Avoid when**: Kernel is already shared-memory-limited (Block Limit Shared Mem ≤ 4) and register-limited (Block Limit Registers ≤ 4) simultaneously. In this regime, increasing BLOCK_K doubles shared memory allocation per block, halves Block Limit Shared Mem, and collapses occupancy further — causing net regression even if bank conflict rate decreases slightly. Also do not apply when BLOCK_N is small (e.g., 32) and the N-dimension is fully covered by one tile, as arithmetic intensity gains from larger BLOCK_K are insufficient to offset occupancy loss.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | revert_block_k_to_restore_shared_memory_occupancy | 1.001× | 1.004× | ✓ |  |


### Metric Comparison

| Metric | Baseline | revert_block_k_to... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 247550 ns | 246530 ns | -1020 ns (-0.4%) |
| SM Throughput (%) | 64.72 % | 64.85 % | +0.13 pp |
| Memory Throughput (%) | 86.15 % | 86.31 % | +0.16 pp |
| DRAM Throughput (%) | 36.33 % | 36.47 % | +0.14 pp |
| Achieved Occupancy (%) | 23.45 % | 23.44 % | -0.01 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 |
| Waves/SM | 3.03 | 3.03 | +0.00 |
| No Eligible (%) | 30.00 % | 29.95 % | -0.05 pp |
| IPC Active | 2.80 inst/cycle | 2.80 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  247550 ns  (1.000×)

Final best: 249570 ns  (0.9919×)
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
+        triton.Config({}, num_stages=3, num_warps=4),
+        triton.Config({}, num_stages=2, num_warps=4),
+        triton.Config({}, num_stages=4, num_warps=8),
+        triton.Config({}, num_stages=3, num_warps=8),
+        triton.Config({}, num_stages=2, num_warps=8),
+        triton.Config({}, num_stages=4, num_warps=4),
+        triton.Config({}, num_stages=5, num_warps=4),
+        triton.Config({}, num_stages=5, num_warps=8),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_mm_view_0(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -12,7 +26,7 @@

     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 128
     BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 32
+    BLOCK_K : tl.constexpr = 64
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -46,14 +60,11 @@

 
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
+
+    # Unconditional vectorization hints for 128-bit loads
+    offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
+    offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
+
     offs_k = tl.arange(0, BLOCK_K)
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
@@ -72,9 +83,7 @@

         xindex = idx_n + 32*idx_m
         b = tl.load(B + (xindex))
 
-
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -86,8 +95,6 @@

     # inductor generates a suffix
     xindex = idx_n + 32*idx_m
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
