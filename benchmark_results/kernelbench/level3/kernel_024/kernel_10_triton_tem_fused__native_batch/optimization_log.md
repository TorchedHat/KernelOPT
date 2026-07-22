# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_10` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 8030 ns — **Best latency**: 5980 ns (**1.3428×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`persistent_kernel_fixed_block_m_and_b_load`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_10`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_10_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3,1,1), Waves/SM: 0.00
- **Registers/thread**: 44
- **Duration**: 8030 ns
- **SM Throughput**: 0.16 % | **Memory**: 1.27 % | **DRAM**: 0.14 %
- **Achieved Occupancy**: 5.64 %
- **IPC Active**: 0.34 inst/cycle | **No Eligible**: 90.17 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.987× | 0.954× | ✓ |  |
| C | split_k_parallelism | 1.077× | 0.848× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | split_k_paralleli... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 8030 ns | 8420 ns | +390 ns (+4.9%) | 9470 ns | +1440 ns (+17.9%) |
| SM Throughput (%) | 0.16 % | 0.16 % | +0.00 pp | 0.13 % | -0.03 pp |
| Memory Throughput (%) | 1.27 % | 1.23 % | -0.04 pp | 0.65 % | -0.62 pp |
| DRAM Throughput (%) | 0.14 % | 0.14 % | +0.00 pp | 0.12 % | -0.02 pp |
| Achieved Occupancy (%) | 5.64 % | 5.74 % | +0.10 pp | 2.98 % | -2.66 pp |
| Registers/Thread | 44.00 register/thread | 44.00 register/thread | +0 | 64.00 register/thread | +20 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 90.17 % | 90.18 % | +0.01 pp | 86.90 % | -3.27 pp |
| IPC Active | 0.34 inst/cycle | 0.34 inst/cycle | +0.00 | 0.27 inst/cycle | -0.07 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | persistent_kernel_fixed_block_m_and_b_load | 1.247× | 1.343× | ✓ | **✓ best** |
| D | persistent_kernel_correct_tile_loop_and_fix_b_load | 0.888× | 0.839× | ✓ |  |
| E | add_triton_autotune | 0.904× | 0.865× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_kernel... | Δ | persistent_kernel... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8030 ns | 5980 ns | -2050 ns (-25.5%) | 9570 ns | +1540 ns (+19.2%) | 9280 ns | +1250 ns (+15.6%) |
| SM Throughput (%) | 0.16 % | 0.10 % | -0.06 pp | 0.12 % | -0.04 pp | 0.13 % | -0.03 pp |
| Memory Throughput (%) | 1.27 % | 0.83 % | -0.44 pp | 1.08 % | -0.19 pp | 1.12 % | -0.15 pp |
| DRAM Throughput (%) | 0.14 % | 0.17 % | +0.03 pp | 0.10 % | -0.04 pp | 0.11 % | -0.03 pp |
| Achieved Occupancy (%) | 5.64 % | 6.24 % | +0.60 pp | 6.25 % | +0.61 pp | 5.62 % | -0.02 pp |
| Registers/Thread | 44.00 register/thread | 50.00 register/thread | +6 | 52.00 register/thread | +8 | 48.00 register/thread | +4 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 90.17 % | 93.86 % | +3.69 pp | 93.60 % | +3.43 pp | 91.97 % | +1.80 pp |
| IPC Active | 0.34 inst/cycle | 0.24 inst/cycle | -0.10 | 0.26 inst/cycle | -0.08 | 0.27 inst/cycle | -0.07 |


### Key Learnings

#### ❌ Plan D: `persistent_kernel_correct_tile_loop_and_fix_b_load` (0.839×)

Restructuring a tiny GEMM kernel (M=2, N=96, K=96, grid=3 on 132-SM GPU) into a persistent tile loop with BLOCK_M=2 and BLOCK_N=16 (doubling tile count to 6) while fixing the B-matrix double-broadcast load caused a regression: duration increased from 8.42 µs to 9.57 µs (0.84x). Despite the grid remaining at 3 blocks (the persistent loop was intended to cover 6 tiles per block), registers per thread increased from 44 to 52, reducing theoretical occupancy from 62.5% to 56.25%, and Warp Cycles Per Issued Instruction worsened from 10.59 to 15.36 cycles. The SharedMemoryConflicts speedup_pct remained nearly unchanged (56.12% → 49.47%), indicating the B-load fix did not meaningfully reduce bank conflicts. The persistent loop structure added control-flow overhead (while loop, tile_id arithmetic) that increased instruction count and stall cycles without delivering the expected parallelism benefit on this already-tiny workload.

> **Applicable when**: Never: this specific combination of persistent-loop restructuring + BLOCK_M reduction on a tiny GEMM (M=2, total tiles=6, grid=3) caused regression. The persistent loop overhead and increased register pressure outweigh any benefit.

> **Avoid when**: When the kernel is already extremely small (total tiles ≤ 6, grid << SM count) and the persistent loop adds while-loop control flow overhead. When the change increases registers per thread (44→52), reducing theoretical occupancy and worsening Warp Cycles Per Issued Instruction (10.59→15.36). When SharedMemoryConflicts remain nearly unchanged despite the B-load restructuring, indicating the bank conflict root cause was not actually addressed by removing the double-broadcast.

---

## Iteration 3

**Best so far**: 1.3428× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_shared_memory_bank_conflicts_via_register_dot | 1.225× | 1.249× | ✓ |  |
| C | reduce_block_m_to_match_problem_size_and_fix_b_load_double_broadcast | 1.186× | 1.293× | ✓ |  |
| D | add_triton_autotune | 1.092× | 0.584× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_shared_... | Δ | reduce_block_m_to... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8030 ns | 6430 ns | -1600 ns (-19.9%) | 6210 ns | -1820 ns (-22.7%) | 13760 ns | +5730 ns (+71.4%) |
| SM Throughput (%) | 0.16 % | 0.38 % | +0.22 pp | 0.09 % | -0.07 pp | 0.07 % | -0.09 pp |
| Memory Throughput (%) | 1.27 % | 0.56 % | -0.71 pp | 0.83 % | -0.44 pp | 0.42 % | -0.85 pp |
| DRAM Throughput (%) | 0.14 % | 0.17 % | +0.03 pp | 0.16 % | +0.02 pp | 0.10 % | -0.04 pp |
| Achieved Occupancy (%) | 5.64 % | 5.51 % | -0.13 pp | 6.23 % | +0.59 pp | 1.56 % | -4.08 pp |
| Registers/Thread | 44.00 register/thread | 96.00 register/thread | +52 | 40.00 register/thread | -4 | 96.00 register/thread | +52 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 90.17 % | 74.60 % | -15.57 pp | 94.85 % | +4.68 pp | 84.61 % | -5.56 pp |
| IPC Active | 0.34 inst/cycle | 0.85 inst/cycle | +0.51 | 0.21 inst/cycle | -0.13 | 0.15 inst/cycle | -0.19 |


### Key Learnings

#### ✅ Plan A: `eliminate_shared_memory_bank_conflicts_via_register_dot` (1.249×)

When NCU reports severe shared memory bank conflicts (17-way, 82% excessive wavefronts) and CPIStall attributes ~43% of stall cycles to MIO (shared memory) latency, replacing tl.dot with a register-level outer-product accumulation via tl.reshape+tl.sum completely eliminates shared memory staging and its associated bank conflicts. The trade-off is a large increase in registers per thread (50→96) and instruction count (4800→17850), but Warp Cycles Per Issued Instruction drops dramatically (16.33→4.07) and Issue Slots Busy improves 3.5x (6.16%→21.45%), yielding a net 1.25x speedup. This strategy is most effective when the GEMM tile is small enough that all operands fit comfortably in the register file (BLOCK_M×BLOCK_K + BLOCK_K×BLOCK_N floats), and the bank conflict penalty dominates over the register pressure cost.

> **Applicable when**: NCU SharedMemoryConflicts rule fires with ≥15-way bank conflicts and ≥70% excessive wavefronts. CPIStall attributes ≥30% of stall cycles to MIO (shared memory) latency. GEMM tile is small: BLOCK_M×BLOCK_K + BLOCK_K×BLOCK_N ≤ ~1200 floats so all operands fit in registers. The B-matrix stride causes aliasing (stride mod BLOCK_K == 0), making tl.dot's shared memory staging pathological.

> **Avoid when**: Large GEMM tiles where BLOCK_K×BLOCK_N exceeds ~1024 floats — register pressure will spill to local memory and cause worse performance than the bank conflicts. When tl.dot is already using tensor cores efficiently (high Compute SM Throughput, low SharedMemoryConflicts). When register count is already near the hardware limit (>96 regs/thread) as further increases reduce theoretical occupancy significantly.

---

## Iteration 4

**Best so far**: 1.3428× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | widen_block_n_to_full_n_eliminate_tiling_overhead | 0.678× | 0.295× | ✓ |  |


### Metric Comparison

| Metric | Baseline | widen_block_n_to_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 8030 ns | 27230 ns | +19200 ns (+239.1%) |
| SM Throughput (%) | 0.16 % | 0.14 % | -0.02 pp |
| Memory Throughput (%) | 1.27 % | 0.27 % | -1.00 pp |
| DRAM Throughput (%) | 0.14 % | 0.09 % | -0.05 pp |
| Achieved Occupancy (%) | 5.64 % | 1.56 % | -4.08 pp |
| Registers/Thread | 44.00 register/thread | 255.00 register/thread | +211 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 90.17 % | 74.13 % | -16.04 pp |
| IPC Active | 0.34 inst/cycle | 0.26 inst/cycle | -0.08 |


### Key Learnings

#### ❌ Plan A: `widen_block_n_to_full_n_eliminate_tiling_overhead` (0.295×)

Increasing BLOCK_N from 32 to 128 (beyond the actual N=96) while simultaneously adding a large autotune config set caused registers per thread to explode from 40 to 255 (the hardware maximum), collapsing theoretical occupancy from 75% to 12.5% and achieved occupancy from 6.23% to 1.56%. Duration increased 4.4x (6.21 µs → 27.23 µs) despite SharedMemoryConflicts bank-conflict-way dropping from 17 to 8. The autotune selected a 1-warp (32-thread) block, which combined with the oversized accumulator tile [16×128] forced maximum register spill. The lesson is that widening BLOCK_N beyond what the register file can hold (BLOCK_M×BLOCK_N floats >> register budget) causes catastrophic register pressure that far outweighs any bank-conflict reduction benefit.

> **Applicable when**: Never: widening BLOCK_N to 128 (exceeding actual N=96) with BLOCK_M=16 creates an accumulator of 2048 floats that cannot fit in the register file, causing maximum register spill regardless of num_warps selection.

> **Avoid when**: When BLOCK_M × BLOCK_N exceeds ~512 floats for a single warp (32 threads), as the register file cannot hold the accumulator and operand tiles simultaneously. When the autotune config set includes low-warp configs (num_warps=1 or 2) that reduce the thread count, making per-thread register demand even higher. When BLOCK_N is set larger than the actual N dimension, wasting computation on masked-out columns. This is the fourth consecutive confirmation that autotuning num_warps/num_stages without controlling tile sizes causes register explosion on tiny GEMMs.

---

## Cumulative Progress

```
Baseline:  8030 ns  (1.000×)
Iter 2     5980 ns  (1.343×)  — persistent_kernel_fixed_block_m_and_b_load

Final best: 5980 ns  (1.3428×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,12 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=1),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=1),
+        triton.Config({}, num_warps=8, num_stages=2),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_10(arg_A, arg_B, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1):
     EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 16
+    BLOCK_M : tl.constexpr = 2
     BLOCK_N : tl.constexpr = 32
     BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
@@ -60,18 +70,15 @@

     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
 
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 96*idx_m
         a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 96*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 96*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
+        b_k = offs_k[:, None] + (k_idx * BLOCK_K)  # shape [BLOCK_K, 1]
+        b_n = offs_b_n[None, :]                       # shape [1, BLOCK_N]
+        b = tl.load(B + (b_k + 96 * b_n))            # direct 2D index, no double-broadcast
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
@@ -101,8 +108,6 @@

     tmp12 = tmp10 * tmp11
     tmp14 = tmp12 + tmp13
     tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 96*idx_m, [BLOCK_M, BLOCK_N])), tmp14, mask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
