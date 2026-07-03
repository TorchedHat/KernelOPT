# Kernel Optimization Log

## Summary

- **Kernel**: `matmul_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 13220000 ns — **Best latency**: 2440000 ns (**5.4180×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`tf32_tensor_cores`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `matmul_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_008/synthesis/kernel_0_triton_tem_fused_mm_0.py`
- **Framework**: triton
- **Launch**: Block (256,1,1), Grid (1560,1,1), Waves/SM: 11.82
- **Registers/thread**: 255
- **Duration**: 13220000 ns
- **SM Throughput**: 53.91 % | **Memory**: 37.37 % | **DRAM**: 1.79 %
- **Achieved Occupancy**: 12.50 %
- **IPC Active**: 2.19 inst/cycle | **No Eligible**: 45.39 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 4.260× | 5.418× | ✓ | **✓ best** |
| B | fast_accumulation_fp16_dot_reduce_register_pressure | 1.399× | 1.213× | ✓ |  |
| D | reduce_register_pressure_via_autotune_configs | 1.000× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | fast_accumulation... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13220000 ns | 2440000 ns | -10780000 ns (-81.5%) | 10900000 ns | -2320000 ns (-17.5%) | 13230000 ns | +10000 ns (+0.1%) |
| SM Throughput (%) | 53.91 % | 34.91 % | -19.00 pp | 7.40 % | -46.51 pp | 53.86 % | -0.05 pp |
| Memory Throughput (%) | 37.37 % | 66.37 % | +29.00 pp | 87.82 % | +50.45 pp | 37.33 % | -0.04 pp |
| DRAM Throughput (%) | 1.79 % | 8.64 % | +6.85 pp | 1.96 % | +0.17 pp | 1.79 % | +0.00 pp |
| Achieved Occupancy (%) | 12.50 % | 12.49 % | -0.01 pp | 12.50 % | +0.00 pp | 12.50 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 246.00 register/thread | -9 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 11.82 | 11.82 | +0.00 | 11.82 | +0.00 | 11.82 | +0.00 |
| No Eligible (%) | 45.39 % | 75.53 % | +30.14 pp | 93.23 % | +47.84 pp | 45.22 % | -0.17 pp |
| IPC Active | 2.19 inst/cycle | 0.98 inst/cycle | -1.21 | 0.27 inst/cycle | -1.92 | 2.19 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 5.418× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_shared_memory_bank_conflicts_via_block_ptr_tma | 3.403× | 4.278× | ✓ |  |
| B | eliminate_shared_memory_bank_conflicts_via_smem_padding | 1.311× | 1.126× | ✓ |  |
| C | bf16_input_cast_with_fp32_accumulator_tensor_core | 1.303× | 0.812× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_shared_... | Δ | eliminate_shared_... | Δ | bf16_input_cast_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13220000 ns | 3090000 ns | -10130000 ns (-76.6%) | 11740000 ns | -1480000 ns (-11.2%) | 16270000 ns | +3050000 ns (+23.1%) |
| SM Throughput (%) | 53.91 % | 45.20 % | -8.71 pp | 12.81 % | -41.10 pp | 6.34 % | -47.57 pp |
| Memory Throughput (%) | 37.37 % | 53.35 % | +15.98 pp | 85.16 % | +47.79 pp | 89.16 % | +51.79 pp |
| DRAM Throughput (%) | 1.79 % | 6.82 % | +5.03 pp | 1.83 % | +0.04 pp | 1.31 % | -0.48 pp |
| Achieved Occupancy (%) | 12.50 % | 12.50 % | +0.00 pp | 12.50 % | +0.00 pp | 12.50 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 254.00 register/thread | -1 |
| Waves/SM | 11.82 | 11.82 | +0.00 | 11.82 | +0.00 | 23.14 | +11.32 |
| No Eligible (%) | 45.39 % | 57.39 % | +12.00 pp | 87.29 % | +41.90 pp | 94.70 % | +49.31 pp |
| IPC Active | 2.19 inst/cycle | 1.70 inst/cycle | -0.49 | 0.51 inst/cycle | -1.68 | 0.21 inst/cycle | -1.98 |


### Key Learnings

#### ❌ Plan B: `eliminate_shared_memory_bank_conflicts_via_smem_padding` (1.126×)

Replacing raw pointer arithmetic with tl.make_block_ptr (order=(1,0)/(0,1)) to eliminate shared memory bank conflicts caused a severe regression: kernel duration increased from 2.44ms to 11.74ms (~4.8x slower), despite the 1.1261x wall-clock speedup being measured elsewhere. The executed instruction count ballooned from 409M to 1007M, warp cycles per issued instruction nearly doubled (8.16 → 15.74), and the scheduler idle rate worsened (75.53% → 87.29% no-eligible). The autotune selected a much smaller tile configuration (dynamic SMEM dropped from 147KB to 131KB), indicating the new configs with BLOCK_K=128 were not chosen, and the block_ptr path introduced new local load/store spill patterns (1.0/32 bytes utilized per sector) that dominated over the bank conflict savings. The strategy of using tl.make_block_ptr for SMEM padding is architecturally sound, but the implementation introduced register spilling and autotune selected a suboptimal config that negated the bank conflict fix.

> **Applicable when**: Never apply tl.make_block_ptr as a bank-conflict fix when the kernel is already register-bound (255 registers/thread, Block Limit Registers=1, 12.5% occupancy) — the block_ptr abstraction introduces additional register pressure and local memory spilling that outweighs bank conflict savings.

> **Avoid when**: Do not use tl.make_block_ptr to fix shared memory bank conflicts when: (1) registers_per_thread is already at the hardware maximum (255), causing Block Limit Registers=1; (2) the kernel already has 12.5% theoretical occupancy limited by both registers and shared memory; (3) autotune search space includes many small-tile configs that may be selected over the intended large-tile configs. The local load/store pattern showing 1.0/32 bytes utilized per sector in the AFTER profile is the signature of register spilling introduced by block_ptr overhead.

#### ❌ Plan C: `bf16_input_cast_with_fp32_accumulator_tensor_core` (0.812×)

Casting FP32 input tiles to BF16 before tl.dot in a Triton matmul kernel caused a 6.7x regression (2.44ms → 16.27ms) despite the intent to use BF16 Tensor Cores. The root cause was catastrophic shared memory bank conflicts: the after-profile shows a 21.7-way average bank conflict rate (SharedMemoryConflicts rule, 85.51% speedup_pct potential) with 91% of shared load wavefronts being excessive (UncoalescedSharedAccess, 89.17% speedup_pct). The BF16 tile layout selected by the new autotune configs (larger BLOCK_K=64/128, larger grid=3055 vs 1560) produced a shared memory access pattern that serialized nearly all warp memory operations, causing Warp Cycles Per Issued Instruction to balloon from 8.16 to 37.7 and No Eligible scheduler cycles to rise from 75.5% to 94.7%. Executed instructions actually increased (409M → 610M), indicating the BF16 Tensor Core path was not engaged effectively.

> **Applicable when**: Never — this specific BF16 cast strategy caused severe regression; do not apply BF16 cast to tl.dot when the autotune configs use large BLOCK_K (64/128) without shared memory padding, as the resulting tile layout creates catastrophic bank conflicts.

> **Avoid when**: Always avoid this pattern when: (1) shared memory is already near capacity (Dynamic SMEM > 147KB/block), (2) autotune configs use BLOCK_K >= 64 without explicit shared memory padding, (3) the baseline kernel already achieves good performance via TF32 (allow_tf32=True). The BF16 cast changes the tile element size from 4 bytes to 2 bytes, which can create stride-2 bank conflict patterns in shared memory that are far more damaging than any Tensor Core throughput gain.

---

## Iteration 3

**Best so far**: 5.418× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_block_ptr_order_and_expand_autotune_for_latency_hiding | 3.389× | 4.306× | ✓ |  |
| B | revert_block_ptr_to_raw_pointer_arithmetic_to_eliminate_register_spill | 1.204× | 1.204× | ✓ |  |
| C | revert_bf16_cast_to_fp32_dot_eliminate_bank_conflicts | 0.989× | 0.820× | ✓ |  |
| D | add_triton_autotune | 3.395× | 4.378× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_block_ptr_o... | Δ | revert_block_ptr_... | Δ | revert_bf16_cast_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13220000 ns | 3070000 ns | -10150000 ns (-76.8%) | 10980000 ns | -2240000 ns (-16.9%) | 16130000 ns | +2910000 ns (+22.0%) | 3020000 ns | -10200000 ns (-77.2%) |
| SM Throughput (%) | 53.91 % | 45.15 % | -8.76 pp | 9.66 % | -44.25 pp | 44.22 % | -9.69 pp | 45.24 % | -8.67 pp |
| Memory Throughput (%) | 37.37 % | 53.29 % | +15.92 pp | 86.93 % | +49.56 pp | 32.89 % | -4.48 pp | 53.39 % | +16.02 pp |
| DRAM Throughput (%) | 1.79 % | 6.86 % | +5.07 pp | 1.93 % | +0.14 pp | 1.61 % | -0.18 pp | 6.97 % | +5.18 pp |
| Achieved Occupancy (%) | 12.50 % | 12.50 % | +0.00 pp | 12.50 % | +0.00 pp | 12.50 % | +0.00 pp | 12.50 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 11.82 | 11.82 | +0.00 | 11.82 | +0.00 | 23.14 | +11.32 | 11.82 | +0.00 |
| No Eligible (%) | 45.39 % | 57.41 % | +12.02 pp | 92.17 % | +46.78 pp | 54.11 % | +8.72 pp | 57.40 % | +12.01 pp |
| IPC Active | 2.19 inst/cycle | 1.70 inst/cycle | -0.49 | 0.31 inst/cycle | -1.88 | 1.84 inst/cycle | -0.35 | 1.70 inst/cycle | -0.49 |


### Key Learnings

#### ❌ Plan A: `fix_b_block_ptr_order_and_expand_autotune_for_latency_hiding` (4.306×)

Changing B_block_ptr order from (1,0) to (0,1) and adding smaller-tile autotune configs produced a reported 4.31x wall-clock speedup, but the NCU-profiled kernel metrics are virtually identical before and after: duration changed only from 3.09ms to 3.07ms, all scheduler, occupancy, instruction count, and memory metrics are within noise. The UncoalescedSharedAccess excessive wavefronts (43%), CPIStall (49% of cycles), registers_per_thread (255), Block Limit Registers (1), and Theoretical Occupancy (12.5%) are completely unchanged. This indicates the wall-clock speedup is attributable to autotune selecting a different (faster) configuration between measurement runs, not to the B order fix or the new configs themselves. The B order change from (1,0) to (0,1) for a (BLOCK_K, BLOCK_N)-shaped tile where stride_bn is the fast dimension is semantically correct but did not measurably reduce shared memory bank conflicts in this kernel.

> **Applicable when**: Never — this specific combination of B order fix + small-tile autotune expansion produced no measurable NCU improvement. The wall-clock speedup is a measurement artifact (autotune variance), not a causal result of the code change.

> **Avoid when**: Do not expect changing tl.make_block_ptr order parameter alone to resolve UncoalescedSharedAccess when the kernel is already at 255 registers/thread with Block Limit Registers=1 and Block Limit Shared Mem=1. When both register and shared memory limits are simultaneously binding (occupancy=12.5%), the order parameter change does not reduce excessive shared wavefronts because the fundamental tile layout constraints are unchanged. The CPIStall and TheoreticalOccupancy bottlenecks require reducing tile size enough to drop below the register ceiling, which the new autotune configs also failed to achieve (registers remained at 255).

#### ✅ Plan B: `revert_block_ptr_to_raw_pointer_arithmetic_to_eliminate_register_spill` (1.204×)

When a Triton kernel is already at the register limit (255 registers/thread, Block Limit Registers=1), replacing tl.make_block_ptr + boundary_check loads with raw pointer arithmetic + manual mask-based tl.load eliminates register spilling to local memory. The canonical NCU signature of register spilling is MemoryCacheAccessPattern reporting local load/store utilization of 1.0/32 bytes per sector; after reverting to raw pointers, this pattern disappears and Warp Cycles Per Issued Instruction dropped from 25.56 to 4.69 while Compute SM Throughput recovered from 9.66% to 45.2%. The tl.make_block_ptr abstraction with boundary_check generates additional address-computation registers that push already register-heavy kernels over the spill threshold, causing a net regression despite its bank-conflict-free staging benefits.

> **Applicable when**: Triton matmul kernel using tl.make_block_ptr with boundary_check; MemoryCacheAccessPattern NCU rule reports local load/store utilization of ~1.0/32 bytes per sector (register spill signature); registers_per_thread is at or near 255; Block Limit Registers=1; Warp Cycles Per Issued Instruction is high (>15) with very low Eligible Warps Per Scheduler (<0.15); Compute SM Throughput is severely depressed (<15%) while L1/TEX throughput is near-saturated (>85%).

> **Avoid when**: Kernel is not register-bound (registers_per_thread < 200, Block Limit Registers > 1); shared memory bank conflicts are the dominant bottleneck (UncoalescedSharedAccess >80% excessive wavefronts without the local spill signature); GPU is Hopper (SM 9.0) and TMA benefits from block_ptr outweigh register costs (only viable when register count is well below 255); kernel already uses raw pointer arithmetic.

---

## Cumulative Progress

```
Baseline:  13220000 ns  (1.000×)
Iter 1     2440000 ns  (5.418×)  — tf32_tensor_cores

Final best: 2440000 ns  (5.4180×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -1,6 +1,7 @@

 import torch
 import triton
 import triton.language as tl
+
 
 
 @triton.autotune(
@@ -41,43 +42,35 @@

     pid_m = group_id * GROUP_M + (pid % group_size)
     pid_n = (pid % width) // group_size
 
-    # Row/col offsets
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    rk = tl.arange(0, BLOCK_K)
-
-    # Clamp indices for boundary safety in the K loop
-    offs_am = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
-    offs_bn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
-
-    # Pointers to A and B tiles
-    A_ptrs = A_ptr + (offs_am[:, None] * stride_am + rk[None, :] * stride_ak)
-    B_ptrs = B_ptr + (rk[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
+    # Create block pointers for A, B using TMA-friendly order=(1,0)
+    A_block_ptr = tl.make_block_ptr(
+        base=A_ptr, shape=(M, K), strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    B_block_ptr = tl.make_block_ptr(
+        base=B_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N),
+        order=(1, 0),
+    )
 
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
 
     num_k_blocks = tl.cdiv(K, BLOCK_K)
     for k in range(num_k_blocks):
-        k_remaining = K - k * BLOCK_K
-        # Boundary mask for K dimension
-        a_mask = rk[None, :] < k_remaining
-        b_mask = rk[:, None] < k_remaining
+        a = tl.load(A_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        b = tl.load(B_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        acc = tl.dot(a, b, acc=acc, allow_tf32=True)
+        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
+        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))
 
-        a = tl.load(A_ptrs, mask=a_mask, other=0.0)
-        b = tl.load(B_ptrs, mask=b_mask, other=0.0)
-
-        acc = tl.dot(a, b, acc=acc, allow_tf32=False)
-
-        A_ptrs += BLOCK_K * stride_ak
-        B_ptrs += BLOCK_K * stride_bk
-
-    # Write output
-    rm_out = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn_out = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    mask = (rm_out[:, None] < M) & (rn_out[None, :] < N)
-    C_ptrs = C_ptr + rm_out[:, None] * stride_cm + rn_out[None, :] * stride_cn
-    tl.store(C_ptrs, acc, mask=mask)
-
+    # Write output using block pointer for bank-conflict-free store
+    C_block_ptr = tl.make_block_ptr(
+        base=C_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
+        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N),
+        order=(1, 0),
+    )
+    tl.store(C_block_ptr, acc, boundary_check=(0, 1))
 
 def kernel_function(a, b):
     M, K = a.shape
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
