# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_22` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 13340 ns — **Best latency**: 36320 ns (**0.3673×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_22`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_22_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (36,1,1), Waves/SM: 0.03
- **Registers/thread**: 56
- **Duration**: 13340 ns
- **SM Throughput**: 2.03 % | **Memory**: 17.75 % | **DRAM**: 1.40 %
- **Achieved Occupancy**: 5.84 %
- **IPC Active**: 0.34 inst/cycle | **No Eligible**: 90.82 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | split_k_parallelism | 1.345× | 1.275× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | split_k_paralleli... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 13340 ns | 10460 ns | -2880 ns (-21.6%) |
| SM Throughput (%) | 2.03 % | 0.50 % | -1.53 pp |
| Memory Throughput (%) | 17.75 % | 3.78 % | -13.97 pp |
| DRAM Throughput (%) | 1.40 % | 1.79 % | +0.39 pp |
| Achieved Occupancy (%) | 5.84 % | 1.56 % | -4.28 pp |
| Registers/Thread | 56.00 register/thread | 70.00 register/thread | +14 |
| Waves/SM | 0.03 | 0.01 | -0.02 |
| No Eligible (%) | 90.82 % | 91.07 % | +0.25 pp |
| IPC Active | 0.34 inst/cycle | 0.09 inst/cycle | -0.25 |

---

## Iteration 2

**Best so far**: 1.2753× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_block_n_for_better_sm_utilization | 1.356× | 0.367× | ✓ | **✓ best** |
| B | persistent_kernel_one_block_per_sm | 1.344× | 1.115× | ✓ |  |
| C | increase_block_n_and_pipeline_stages_to_hide_l1tex_latency | 1.347× | 0.751× | ✓ |  |
| D | increase_block_n_and_num_warps_for_tiny_m_gemm | 1.325× | 0.691× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_block_n_... | Δ | persistent_kernel... | Δ | increase_block_n_... | Δ | increase_block_n_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13340 ns | 36320 ns | +22980 ns (+172.3%) | 11970 ns | -1370 ns (-10.3%) | 17760 ns | +4420 ns (+33.1%) | 19300 ns | +5960 ns (+44.7%) |
| SM Throughput (%) | 2.03 % | 0.50 % | -1.53 pp | 0.40 % | -1.63 pp | 1.15 % | -0.88 pp | 0.92 % | -1.11 pp |
| Memory Throughput (%) | 17.75 % | 3.95 % | -13.80 pp | 3.24 % | -14.51 pp | 14.46 % | -3.29 pp | 7.66 % | -10.09 pp |
| DRAM Throughput (%) | 1.40 % | 0.53 % | -0.87 pp | 1.56 % | +0.16 pp | 1.06 % | -0.34 pp | 0.97 % | -0.43 pp |
| Achieved Occupancy (%) | 5.84 % | 1.56 % | -4.28 pp | 1.56 % | -4.28 pp | 3.12 % | -2.72 pp | 3.12 % | -2.72 pp |
| Registers/Thread | 56.00 register/thread | 190.00 register/thread | +134 | 68.00 register/thread | +12 | 72.00 register/thread | +16 | 88.00 register/thread | +32 |
| Waves/SM | 0.03 | 0.05 | +0.02 | 0.01 | -0.02 | 0.02 | -0.01 | 0.03 | +0.00 |
| No Eligible (%) | 90.82 % | 91.98 % | +1.16 pp | 92.96 % | +2.14 pp | 90.18 % | -0.64 pp | 92.19 % | +1.37 pp |
| IPC Active | 0.34 inst/cycle | 0.08 inst/cycle | -0.26 | 0.07 inst/cycle | -0.27 | 0.20 inst/cycle | -0.14 | 0.16 inst/cycle | -0.18 |


### Key Learnings

#### ❌ Plan C: `increase_block_n_and_pipeline_stages_to_hide_l1tex_latency` (0.751×)

For a tiny-M GEMM (M=2, N=1152, K=192) with a fixed 36-block grid, increasing BLOCK_N from 32 to 128 (by hardcoding it in the kernel body rather than making it an autotune key) caused the block size to double (32→64 threads), dynamic shared memory to quadruple (4.35→16.64 KB/block), and executed instructions to increase nearly 4x (40932→161100). Critically, shared memory bank conflicts worsened dramatically: the average conflict way increased from 12-way to 22.5-way, and the conflict wavefront fraction rose from 77.78% to 91.61%, because the wider tile mapped more threads onto the same shared memory banks under this access pattern. The kernel duration increased from 10.46 µs to 17.76 µs (0.75x speedup). The root cause is that for this extremely small problem (total output = 2304 floats), the kernel is fundamentally latency-bound and launch-overhead-bound — widening tiles increases per-block work but does not reduce the dominant bottleneck (too few blocks to fill 132 SMs), and the wider B-matrix tile layout actually amplified bank conflicts rather than reducing them.

> **Applicable when**: Never: this specific change (widening BLOCK_N without making it a true autotune key with grid adaptation) consistently worsens performance for tiny-M GEMMs with fixed small grids.

> **Avoid when**: Any tiny-M GEMM (M≤4) where the grid is already small (≤36 blocks vs 132 SMs). Widening BLOCK_N without proportionally reducing grid size does not increase SM utilization and amplifies shared memory bank conflicts when the B-matrix access stride maps wider tiles onto fewer distinct banks. The L1/TEX Cache Throughput rising from 16.9% to 62.9% and bank conflict way increasing from 12 to 22.5 confirm the wider tile made memory access patterns worse, not better.

---

## Iteration 3

**Best so far**: 1.2753× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_b_tile_bank_conflicts_via_smem_transpose_with_padding | 1.287× | 0.560× | ✓ |  |
| C | fix_smem_bank_conflicts_via_padding | 1.335× | 0.411× | ✓ |  |
| D | reduce_register_pressure_via_smaller_tile_autotune | 1.345× | 1.241× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_tile_bank_c... | Δ | fix_smem_bank_con... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13340 ns | 23840 ns | +10500 ns (+78.7%) | 32480 ns | +19140 ns (+143.5%) | 10750 ns | -2590 ns (-19.4%) |
| SM Throughput (%) | 2.03 % | 0.77 % | -1.26 pp | 0.51 % | -1.52 pp | 0.48 % | -1.55 pp |
| Memory Throughput (%) | 17.75 % | 8.46 % | -9.29 pp | 6.91 % | -10.84 pp | 3.67 % | -14.08 pp |
| DRAM Throughput (%) | 1.40 % | 0.80 % | -0.60 pp | 0.58 % | -0.82 pp | 1.74 % | +0.34 pp |
| Achieved Occupancy (%) | 5.84 % | 1.56 % | -4.28 pp | 1.56 % | -4.28 pp | 1.56 % | -4.28 pp |
| Registers/Thread | 56.00 register/thread | 128.00 register/thread | +72 | 128.00 register/thread | +72 | 70.00 register/thread | +14 |
| Waves/SM | 0.03 | 0.02 | -0.01 | 0.02 | -0.01 | 0.01 | -0.02 |
| No Eligible (%) | 90.82 % | 87.19 % | -3.63 pp | 91.71 % | +0.89 pp | 90.72 % | -0.10 pp |
| IPC Active | 0.34 inst/cycle | 0.13 inst/cycle | -0.21 | 0.08 inst/cycle | -0.26 | 0.09 inst/cycle | -0.25 |


### Key Learnings

#### ❌ Plan B: `fix_b_tile_bank_conflicts_via_smem_transpose_with_padding` (0.560×)

Attempting to fix B-tile shared memory bank conflicts by changing BLOCK_N from 32 to 128 (combined with autotune config changes) caused registers per thread to nearly double (68→128) and dynamic shared memory per block to quadruple (4.35→16.64 KB), collapsing theoretical occupancy from 43.75% to 20.31%. Despite the intent to pad shared memory, the actual diff did not implement explicit smem padding — it only widened BLOCK_N, which amplified bank conflicts from 12-way to 25.3-way and increased executed instructions by ~4x (37980→146574), causing duration to double from 11.97 µs to 23.84 µs. The stall pattern shifted from L1TEX scoreboard stalls to MIO (shared memory) stalls, confirming the wider tile made the conflict pattern worse, not better.

> **Applicable when**: Never: this specific combination of BLOCK_N=128 without true smem padding consistently worsens performance for this tiny-M GEMM with fixed 36-block grid.

> **Avoid when**: Any tiny-M GEMM (M≤4) where SharedMemoryConflicts is already firing and the grid is small (≤36 blocks vs 132 SMs). Widening BLOCK_N without implementing explicit +1-padded shared memory staging causes register explosion (68→128 regs/thread), smem explosion (4.35→16.64 KB), and amplifies bank conflicts proportionally. The autotune config changes (removing num_stages=1 configs, adding num_stages=3/4) did not compensate for the structural regression from the wider tile. Do not conflate 'changing autotune configs' with 'implementing smem padding' — the actual code must allocate a [BLOCK_K, BLOCK_N+1] smem buffer and use tl.store/tl.load through it.

#### ❌ Plan C: `fix_smem_bank_conflicts_via_padding` (0.411×)

The plan attempted to fix shared memory bank conflicts by staging the B-matrix tile through a padded intermediate tensor, but the actual diff changed BLOCK_N from 32 to 128 (hardcoded) without successfully implementing the padding — the code comment explicitly notes 'just use b_global directly'. This caused registers per thread to nearly double (68→128), dynamic shared memory to quadruple (4.35→16.64 KB/block), and theoretical occupancy to halve (43.75%→20.31%), while bank conflict severity worsened from 12-way to 25.3-way and executed instructions tripled (37980→133326). Duration increased 2.7x (11.97→32.48 µs). The implementation flaw was that Triton does not support non-power-of-2 intermediate tensor shapes for smem padding, so the padding was silently dropped and only the larger BLOCK_N remained, amplifying the conflict pattern.

> **Applicable when**: Never: this specific approach (increasing BLOCK_N to 128 while attempting smem padding via non-power-of-2 intermediate tensors) consistently worsens performance for this tiny-M GEMM. Triton does not support non-power-of-2 intermediate tensor shapes for smem layout control, so the padding is silently dropped.

> **Avoid when**: Any tiny-M GEMM (M≤4) with a fixed small grid (≤36 blocks on 132-SM GPU) where SharedMemoryConflicts is already active. Increasing BLOCK_N from 32 to 128 without a working padding mechanism amplifies bank conflicts proportionally (12-way→25.3-way), doubles register pressure (68→128 regs/thread), halves theoretical occupancy (43.75%→20.31%), and triples instruction count. Triton's smem padding via non-power-of-2 intermediate tensor shapes is not reliably supported — the compiler silently falls back to the unpadded layout.

---

## Iteration 4

**Best so far**: 1.2753× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  13340 ns  (1.000×)
Iter 1     10460 ns  (1.275×)  — split_k_parallelism

Final best: 36320 ns  (0.3673×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,17 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({}, num_warps=1, num_stages=1),
+    triton.Config({}, num_warps=2, num_stages=1),
+    triton.Config({}, num_warps=4, num_stages=1),
+    triton.Config({}, num_warps=1, num_stages=2),
+    triton.Config({}, num_warps=2, num_stages=2),
+    triton.Config({}, num_warps=4, num_stages=2),
+], key=[])
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_22(arg_A, arg_B, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1):
     EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 16
+    BLOCK_M : tl.constexpr = 2
     BLOCK_N : tl.constexpr = 32
     BLOCK_K : tl.constexpr = 32
-    GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
-    INDEX_DTYPE : tl.constexpr = tl.int32
     A = arg_A
     B = arg_B
 
@@ -25,86 +32,66 @@

     if M * N == 0:
         # early exit due to zero-size input(s)
         return
-    stride_am = 192
-    stride_ak = 1
-    stride_bk = 1
-    stride_bn = 192
 
-    # based on triton.ops.matmul
-    pid = tl.program_id(0).to(INDEX_DTYPE)
-    grid_m = (M + BLOCK_M - 1) // BLOCK_M
-    grid_n = (N + BLOCK_N - 1) // BLOCK_N
+    pid = tl.program_id(0).to(tl.int32)
+    num_progs = tl.num_programs(0).to(tl.int32)
+    grid_m = tl.cdiv(M, BLOCK_M)   # = 1
+    grid_n = tl.cdiv(N, BLOCK_N)   # = 36
+    num_tiles = grid_m * grid_n     # = 36
 
-    # re-order program ID for better L2 performance
-    width = GROUP_M * grid_n
-    group_id = pid // width
-    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
-    pid_m = group_id * GROUP_M + (pid % group_size)
-    pid_n = (pid % width) // (group_size)
-    tl.assume(pid_m >= 0)
-    tl.assume(pid_n >= 0)
+    tile_id = pid  # start at this block's SM index
+    while tile_id < num_tiles:
+        pid_m = tile_id // grid_n   # = 0 always (M=2, BLOCK_M=2)
+        pid_n = tile_id % grid_n    # 0..35
 
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
-        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
-    else:
+        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
+        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
         offs_a_m = rm % M
-    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
-        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
-    else:
         offs_b_n = rn % N
-    offs_k = tl.arange(0, BLOCK_K)
-    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
+        offs_k = tl.arange(0, BLOCK_K)
+        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
-    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
+        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
+            a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
+            b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 192*idx_m
-        a = tl.load(A + (xindex))
+            idx_m = offs_a_m[:, None]
+            idx_n_a = a_k_idx_vals
+            a = tl.load(A + (idx_n_a + 192 * idx_m))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 1152*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 192*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+            idx_m_b = b_k_idx_vals
+            idx_n_b = offs_b_n[None, :]
+            b = tl.load(B + tl.broadcast_to(idx_m_b + 192 * idx_n_b, [BLOCK_K, BLOCK_N]))
 
-        
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
+            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
-    # rematerialize rm and rn to save registers
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    idx_m = rm[:, None]
-    idx_n = rn[None, :]
-    mask = (idx_m < M) & (idx_n < N)
+        # Epilogue: BN + ReLU
+        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
+        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
+        idx_m = rm[:, None]
+        idx_n = rn[None, :]
+        mask = (idx_m < M) & (idx_n < N)
 
-    # inductor generates a suffix
-    xindex = idx_n + 1152*idx_m
-    tmp0 = tl.load(in_ptr2 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
-    tmp2 = tl.load(in_ptr3 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
-    tmp11 = tl.load(in_ptr4 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
-    tmp13 = tl.load(in_ptr5 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
-    tmp1 = acc - tmp0
-    tmp3 = tl.full([1], 1e-05, tl.float32)
-    tmp4 = tmp2 + tmp3
-    tmp5 = tl.sqrt_rn(tmp4)
-    tmp6 = tl.full([1], 1, tl.int32)
-    tmp7 = (tmp6 / tmp5)
-    tmp8 = tl.full([1], 1.0, tl.float32)
-    tmp9 = tmp7 * tmp8
-    tmp10 = tmp1 * tmp9
-    tmp12 = tmp10 * tmp11
-    tmp14 = tmp12 + tmp13
-    tmp15 = tl.full([1], 0, tl.int32)
-    tmp16 = tl.maximum(tmp15, tmp14)
-    tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 1152*idx_m, [BLOCK_M, BLOCK_N])), tmp16, mask)
+        tmp0  = tl.load(in_ptr2 + tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N]), mask, eviction_policy='evict_last')
+        tmp2  = tl.load(in_ptr3 + tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N]), mask, eviction_policy='evict_last')
+        tmp11 = tl.load(in_ptr4 + tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N]), mask, eviction_policy='evict_last')
+        tmp13 = tl.load(in_ptr5 + tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N]), mask, eviction_policy='evict_last')
+        tmp1  = acc - tmp0
+        tmp3  = tl.full([1], 1e-05, tl.float32)
+        tmp4  = tmp2 + tmp3
+        tmp5  = tl.sqrt_rn(tmp4)
+        tmp6  = tl.full([1], 1, tl.int32)
+        tmp7  = (tmp6 / tmp5)
+        tmp8  = tl.full([1], 1.0, tl.float32)
+        tmp9  = tmp7 * tmp8
+        tmp10 = tmp1 * tmp9
+        tmp12 = tmp10 * tmp11
+        tmp14 = tmp12 + tmp13
+        tmp15 = tl.full([1], 0, tl.int32)
+        tmp16 = tl.maximum(tmp15, tmp14)
+        tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 1152 * idx_m, [BLOCK_M, BLOCK_N])), tmp16, mask)
 
-
+        tile_id += num_progs  # stride by number of SMs (132)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
