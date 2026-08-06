# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_mm_relu_t_view_9` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 28960 ns — **Best latency**: 11650 ns (**2.4858×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`persistent_kernel_with_autotuned_tile_sizes`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_mm_relu_t_view_9`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_030/kernel_9_triton_tem_fused_addmm_mm_relu/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (6,1,1), Waves/SM: 0.01
- **Registers/thread**: 53
- **Duration**: 28960 ns
- **SM Throughput**: 0.36 % | **Memory**: 3.90 % | **DRAM**: 0.25 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.33 inst/cycle | **No Eligible**: 91.59 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | persistent_kernel_with_autotuned_tile_sizes | 2.251× | 2.486× | ✓ | **✓ best** |
| C | split_k_to_increase_parallelism | 0.999× | 0.824× | ✓ |  |
| D | increase_grid_parallelism_via_tile_size_autotune | 1.802× | 1.963× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_kernel... | Δ | split_k_to_increa... | Δ | increase_grid_par... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 28960 ns | 11650 ns | -17310 ns (-59.8%) | 35140 ns | +6180 ns (+21.3%) | 14750 ns | -14210 ns (-49.1%) |
| SM Throughput (%) | 0.36 % | 0.38 % | +0.02 pp | 0.27 % | -0.09 pp | 0.36 % | +0.00 pp |
| Memory Throughput (%) | 3.90 % | 3.37 % | -0.53 pp | 3.22 % | -0.68 pp | 2.18 % | -1.72 pp |
| DRAM Throughput (%) | 0.25 % | 0.65 % | +0.40 pp | 0.21 % | -0.04 pp | 0.51 % | +0.26 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 53.00 register/thread | 46.00 register/thread | -7 | 48.00 register/thread | -5 | 80.00 register/thread | +27 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.00 | -0.01 | 0.00 | -0.01 |
| No Eligible (%) | 91.59 % | 93.05 % | +1.46 pp | 93.56 % | +1.97 pp | 81.77 % | -9.82 pp |
| IPC Active | 0.33 inst/cycle | 0.27 inst/cycle | -0.06 | 0.26 inst/cycle | -0.07 | 0.35 inst/cycle | +0.02 |

---

## Iteration 2

**Best so far**: 2.4858× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_block_m_with_autotune_to_reduce_shared_memory_bank_conflicts | 1.469× | 1.264× | ✓ |  |
| B | persistent_kernel_to_fill_all_sms | 1.821× | 1.954× | ✓ |  |
| C | persistent_kernel_with_block_n_matched_to_n_dim | 2.222× | 2.501× | ✓ |  |
| D | add_triton_autotune | 1.747× | 1.757× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_block_m_... | Δ | persistent_kernel... | Δ | persistent_kernel... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 28960 ns | 22910 ns | -6050 ns (-20.9%) | 14820 ns | -14140 ns (-48.8%) | 11580 ns | -17380 ns (-60.0%) | 16480 ns | -12480 ns (-43.1%) |
| SM Throughput (%) | 0.36 % | 0.14 % | -0.22 pp | 0.17 % | -0.19 pp | 0.37 % | +0.01 pp | 0.24 % | -0.12 pp |
| Memory Throughput (%) | 3.90 % | 1.00 % | -2.90 pp | 1.05 % | -2.85 pp | 3.30 % | -0.60 pp | 2.41 % | -1.49 pp |
| DRAM Throughput (%) | 0.25 % | 0.33 % | +0.08 pp | 0.50 % | +0.25 pp | 0.65 % | +0.40 pp | 0.44 % | +0.19 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp | 6.24 % | -0.01 pp |
| Registers/Thread | 53.00 register/thread | 216.00 register/thread | +163 | 70.00 register/thread | +17 | 46.00 register/thread | -7 | 38.00 register/thread | -15 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.00 | -0.01 | 0.00 | -0.01 | 0.00 | -0.01 |
| No Eligible (%) | 91.59 % | 80.18 % | -11.41 pp | 83.28 % | -8.31 pp | 93.10 % | +1.51 pp | 95.27 % | +3.68 pp |
| IPC Active | 0.33 inst/cycle | 0.40 inst/cycle | +0.07 | 0.34 inst/cycle | +0.01 | 0.28 inst/cycle | -0.05 | 0.18 inst/cycle | -0.15 |


### Key Learnings

#### ✅ Plan B: `persistent_kernel_to_fill_all_sms` (1.954×)

When a GEMM kernel launches far fewer CTAs than available SMs (e.g., 6 CTAs on 132 SMs, 0 waves/SM, <7% achieved occupancy), a persistent kernel pattern can distribute work across all SMs: launch NUM_SMS CTAs and wrap the tile-decode+accumulate+store body in an outer loop `for tile_id in tl.range(sm_id, total_tiles, NUM_SMS)`. This eliminates idle SMs and reduces MIO stall cycles (from 9.9 to 2.2 cycles per warp) by spreading the small number of tiles across more active warps. The LaunchConfiguration NCU rule (95.45% estimated speedup) is the primary signal; the pattern is safe when total_tiles < NUM_SMS so each tile is owned by exactly one SM with no atomic operations needed.

> **Applicable when**: Triton GEMM kernel where grid_size << sm_count (waves_per_sm < 0.1, achieved_occupancy < 10%); LaunchConfiguration NCU rule fires with high estimated speedup (>90%); total output tiles is small (e.g., <20) relative to SM count; each output tile is independent (no reduction across tiles needed, so no atomic_add required).

> **Avoid when**: total_tiles >= NUM_SMS (persistent loop adds overhead without benefit); kernel requires atomic accumulation across tiles (Split-K pattern); ALLOW_TF32=True causes correctness failures for this kernel (confirmed for K=512 reduction with atol=1e-3); grid is already large enough to fill all SMs (waves_per_sm >= 1).

---

## Iteration 3

**Best so far**: 2.4858× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_block_k_to_break_shared_memory_bank_conflicts | 2.136× | 2.452× | ✓ |  |
| B | eliminate_smem_bank_conflicts_via_register_cached_b_tile | 2.095× | 1.138× | ✓ |  |
| C | increase_block_m_and_widen_autotune_configs_to_reduce_mio_stalls | 1.471× | 1.250× | ✓ |  |
| D | reduce_block_m_and_autotune_tile_sizes_to_cut_bank_conflicts | 2.222× | 2.268× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_block_k_to... | Δ | eliminate_smem_ba... | Δ | increase_block_m_... | Δ | reduce_block_m_an... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 28960 ns | 11810 ns | -17150 ns (-59.2%) | 25440 ns | -3520 ns (-12.2%) | 23170 ns | -5790 ns (-20.0%) | 12770 ns | -16190 ns (-55.9%) |
| SM Throughput (%) | 0.36 % | 0.40 % | +0.04 pp | 0.09 % | -0.27 pp | 0.14 % | -0.22 pp | 0.24 % | -0.12 pp |
| Memory Throughput (%) | 3.90 % | 3.19 % | -0.71 pp | 0.56 % | -3.34 pp | 1.00 % | -2.90 pp | 2.12 % | -1.78 pp |
| DRAM Throughput (%) | 0.25 % | 0.64 % | +0.39 pp | 0.29 % | +0.04 pp | 0.33 % | +0.08 pp | 0.58 % | +0.33 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 1.56 % | -4.69 pp | 3.12 % | -3.13 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 53.00 register/thread | 40.00 register/thread | -13 | 80.00 register/thread | +27 | 216.00 register/thread | +163 | 63.00 register/thread | +10 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.00 | -0.01 | 0.01 | +0.00 | 0.00 | -0.01 |
| No Eligible (%) | 91.59 % | 93.04 % | +1.45 pp | 82.59 % | -9.00 pp | 80.44 % | -11.15 pp | 88.64 % | -2.95 pp |
| IPC Active | 0.33 inst/cycle | 0.27 inst/cycle | -0.06 | 0.17 inst/cycle | -0.16 | 0.39 inst/cycle | +0.06 | 0.23 inst/cycle | -0.10 |


### Key Learnings

#### ✅ Plan A: `reduce_block_k_to_break_shared_memory_bank_conflicts` (2.452×)

Reducing BLOCK_K from 64 to 32 in a Triton GEMM kernel (halving the shared memory tile stride from 256 bytes to 128 bytes) did not reduce the 12-way shared memory bank conflict count or the 75% excessive wavefront fraction — both metrics remained identical before and after. Dynamic shared memory per block dropped from 18.43 KB to 9.22 KB and registers per thread fell from 46 to 40 (improving theoretical occupancy from 62.5% to 75%), but MIO stall cycles per warp only marginally improved from 9.9 to 9.5. The 2.45x reported speedup is therefore attributable to the surrounding persistent kernel and BLOCK_N=4 changes already in place, not to the BLOCK_K reduction itself. When Triton's tl.dot compiler-controlled SMEM layout produces 12-way conflicts, changing BLOCK_K alone does not alter the conflict multiplicity because the conflict pattern is determined by the internal MMA tile layout, not the outer K-stride.

> **Applicable when**: Never as a standalone fix for tl.dot-induced shared memory bank conflicts. Reducing BLOCK_K may be useful for reducing register pressure or shared memory footprint, but it does not address bank conflicts generated by Triton's internal MMA tile layout.

> **Avoid when**: When SharedMemoryConflicts NCU rule fires with >=12-way bank conflicts and >=75% excessive wavefronts in a Triton kernel using tl.dot — halving BLOCK_K will not reduce the conflict count because the conflict pattern is determined by the compiler-controlled MMA SMEM layout, not the outer K-dimension stride.

---

## Cumulative Progress

```
Baseline:  28960 ns  (1.000×)
Iter 1     11650 ns  (2.486×)  — persistent_kernel_with_autotuned_tile_sizes

Final best: 11650 ns  (2.4858×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,89 +5,80 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_tem_fused_addmm_mm_relu_t_view_9(arg_A, arg_B, out_ptr0):
-    EVEN_K : tl.constexpr = True
-    USE_FAST_ACCUM : tl.constexpr = False
-    ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 32
-    BLOCK_N : tl.constexpr = 16
-    BLOCK_K : tl.constexpr = 32
-    GROUP_M : tl.constexpr = 8
-    ALLOW_TF32 : tl.constexpr = False
-    INDEX_DTYPE : tl.constexpr = tl.int32
+    ACC_TYPE: tl.constexpr = tl.float32
+    BLOCK_M: tl.constexpr = 32
+    BLOCK_N: tl.constexpr = 4
+    BLOCK_K: tl.constexpr = 64
+    GROUP_M: tl.constexpr = 8
+    ALLOW_TF32: tl.constexpr = False
+    INDEX_DTYPE: tl.constexpr = tl.int32
+    NUM_SMS: tl.constexpr = 132
+
     A = arg_A
     B = arg_B
 
     M = 169
     N = 3
     K = 512
+
     if M * N == 0:
-        # early exit due to zero-size input(s)
         return
+
     stride_am = 512
     stride_ak = 1
     stride_bk = 1
     stride_bn = 512
 
-    # based on triton.ops.matmul
-    pid = tl.program_id(0).to(INDEX_DTYPE)
-    grid_m = (M + BLOCK_M - 1) // BLOCK_M
-    grid_n = (N + BLOCK_N - 1) // BLOCK_N
+    grid_m = tl.cdiv(M, BLOCK_M)
+    grid_n = tl.cdiv(N, BLOCK_N)
+    total_tiles = grid_m * grid_n
 
-    # re-order program ID for better L2 performance
-    width = GROUP_M * grid_n
-    group_id = pid // width
-    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
-    pid_m = group_id * GROUP_M + (pid % group_size)
-    pid_n = (pid % width) // (group_size)
-    tl.assume(pid_m >= 0)
-    tl.assume(pid_n >= 0)
+    sm_id = tl.program_id(0).to(INDEX_DTYPE)
 
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
-        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
-    else:
+    for tile_id in tl.range(sm_id, total_tiles, NUM_SMS):
+        # re-order program ID for better L2 performance
+        width = GROUP_M * grid_n
+        group_id = tile_id // width
+        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
+        pid_m = group_id * GROUP_M + (tile_id % group_size)
+        pid_n = (tile_id % width) // group_size
+
+        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
+        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
+
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
+        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
+            a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
+            b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
+            idx_m = offs_a_m[:, None]
+            idx_n = a_k_idx_vals
+            xindex = idx_n + 512 * idx_m
+            a = tl.load(A + xindex)
 
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 512*idx_m
-        a = tl.load(A + (xindex))
+            idx_m = b_k_idx_vals
+            idx_n = offs_b_n[None, :]
+            xindex = idx_n + 3 * idx_m
+            b = tl.load(B + (tl.broadcast_to(idx_m + 512 * idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 3*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
+        # rematerialize rm and rn to save registers
+        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
+        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
+        idx_m = rm[:, None]
+        idx_n = rn[None, :]
+        mask = (idx_m < M) & (idx_n < N)
 
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
-
-    # rematerialize rm and rn to save registers
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    idx_m = rm[:, None]
-    idx_n = rn[None, :]
-    mask = (idx_m < M) & (idx_n < N)
-
-    # inductor generates a suffix
-    xindex = idx_n + 3*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+        xindex = idx_n + 3 * idx_m
+        tl.store(out_ptr0 + tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N]), acc, mask=mask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
