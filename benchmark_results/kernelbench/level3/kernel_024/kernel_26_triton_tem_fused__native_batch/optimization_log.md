# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_26` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 57980 ns — **Best latency**: 27100 ns (**2.1395×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`increase_tile_sizes_and_autotune_for_tiny_gemm`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_26`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_26_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (9,1,1), Waves/SM: 0.02
- **Registers/thread**: 62
- **Duration**: 57980 ns
- **SM Throughput**: 0.54 % | **Memory**: 6.19 % | **DRAM**: 0.48 %
- **Achieved Occupancy**: 6.18 %
- **IPC Active**: 0.33 inst/cycle | **No Eligible**: 91.77 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.997× | 0.999× | ✓ |  |
| B | autotune_tile_sizes_for_tiny_m_gemm_with_bank_conflict_fix | 2.767× | 2.577× | ✓ |  |
| C | split_k_gemm_parallelism | 2.603× | 2.784× | ✓ |  |
| D | increase_tile_sizes_and_autotune_for_tiny_gemm | 2.906× | 2.139× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | autotune_tile_siz... | Δ | split_k_gemm_para... | Δ | increase_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 57980 ns | 58020 ns | +40 ns (+0.1%) | 22500 ns | -35480 ns (-61.2%) | 20830 ns | -37150 ns (-64.1%) | 27100 ns | -30880 ns (-53.3%) |
| SM Throughput (%) | 0.54 % | 0.54 % | +0.00 pp | 0.32 % | -0.22 pp | 0.38 % | -0.16 pp | 0.29 % | -0.25 pp |
| Memory Throughput (%) | 6.19 % | 6.20 % | +0.01 pp | 4.00 % | -2.19 pp | 4.40 % | -1.79 pp | 3.37 % | -2.82 pp |
| DRAM Throughput (%) | 0.48 % | 0.48 % | +0.00 pp | 1.23 % | +0.75 pp | 1.32 % | +0.84 pp | 1.02 % | +0.54 pp |
| Achieved Occupancy (%) | 6.18 % | 6.19 % | +0.01 pp | 3.12 % | -3.06 pp | 3.12 % | -3.06 pp | 3.12 % | -3.06 pp |
| Registers/Thread | 62.00 register/thread | 62.00 register/thread | +0 | 72.00 register/thread | +10 | 48.00 register/thread | -14 | 44.00 register/thread | -18 |
| Waves/SM | 0.02 | 0.02 | +0.00 | 0.01 | -0.01 | 0.01 | -0.01 | 0.00 | -0.02 |
| No Eligible (%) | 91.77 % | 91.78 % | +0.01 pp | 90.11 % | -1.66 pp | 88.68 % | -3.09 pp | 92.79 % | +1.02 pp |
| IPC Active | 0.33 inst/cycle | 0.33 inst/cycle | +0.00 | 0.20 inst/cycle | -0.13 | 0.23 inst/cycle | -0.10 | 0.14 inst/cycle | -0.19 |

---

## Iteration 2

**Best so far**: 2.1395× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_shared_memory_bank_conflicts_via_register_matmul | 2.621× | 1.938× | ✓ |  |
| B | reduce_block_k_to_cut_shared_memory_bank_conflicts | 2.523× | 1.505× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_shared_... | Δ | reduce_block_k_to... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 57980 ns | 29920 ns | -28060 ns (-48.4%) | 38530 ns | -19450 ns (-33.5%) |
| SM Throughput (%) | 0.54 % | 0.30 % | -0.24 pp | 0.16 % | -0.38 pp |
| Memory Throughput (%) | 6.19 % | 0.94 % | -5.25 pp | 1.27 % | -4.92 pp |
| DRAM Throughput (%) | 0.48 % | 0.94 % | +0.46 pp | 0.72 % | +0.24 pp |
| Achieved Occupancy (%) | 6.18 % | 1.56 % | -4.62 pp | 1.56 % | -4.62 pp |
| Registers/Thread | 62.00 register/thread | 255.00 register/thread | +193 | 70.00 register/thread | +8 |
| Waves/SM | 0.02 | 0.01 | -0.01 | 0.00 | -0.02 |
| No Eligible (%) | 91.77 % | 83.57 % | -8.20 pp | 90.67 % | -1.10 pp |
| IPC Active | 0.33 inst/cycle | 0.17 inst/cycle | -0.16 | 0.09 inst/cycle | -0.24 |

---

## Iteration 3

**Best so far**: 2.1395× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_block_k_to_cut_register_pressure_and_restore_occupancy | 1.690× | 1.791× | ✓ |  |
| C | eliminate_shared_memory_via_register_level_outer_product_accumulation | 1.692× | 1.821× | ✓ |  |
| D | persistent_kernel_one_block_per_sm_tiny_gemm | 1.573× | 0.794× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_block_k_to... | Δ | eliminate_shared_... | Δ | persistent_kernel... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 57980 ns | 32380 ns | -25600 ns (-44.2%) | 31840 ns | -26140 ns (-45.1%) | 72990 ns | +15010 ns (+25.9%) |
| SM Throughput (%) | 0.54 % | 0.23 % | -0.31 pp | 0.23 % | -0.31 pp | 0.12 % | -0.42 pp |
| Memory Throughput (%) | 6.19 % | 0.85 % | -5.34 pp | 0.86 % | -5.33 pp | 1.28 % | -4.91 pp |
| DRAM Throughput (%) | 0.48 % | 0.85 % | +0.37 pp | 0.86 % | +0.38 pp | 0.38 % | -0.10 pp |
| Achieved Occupancy (%) | 6.18 % | 1.56 % | -4.62 pp | 1.56 % | -4.62 pp | 1.56 % | -4.62 pp |
| Registers/Thread | 62.00 register/thread | 158.00 register/thread | +96 | 158.00 register/thread | +96 | 64.00 register/thread | +2 |
| Waves/SM | 0.02 | 0.01 | -0.01 | 0.01 | -0.01 | 0.00 | -0.02 |
| No Eligible (%) | 91.77 % | 85.16 % | -6.61 pp | 85.28 % | -6.49 pp | 94.56 % | +2.79 pp |
| IPC Active | 0.33 inst/cycle | 0.15 inst/cycle | -0.18 | 0.15 inst/cycle | -0.18 | 0.05 inst/cycle | -0.28 |


### Key Learnings

#### ✅ Plan A: `reduce_block_k_to_cut_register_pressure_and_restore_occupancy` (1.791×)

When a register outer-product accumulation pattern (tl.sum(a[:,:,None]*b[None,:,:], axis=1)) uses a large BLOCK_K, the 3D intermediate tensor [BLOCK_M, BLOCK_K, BLOCK_N] saturates register usage (255/thread). Halving BLOCK_K from 64 to 32 reduced registers from 255 to 158 and raised Block Limit Registers from 8 to 12, improving theoretical occupancy from 12.5% to 18.75%. However, achieved occupancy remained at 1.56% (grid of only 9 blocks across 132 SMs is the true binding constraint), and the doubled K-loop iteration count (18→36) increased elapsed cycles from 44803 to 48509, resulting in a net regression in kernel duration (29.92 µs → 32.38 µs). The reported 1.79x speedup must reflect end-to-end wall-clock improvement from a different source (e.g., reduced register spill traffic or compiler effects), not the isolated kernel duration.

> **Applicable when**: Register outer-product GEMM (tl.sum 3D pattern) where registers_per_thread is at or near 255 due to a large [BLOCK_M, BLOCK_K, BLOCK_N] intermediate; Block Limit Registers is the binding occupancy constraint; grid size is large enough that improved theoretical occupancy can translate to better achieved occupancy (waves_per_sm >= 0.5).

> **Avoid when**: Grid size is tiny (waves_per_sm << 0.1, e.g., 9 blocks for 132 SMs) — in this regime achieved occupancy is pinned by grid size regardless of register count, and the doubled K-loop iterations increase elapsed cycles without benefit. Also avoid when BLOCK_K reduction causes K to not evenly divide (requires masking overhead).

#### ✅ Plan C: `eliminate_shared_memory_via_register_level_outer_product_accumulation` (1.938×)

When a Triton GEMM kernel uses tl.dot with BLOCK_M < 16 (below the Tensor Core minimum), the dot product falls back to FFMA with shared memory staging, which causes severe bank conflicts when the B matrix has a large column stride (e.g., stride_bn >> BLOCK_K). Replacing tl.dot with an explicit register-level outer product — tl.sum(a[:, :, None] * b[None, :, :], axis=1) — eliminates all shared memory staging entirely, dropping SharedMemoryConflicts from 17-way at 82% of wavefronts to zero, and reducing MIO CPI stalls from 61% to 0%. The tradeoff is a large increase in register usage (48 → 255 registers/thread) and reduced theoretical occupancy (37.5% → 12.5%), but the elimination of the dominant bank-conflict bottleneck yields a net ~1.94x speedup for this tiny-M regime.

> **Applicable when**: Triton GEMM kernel where BLOCK_M < 16 (Tensor Core minimum), tl.dot falls back to FFMA with shared memory staging, B matrix has large column stride causing 17-way (or higher) shared memory bank conflicts, MIO CPI stall fraction is >50% of warp cycles, and the 3D intermediate tensor [BLOCK_M, BLOCK_K, BLOCK_N] fits in registers (e.g., 2*64*32=4096 fp32 elements = 16 KB, well within 256 KB register file per SM).

> **Avoid when**: BLOCK_M >= 16 (Tensor Cores are available and tl.dot is efficient); the 3D intermediate tensor [BLOCK_M, BLOCK_K, BLOCK_N] is too large to fit in registers (would cause register spilling to local memory, which is worse); the kernel is already register-limited (registers_per_thread near 255); bank conflicts are not the dominant bottleneck (MIO stall fraction < 30%).

#### ❌ Plan D: `persistent_kernel_one_block_per_sm_tiny_gemm` (0.794×)

Converting a tiny-GEMM kernel from a register-level outer-product accumulation (tl.sum outer product, zero shared memory) to a persistent kernel using tl.dot caused a 2.44x duration regression (29.92 µs → 72.99 µs). The persistent loop structure forced a switch from the register outer-product pattern back to tl.dot, which reintroduced 17-way shared memory bank conflicts (0 → 82.35% of wavefronts) and MIO CPI stalls (0% → 32.74% of cycles), while Warp Cycles Per Issued Instruction tripled (6.01 → 18.18). Despite reducing registers per thread from 255 to 64 and raising theoretical occupancy from 12.5% to 50%, achieved occupancy remained identical at 1.56% because the grid (9 blocks) is still far below the 132 SM count — the persistent loop provided no additional parallelism. The lesson is that for kernels already bottlenecked by bank conflicts, adding a persistent loop wrapper does not help if it forces reintroduction of the conflicting memory access pattern; and for kernels where the register outer-product has already eliminated bank conflicts, switching to tl.dot to enable a persistent loop is counterproductive.

> **Applicable when**: Never apply persistent kernel loop wrapping when the kernel already uses register-level outer-product accumulation to avoid shared memory bank conflicts; the persistent loop forces tl.dot which reintroduces the conflict.

> **Avoid when**: Kernel is already using register outer-product (tl.sum(a[:,:,None]*b[None,:,:], axis=1)) to avoid bank conflicts; grid size is far below SM count (waves_per_sm < 0.1) and total tiles cannot be increased enough to saturate SMs; achieved occupancy is already at its practical ceiling given the tiny grid.

---

## Cumulative Progress

```
Baseline:  57980 ns  (1.000×)
Iter 1     27100 ns  (2.139×)  — increase_tile_sizes_and_autotune_for_tiny_gemm

Final best: 27100 ns  (2.1395×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,14 +5,29 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=1, num_stages=2),
+        triton.Config({}, num_warps=1, num_stages=3),
+        triton.Config({}, num_warps=1, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_26(arg_A, arg_B, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1):
     EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 16
+    BLOCK_M : tl.constexpr = 2
     BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 128
+    BLOCK_K : tl.constexpr = 64
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -67,14 +82,15 @@

         xindex = idx_n + 1152*idx_m
         a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 288*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 1152*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        # Direct global load of B as [BLOCK_K, BLOCK_N] — no shared memory staging
+        # B[k, n] = B_ptr + k*stride_bk + n*stride_bn = B_ptr + k + n*1152
+        idx_n = offs_b_n[None, :]  # [1, BLOCK_N]
+        b = tl.load(B + (b_k_idx_vals + 1152 * idx_n))  # [BLOCK_K, BLOCK_N]
 
-
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
+        # Register-level accumulation: expand a to [BLOCK_M, BLOCK_K, 1],
+        # b to [1, BLOCK_K, BLOCK_N], multiply -> [BLOCK_M, BLOCK_K, BLOCK_N],
+        # reduce axis=1 -> [BLOCK_M, BLOCK_N]. Zero shared memory, zero bank conflicts.
+        acc = acc + tl.sum(a[:, :, None] * b[None, :, :], axis=1).to(ACC_TYPE)
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -101,8 +117,6 @@

     tmp12 = tmp10 * tmp11
     tmp14 = tmp12 + tmp13
     tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 288*idx_m, [BLOCK_M, BLOCK_N])), tmp14, mask)
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
