# Kernel Optimization Log

## Summary

- **Kernel**: `fused_linear_sub_mul_relu_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7200000 ns — **Best latency**: 4010000 ns (**1.7955×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_w_block_ptr_order_to_eliminate_tl_trans_and_reduce_register_pressure`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `fused_linear_sub_mul_relu_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_009/synthesis/kernel_0_fused_fused_addmm.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 1.94
- **Registers/thread**: 255
- **Duration**: 7200000 ns
- **SM Throughput**: 45.61 % | **Memory**: 60.62 % | **DRAM**: 1.01 %
- **Achieved Occupancy**: 12.48 %
- **IPC Active**: 1.93 inst/cycle | **No Eligible**: 51.79 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reduce_register_spill_via_hopper_autotune_configs | 0.660× | 0.480× | ✓ |  |
| C | reduce_register_pressure_via_smaller_tile_autotune_configs | 0.999× | 1.000× | ✓ |  |
| D | reduce_register_pressure_via_autotune_configs | 0.941× | 0.480× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_s... | Δ | reduce_register_p... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7200000 ns | 15000000 ns | +7800000 ns (+108.3%) | 7200000 ns | +0 ns (+0.0%) | 15000000 ns | +7800000 ns (+108.3%) |
| SM Throughput (%) | 45.61 % | 21.06 % | -24.55 pp | 45.72 % | +0.11 pp | 21.06 % | -24.55 pp |
| Memory Throughput (%) | 60.62 % | 81.69 % | +21.07 pp | 60.78 % | +0.16 pp | 81.67 % | +21.05 pp |
| DRAM Throughput (%) | 1.01 % | 1.18 % | +0.17 pp | 1.01 % | +0.00 pp | 1.18 % | +0.17 pp |
| Achieved Occupancy (%) | 12.48 % | 6.25 % | -6.23 pp | 12.48 % | +0.00 pp | 6.25 % | -6.23 pp |
| Registers/Thread | 255.00 register/thread | 128.00 register/thread | -127 | 255.00 register/thread | +0 | 128.00 register/thread | -127 |
| Waves/SM | 1.94 | 15.52 | +13.58 | 1.94 | +0.00 | 15.52 | +13.58 |
| No Eligible (%) | 51.79 % | 78.23 % | +26.44 pp | 51.78 % | -0.01 pp | 78.24 % | +26.45 pp |
| IPC Active | 1.93 inst/cycle | 0.87 inst/cycle | -1.06 | 1.93 inst/cycle | +0.00 | 0.87 inst/cycle | -1.06 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_shared_memory_bank_conflicts_via_block_ptr | 1.000× | 0.930× | ✓ |  |
| B | restore_occupancy_via_small_block_k_and_fewer_stages | 1.033× | 0.550× | ✓ |  |
| C | reduce_shared_memory_per_block_via_small_block_k_configs | 1.034× | 1.014× | ✓ | **✓ best** |
| D | add_triton_autotune | 0.996× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_shared_... | Δ | restore_occupancy... | Δ | reduce_shared_mem... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7200000 ns | 7740000 ns | +540000 ns (+7.5%) | 13080000 ns | +5880000 ns (+81.7%) | 7100000 ns | -100000 ns (-1.4%) | 7210000 ns | +10000 ns (+0.1%) |
| SM Throughput (%) | 45.61 % | 44.13 % | -1.48 pp | 23.92 % | -21.69 pp | 46.26 % | +0.65 pp | 45.63 % | +0.02 pp |
| Memory Throughput (%) | 60.62 % | 59.91 % | -0.71 pp | 93.83 % | +33.21 pp | 62.79 % | +2.17 pp | 60.67 % | +0.05 pp |
| DRAM Throughput (%) | 1.01 % | 0.99 % | -0.02 pp | 1.12 % | +0.11 pp | 1.05 % | +0.04 pp | 1.01 % | +0.00 pp |
| Achieved Occupancy (%) | 12.48 % | 12.42 % | -0.06 pp | 24.03 % | +11.55 pp | 12.47 % | -0.01 pp | 12.48 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 128.00 register/thread | -127 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 3.88 | +1.94 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 51.79 % | 53.26 % | +1.47 pp | 75.33 % | +23.54 pp | 51.49 % | -0.30 pp | 51.78 % | -0.01 pp |
| IPC Active | 1.93 inst/cycle | 1.87 inst/cycle | -0.06 | 0.99 inst/cycle | -0.94 | 1.94 inst/cycle | +0.01 | 1.93 inst/cycle | +0.00 |


### Key Learnings

#### ❌ Plan B: `restore_occupancy_via_small_block_k_and_fewer_stages` (0.550×)

Switching Triton GEMM autotune configs to BLOCK_K=32 with num_stages=2/3 reduced dynamic shared memory per block from 65 KB to 16 KB and improved theoretical occupancy from 12.5% to 25% (Block Limit Registers rose from 2 to 4). However, the increased block count (grid 512→2048, waves 1.94→3.88) caused L1/TEX cache throughput to saturate at 96.8% (up from 64.1%), Mem Busy to rise to 93.8%, and warp cycles per issued instruction to jump from 4.1 to 15.6 cycles — indicating severe L1 cache contention from the higher concurrency. The net result was a 1.8× slowdown (7.2 ms → 13.1 ms): more occupancy does not help when the memory subsystem becomes the new binding bottleneck due to increased concurrent tile traffic. When the baseline kernel already has severe shared memory bank conflicts (5.4-way) and uncoalesced accesses (56% excess wavefronts), reducing tile size increases the number of tiles and amplifies L1 pressure rather than relieving it.

> **Applicable when**: Never apply this specific pattern. Reducing num_stages to lower shared memory per block while keeping BLOCK_K=32 increases concurrent block count and saturates L1/TEX cache when the kernel already has structural shared memory bank conflicts.

> **Avoid when**: Do not reduce num_stages to 2 when the kernel already has severe shared memory bank conflicts (>5-way) and uncoalesced shared accesses (>50% excess wavefronts). The increased block concurrency from lower smem/block will saturate L1/TEX (>90% Mem Busy) and cause warp stall cycles to multiply (4→15 cycles/inst), negating any occupancy gain. The root cause (bank conflicts from strided shared memory access pattern) must be fixed first — e.g. via shared memory padding or TMA-based access — before tuning occupancy via num_stages.

---

## Iteration 3

**Best so far**: 1.0141× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | eliminate_shared_memory_bank_conflicts_via_block_ptr_tma_loads | 0.936× | 0.536× | ✓ |  |
| D | tf32_tensor_cores_with_ieee_precision_emulation | 2.965× | 1.809× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | eliminate_shared_... | Δ | tf32_tensor_cores... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7200000 ns | 13420000 ns | +6220000 ns (+86.4%) | 3980000 ns | -3220000 ns (-44.7%) |
| SM Throughput (%) | 45.61 % | 24.11 % | -21.50 pp | 27.88 % | -17.73 pp |
| Memory Throughput (%) | 60.62 % | 91.46 % | +30.84 pp | 54.26 % | -6.36 pp |
| DRAM Throughput (%) | 1.01 % | 1.05 % | +0.04 pp | 3.72 % | +2.71 pp |
| Achieved Occupancy (%) | 12.48 % | 18.16 % | +5.68 pp | 12.30 % | -0.18 pp |
| Registers/Thread | 255.00 register/thread | 158.00 register/thread | -97 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 5.17 | +3.23 | 1.94 | +0.00 |
| No Eligible (%) | 51.79 % | 75.01 % | +23.22 pp | 76.93 % | +25.14 pp |
| IPC Active | 1.93 inst/cycle | 1.00 inst/cycle | -0.93 | 0.91 inst/cycle | -1.02 |


### Key Learnings

#### ❌ Plan C: `eliminate_shared_memory_bank_conflicts_via_block_ptr_tma_loads` (0.536×)

Replacing raw pointer arithmetic loads with tl.make_block_ptr + tl.advance in a Triton GEMM kernel (intended to trigger TMA swizzled shared memory layout on Hopper) actually worsened shared memory bank conflicts from 5.3-way to 12-way and increased uncoalesced shared wavefronts from 56% to 76%. The L1/TEX hit rate collapsed from 64.1% to 0.02%, L1/TEX throughput saturated at 94.8%, Mem Busy rose from 62.8% to 91.5%, and warp cycles per issued instruction jumped from 4.1 to 11.6 — all indicating the block pointer path did not engage TMA swizzling as expected, and instead increased memory pressure by changing the tile configuration (smaller smem/block, more concurrent blocks, grid 512→2048). The autotune config changes that accompanied the block pointer switch (adding smaller BLOCK_M/BLOCK_N configs, changing num_stages) likely selected a config with more concurrent blocks that amplified L1 contention, masking or preventing any TMA benefit.

> **Applicable when**: Never apply this specific pattern in isolation. tl.make_block_ptr alone does not guarantee TMA swizzled layout or bank-conflict elimination; the autotune config selection (tile sizes, num_stages) must also be controlled to avoid increasing concurrent block count and L1 pressure.

> **Avoid when**: Do not switch from raw pointer arithmetic to tl.make_block_ptr when: (1) the kernel already has structural shared memory bank conflicts (>5-way) that are not caused by the pointer style alone, (2) the autotune config space is simultaneously changed to include smaller tiles that increase grid size and concurrent block count, (3) L1/TEX is already under moderate pressure (>60% throughput) — adding more concurrent blocks will saturate it. The block pointer change must be isolated from config changes to determine its true effect. Also do not apply when the kernel's shared memory access pattern has stride issues that tl.make_block_ptr does not fix (e.g. non-power-of-two strides or layouts incompatible with TMA swizzling).

---

## Iteration 4

**Best so far**: 1.809× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 2.997× | 1.800× | ✓ |  |
| B | eliminate_register_spill_via_ieee_precision_and_grid_alignment | 1.000× | 0.930× | ✓ |  |
| C | tf32x3_tensor_cores_for_fp32_accurate_hmma | 2.831× | 2.271× | ✓ | **✓ best** |
| D | tf32x3_tensor_cores_via_input_precision_ozaki_splitting | — | fail | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | eliminate_registe... | Δ | tf32x3_tensor_cor... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7200000 ns | 4000000 ns | -3200000 ns (-44.4%) | 7740000 ns | +540000 ns (+7.5%) | 3170000 ns | -4030000 ns (-56.0%) |
| SM Throughput (%) | 45.61 % | 27.83 % | -17.78 pp | 44.13 % | -1.48 pp | 38.05 % | -7.56 pp |
| Memory Throughput (%) | 60.62 % | 54.20 % | -6.42 pp | 59.92 % | -0.70 pp | 49.87 % | -10.75 pp |
| DRAM Throughput (%) | 1.01 % | 3.62 % | +2.61 pp | 1.02 % | +0.01 pp | 4.46 % | +3.45 pp |
| Achieved Occupancy (%) | 12.48 % | 12.26 % | -0.22 pp | 12.42 % | -0.06 pp | 17.78 % | +5.30 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 142.00 register/thread | -113 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 5.17 | +3.23 |
| No Eligible (%) | 51.79 % | 77.18 % | +25.39 pp | 53.28 % | +1.49 pp | 60.80 % | +9.01 pp |
| IPC Active | 1.93 inst/cycle | 0.92 inst/cycle | -1.01 | 1.87 inst/cycle | -0.06 | 1.57 inst/cycle | -0.36 |

---

## Iteration 5

**Best so far**: 2.2713× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_w_block_ptr_order_to_eliminate_tl_trans_and_reduce_register_pressure | 2.961× | 1.796× | ✓ | **✓ best** |
| C | fix_w_block_ptr_layout_to_eliminate_tl_trans_register_pressure | 1.005× | 0.929× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_w_block_ptr_o... | Δ | fix_w_block_ptr_l... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7200000 ns | 4010000 ns | -3190000 ns (-44.3%) | 7750000 ns | +550000 ns (+7.6%) |
| SM Throughput (%) | 45.61 % | 27.81 % | -17.80 pp | 44.16 % | -1.45 pp |
| Memory Throughput (%) | 60.62 % | 53.89 % | -6.73 pp | 59.99 % | -0.63 pp |
| DRAM Throughput (%) | 1.01 % | 3.64 % | +2.63 pp | 1.02 % | +0.01 pp |
| Achieved Occupancy (%) | 12.48 % | 12.26 % | -0.22 pp | 12.42 % | -0.06 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 51.79 % | 76.96 % | +25.17 pp | 53.18 % | +1.39 pp |
| IPC Active | 1.93 inst/cycle | 0.91 inst/cycle | -1.02 | 1.87 inst/cycle | -0.06 |

---

## Cumulative Progress

```
Baseline:  7200000 ns  (1.000×)
Iter 2     7100000 ns  (1.014×)  — reduce_shared_memory_per_block_via_small_block_k_configs
Iter 3     3980000 ns  (1.809×)  — tf32_tensor_cores_with_ieee_precision_emulation
Iter 4     3170000 ns  (2.271×)  — tf32x3_tensor_cores_for_fp32_accurate_hmma

Final best: 4010000 ns  (1.7955×)
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
@@ -13,6 +14,14 @@

         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
         triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=2),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=2),
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=2),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=4),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=3, num_warps=8),
     ],
     key=['M', 'N', 'K'],
 )
@@ -46,37 +55,43 @@

     pid_m = group_id * GROUP_M + (pid % group_size)
     pid_n = (pid % width) // group_size
     
-    # Row/col offsets for this tile
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    rk = tl.arange(0, BLOCK_K)
-    
-    # Boundary-safe offsets
-    offs_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
-    offs_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
-    
-    # Pointers to first tiles
-    X_ptrs = X_ptr + offs_m[:, None] * stride_xm + rk[None, :] * stride_xk
-    # W is stored as (N, K) i.e. weight matrix transposed from (K, N)
-    W_ptrs = W_ptr + offs_n[None, :] * stride_wn + rk[:, None] * stride_wk
+    # Block pointer descriptors for TMA-friendly, conflict-free shared memory staging
+    # X is [M, K], load tile [BLOCK_M, BLOCK_K]
+    X_block_ptr = tl.make_block_ptr(
+        base=X_ptr,
+        shape=(M, K),
+        strides=(stride_xm, stride_xk),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    # W is stored as [N, K] row-major, reinterpreted as [K, N] with swapped strides.
+    # This loads tile [BLOCK_K, BLOCK_N] directly, eliminating tl.trans() overhead.
+    W_block_ptr = tl.make_block_ptr(
+        base=W_ptr,
+        shape=(K, N),
+        strides=(stride_wk, stride_wn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(1, 0),
+    )
     
     # Accumulate in float32
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
     
     for k in range(0, tl.cdiv(K, BLOCK_K)):
-        k_remaining = K - k * BLOCK_K
         # Load X tile [BLOCK_M, BLOCK_K]
-        x_mask = (offs_m[:, None] < M) & (rk[None, :] < k_remaining)
-        x = tl.load(X_ptrs, mask=x_mask, other=0.0)
+        x = tl.load(X_block_ptr, boundary_check=(0, 1), padding_option='zero')
         
-        # Load W tile [BLOCK_K, BLOCK_N]
-        w_mask = (rk[:, None] < k_remaining) & (offs_n[None, :] < N)
-        w = tl.load(W_ptrs, mask=w_mask, other=0.0)
+        # Load W tile [BLOCK_K, BLOCK_N] directly (no tl.trans needed)
+        w = tl.load(W_block_ptr, boundary_check=(0, 1), padding_option='zero')
         
-        acc = tl.dot(x, w, acc, allow_tf32=False)
+        # tl.dot(x[BLOCK_M,BLOCK_K], w[BLOCK_K,BLOCK_N]) -> acc[BLOCK_M,BLOCK_N]
+        # input_precision='tf32x3': Ozaki splitting for FP32-accurate Tensor Core HMMA
+        acc = tl.dot(x, w, acc, input_precision='tf32x3')
         
-        X_ptrs += BLOCK_K * stride_xk
-        W_ptrs += BLOCK_K * stride_wk
+        X_block_ptr = tl.advance(X_block_ptr, (0, BLOCK_K))
+        W_block_ptr = tl.advance(W_block_ptr, (BLOCK_K, 0))
     
     # Recompute rm, rn for output
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -96,7 +111,6 @@

     out_mask = (rm[:, None] < M) & (rn[None, :] < N)
     Out_ptrs = Out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
     tl.store(Out_ptrs, result.to(tl.float32), mask=out_mask)
-
 
 def kernel_function(p_linear_weight, p_linear_bias, x):
     # x: [M, K], weight: [N, K] (linear weight is stored as [out, in])
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
