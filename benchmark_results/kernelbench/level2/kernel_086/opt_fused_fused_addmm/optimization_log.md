# Kernel Optimization Log

## Summary

- **Kernel**: `linear_div_gelu_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 11020000 ns — **Best latency**: 10990000 ns (**1.0027×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`reduce_register_pressure_and_add_eviction_policies`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `linear_div_gelu_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_086/synthesis/kernel_0_fused_fused_addmm.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 1.94
- **Registers/thread**: 255
- **Duration**: 11020000 ns
- **SM Throughput**: 29.03 % | **Memory**: 44.35 % | **DRAM**: 1.09 %
- **Achieved Occupancy**: 12.33 %
- **IPC Active**: 1.24 inst/cycle | **No Eligible**: 68.95 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reduce_register_pressure_via_tma_and_pipeline_tuning | 0.346× | 0.345× | ✓ |  |
| C | reduce_register_pressure_and_add_eviction_policies | 1.006× | 1.003× | ✓ | **✓ best** |
| D | reduce_register_pressure_via_smaller_tile_configs | 1.005× | 0.429× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | reduce_register_p... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 11020000 ns | 31970000 ns | +20950000 ns (+190.1%) | 10990000 ns | -30000 ns (-0.3%) | 25680000 ns | +14660000 ns (+133.0%) |
| SM Throughput (%) | 29.03 % | 10.91 % | -18.12 pp | 29.27 % | +0.24 pp | 13.17 % | -15.86 pp |
| Memory Throughput (%) | 44.35 % | 43.59 % | -0.76 pp | 44.73 % | +0.38 pp | 95.66 % | +51.31 pp |
| DRAM Throughput (%) | 1.09 % | 0.58 % | -0.51 pp | 1.09 % | +0.00 pp | 1.08 % | -0.01 pp |
| Achieved Occupancy (%) | 12.33 % | 6.25 % | -6.08 pp | 12.33 % | +0.00 pp | 18.42 % | +6.09 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 168.00 register/thread | -87 |
| Waves/SM | 1.94 | 15.52 | +13.58 | 1.94 | +0.00 | 10.34 | +8.40 |
| No Eligible (%) | 68.95 % | 88.74 % | +19.79 pp | 68.91 % | -0.04 pp | 86.61 % | +17.66 pp |
| IPC Active | 1.24 inst/cycle | 0.45 inst/cycle | -0.79 | 1.24 inst/cycle | +0.00 | 0.54 inst/cycle | -0.70 |

---

## Iteration 2

**Best so far**: 1.0027× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_weight_tile_load_layout_to_eliminate_shared_memory_bank_conflicts | 1.004× | 1.004× | ✓ |  |
| B | fix_shared_memory_bank_conflicts_via_w_tile_transpose | 1.004× | 0.429× | ✓ |  |
| C | fix_shared_memory_bank_conflicts_via_smem_padding | 1.004× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_weight_tile_l... | Δ | fix_shared_memory... | Δ | fix_shared_memory... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 11020000 ns | 10980000 ns | -40000 ns (-0.4%) | 25680000 ns | +14660000 ns (+133.0%) | 11010000 ns | -10000 ns (-0.1%) |
| SM Throughput (%) | 29.03 % | 29.19 % | +0.16 pp | 13.17 % | -15.86 pp | 29.20 % | +0.17 pp |
| Memory Throughput (%) | 44.35 % | 44.63 % | +0.28 pp | 95.66 % | +51.31 pp | 44.62 % | +0.27 pp |
| DRAM Throughput (%) | 1.09 % | 1.06 % | -0.03 pp | 1.08 % | -0.01 pp | 1.08 % | -0.01 pp |
| Achieved Occupancy (%) | 12.33 % | 12.41 % | +0.08 pp | 18.42 % | +6.09 pp | 12.37 % | +0.04 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 168.00 register/thread | -87 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 10.34 | +8.40 | 1.94 | +0.00 |
| No Eligible (%) | 68.95 % | 68.99 % | +0.04 pp | 86.61 % | +17.66 pp | 68.81 % | -0.14 pp |
| IPC Active | 1.24 inst/cycle | 1.24 inst/cycle | +0.00 | 0.54 inst/cycle | -0.70 | 1.24 inst/cycle | +0.00 |


### Key Learnings

#### ❌ Plan B: `fix_shared_memory_bank_conflicts_via_w_tile_transpose` (0.429×)

Changing the weight tile load from a strided [BLOCK_K, BLOCK_N] layout to a row-major [BLOCK_N, BLOCK_K] layout followed by tl.trans() was intended to eliminate shared memory bank conflicts, but instead worsened them dramatically: bank conflict rate rose from 58.79% to 83.33% (5.4-way to 18-way conflicts) and uncoalesced shared wavefronts increased from 56% to 76%. The autotuner selected smaller tile configs (block_size 128→64, grid_size 512→8192) alongside the transpose change, which together caused L1/TEX cache throughput to saturate at 97% (up from 47%) and Mem Busy to reach 96% (up from 45%), tripling Warp Cycles Per Issued Instruction from 6.37 to 22.02 and causing a 2.3x slowdown. The tl.trans() approach did not eliminate the shared memory staging bottleneck — Triton still routed the transposed tile through shared memory with a conflicting access pattern, and the smaller tiles selected by autotuning amplified the per-element overhead of those conflicts.

> **Applicable when**: Never — this strategy caused a 2.3x regression. Do not apply tl.trans() as a fix for shared memory bank conflicts in Triton GEMM kernels when the autotuner is free to select smaller tile sizes, as smaller tiles amplify the per-element conflict overhead.

> **Avoid when**: When SharedMemoryConflicts bank conflict rate is already above 50% of wavefronts: tl.trans() does not eliminate Triton's shared memory staging for tl.dot operands and can worsen conflicts when combined with smaller tile configs selected by autotuning. Also avoid when the autotuner config space includes small tiles (BLOCK_N/BLOCK_K < 64), as these amplify bank conflict overhead relative to useful compute.

---

## Iteration 3

**Best so far**: 1.0027× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cap_register_usage_via_maxnreg_to_improve_occupancy | 0.981× | 0.199× | ✓ |  |
| B | cast_dot_operands_to_fp16_for_tensor_core_hmma | 0.894× | 0.199× | ✓ |  |
| C | revert_w_tile_transpose_and_expand_large_tile_autotune_configs | 1.004× | 0.429× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_register_usag... | Δ | cast_dot_operands... | Δ | revert_w_tile_tra... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 11020000 ns | 55450000 ns | +44430000 ns (+403.2%) | 55450000 ns | +44430000 ns (+403.2%) | 25690000 ns | +14670000 ns (+133.1%) |
| SM Throughput (%) | 29.03 % | 10.85 % | -18.18 pp | 10.84 % | -18.19 pp | 13.17 % | -15.86 pp |
| Memory Throughput (%) | 44.35 % | 23.68 % | -20.67 pp | 23.69 % | -20.66 pp | 95.68 % | +51.33 pp |
| DRAM Throughput (%) | 1.09 % | 16.38 % | +15.29 pp | 16.37 % | +15.28 pp | 1.08 % | -0.01 pp |
| Achieved Occupancy (%) | 12.33 % | 17.29 % | +4.96 pp | 17.35 % | +5.02 pp | 18.42 % | +6.09 pp |
| Registers/Thread | 255.00 register/thread | 128.00 register/thread | -127 | 128.00 register/thread | -127 | 168.00 register/thread | -87 |
| Waves/SM | 1.94 | 1.29 | -0.65 | 1.29 | -0.65 | 10.34 | +8.40 |
| No Eligible (%) | 68.95 % | 89.27 % | +20.32 pp | 89.34 % | +20.39 pp | 86.61 % | +17.66 pp |
| IPC Active | 1.24 inst/cycle | 0.43 inst/cycle | -0.81 | 0.43 inst/cycle | -0.81 | 0.54 inst/cycle | -0.70 |


### Key Learnings

#### ❌ Plan A: `cap_register_usage_via_maxnreg_to_improve_occupancy` (0.199×)

Adding maxnreg=128 to Triton autotune configs successfully reduced registers/thread from 255 to 128 and raised theoretical occupancy from 12.5% to 18.75%, but caused a catastrophic 5x slowdown (10.98ms → 55.45ms). The compiler responded to the register cap by spilling hot accumulator values to local memory: DRAM throughput jumped from 1.06% to 16.38%, memory throughput exploded from 51.87 GB/s to 804.96 GB/s, Warp Cycles Per Issued Instruction increased 4x (6.38 → 25.87), and CPIStall for L1TEX dependency grew from 32% to 84.8% of cycles. The stack size also grew from 1024 to 2432 bytes, confirming heavy spilling. When the kernel has a large accumulator tile (BLOCK_M × BLOCK_N fp32 values) that cannot fit in the capped register budget, maxnreg forces spills that far outweigh any occupancy benefit.

> **Applicable when**: Never for this kernel class — maxnreg causes catastrophic spill when the accumulator tile (BLOCK_M × BLOCK_N fp32 values) is large relative to the capped register budget. Do not apply when the kernel already has high DRAM throughput or when stack_size grows significantly after applying the cap.

> **Avoid when**: When the kernel has a large GEMM accumulator (e.g., BLOCK_M=128, BLOCK_N=128-256 fp32 values = 16K-32K bytes) that requires many registers to hold: capping maxnreg below the compiler's natural allocation will force the accumulator to spill to local memory, causing DRAM throughput to spike (here 1% → 16%), Warp Cycles Per Issued Instruction to multiply (6x → 25x), and CPIStall for L1TEX to dominate (84%+ of cycles). The occupancy gain (12.5% → 18.75%) is completely overwhelmed by the spill penalty. Also avoid when stack_size increases substantially after applying maxnreg — this is a reliable indicator of register spilling.

#### ❌ Plan B: `cast_dot_operands_to_fp16_for_tensor_core_hmma` (0.199×)

Adding maxnreg=128 to triton.Config successfully reduced registers/thread from 255 to 128 and raised Block Limit Registers from 2 to 4, improving theoretical occupancy from 12.5% to 18.75%. However, the register cap forced the compiler to spill variables to the local memory stack (stack size grew from 1024 to 2432 bytes), causing Warp Cycles Per Issued Instruction to explode from 6.38 to 25.86 cycles and CPIStall from L1TEX to jump from 32% to 84.8% of stall cycles. The net effect was a 5x slowdown (10.98ms → 55.45ms) because the increased occupancy was completely overwhelmed by the massive latency from local memory spill traffic. The instruction count also grew from 2.53B to 3.44B, indicating significant spill-reload overhead. This anti-pattern occurs when maxnreg is set below the compiler's natural register allocation for a kernel with complex epilogues (GELU + division), forcing spills that are far more expensive than the occupancy gain.

> **Applicable when**: Never — this strategy caused a 5x regression. Do not apply maxnreg to cap registers below the compiler's natural allocation when the kernel has a complex epilogue (e.g., GELU, division, erf) that requires many live values simultaneously.

> **Avoid when**: When the kernel's natural register count is at or near 255 due to a complex epilogue (GELU, erf, division chains): capping registers with maxnreg will force local memory spills. The spill traffic (stack_size growth from 1024 → 2432+) causes L1TEX stall cycles to dominate (84%+ of all stall cycles), making the occupancy gain from lower register count irrelevant. Avoid maxnreg when stack_size increases by more than 2x after applying the cap.

#### ❌ Plan C: `revert_w_tile_transpose_and_expand_large_tile_autotune_configs` (0.429×)

Reverting the W-tile load from [BLOCK_N, BLOCK_K]+tl.trans() back to the strided [BLOCK_K, BLOCK_N] layout (no tl.trans) while simultaneously expanding the autotune config space with num_warps=8 large-tile configs failed to recover the baseline: the kernel still ran at 25.69ms vs the 10.98ms baseline. The profiling after is identical to the previous tl.trans() regression (iteration 2), indicating the autotuner re-selected the same small-tile configs (block_size=64, grid_size=8192, waves_per_sm=10.34) that saturate L1/TEX at 97% and Mem Busy at 96% with 18-way bank conflicts. The root cause is that the autotuner's selection of small tiles (driven by the expanded config space) dominates over the load-layout revert, and the structural shared memory bank conflict pattern persists regardless of whether tl.trans() is present.

> **Applicable when**: Never — this strategy caused a 2.3x regression identical to the prior iteration. Do not apply when the autotune config space includes small-tile configs (BLOCK_M/N <= 64) alongside large-tile configs, as the autotuner will select small tiles that amplify shared memory bank conflict overhead.

> **Avoid when**: When the autotune config space includes small tiles (BLOCK_M or BLOCK_N <= 64): the autotuner will prefer these configs due to lower per-block resource usage, but they cause more total shared memory traffic and higher bank conflict rates per unit of useful compute. Expanding the config space with num_warps=8 large-tile configs does not prevent the autotuner from selecting small-tile configs that cause L1/TEX saturation. The load-layout revert alone is insufficient if the autotuner config space is not constrained to large tiles only.

---

## Cumulative Progress

```
Baseline:  11020000 ns  (1.000×)
Iter 1     10990000 ns  (1.003×)  — reduce_register_pressure_and_add_eviction_policies

Final best: 10990000 ns  (1.0027×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -2,6 +2,7 @@

 import triton
 import triton.language as tl
 from triton.language.extra import libdevice
+
 
 
 @triton.autotune(
@@ -14,6 +15,15 @@

         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
         triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        # New small-tile configs to reduce register pressure
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=2),
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=2),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=2),
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=2),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=2),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=2),
     ],
     key=['M', 'N', 'K'],
 )
@@ -60,17 +70,18 @@

 
         # Load X tile: [BLOCK_M, BLOCK_K]
         x_ptrs = X_ptr + rm[:, None] * stride_xm + offs_k[None, :] * stride_xk
-        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
+        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0, eviction_policy='evict_last')
 
-        # Load W tile: [BLOCK_K, BLOCK_N] (W is N x K, we want K x N for dot)
-        # W_ptr[n, k] -> we need [k, n] layout for tl.dot
-        w_ptrs = W_ptr + rn[None, :] * stride_wn + offs_k[:, None] * stride_wk
-        w_tile = tl.load(w_ptrs, mask=mask_n[None, :] & mask_k[:, None], other=0.0)
+        # Load W tile: [BLOCK_N, BLOCK_K] -- COALESCED along K (fast axis for row-major W stored as [N, K])
+        # w_tile[n, k] = W[rn[n], offs_k[k]], then tl.trans gives [BLOCK_K, BLOCK_N] for tl.dot
+        w_ptrs = W_ptr + rn[:, None] * stride_wn + offs_k[None, :] * stride_wk
+        w_tile = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0, eviction_policy='evict_last')
 
-        acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=False)
+        # tl.trans() is a zero-cost metadata op: [BLOCK_N, BLOCK_K] -> [BLOCK_K, BLOCK_N] for tl.dot
+        acc = tl.dot(x_tile, tl.trans(w_tile), acc=acc, allow_tf32=False)
 
     # Add bias
-    bias = tl.load(B_ptr + rn, mask=mask_n, other=0.0)
+    bias = tl.load(B_ptr + rn, mask=mask_n, other=0.0, eviction_policy='evict_last')
     acc = acc + bias[None, :]
 
     # Divide by divisor
@@ -85,8 +96,7 @@

 
     # Store output
     out_ptrs = OUT_ptr + rm[:, None] * N + rn[None, :]
-    tl.store(out_ptrs, result.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])
-
+    tl.store(out_ptrs, result.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :], eviction_policy='evict_first')
 
 def kernel_function(weight, bias, x):
     # weight: (N, K) = (8192, 8192)
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
