# Kernel Optimization Log

## Summary

- **Kernel**: `matmul_bias_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7880000 ns — **Best latency**: 7880000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `matmul_bias_kernel`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_098/synthesis/kernel_0_fused_fused_addmm.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 1.94
- **Registers/thread**: 255
- **Duration**: 7880000 ns
- **SM Throughput**: 43.89 % | **Memory**: 60.50 % | **DRAM**: 1.00 %
- **Achieved Occupancy**: 12.46 %
- **IPC Active**: 1.85 inst/cycle | **No Eligible**: 53.76 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | enable_tf32_and_expand_autotune_with_large_block_k | 0.930× | 0.932× | ✓ |  |
| D | reduce_register_spilling_via_smaller_tiles | 0.914× | 0.325× | ✓ |  |


### Metric Comparison

| Metric | Baseline | enable_tf32_and_e... | Δ | reduce_register_s... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7880000 ns | 8450000 ns | +570000 ns (+7.2%) | 24280000 ns | +16400000 ns (+208.1%) |
| SM Throughput (%) | 43.89 % | 36.39 % | -7.50 pp | 14.35 % | -29.54 pp |
| Memory Throughput (%) | 60.50 % | 80.96 % | +20.46 pp | 95.55 % | +35.05 pp |
| DRAM Throughput (%) | 1.00 % | 1.06 % | +0.06 pp | 1.01 % | +0.01 pp |
| Achieved Occupancy (%) | 12.46 % | 12.50 % | +0.04 pp | 24.46 % | +12.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 96.00 register/thread | -159 |
| Waves/SM | 1.94 | 3.88 | +1.94 | 7.76 | +5.82 |
| No Eligible (%) | 53.76 % | 62.42 % | +8.66 pp | 85.21 % | +31.45 pp |
| IPC Active | 1.85 inst/cycle | 1.50 inst/cycle | -0.35 | 0.59 inst/cycle | -1.26 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | enable_tensor_cores_via_input_precision_ieee | 0.931× | 0.935× | ✓ |  |
| C | eliminate_shared_memory_bank_conflicts_via_k_padding | 0.956× | 0.599× | ✓ |  |
| D | eliminate_shared_memory_bank_conflicts_via_block_ptr | 0.959× | 0.325× | ✓ |  |


### Metric Comparison

| Metric | Baseline | enable_tensor_cor... | Δ | eliminate_shared_... | Δ | eliminate_shared_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7880000 ns | 8430000 ns | +550000 ns (+7.0%) | 13150000 ns | +5270000 ns (+66.9%) | 24280000 ns | +16400000 ns (+208.1%) |
| SM Throughput (%) | 43.89 % | 36.39 % | -7.50 pp | 24.70 % | -19.19 pp | 14.34 % | -29.55 pp |
| Memory Throughput (%) | 60.50 % | 80.94 % | +20.44 pp | 93.37 % | +32.87 pp | 95.49 % | +34.99 pp |
| DRAM Throughput (%) | 1.00 % | 1.06 % | +0.06 pp | 1.10 % | +0.10 pp | 1.01 % | +0.01 pp |
| Achieved Occupancy (%) | 12.46 % | 12.50 % | +0.04 pp | 18.17 % | +5.71 pp | 24.46 % | +12.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 164.00 register/thread | -91 | 96.00 register/thread | -159 |
| Waves/SM | 1.94 | 3.88 | +1.94 | 5.17 | +3.23 | 7.76 | +5.82 |
| No Eligible (%) | 53.76 % | 62.42 % | +8.66 pp | 74.42 % | +20.66 pp | 85.21 % | +31.45 pp |
| IPC Active | 1.85 inst/cycle | 1.50 inst/cycle | -0.35 | 1.02 inst/cycle | -0.83 | 0.59 inst/cycle | -1.26 |


### Key Learnings

#### ❌ Plan D: `eliminate_shared_memory_bank_conflicts_via_block_ptr` (0.325×)

The plan intended to use tl.make_block_ptr with order tuples to eliminate shared memory bank conflicts, but the actual diff only removed cache_modifier='.ca' from tl.load calls and replaced the autotune configs with smaller tiles — the block_ptr refactor was not implemented. The result was identical to the prior iteration's regression: smaller tiles increased grid size from 512 to 4096, worsened bank conflicts from 8-way to 17-way, saturated L1/TEX at 98.51%, and drove warp cycles per issued instruction from 5.32 to 26.46. Removing cache_modifier hints had no measurable positive effect; the regression was entirely caused by the tile size reduction.

> **Applicable when**: Never: this change (removing cache_modifier without implementing block_ptr) does not address bank conflicts and causes regression when combined with smaller tiles.

> **Avoid when**: Always avoid: removing cache_modifier='.ca' alone does not fix shared memory bank conflicts. Do not reduce tile sizes when SharedMemoryConflicts rule is already triggered — smaller tiles worsen bank conflict degree (8-way → 17-way) and saturate L1/TEX. The intended tl.make_block_ptr refactor was not actually implemented in the diff.

---

## Cumulative Progress

```
Baseline:  7880000 ns  (1.000×)

Final best: 7880000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,14 +5,19 @@

 
 
 # GEMM kernel: computes x @ W^T + bias, output shape [M, N]
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=5, num_warps=8),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
     ],
     key=['M', 'N', 'K'],
 )
@@ -49,8 +54,8 @@

         k_remaining = K - k * BLOCK_K
         a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
         b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
-        a = tl.load(A_ptrs, mask=a_mask, other=0.0)
-        b = tl.load(B_ptrs, mask=b_mask, other=0.0)
+        a = tl.load(A_ptrs, mask=a_mask, other=0.0, cache_modifier='.ca')
+        b = tl.load(B_ptrs, mask=b_mask, other=0.0, cache_modifier='.ca')
         acc = tl.dot(a, b, acc, allow_tf32=False)
         A_ptrs += BLOCK_K * stride_ak
         B_ptrs += BLOCK_K * stride_bk
@@ -65,9 +70,6 @@

     tl.store(C_ptrs, acc, mask=mask)
 
 
-# Fused kernel: avg_pool1d (kernel=16) + gelu + scale + max reduction
-# Input: linear output [M, N] where N=8192, pool_size=16 -> pooled_N=512
-# Output: max values [M]
 @triton.autotune(
     configs=[
         triton.Config({'XBLOCK': 1, 'R0_BLOCK': 512}, num_stages=2, num_warps=4),
@@ -80,6 +82,7 @@

     key=['xnumel', 'r0_numel'],
 )
 @triton.jit
+
 def pool_gelu_scale_max_kernel(
     in_ptr,
     out_max_ptr,
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
