# Kernel Optimization Log

## Summary

- **Kernel**: `gemm_bias_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7780000 ns — **Best latency**: 7990000 ns (**0.9737×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `gemm_bias_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_062/synthesis/kernel_0_fused_fused_addmm.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 1.94
- **Registers/thread**: 255
- **Duration**: 7780000 ns
- **SM Throughput**: 42.39 % | **Memory**: 58.56 % | **DRAM**: 0.95 %
- **Achieved Occupancy**: 12.50 %
- **IPC Active**: 1.80 inst/cycle | **No Eligible**: 54.97 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | fix_w_tile_access_pattern_and_reduce_register_pressure | 1.000× | 1.000× | ✓ |  |
| D | fix_w_tile_load_coalescing_via_transpose | 1.000× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_w_tile_access... | Δ | fix_w_tile_load_c... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7780000 ns | 7780000 ns | +0 ns (+0.0%) | 7780000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 42.39 % | 42.55 % | +0.16 pp | 42.42 % | +0.03 pp |
| Memory Throughput (%) | 58.56 % | 58.77 % | +0.21 pp | 58.59 % | +0.03 pp |
| DRAM Throughput (%) | 0.95 % | 0.95 % | +0.00 pp | 0.95 % | +0.00 pp |
| Achieved Occupancy (%) | 12.50 % | 12.50 % | +0.00 pp | 12.50 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 54.97 % | 55.05 % | +0.08 pp | 54.92 % | -0.05 pp |
| IPC Active | 1.80 inst/cycle | 1.80 inst/cycle | +0.00 | 1.80 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reduce_register_pressure_via_maxnreg_to_improve_occupancy | 0.796× | 0.180× | ✓ |  |
| C | reduce_register_pressure_via_smaller_tiles_and_num_warps | 1.016× | 0.594× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7780000 ns | 43100000 ns | +35320000 ns (+454.0%) | 13090000 ns | +5310000 ns (+68.3%) |
| SM Throughput (%) | 42.39 % | 19.01 % | -23.38 pp | 24.31 % | -18.08 pp |
| Memory Throughput (%) | 58.56 % | 54.43 % | -4.13 pp | 93.86 % | +35.30 pp |
| DRAM Throughput (%) | 0.95 % | 32.77 % | +31.82 pp | 3.30 % | +2.35 pp |
| Achieved Occupancy (%) | 12.50 % | 16.90 % | +4.40 pp | 18.15 % | +5.65 pp |
| Registers/Thread | 255.00 register/thread | 128.00 register/thread | -127 | 158.00 register/thread | -97 |
| Waves/SM | 1.94 | 1.29 | -0.65 | 5.17 | +3.23 |
| No Eligible (%) | 54.97 % | 85.71 % | +30.74 pp | 74.76 % | +19.79 pp |
| IPC Active | 1.80 inst/cycle | 0.57 inst/cycle | -1.23 | 1.01 inst/cycle | -0.79 |


### Key Learnings

#### ❌ Plan C: `reduce_register_pressure_via_smaller_tiles_and_num_warps` (0.594×)

Reducing tile sizes (e.g. BLOCK_M=32/64, BLOCK_N=64/128) and num_warps (from 4/8 to 2) to lower register pressure in a Triton GEMM kernel backfired: registers/thread dropped from 255 to 158 and occupancy improved from 12.5% to 18.75%, but the autotuner selected a small-tile config that dramatically worsened shared memory bank conflicts (5.4-way → 12.0-way) and increased warp stall cycles (4.43 → 11.56 cycles/instruction), causing a 1.68x slowdown. The smaller tiles reduced L2 hit rate (96% → 73.5%) and caused the kernel to become L1/TEX memory-bound (93.86% throughput) rather than compute-bound, with compute throughput dropping from 42.55% to 24.31%. When the primary bottleneck is register-limited occupancy in a GEMM kernel, simply shrinking tiles to reduce register pressure can trade one bottleneck for a worse one: increased bank conflicts and cache thrashing that more than offset the occupancy gain.

> **Applicable when**: Never: this specific strategy (adding small-tile num_warps=2 configs to a register-limited Triton GEMM) caused regression. The occupancy gain was outweighed by bank conflict explosion and cache thrashing.

> **Avoid when**: When the kernel already has significant shared memory bank conflicts (>5-way): reducing tile sizes to improve occupancy will worsen bank conflicts further (5.4-way → 12.0-way observed), increasing warp stall cycles from 4.43 to 11.56 and causing overall slowdown. Also avoid when L2 hit rate is high (>90%): smaller tiles reduce temporal reuse and degrade L2 hit rate, increasing effective memory traffic.

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | eliminate_smem_bank_conflicts_via_block_ptr_and_prune_autotune_configs | 1.037× | 0.974× | ✓ | **✓ best** |
| D | enable_tf32_with_evict_first_streaming_tiles | 0.795× | 0.180× | ✓ |  |
| E | add_triton_autotune | 0.940× | 0.939× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_smem_ba... | Δ | enable_tf32_with_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7780000 ns | 7990000 ns | +210000 ns (+2.7%) | 43150000 ns | +35370000 ns (+454.6%) | 8290000 ns | +510000 ns (+6.6%) |
| SM Throughput (%) | 42.39 % | 41.80 % | -0.59 pp | 19.03 % | -23.36 pp | 36.45 % | -5.94 pp |
| Memory Throughput (%) | 58.56 % | 57.75 % | -0.81 pp | 54.37 % | -4.19 pp | 82.07 % | +23.51 pp |
| DRAM Throughput (%) | 0.95 % | 0.96 % | +0.01 pp | 32.79 % | +31.84 pp | 1.08 % | +0.13 pp |
| Achieved Occupancy (%) | 12.50 % | 12.50 % | +0.00 pp | 16.79 % | +4.29 pp | 12.50 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 128.00 register/thread | -127 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.29 | -0.65 | 3.88 | +1.94 |
| No Eligible (%) | 54.97 % | 54.69 % | -0.28 pp | 85.76 % | +30.79 pp | 62.30 % | +7.33 pp |
| IPC Active | 1.80 inst/cycle | 1.81 inst/cycle | +0.01 | 0.57 inst/cycle | -1.23 | 1.51 inst/cycle | -0.29 |


### Key Learnings

#### ❌ Plan D: `enable_tf32_with_evict_first_streaming_tiles` (0.180×)

Adding maxnreg constraints to autotune configs (to reduce registers/thread from 158 to 128) combined with eviction_policy='evict_first' on tile loads and BLOCK_K=128 configs caused a 5.5x slowdown (13.09ms → 43.15ms). The register cap forced heavy spill to local memory: DRAM throughput jumped from 3.3% to 32.79%, L2 compression input sectors grew ~2800x (from ~1M to ~2.97B sectors), and warp cycles per issued instruction increased from 11.56 to 18.88 — all consistent with severe L1TEX stall from spill/fill traffic. The CPIStall rule confirmed 70.9% of stall cycles attributed to L1TEX dependency. Despite the autotuner selecting a config with fewer bank conflicts (12-way → 5.6-way), the spill traffic completely overwhelmed any benefit, and the reduced grid (4096 → 512 blocks) also dropped waves_per_sm from 5.17 to 1.29, adding a severe partial-wave tail penalty.

> **Applicable when**: Never: this combined strategy (maxnreg caps + evict_first + BLOCK_K=128 configs) is a dead end for this kernel. The maxnreg cap is the primary cause of regression via register spill.

> **Avoid when**: When the kernel has large accumulation tiles (BLOCK_M x BLOCK_N fp32 accumulators) that inherently require many registers — capping maxnreg forces spill to local memory, increasing DRAM traffic by 10-30x and warp stall cycles by 1.5-4x. Also do not apply when the current grid already provides good wave coverage (waves_per_sm > 3): reducing grid size via larger tiles with maxnreg will collapse wave coverage and add a severe partial-wave tail penalty. evict_first alone is harmless but provides no benefit when the bottleneck is register spill rather than L2 cache pollution.

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  7780000 ns  (1.000×)

Final best: 7990000 ns  (0.9737×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,16 +4,19 @@

 from triton.language.extra import libdevice
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
-        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=8),
     ],
     key=['M', 'N', 'K'],
 )
@@ -49,11 +52,20 @@

         x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
         x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
 
-        # W has shape [N, K], stride_wn=K, stride_wk=1
-        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
-        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
-        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
-
+        # Load W as [BLOCK_K, BLOCK_N] via block_ptr treating W as W^T with shape [K, N]
+        # W is [N, K] row-major: stride_wn=K=8192 (slow), stride_wk=1 (fast)
+        # Treating as W^T [K, N]: strides are (stride_wk=1, stride_wn=8192)
+        # order=(1,0) = column-major hint enables Hopper TMA swizzling
+        w_block_ptr = tl.make_block_ptr(
+            base=W_ptr,
+            shape=(K, N),
+            strides=(stride_wk, stride_wn),
+            offsets=(k * BLOCK_K, pid_n * BLOCK_N),
+            block_shape=(BLOCK_K, BLOCK_N),
+            order=(1, 0),
+        )
+        w_tile = tl.load(w_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        # w_tile shape: [BLOCK_K, BLOCK_N] — no .T needed
         acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
 
     # Add bias
@@ -75,6 +87,7 @@

     key=['group_size'],
 )
 @triton.jit
+
 def groupnorm_leakyrelu_add_kernel(
     X_ptr, GN_W_ptr, GN_B_ptr, Out_ptr,
     num_rows, N, num_groups, group_size,
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
