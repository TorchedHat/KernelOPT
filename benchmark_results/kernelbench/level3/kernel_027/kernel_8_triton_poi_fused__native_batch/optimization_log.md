# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_8` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 18080 ns — **Best latency**: 33340 ns (**0.5423×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_8`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_027/kernel_8_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (25088,1,1), Waves/SM: 11.88
- **Registers/thread**: 16
- **Duration**: 18080 ns
- **SM Throughput**: 33.25 % | **Memory**: 65.29 % | **DRAM**: 65.29 %
- **Achieved Occupancy**: 55.32 %
- **IPC Active**: 1.50 inst/cycle | **No Eligible**: 62.21 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_evict_first_cache_policy_with_larger_xblock | 1.123× | 0.542× | ✓ | **✓ best** |
| B | vectorized_loads_with_larger_xblock | 0.795× | 0.712× | ✓ |  |
| C | memory_coalescing_increase_xblock_autotune | 1.003× | 0.997× | ✓ |  |
| D | increase_xblock_tile_size_with_autotune | 0.794× | 0.708× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_evict_f... | Δ | vectorized_loads_... | Δ | memory_coalescing... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 18080 ns | 33340 ns | +15260 ns (+84.4%) | 25380 ns | +7300 ns (+40.4%) | 18140 ns | +60 ns (+0.3%) | 25540 ns | +7460 ns (+41.3%) |
| SM Throughput (%) | 33.25 % | 55.63 % | +22.38 pp | 34.99 % | +1.74 pp | 36.86 % | +3.61 pp | 40.94 % | +7.69 pp |
| Memory Throughput (%) | 65.29 % | 33.16 % | -32.13 pp | 44.76 % | -20.53 pp | 64.94 % | -0.35 pp | 44.47 % | -20.82 pp |
| DRAM Throughput (%) | 65.29 % | 33.16 % | -32.13 pp | 44.76 % | -20.53 pp | 64.94 % | -0.35 pp | 44.47 % | -20.82 pp |
| Achieved Occupancy (%) | 55.32 % | 82.21 % | +26.89 pp | 55.16 % | -0.16 pp | 56.56 % | +1.24 pp | 56.12 % | +0.80 pp |
| Registers/Thread | 16.00 register/thread | 28.00 register/thread | +12 | 26.00 register/thread | +10 | 17.00 register/thread | +1 | 32.00 register/thread | +16 |
| Waves/SM | 11.88 | 23.76 | +11.88 | 11.88 | +0.00 | 11.88 | +0.00 | 11.88 | +0.00 |
| No Eligible (%) | 62.21 % | 42.12 % | -20.09 pp | 68.06 % | +5.85 pp | 58.15 % | -4.06 pp | 62.74 % | +0.53 pp |
| IPC Active | 1.50 inst/cycle | 2.31 inst/cycle | +0.81 | 1.28 inst/cycle | -0.22 | 1.66 inst/cycle | +0.16 | 1.47 inst/cycle | -0.03 |


### Key Learnings

#### ❌ Plan B: `vectorized_loads_with_larger_xblock` (0.712×)

Adding @triton.autotune with larger XBLOCK configs (512/1024/2048) and tl.max_contiguous/tl.multiple_of vectorization hints caused a 29% slowdown. The register count per thread increased from 16 to 26, which reduced Block Limit Registers from 32 to 16, cutting the effective occupancy headroom. Despite DRAM and L2 throughput dropping (65%→45% and 80%→53%), elapsed cycles grew from 27017 to 37986 and executed instructions increased by ~20%, indicating the autotune overhead and register pressure increase outweighed any memory access improvement. The stall cycles per issued instruction worsened (23.7→27.1 cycles), suggesting the larger tile size did not improve memory-level parallelism but instead increased register spill pressure and instruction count.

> **Applicable when**: Never — this combination caused regression in all tested configurations

> **Avoid when**: When the kernel already has low register count (≤16 regs/thread) and the block limit is register-constrained; increasing XBLOCK via autotune with tl.max_contiguous/tl.multiple_of hints can inflate register usage (16→26 regs/thread), halving Block Limit Registers (32→16) and increasing stall cycles. Also avoid when the kernel is already achieving ~55% occupancy and the bottleneck is L1TEX stalls — larger tiles do not improve cache hit rate if the access pattern has no spatial reuse across the tile.

#### ❌ Plan D: `increase_xblock_tile_size_with_autotune` (0.708×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) to a memory-bound pointwise kernel caused a 41% slowdown (18.08 µs → 25.54 µs). The register count doubled from 16 to 32 per thread (Block Limit Registers dropped from 32 to 16 blocks), yet achieved occupancy barely changed (55.3% → 56.1%) and eligible warps per scheduler improved only marginally (0.60 → 0.69). The dominant effect was a 39% increase in executed instructions (4.7M → 6.6M) with no improvement in warp stall cycles (23.71 → 23.72 cycles/instruction), indicating the autotune selected a config that added register pressure and instruction overhead without meaningfully hiding L1TEX latency. When the kernel is already at ~12 waves/SM and the stall is L1TEX scoreboard (not occupancy-limited), increasing XBLOCK inflates register usage and instruction count without improving memory latency hiding.

> **Applicable when**: Kernel has very low register usage (<16/thread), large grid with many waves (>20), and L1TEX stall is caused by insufficient in-flight requests rather than register pressure. Eligible warps/scheduler is very low (<0.3) AND occupancy is genuinely register-limited.

> **Avoid when**: Kernel already uses ~12 waves/SM with moderate occupancy (~55%); L1TEX stall cycles per instruction are unchanged despite occupancy changes; register count is already at 16/thread and doubling it would halve Block Limit Registers; xnumel is not cleanly divisible by the new XBLOCK sizes (requiring masking overhead); the kernel is a simple pointwise with no shared memory reuse to benefit from larger tiles.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | independent_load_reordering_to_hide_l1tex_latency | 1.106× | 1.000× | ✓ |  |
| B | restore_small_xblock_increase_num_warps_for_latency_hiding | 1.001× | 1.002× | ✓ |  |
| C | reduce_num_warps_to_increase_concurrent_blocks_per_sm | 1.106× | 0.998× | ✓ |  |
| D | reduce_register_pressure_restore_small_xblock_remove_evict_first | 1.006× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | independent_load_... | Δ | restore_small_xbl... | Δ | reduce_num_warps_... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 18080 ns | 18080 ns | +0 ns (+0.0%) | 18050 ns | -30 ns (-0.2%) | 18110 ns | +30 ns (+0.2%) | 18080 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 33.25 % | 36.81 % | +3.56 pp | 26.64 % | -6.61 pp | 14.31 % | -18.94 pp | 33.26 % | +0.01 pp |
| Memory Throughput (%) | 65.29 % | 65.40 % | +0.11 pp | 65.41 % | +0.12 pp | 64.20 % | -1.09 pp | 65.20 % | -0.09 pp |
| DRAM Throughput (%) | 65.29 % | 65.40 % | +0.11 pp | 65.41 % | +0.12 pp | 64.20 % | -1.09 pp | 65.20 % | -0.09 pp |
| Achieved Occupancy (%) | 55.32 % | 58.72 % | +3.40 pp | 58.93 % | +3.61 pp | 27.87 % | -27.45 pp | 56.81 % | +1.49 pp |
| Registers/Thread | 16.00 register/thread | 17.00 register/thread | +1 | 16.00 register/thread | +0 | 32.00 register/thread | +16 | 16.00 register/thread | +0 |
| Waves/SM | 11.88 | 11.88 | +0.00 | 11.88 | +0.00 | 5.94 | -5.94 | 11.88 | +0.00 |
| No Eligible (%) | 62.21 % | 58.04 % | -4.17 pp | 69.43 % | +7.22 pp | 85.47 % | +23.26 pp | 62.91 % | +0.70 pp |
| IPC Active | 1.50 inst/cycle | 1.61 inst/cycle | +0.11 | 1.18 inst/cycle | -0.32 | 0.57 inst/cycle | -0.93 | 1.50 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_fixed_xblock_latency_hiding | 0.998× | 0.961× | ✓ |  |
| B | evict_last_cache_policy_on_reused_row_loads_fixed_xblock | 1.113× | 0.951× | ✓ |  |
| C | increase_num_warps_fixed_xblock_for_l1tex_latency_hiding | 0.803× | 0.774× | ✓ |  |
| D | increase_warps_per_block_for_l1tex_latency_hiding_fixed_xblock | 1.106× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | evict_last_cache_... | Δ | increase_num_warp... | Δ | increase_warps_pe... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 18080 ns | 18820 ns | +740 ns (+4.1%) | 19010 ns | +930 ns (+5.1%) | 23360 ns | +5280 ns (+29.2%) | 18050 ns | -30 ns (-0.2%) |
| SM Throughput (%) | 33.25 % | 26.73 % | -6.52 pp | 45.33 % | +12.08 pp | 53.74 % | +20.49 pp | 21.78 % | -11.47 pp |
| Memory Throughput (%) | 65.29 % | 62.92 % | -2.37 pp | 62.25 % | -3.04 pp | 50.56 % | -14.73 pp | 64.26 % | -1.03 pp |
| DRAM Throughput (%) | 65.29 % | 62.92 % | -2.37 pp | 62.25 % | -3.04 pp | 50.56 % | -14.73 pp | 64.26 % | -1.03 pp |
| Achieved Occupancy (%) | 55.32 % | 58.44 % | +3.12 pp | 58.17 % | +2.85 pp | 75.99 % | +20.67 pp | 48.07 % | -7.25 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 17.00 register/thread | +1 | 16.00 register/thread | +0 | 22.00 register/thread | +6 |
| Waves/SM | 11.88 | 11.88 | +0.00 | 11.88 | +0.00 | 23.76 | +11.88 | 5.94 | -5.94 |
| No Eligible (%) | 62.21 % | 69.52 % | +7.31 pp | 48.33 % | -13.88 pp | 39.72 % | -22.49 pp | 75.16 % | +12.95 pp |
| IPC Active | 1.50 inst/cycle | 1.21 inst/cycle | -0.29 | 2.02 inst/cycle | +0.52 | 2.35 inst/cycle | +0.85 | 0.98 inst/cycle | -0.52 |


### Key Learnings

#### ❌ Plan C: `increase_num_warps_fixed_xblock_for_l1tex_latency_hiding` (0.774×)

Increasing num_warps from 4 to 8 (via @triton.autotune with XBLOCK=128) doubled the block size from 128 to 256 threads, which doubled waves_per_SM (11.88→23.76) and more than doubled executed instructions (3.8M→9.6M). Although eligible warps/scheduler improved substantially (0.45→1.47) and achieved occupancy rose (58.9%→76.0%), the instruction count explosion caused duration to increase from 18.05 µs to 23.36 µs — a 29% regression. The L1/TEX hit rate surprisingly jumped from 0% to 44.4%, indicating the larger block size caused intra-block data reuse, but this was insufficient to offset the 2.5x instruction count increase. The key failure mode: when autotune doubles block_size (128→256) instead of keeping XBLOCK fixed, the kernel processes twice as many elements per block, doubling total work and instruction count even though registers per thread stayed at 16.

> **Applicable when**: Never — this approach caused regression. Avoid using @triton.autotune with num_warps>4 when XBLOCK is fixed and the grid is already well-saturated (~12 waves/SM), as the autotune framework doubles block_size (threads/block) which doubles instruction count.

> **Avoid when**: Do not increase num_warps via @triton.autotune when: (1) XBLOCK is fixed and the kernel already covers all xnumel elements with the current grid — doubling num_warps doubles block_size and thus doubles total instructions; (2) the kernel is already at ~12 waves/SM with moderate occupancy; (3) the grid size is fixed (not lambda-based) — the extra warps do not reduce grid size but do double per-block work. Even if eligible warps/scheduler improves, the instruction count explosion dominates.

---

## Cumulative Progress

```
Baseline:  18080 ns  (1.000×)

Final best: 33340 ns  (0.5423×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,26 +5,25 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 3211264
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
     x0 = (xindex % 128)
     x1 = ((xindex // 128) % 56)
     x2 = xindex // 7168
     x3 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0 + 256*x1 + 28672*x2), None)
-    tmp1 = tl.load(in_ptr0 + (128 + x0 + 256*x1 + 28672*x2), None)
-    tmp3 = tl.load(in_ptr0 + (14336 + x0 + 256*x1 + 28672*x2), None)
-    tmp5 = tl.load(in_ptr0 + (14464 + x0 + 256*x1 + 28672*x2), None)
+    base = x0 + 256*x1 + 28672*x2
+    tmp0 = tl.load(in_ptr0 + base)
+    tmp1 = tl.load(in_ptr0 + base + 128)
+    tmp3 = tl.load(in_ptr0 + base + 14336)
+    tmp5 = tl.load(in_ptr0 + base + 14464)
     tmp2 = tl.maximum(tmp0, tmp1)
     tmp4 = tl.maximum(tmp2, tmp3)
     tmp6 = tl.maximum(tmp4, tmp5)
-    tl.store(out_ptr0 + (x3), tmp6, None)
-
-
+    tl.store(out_ptr0 + x3, tmp6)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
