# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_tanh_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 22720 ns — **Best latency**: 22720 ns (**1.0000×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_addmm_tanh_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_033/kernel_2_triton_poi_fused_addmm_tanh_3/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (32768,1,1), Waves/SM: 15.52
- **Registers/thread**: 16
- **Duration**: 22720 ns
- **SM Throughput**: 37.20 % | **Memory**: 19.96 % | **DRAM**: 15.12 %
- **Achieved Occupancy**: 45.07 %
- **IPC Active**: 1.66 inst/cycle | **No Eligible**: 58.35 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_via_autotune_to_reduce_launch_overhead_and_improve_bias_reuse | 0.893× | 0.922× | ✓ |  |
| B | increase_xblock_tile_size_with_autotune | 0.723× | 0.714× | ✓ |  |
| C | increase_xblock_with_autotune_for_broadcast_reuse | 0.891× | 0.920× | ✓ |  |
| D | vectorized_loads_larger_xblock | 1.001× | 0.997× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_v... | Δ | increase_xblock_t... | Δ | increase_xblock_w... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 22720 ns | 24640 ns | +1920 ns (+8.5%) | 31840 ns | +9120 ns (+40.1%) | 24700 ns | +1980 ns (+8.7%) | 22780 ns | +60 ns (+0.3%) |
| SM Throughput (%) | 37.20 % | 65.06 % | +27.86 pp | 85.86 % | +48.66 pp | 65.16 % | +27.96 pp | 36.84 % | -0.36 pp |
| Memory Throughput (%) | 19.96 % | 17.94 % | -2.02 pp | 19.08 % | -0.88 pp | 17.91 % | -2.05 pp | 19.89 % | -0.07 pp |
| DRAM Throughput (%) | 15.12 % | 13.95 % | -1.17 pp | 10.78 % | -4.34 pp | 13.92 % | -1.20 pp | 15.09 % | -0.03 pp |
| Achieved Occupancy (%) | 45.07 % | 47.54 % | +2.47 pp | 86.23 % | +41.16 pp | 47.78 % | +2.71 pp | 45.10 % | +0.03 pp |
| Registers/Thread | 16.00 register/thread | 18.00 register/thread | +2 | 27.00 register/thread | +11 | 18.00 register/thread | +2 | 16.00 register/thread | +0 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 58.35 % | 26.27 % | -32.08 pp | 7.78 % | -50.57 pp | 26.35 % | -32.00 pp | 58.22 % | -0.13 pp |
| IPC Active | 1.66 inst/cycle | 2.94 inst/cycle | +1.28 | 3.67 inst/cycle | +2.01 | 2.93 inst/cycle | +1.27 | 1.66 inst/cycle | +0.00 |


### Key Learnings

#### ❌ Plan B: `increase_xblock_tile_size_with_autotune` (0.714×)

Increasing XBLOCK tile size via @triton.autotune for a pointwise kernel successfully resolved the L1TEX scoreboard stall bottleneck (No Eligible warps dropped from 58.35% to 7.78%, achieved occupancy rose from 45% to 86%), but caused a 3.2x explosion in executed instructions (6.7M → 21.6M) and registers per thread jumped from 16 to 27. The net effect was a 40% slowdown despite dramatically better scheduler utilization, because the larger tile introduced substantially more computation (likely from predicated/masked operations and loop overhead at larger XBLOCK) that outweighed the latency-hiding benefit. When the original kernel already has very low register count (≤16) and the instruction count balloons by >3x after tiling, the compute overhead dominates over the latency-hiding gain.

> **Applicable when**: Kernel is latency-bound with high No Eligible warp % (>50%) from L1TEX stalls, low achieved occupancy (<50%), and very small tile size relative to problem size. Registers per thread are low (≤16) and instruction count is expected to remain stable with larger tiles.

> **Avoid when**: The xnumel is not evenly divisible by the larger XBLOCK values (requires masking), causing the autotune to select configs that introduce predicated/masked instructions and significantly inflate instruction count (>2x). Also avoid when the kernel's register count will increase substantially (>20%) with larger XBLOCK, as this can reduce theoretical occupancy and negate latency-hiding gains. Do not apply when instruction count inflation from masking overhead will dominate the latency-hiding benefit.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 1.002× | 1.000× | ✓ | **✓ best** |
| B | persistent_tile_scheduling | 1.000× | 1.000× | ✓ |  |
| C | persistent_tile_scheduling | 0.897× | 0.918× | ✓ |  |
| D | persistent_tile_scheduling_to_eliminate_instruction_inflation | 0.999× | 1.000× | ✓ |  |
| E | add_triton_autotune | 0.996× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 22720 ns | 22720 ns | +0 ns (+0.0%) | 22720 ns | +0 ns (+0.0%) | 24740 ns | +2020 ns (+8.9%) | 22720 ns | +0 ns (+0.0%) | 22720 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 37.20 % | 44.48 % | +7.28 pp | 48.09 % | +10.89 pp | 27.33 % | -9.87 pp | 43.73 % | +6.53 pp | 40.09 % | +2.89 pp |
| Memory Throughput (%) | 19.96 % | 20.05 % | +0.09 pp | 19.98 % | +0.02 pp | 17.78 % | -2.18 pp | 20.13 % | +0.17 pp | 20.03 % | +0.07 pp |
| DRAM Throughput (%) | 15.12 % | 15.13 % | +0.01 pp | 15.13 % | +0.01 pp | 13.91 % | -1.21 pp | 15.12 % | +0.00 pp | 15.12 % | +0.00 pp |
| Achieved Occupancy (%) | 45.07 % | 47.15 % | +2.08 pp | 49.25 % | +4.18 pp | 38.00 % | -7.07 pp | 46.96 % | +1.89 pp | 45.70 % | +0.63 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 21.00 register/thread | +5 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 58.35 % | 50.08 % | -8.27 pp | 45.96 % | -12.39 pp | 60.11 % | +1.76 pp | 50.85 % | -7.50 pp | 54.93 % | -3.42 pp |
| IPC Active | 1.66 inst/cycle | 1.99 inst/cycle | +0.33 | 2.15 inst/cycle | +0.49 | 1.43 inst/cycle | -0.23 | 1.95 inst/cycle | +0.29 | 1.78 inst/cycle | +0.12 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_scoreboard_stall_latency | 0.831× | 0.871× | ✓ |  |
| B | replace_integer_modulo_with_bitwise_and_for_bias_index_and_enable_fma_fusion | 0.998× | 0.993× | ✓ |  |
| C | increase_num_warps_for_l1tex_latency_hiding | 0.999× | 1.001× | ✓ |  |
| D | fix_persistent_kernel_grid_to_use_sm_count_not_xblock_granularity | 0.898× | 0.917× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | replace_integer_m... | Δ | increase_num_warp... | Δ | fix_persistent_ke... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 22720 ns | 26080 ns | +3360 ns (+14.8%) | 22880 ns | +160 ns (+0.7%) | 22690 ns | -30 ns (-0.1%) | 24770 ns | +2050 ns (+9.0%) |
| SM Throughput (%) | 37.20 % | 47.19 % | +9.99 pp | 45.80 % | +8.60 pp | 43.68 % | +6.48 pp | 27.28 % | -9.92 pp |
| Memory Throughput (%) | 19.96 % | 17.65 % | -2.31 pp | 19.85 % | -0.11 pp | 20.13 % | +0.17 pp | 17.79 % | -2.17 pp |
| DRAM Throughput (%) | 15.12 % | 13.16 % | -1.96 pp | 15.03 % | -0.09 pp | 15.14 % | +0.02 pp | 13.87 % | -1.25 pp |
| Achieved Occupancy (%) | 45.07 % | 56.47 % | +11.40 pp | 49.44 % | +4.37 pp | 47.03 % | +1.96 pp | 38.16 % | -6.91 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 21.00 register/thread | +5 |
| Waves/SM | 15.52 | 31.03 | +15.51 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 58.35 % | 38.68 % | -19.67 pp | 48.95 % | -9.40 pp | 50.98 % | -7.37 pp | 60.90 % | +2.55 pp |
| IPC Active | 1.66 inst/cycle | 2.31 inst/cycle | +0.65 | 2.04 inst/cycle | +0.38 | 1.96 inst/cycle | +0.30 | 1.44 inst/cycle | -0.22 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fma_fusion_via_explicit_fma_and_bitwise_bias_index | 0.998× | 1.001× | ✓ |  |
| B | fix_persistent_kernel_grid_to_sm_count_for_bias_l1_reuse | 1.000× | 1.000× | ✓ |  |
| D | preload_bias_reorder_loads_to_eliminate_l1tex_scoreboard_stall | 0.833× | 0.872× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fma_fusion_via_ex... | Δ | fix_persistent_ke... | Δ | preload_bias_reor... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 22720 ns | 22690 ns | -30 ns (-0.1%) | 22720 ns | +0 ns (+0.0%) | 26050 ns | +3330 ns (+14.7%) |
| SM Throughput (%) | 37.20 % | 41.49 % | +4.29 pp | 45.97 % | +8.77 pp | 45.24 % | +8.04 pp |
| Memory Throughput (%) | 19.96 % | 20.06 % | +0.10 pp | 20.03 % | +0.07 pp | 17.68 % | -2.28 pp |
| DRAM Throughput (%) | 15.12 % | 15.15 % | +0.03 pp | 15.12 % | +0.00 pp | 13.19 % | -1.93 pp |
| Achieved Occupancy (%) | 45.07 % | 47.66 % | +2.59 pp | 49.15 % | +4.08 pp | 56.36 % | +11.29 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 | 31.03 | +15.51 |
| No Eligible (%) | 58.35 % | 53.55 % | -4.80 pp | 48.48 % | -9.87 pp | 40.94 % | -17.41 pp |
| IPC Active | 1.66 inst/cycle | 1.86 inst/cycle | +0.20 | 2.03 inst/cycle | +0.37 | 2.24 inst/cycle | +0.58 |

---

## Cumulative Progress

```
Baseline:  22720 ns  (1.000×)

Final best: 22720 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,28 +5,40 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 128}, num_warps=16),
+    ],
+    key=['xnumel'],
+    restore_value=['in_out_ptr0'],
+)
 @triton.jit
-def triton_poi_fused_addmm_tanh_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
-    xnumel = 4194304
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
-    x0 = (xindex % 16384)
-    x2 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_out_ptr0 + (x2), None)
-    tmp2 = tmp0 + tmp1
-    tmp3 = libdevice.tanh(tmp2)
-    tl.store(in_out_ptr0 + (x2), tmp3, None)
-
-
+def triton_poi_fused_addmm_tanh_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
+    total_tiles = xnumel // XBLOCK  # = 32768, exact division (4194304 / 128)
+    num_sms = tl.num_programs(0)    # = 132 CTAs launched
+    pid = tl.program_id(0)          # 0..131
+    tile_id = pid
+    while tile_id < total_tiles:
+        xoffset = tile_id * XBLOCK
+        xindex = xoffset + tl.arange(0, XBLOCK)  # no mask needed: exact divisor
+        x0 = xindex % 16384  # bias index
+        # evict_last keeps bias hot in L1 across loop iterations
+        tmp0 = tl.load(in_ptr0 + x0, eviction_policy='evict_last')
+        tmp1 = tl.load(in_out_ptr0 + xindex)
+        tmp2 = tmp0 + tmp1
+        tmp3 = libdevice.tanh(tmp2)
+        tl.store(in_out_ptr0 + xindex, tmp3)
+        tile_id += num_sms
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 4194304
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_addmm_tanh_3[grid](arg_0, arg_1, 4194304, XBLOCK=128)
+    triton_poi_fused_addmm_tanh_3[grid](arg_0, arg_1, 4194304)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
