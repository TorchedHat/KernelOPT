# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 63170 ns — **Best latency**: 63200 ns (**0.9995×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_031/kernel_3_triton_poi_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (98304,1,1), Waves/SM: 46.55
- **Registers/thread**: 16
- **Duration**: 63170 ns
- **SM Throughput**: 31.89 % | **Memory**: 26.44 % | **DRAM**: 26.44 %
- **Achieved Occupancy**: 40.98 %
- **IPC Active**: 1.31 inst/cycle | **No Eligible**: 66.59 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_for_latency_hiding | 0.838× | 0.263× | ✓ |  |
| B | increase_xblock_with_autotune_to_hide_l1tex_latency | 0.888× | 0.926× | ✓ |  |
| C | autotune_xblock_and_num_warps_for_latency_hiding | 0.997× | 1.005× | ✓ |  |
| D | autotune_xblock_vectorized_loads | 0.800× | 0.834× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | increase_xblock_w... | Δ | autotune_xblock_a... | Δ | autotune_xblock_v... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 63170 ns | 240350 ns | +177180 ns (+280.5%) | 68190 ns | +5020 ns (+7.9%) | 62850 ns | -320 ns (-0.5%) | 75780 ns | +12610 ns (+20.0%) |
| SM Throughput (%) | 31.89 % | 79.13 % | +47.24 pp | 35.23 % | +3.34 pp | 35.63 % | +3.74 pp | 64.23 % | +32.34 pp |
| Memory Throughput (%) | 26.44 % | 22.74 % | -3.70 pp | 23.35 % | -3.09 pp | 26.37 % | -0.07 pp | 24.18 % | -2.26 pp |
| DRAM Throughput (%) | 26.44 % | 7.86 % | -18.58 pp | 23.35 % | -3.09 pp | 26.37 % | -0.07 pp | 21.05 % | -5.39 pp |
| Achieved Occupancy (%) | 40.98 % | 66.77 % | +25.79 pp | 33.82 % | -7.16 pp | 30.05 % | -10.93 pp | 40.25 % | -0.73 pp |
| Registers/Thread | 16.00 register/thread | 18.00 register/thread | +2 | 18.00 register/thread | +2 | 16.00 register/thread | +0 | 18.00 register/thread | +2 |
| Waves/SM | 46.55 | 372.36 | +325.81 | 46.55 | +0.00 | 46.55 | +0.00 | 93.09 | +46.54 |
| No Eligible (%) | 66.59 % | 20.17 % | -46.42 pp | 60.80 % | -5.79 pp | 60.56 % | -6.03 pp | 34.59 % | -32.00 pp |
| IPC Active | 1.31 inst/cycle | 3.18 inst/cycle | +1.87 | 1.53 inst/cycle | +0.22 | 1.56 inst/cycle | +0.25 | 2.61 inst/cycle | +1.30 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune_for_latency_hiding` (0.263×)

Adding @triton.autotune with larger XBLOCK configs (512–4096) to a pointwise kernel that already has a fixed grid computed from XBLOCK=128 causes a critical bug: the grid remains fixed at cdiv(xnumel, 128) while the kernel now processes XBLOCK=1024 elements per thread block, resulting in 8x more total work (executed instructions jumped from 15.7M to 148M). Additionally, introducing a non-trivially-true xmask (xindex < xnumel) when xnumel is divisible by all XBLOCK values adds unnecessary predication overhead. The net effect was a 3.8x slowdown despite improved scheduler utilization metrics, because the total instruction count exploded due to the grid/XBLOCK mismatch.

> **Applicable when**: Never apply this pattern as implemented — the grid must be updated to a lambda when using autotune with variable XBLOCK.

> **Avoid when**: Do not add @triton.autotune with variable XBLOCK without simultaneously changing the grid from a fixed expression to `lambda meta: (cdiv(xnumel, meta['XBLOCK']),)`. Do not introduce xmask = xindex < xnumel when xnumel is guaranteed divisible by all XBLOCK candidates — this adds unnecessary predication. Do not apply when the kernel call site still passes a fixed XBLOCK kwarg or uses a fixed grid.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.997× | 1.009× | ✓ |  |
| B | persistent_tile_scheduling | 1.000× | 1.003× | ✓ | **✓ best** |
| C | fix_autotune_grid_lambda_to_eliminate_instruction_explosion | 0.798× | 0.829× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 63170 ns | 62620 ns | -550 ns (-0.9%) | 62980 ns | -190 ns (-0.3%) | 76220 ns | +13050 ns (+20.7%) |
| SM Throughput (%) | 31.89 % | 36.82 % | +4.93 pp | 36.35 % | +4.46 pp | 63.61 % | +31.72 pp |
| Memory Throughput (%) | 26.44 % | 26.73 % | +0.29 pp | 26.57 % | +0.13 pp | 23.90 % | -2.54 pp |
| DRAM Throughput (%) | 26.44 % | 26.73 % | +0.29 pp | 26.57 % | +0.13 pp | 20.85 % | -5.59 pp |
| Achieved Occupancy (%) | 40.98 % | 42.78 % | +1.80 pp | 43.68 % | +2.70 pp | 40.57 % | -0.41 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 18.00 register/thread | +2 |
| Waves/SM | 46.55 | 46.55 | +0.00 | 46.55 | +0.00 | 93.09 | +46.54 |
| No Eligible (%) | 66.59 % | 61.59 % | -5.00 pp | 61.97 % | -4.62 pp | 33.79 % | -32.80 pp |
| IPC Active | 1.31 inst/cycle | 1.51 inst/cycle | +0.20 | 1.52 inst/cycle | +0.21 | 2.64 inst/cycle | +1.33 |

---

## Iteration 3

**Best so far**: 1.003× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_for_l1tex_latency_hiding | 0.357× | 1.004× | ✓ |  |
| B | increase_num_warps_for_l1tex_latency_hiding | 0.335× | 1.008× | ✓ |  |
| C | remove_broken_autotune_restore_fixed_xblock_128_unmasked | 1.000× | 0.996× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | increase_num_warp... | Δ | remove_broken_aut... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 63170 ns | 62940 ns | -230 ns (-0.4%) | 62660 ns | -510 ns (-0.8%) | 63420 ns | +250 ns (+0.4%) |
| SM Throughput (%) | 31.89 % | 36.38 % | +4.49 pp | 41.03 % | +9.14 pp | 31.56 % | -0.33 pp |
| Memory Throughput (%) | 26.44 % | 35.20 % | +8.76 pp | 29.96 % | +3.52 pp | 34.47 % | +8.03 pp |
| DRAM Throughput (%) | 26.44 % | 35.20 % | +8.76 pp | 29.96 % | +3.52 pp | 34.47 % | +8.03 pp |
| Achieved Occupancy (%) | 40.98 % | 46.89 % | +5.91 pp | 45.55 % | +4.57 pp | 44.94 % | +3.96 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 46.55 | 46.55 | +0.00 | 46.55 | +0.00 | 46.55 | +0.00 |
| No Eligible (%) | 66.59 % | 61.85 % | -4.74 pp | 57.72 % | -8.87 pp | 67.02 % | +0.43 pp |
| IPC Active | 1.31 inst/cycle | 1.51 inst/cycle | +0.20 | 1.68 inst/cycle | +0.37 | 1.32 inst/cycle | +0.01 |

---

## Iteration 4

**Best so far**: 1.003× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_broken_autotune_grid_and_add_streaming_eviction_for_l2_locality | 1.000× | 1.007× | ✓ |  |
| B | l2_swizzle_program_id_for_bias_broadcast_reuse | 1.000× | 1.008× | ✓ | **✓ best** |
| C | increase_num_warps_fixed_xblock_for_l1tex_latency_hiding | 0.996× | 0.852× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_broken_autotu... | Δ | l2_swizzle_progra... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 63170 ns | 62750 ns | -420 ns (-0.7%) | 62690 ns | -480 ns (-0.8%) | 74180 ns | +11010 ns (+17.4%) |
| SM Throughput (%) | 31.89 % | 35.76 % | +3.87 pp | 40.65 % | +8.76 pp | 59.37 % | +27.48 pp |
| Memory Throughput (%) | 26.44 % | 24.33 % | -2.11 pp | 26.69 % | +0.25 pp | 22.70 % | -3.74 pp |
| DRAM Throughput (%) | 26.44 % | 24.33 % | -2.11 pp | 26.69 % | +0.25 pp | 22.70 % | -3.74 pp |
| Achieved Occupancy (%) | 40.98 % | 39.35 % | -1.63 pp | 44.15 % | +3.17 pp | 74.10 % | +33.12 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 46.55 | 46.55 | +0.00 | 46.55 | +0.00 | 93.09 | +46.54 |
| No Eligible (%) | 66.59 % | 62.48 % | -4.11 pp | 57.96 % | -8.63 pp | 38.94 % | -27.65 pp |
| IPC Active | 1.31 inst/cycle | 1.48 inst/cycle | +0.17 | 1.68 inst/cycle | +0.37 | 2.42 inst/cycle | +1.11 |


### Key Learnings

#### ❌ Plan C: `increase_num_warps_fixed_xblock_for_l1tex_latency_hiding` (0.852×)

Increasing num_warps from 4 to 8 (block size 128→256) while keeping XBLOCK=128 fixed improved occupancy (45.6%→74.1%) and eligible warps per scheduler (0.6→1.61), but the autotuner selected num_warps=8 (block_size=256) which doubled total executed instructions from 20M to 34.6M, causing duration to increase from 62.7µs to 74.2µs. Although scheduler 'No Eligible' cycles dropped from 57.7% to 38.9%, the warp cycles per issued instruction actually worsened slightly (17.21→18.94), indicating the L1TEX stall latency was not meaningfully hidden. The instruction explosion from doubling threads per block (while keeping the same grid of 98304 CTAs) caused each CTA to process the same XBLOCK=128 elements with 2x the threads, resulting in 2x redundant work and a net regression.

> **Applicable when**: Never apply num_warps > XBLOCK/32 for a pointwise kernel where each thread processes exactly one element. num_warps should not exceed XBLOCK//32 (the number of warps needed to cover XBLOCK elements with 32 threads/warp).

> **Avoid when**: Do not increase num_warps beyond XBLOCK//32 for pointwise kernels where each thread handles one element. With XBLOCK=128, the maximum useful num_warps is 4 (128 threads / 32 threads per warp). Setting num_warps=8 with XBLOCK=128 means 256 threads are launched but only 128 elements exist per CTA — half the threads are wasted, doubling instruction count. Also do not introduce a real xmask (xindex < xnumel) when xnumel is divisible by XBLOCK, as this adds predication overhead with no correctness benefit.

---

## Iteration 5

**Best so far**: 1.0077× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_eviction_on_main_load_to_boost_bias_l1_hit_rate | 0.996× | 1.008× | ✓ |  |
| B | swizzle_program_id_by_head_dimension_for_bias_l2_reuse | 0.996× | 1.006× | ✓ |  |
| C | fix_num_warps_xblock_mismatch_restore_correct_single_config | 1.000× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | streaming_evictio... | Δ | swizzle_program_i... | Δ | fix_num_warps_xbl... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 63170 ns | 62660 ns | -510 ns (-0.8%) | 62820 ns | -350 ns (-0.6%) | 63200 ns | +30 ns (+0.0%) |
| SM Throughput (%) | 31.89 % | 44.32 % | +12.43 pp | 35.78 % | +3.89 pp | 31.71 % | -0.18 pp |
| Memory Throughput (%) | 26.44 % | 24.47 % | -1.97 pp | 24.50 % | -1.94 pp | 26.49 % | +0.05 pp |
| DRAM Throughput (%) | 26.44 % | 24.47 % | -1.97 pp | 24.50 % | -1.94 pp | 26.49 % | +0.05 pp |
| Achieved Occupancy (%) | 40.98 % | 44.48 % | +3.50 pp | 39.73 % | -1.25 pp | 40.21 % | -0.77 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 46.55 | 46.55 | +0.00 | 46.55 | +0.00 | 46.55 | +0.00 |
| No Eligible (%) | 66.59 % | 53.20 % | -13.39 pp | 62.79 % | -3.80 pp | 66.90 % | +0.31 pp |
| IPC Active | 1.31 inst/cycle | 1.87 inst/cycle | +0.56 | 1.47 inst/cycle | +0.16 | 1.32 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  63170 ns  (1.000×)
Iter 2     62980 ns  (1.003×)  — persistent_tile_scheduling
Iter 4     62690 ns  (1.008×)  — l2_swizzle_program_id_for_bias_broadcast_reuse

Final best: 63200 ns  (0.9995×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,22 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 12582912
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
     x0 = (xindex % 128)
     x1 = ((xindex // 128) % 32768)
     x2 = xindex // 4194304
     x3 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2 + 384*x1), None)
-    tmp1 = tl.load(in_ptr1 + (x0 + 128*x2), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2 + 384*x1))
+    tmp1 = tl.load(in_ptr1 + (x0 + 128*x2), eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
-    tl.store(out_ptr0 + (x3), tmp2, None)
-
-
+    tl.store(out_ptr0 + (x3), tmp2)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
