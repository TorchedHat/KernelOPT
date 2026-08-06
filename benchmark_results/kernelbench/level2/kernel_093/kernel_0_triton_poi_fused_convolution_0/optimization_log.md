# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6050 ns — **Best latency**: 3230 ns (**1.8731×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`l2_pid_grouping_to_reduce_workload_imbalance`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_093/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6050 ns
- **SM Throughput**: 15.68 % | **Memory**: 40.65 % | **DRAM**: 20.37 %
- **Achieved Occupancy**: 54.92 %
- **IPC Active**: 1.16 inst/cycle | **No Eligible**: 70.39 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguity_hints | 0.660× | 0.212× | ✓ |  |
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.258× | 0.979× | ✓ |  |
| C | memory_coalescing_swap_tile_dimensions | 1.359× | 1.410× | ✓ | **✓ best** |
| D | autotune_tile_sizes_for_underutilized_kernel | 1.074× | 1.027× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6050 ns | 28580 ns | +22530 ns (+372.4%) | 6180 ns | +130 ns (+2.1%) | 4290 ns | -1760 ns (-29.1%) | 5890 ns | -160 ns (-2.6%) |
| SM Throughput (%) | 15.68 % | 58.02 % | +42.34 pp | 16.18 % | +0.50 pp | 23.84 % | +8.16 pp | 16.73 % | +1.05 pp |
| Memory Throughput (%) | 40.65 % | 70.53 % | +29.88 pp | 39.78 % | -0.87 pp | 21.05 % | -19.60 pp | 41.78 % | +1.13 pp |
| DRAM Throughput (%) | 20.37 % | 4.30 % | -16.07 pp | 19.96 % | -0.41 pp | 7.25 % | -13.12 pp | 20.89 % | +0.52 pp |
| Achieved Occupancy (%) | 54.92 % | 33.96 % | -20.96 pp | 55.95 % | +1.03 pp | 52.48 % | -2.44 pp | 56.01 % | +1.09 pp |
| Registers/Thread | 26.00 register/thread | 32.00 register/thread | +6 | 26.00 register/thread | +0 | 28.00 register/thread | +2 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 3.96 | +3.22 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.39 % | 40.16 % | -30.23 pp | 70.50 % | +0.11 pp | 40.22 % | -30.17 pp | 71.14 % | +0.75 pp |
| IPC Active | 1.16 inst/cycle | 2.37 inst/cycle | +1.21 | 1.10 inst/cycle | -0.06 | 2.38 inst/cycle | +1.22 | 1.10 inst/cycle | -0.06 |

---

## Iteration 2

**Best so far**: 1.4103× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_scalar_contiguity_hints_on_load_index | 1.356× | 1.360× | ✓ |  |
| C | swap_fast_slow_tile_dim_for_coalesced_stores | 1.364× | 1.873× | ✓ | **✓ best** |
| D | swap_fast_slow_tile_dim_for_coalesced_stores_and_remove_bad_smem_hint | 1.168× | 0.872× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | swap_fast_slow_ti... | Δ | swap_fast_slow_ti... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6050 ns | 4450 ns | -1600 ns (-26.4%) | 3230 ns | -2820 ns (-46.6%) | 6940 ns | +890 ns (+14.7%) |
| SM Throughput (%) | 15.68 % | 24.51 % | +8.83 pp | 15.95 % | +0.27 pp | 20.45 % | +4.77 pp |
| Memory Throughput (%) | 40.65 % | 21.55 % | -19.10 pp | 8.93 % | -31.72 pp | 37.34 % | -3.31 pp |
| DRAM Throughput (%) | 20.37 % | 6.98 % | -13.39 pp | 2.44 % | -17.93 pp | 9.45 % | -10.92 pp |
| Achieved Occupancy (%) | 54.92 % | 53.12 % | -1.80 pp | 53.66 % | -1.26 pp | 50.46 % | -4.46 pp |
| Registers/Thread | 26.00 register/thread | 28.00 register/thread | +2 | 16.00 register/thread | -10 | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.39 % | 38.67 % | -31.72 pp | 49.94 % | -20.45 pp | 64.64 % | -5.75 pp |
| IPC Active | 1.16 inst/cycle | 2.32 inst/cycle | +1.16 | 1.63 inst/cycle | +0.47 | 1.38 inst/cycle | +0.22 |

---

## Iteration 3

**Best so far**: 1.8731× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | change_load_eviction_policy_to_evict_first_and_increase_pipeline_stages | 1.332× | 1.856× | ✓ |  |
| B | increase_tile_size_and_num_stages_for_occupancy_and_latency_hiding | 1.343× | 1.381× | ✓ |  |
| C | software_pipelining_num_stages_to_hide_memory_latency | 1.211× | 0.867× | ✓ |  |
| D | add_triton_autotune | 1.370× | 1.873× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | change_load_evict... | Δ | increase_tile_siz... | Δ | software_pipelini... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6050 ns | 3260 ns | -2790 ns (-46.1%) | 4380 ns | -1670 ns (-27.6%) | 6980 ns | +930 ns (+15.4%) | 3230 ns | -2820 ns (-46.6%) |
| SM Throughput (%) | 15.68 % | 15.55 % | -0.13 pp | 25.50 % | +9.82 pp | 18.79 % | +3.11 pp | 15.15 % | -0.53 pp |
| Memory Throughput (%) | 40.65 % | 8.68 % | -31.97 pp | 22.46 % | -18.19 pp | 37.36 % | -3.29 pp | 8.71 % | -31.94 pp |
| DRAM Throughput (%) | 20.37 % | 2.40 % | -17.97 pp | 7.10 % | -13.27 pp | 9.46 % | -10.91 pp | 2.43 % | -17.94 pp |
| Achieved Occupancy (%) | 54.92 % | 48.18 % | -6.74 pp | 53.14 % | -1.78 pp | 50.29 % | -4.63 pp | 50.42 % | -4.50 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 28.00 register/thread | +2 | 32.00 register/thread | +6 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.39 % | 54.46 % | -15.93 pp | 39.51 % | -30.88 pp | 64.55 % | -5.84 pp | 56.49 % | -13.90 pp |
| IPC Active | 1.16 inst/cycle | 1.82 inst/cycle | +0.66 | 2.37 inst/cycle | +1.21 | 1.35 inst/cycle | +0.19 | 1.59 inst/cycle | +0.43 |


### Key Learnings

#### ❌ Plan C: `software_pipelining_num_stages_to_hide_memory_latency` (0.867×)

Replacing small-XBLOCK autotune configs (XBLOCK≤256, YBLOCK≥16) with large-XBLOCK configs (XBLOCK up to 2048, YBLOCK as small as 2) caused the autotuner to select a configuration that dramatically amplified uncoalesced global accesses: excessive sectors jumped from 5.9K (20% of total) to 200K (50% of total), and executed instructions increased 2.5x (399K → 1012K). Registers per thread doubled from 16 to 32, reducing Block Limit Registers from 32 to 16 and slightly lowering achieved occupancy. The net effect was kernel duration more than doubling (3.23 µs → 6.98 µs). The root cause is that large XBLOCK with small YBLOCK (e.g. XBLOCK=2048, YBLOCK=2) causes each warp to stride across a very wide x-range in the store address, amplifying the stride-N uncoalesced store pattern that was already partially present. The num_stages increase (from 3 to 2-3 across configs) did not help because the kernel is too short (3.23 µs, ~4768 cycles) for software pipelining to provide meaningful latency hiding benefit, and the dominant regression came from the tile shape change.

> **Applicable when**: Never: this is a negative pattern. Do not expand autotune configs to include very large XBLOCK (≥512) with very small YBLOCK (≤2) when the store address has a stride-N pattern in the x-dimension, as this amplifies uncoalesced sector count proportionally to XBLOCK size.

> **Avoid when**: Do not add large-XBLOCK (≥512) + small-YBLOCK (≤4) autotune configs when: (1) the store address has stride > 1 in the x-dimension (uncoalesced stores scale with XBLOCK), (2) the kernel is already very short (< 5 µs / < 8000 cycles) making software pipelining overhead non-negligible, (3) registers per thread are already low (≤16) meaning larger tile configs will double register usage and halve Block Limit Registers.

---

## Iteration 4

**Best so far**: 1.8731× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_load_policy_with_scalar_contiguity_hint_on_load_index | 1.310× | 1.873× | ✓ |  |
| C | restrict_autotune_configs_to_small_xblock_large_yblock_to_eliminate_uncoalesced_store_amplification | 1.127× | 0.965× | ✓ |  |
| D | expand_autotune_yblock_num_stages5_for_sm_utilization | 1.282× | 1.891× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_load_... | Δ | restrict_autotune... | Δ | expand_autotune_y... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6050 ns | 3230 ns | -2820 ns (-46.6%) | 6270 ns | +220 ns (+3.6%) | 3200 ns | -2850 ns (-47.1%) |
| SM Throughput (%) | 15.68 % | 15.51 % | -0.17 pp | 33.73 % | +18.05 pp | 15.54 % | -0.14 pp |
| Memory Throughput (%) | 40.65 % | 8.78 % | -31.87 pp | 33.10 % | -7.55 pp | 8.80 % | -31.85 pp |
| DRAM Throughput (%) | 20.37 % | 2.42 % | -17.95 pp | 9.82 % | -10.55 pp | 2.43 % | -17.94 pp |
| Achieved Occupancy (%) | 54.92 % | 49.53 % | -5.39 pp | 56.93 % | +2.01 pp | 48.91 % | -6.01 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 32.00 register/thread | +6 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.39 % | 57.24 % | -13.15 pp | 40.84 % | -29.55 pp | 55.65 % | -14.74 pp |
| IPC Active | 1.16 inst/cycle | 1.76 inst/cycle | +0.60 | 2.37 inst/cycle | +1.21 | 1.80 inst/cycle | +0.64 |

---

## Iteration 5

**Best so far**: 1.8731× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reduce_register_pressure_with_small_num_warps_to_improve_block_limit_and_occupancy | 1.218× | 1.381× | ✓ |  |
| C | l2_pid_grouping_to_reduce_workload_imbalance | 1.363× | 1.873× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | l2_pid_grouping_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6050 ns | 4380 ns | -1670 ns (-27.6%) | 3230 ns | -2820 ns (-46.6%) |
| SM Throughput (%) | 15.68 % | 21.81 % | +6.13 pp | 15.59 % | -0.09 pp |
| Memory Throughput (%) | 40.65 % | 23.45 % | -17.20 pp | 8.67 % | -31.98 pp |
| DRAM Throughput (%) | 20.37 % | 7.06 % | -13.31 pp | 2.42 % | -17.95 pp |
| Achieved Occupancy (%) | 54.92 % | 31.32 % | -23.60 pp | 49.12 % | -5.80 pp |
| Registers/Thread | 26.00 register/thread | 32.00 register/thread | +6 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.37 | -0.37 | 0.74 | +0.00 |
| No Eligible (%) | 70.39 % | 47.47 % | -22.92 pp | 53.01 % | -17.38 pp |
| IPC Active | 1.16 inst/cycle | 2.02 inst/cycle | +0.86 | 1.74 inst/cycle | +0.58 |


### Key Learnings

#### ❌ Plan C: `l2_pid_grouping_to_reduce_workload_imbalance` (1.873×)

Applying a grouped PID swizzle (GROUP_SIZE_X=8) to reduce L2 slice workload imbalance did reduce the L2 imbalance metric (max-above-average dropped from 25% to 17%), but produced no wall-clock speedup — kernel duration remained at ~3.2 µs and the reported 1.87x speedup matches the prior iteration's baseline, indicating the swizzle itself contributed nothing. Warp Cycles Per Issued Instruction actually worsened slightly (16.55 → 18.27 cycles) and Issue Slots Busy dropped (46.6% → 45.1%), suggesting the swizzle added minor arithmetic overhead without compensating benefit. For kernels with <1 wave/SM where the grid-size bottleneck dominates, L2 slice rebalancing via PID swizzle does not translate to measurable latency reduction because the L2 imbalance is not the binding constraint.

> **Applicable when**: Never: this is a negative pattern for short pointwise kernels with <1 wave/SM. The grouped swizzle (pid_group * GROUP_SIZE_X + pid_in_group) is mathematically equivalent to the linear mapping when GROUP_SIZE_X divides the grid evenly, so it cannot reorder CTA execution — it only adds integer division/modulo overhead.

> **Avoid when**: Do not apply L2 PID swizzle grouping when: (1) the kernel has <1 wave/SM and the primary bottleneck is grid size (SOLBottleneck rule fires); (2) the WorkloadImbalance L2 max-above-average is <30% — the imbalance is not severe enough to be the binding constraint; (3) the kernel duration is <5 µs — the arithmetic overhead of the swizzle (integer division/modulo) is non-negligible relative to total kernel time; (4) the grouped swizzle formula (pid_group * G + pid_in_group) * XBLOCK is algebraically identical to pid_x_raw * XBLOCK, meaning no actual reordering occurs.

---

## Cumulative Progress

```
Baseline:  6050 ns  (1.000×)
Iter 1     4290 ns  (1.410×)  — memory_coalescing_swap_tile_dimensions
Iter 2     3230 ns  (1.873×)  — swap_fast_slow_tile_dim_for_coalesced_stores

Final best: 3230 ns  (1.8731×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,47 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 32},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 32},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16},  num_warps=2, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 16},  num_warps=2, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 16},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16},  num_warps=4, num_stages=5),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16},  num_warps=8, num_stages=5),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16},  num_warps=4, num_stages=5),
+        # NEW: YBLOCK=64 covers all ynumel=30 in one tile + deep pipeline
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64},  num_warps=4, num_stages=5),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=5),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8, num_stages=5),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 32},  num_warps=2, num_stages=5),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=5),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
     xnumel = 50176
+    xoffset = tl.program_id(0) * XBLOCK
+    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
+    xmask = xindex < xnumel
     yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
+    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
     ymask = yindex < ynumel
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
-    xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
     y0 = (yindex % 3)
     y1 = yindex // 3
-    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_first')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +53,7 @@

     ynumel = 30
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
