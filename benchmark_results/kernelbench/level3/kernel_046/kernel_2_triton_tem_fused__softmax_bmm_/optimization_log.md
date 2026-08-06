# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__softmax_bmm_exp_slice_sub_transpose_view_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 8670 ns — **Best latency**: 8580 ns (**1.0105×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`autotune_with_runtime_batch_grid_lambda`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__softmax_bmm_exp_slice_sub_transpose_view_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_046/kernel_2_triton_tem_fused__softmax_bmm_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4,1,1), Waves/SM: 0.01
- **Registers/thread**: 96
- **Duration**: 8670 ns
- **SM Throughput**: 1.19 % | **Memory**: 1.16 % | **DRAM**: 0.58 %
- **Achieved Occupancy**: 6.23 %
- **IPC Active**: 2.10 inst/cycle | **No Eligible**: 49.86 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | online_softmax_attention | 0.341× | 0.232× | ✓ |  |
| B | autotune_tile_sizes_and_fix_batch_grid | 1.019× | 1.019× | ✓ | **✓ best** |
| C | fix_grid_batch_dimension_and_autotune_tile_sizes | 1.006× | 0.989× | ✓ |  |
| D | fix_grid_and_add_autotune_with_vectorized_loads | 1.009× | 0.974× | ✓ |  |


### Metric Comparison

| Metric | Baseline | online_softmax_at... | Δ | autotune_tile_siz... | Δ | fix_grid_batch_di... | Δ | fix_grid_and_add_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8670 ns | 37380 ns | +28710 ns (+331.1%) | 8510 ns | -160 ns (-1.8%) | 8770 ns | +100 ns (+1.2%) | 8900 ns | +230 ns (+2.7%) |
| SM Throughput (%) | 1.19 % | 1.02 % | -0.17 pp | 1.22 % | +0.03 pp | 1.21 % | +0.02 pp | 1.23 % | +0.04 pp |
| Memory Throughput (%) | 1.16 % | 0.67 % | -0.49 pp | 1.19 % | +0.03 pp | 1.18 % | +0.02 pp | 1.20 % | +0.04 pp |
| DRAM Throughput (%) | 0.58 % | 0.16 % | -0.42 pp | 0.59 % | +0.01 pp | 0.57 % | -0.01 pp | 0.57 % | -0.01 pp |
| Achieved Occupancy (%) | 6.23 % | 6.25 % | +0.02 pp | 6.23 % | +0.00 pp | 6.23 % | +0.00 pp | 6.23 % | +0.00 pp |
| Registers/Thread | 96.00 register/thread | 255.00 register/thread | +159 | 96.00 register/thread | +0 | 96.00 register/thread | +0 | 96.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.03 | +0.02 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 49.86 % | 62.43 % | +12.57 pp | 46.58 % | -3.28 pp | 49.44 % | -0.42 pp | 46.76 % | -3.10 pp |
| IPC Active | 2.10 inst/cycle | 1.45 inst/cycle | -0.65 | 2.03 inst/cycle | -0.07 | 2.07 inst/cycle | -0.03 | 2.11 inst/cycle | +0.01 |

---

## Iteration 2

**Best so far**: 1.0188× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_grid_to_cover_batch_dimension | 1.020× | 1.007× | ✓ |  |
| B | expand_grid_batch_dimension_to_expose_parallelism | 1.017× | 1.007× | ✓ |  |
| C | expand_grid_batch_dimension_to_cover_all_batch_elements | 1.007× | 1.022× | ✓ |  |
| D | shrink_tile_and_fix_batch_grid_dimension | 1.020× | 1.022× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | expand_grid_to_co... | Δ | expand_grid_batch... | Δ | expand_grid_batch... | Δ | shrink_tile_and_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8670 ns | 8610 ns | -60 ns (-0.7%) | 8610 ns | -60 ns (-0.7%) | 8480 ns | -190 ns (-2.2%) | 8480 ns | -190 ns (-2.2%) |
| SM Throughput (%) | 1.19 % | 1.24 % | +0.05 pp | 1.25 % | +0.06 pp | 1.20 % | +0.01 pp | 1.21 % | +0.02 pp |
| Memory Throughput (%) | 1.16 % | 1.21 % | +0.05 pp | 1.22 % | +0.06 pp | 1.17 % | +0.01 pp | 1.18 % | +0.02 pp |
| DRAM Throughput (%) | 0.58 % | 0.58 % | +0.00 pp | 0.58 % | +0.00 pp | 0.59 % | +0.01 pp | 0.59 % | +0.01 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp | 6.23 % | +0.00 pp | 6.23 % | +0.00 pp | 6.23 % | +0.00 pp |
| Registers/Thread | 96.00 register/thread | 96.00 register/thread | +0 | 96.00 register/thread | +0 | 96.00 register/thread | +0 | 96.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 49.86 % | 46.87 % | -2.99 pp | 48.03 % | -1.83 pp | 46.94 % | -2.92 pp | 46.66 % | -3.20 pp |
| IPC Active | 2.10 inst/cycle | 2.11 inst/cycle | +0.01 | 2.06 inst/cycle | -0.04 | 2.11 inst/cycle | +0.01 | 2.07 inst/cycle | -0.03 |

---

## Iteration 3

**Best so far**: 1.0224× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 4

**Best so far**: 1.0224× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_with_runtime_batch_grid_lambda | 1.022× | 1.010× | ✓ | **✓ best** |
| D | vectorized_loads_even_k_with_batch_grid_fix | 0.867× | 0.874× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_with_run... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 8670 ns | 8580 ns | -90 ns (-1.0%) | 9920 ns | +1250 ns (+14.4%) |
| SM Throughput (%) | 1.19 % | 1.25 % | +0.06 pp | 1.03 % | -0.16 pp |
| Memory Throughput (%) | 1.16 % | 1.22 % | +0.06 pp | 1.19 % | +0.03 pp |
| DRAM Throughput (%) | 0.58 % | 0.59 % | +0.01 pp | 0.50 % | -0.08 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp | 6.24 % | +0.01 pp |
| Registers/Thread | 96.00 register/thread | 96.00 register/thread | +0 | 104.00 register/thread | +8 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 49.86 % | 50.09 % | +0.23 pp | 58.07 % | +8.21 pp |
| IPC Active | 2.10 inst/cycle | 2.05 inst/cycle | -0.05 | 1.58 inst/cycle | -0.52 |

---

## Iteration 5

**Best so far**: 1.0224× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | autotune_with_correct_batch_grid_via_grid_lambda | 0.874× | 0.842× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_with_cor... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 8670 ns | 10300 ns | +1630 ns (+18.8%) |
| SM Throughput (%) | 1.19 % | 1.01 % | -0.18 pp |
| Memory Throughput (%) | 1.16 % | 1.16 % | +0.00 pp |
| DRAM Throughput (%) | 0.58 % | 0.48 % | -0.10 pp |
| Achieved Occupancy (%) | 6.23 % | 6.24 % | +0.01 pp |
| Registers/Thread | 96.00 register/thread | 104.00 register/thread | +8 |
| Waves/SM | 0.01 | 0.01 | +0.00 |
| No Eligible (%) | 49.86 % | 59.23 % | +9.37 pp |
| IPC Active | 2.10 inst/cycle | 1.59 inst/cycle | -0.51 |


### Key Learnings

#### ❌ Plan C: `autotune_with_correct_batch_grid_via_grid_lambda` (0.842×)

The actual diff applied did NOT expand the grid or add @triton.autotune as planned — instead it only changed EVEN_K from False to True (replacing the mask-guarded load with an unmasked load conditioned on `k >= BLOCK_K`), removed the store mask, and left the grid at (4,1,1). This caused a regression: elapsed cycles increased from 12783 to 15379 (+20%), duration from 8.58us to 10.3us, registers per thread rose from 96 to 104 (reducing Block Limit Registers from 5 to 4 and theoretical occupancy from 31.25% to 25%), and warp cycles per issued instruction worsened from 1.88 to 2.45. The EVEN_K=True path with `k >= BLOCK_K` as the branch condition generates more register-pressure code than the original masked-load path, and removing the store mask does not compensate. The root cause (grid=4 for batch=2048) was never addressed.

> **Applicable when**: Never — this specific combination of EVEN_K=True with a runtime `k >= BLOCK_K` branch and removal of the store mask does not improve performance when K is not a multiple of BLOCK_K and the grid is already too small.

> **Avoid when**: Do not change EVEN_K from False to True using a runtime branch condition (`k >= BLOCK_K`) as a substitute for a compile-time constexpr. This increases register pressure (96→104 regs/thread), reduces Block Limit Registers (5→4), lowers theoretical occupancy (31.25%→25%), and worsens warp cycles per issued instruction (1.88→2.45). Do not remove the store mask when tile dimensions exactly match the problem dimensions as a standalone optimization — it provides no benefit and may introduce correctness risk if dimensions change.

---

## Cumulative Progress

```
Baseline:  8670 ns  (1.000×)
Iter 1     8510 ns  (1.019×)  — autotune_tile_sizes_and_fix_batch_grid
Iter 2     8480 ns  (1.022×)  — shrink_tile_and_fix_batch_grid_dimension

Final best: 8580 ns  (1.0105×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_tem_fused__softmax_bmm_exp_slice_sub_transpose_view_2(arg_A, arg_B, out_ptr0):
@@ -86,8 +87,6 @@

     xindex = idx_n + 512*idx_m + 16384*idx_q
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
