# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_clone_exp_mul_permute_slice_sub_unsqueeze_view_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2370 ns — **Best latency**: 2370 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_clone_exp_mul_permute_slice_sub_unsqueeze_view_5`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_048/kernel_5_triton_poi_fused_clone_exp_mul/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2,1,1), Waves/SM: 0.00
- **Registers/thread**: 32
- **Duration**: 2370 ns
- **SM Throughput**: 0.07 % | **Memory**: 0.44 % | **DRAM**: 0.05 %
- **Achieved Occupancy**: 6.22 %
- **IPC Active**: 0.48 inst/cycle | **No Eligible**: 86.69 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | fix_launch_grid_to_cover_full_ynumel | 0.555× | 0.814× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_launch_grid_t... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2370 ns | 2910 ns | +540 ns (+22.8%) |
| SM Throughput (%) | 0.07 % | 0.08 % | +0.01 pp |
| Memory Throughput (%) | 0.44 % | 0.37 % | -0.07 pp |
| DRAM Throughput (%) | 0.05 % | 0.05 % | +0.00 pp |
| Achieved Occupancy (%) | 6.22 % | 6.22 % | +0.00 pp |
| Registers/Thread | 32.00 register/thread | 34.00 register/thread | +2 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 86.69 % | 84.56 % | -2.13 pp |
| IPC Active | 0.48 inst/cycle | 0.63 inst/cycle | +0.15 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_grid_with_correct_xblock_yblock_and_2d_launch | 0.935× | 0.814× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_with_cor... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2370 ns | 2910 ns | +540 ns (+22.8%) |
| SM Throughput (%) | 0.07 % | 0.08 % | +0.01 pp |
| Memory Throughput (%) | 0.44 % | 0.37 % | -0.07 pp |
| DRAM Throughput (%) | 0.05 % | 0.05 % | +0.00 pp |
| Achieved Occupancy (%) | 6.22 % | 6.22 % | +0.00 pp |
| Registers/Thread | 32.00 register/thread | 34.00 register/thread | +2 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 86.69 % | 85.80 % | -0.89 pp |
| IPC Active | 0.48 inst/cycle | 0.60 inst/cycle | +0.12 |


### Key Learnings

#### ⚠️ Plan B: `fix_grid_with_correct_xblock_yblock_and_2d_launch` (0.814×)

The plan to fix the 1D→2D grid launch (XBLOCK=16, YBLOCK=64, grid=(1, 32768)) was applied but produced no observable change in launch configuration: grid_size remained 2, achieved occupancy stayed at 6.22%, and duration was unchanged at 2.91 µs. The diff was empty, indicating the code change was either not applied or was silently reverted. This is the fourth consecutive attempt (iterations 1 and 2) to fix the same LaunchConfiguration bug with the same diagnosis but all resulting in either correctness failures or no change. The root cause of the repeated failure is unclear from profiling alone — the kernel body and launch parameters appear identical before and after, suggesting the modification target (the Python wrapper/launch site) was not correctly identified or edited.

> **Applicable when**: Triton kernel with LaunchConfiguration rule triggered, grid_size << sm_count, achieved occupancy << theoretical occupancy, and a 1D grid that fails to cover a large Y dimension (ynumel >> xnumel). The fix is conceptually correct but has failed to apply successfully across 4 iterations.

> **Avoid when**: Do not attempt further variations of this same grid-fix plan without first verifying that the code change is actually being applied (non-empty diff). All four prior attempts (fix_grid_launch_to_cover_full_ynumel, increase_tile_sizes_for_grid_parallelism, fix_grid_and_add_vectorized_loads_stores, fix_grid_with_correct_xblock_yblock_and_2d_launch) have failed — either with correctness errors or with no change at all. The repeated failure pattern suggests the launch wrapper is auto-generated and cannot be modified directly, or the index arithmetic in the kernel body is more complex than assumed.

---

## Cumulative Progress

```
Baseline:  2370 ns  (1.000×)

Final best: 2370 ns  (1.0000×)
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
 def triton_poi_fused_clone_exp_mul_permute_slice_sub_unsqueeze_view_5(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
@@ -22,15 +23,13 @@

     y4 = yindex // 64
     y5 = yindex
     tmp0 = tl.load(in_ptr0 + (x3 + 16*y1 + 128*y0 + 8192*y2), xmask & ymask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (63 + 64*y4), ymask, eviction_policy='evict_last')
-    tmp2 = tl.load(in_ptr1 + (y5), ymask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr1 + (63 + 64*y4), xmask & ymask, eviction_policy='evict_last')
+    tmp2 = tl.load(in_ptr1 + (y5), xmask & ymask, eviction_policy='evict_last')
     tmp3 = tmp1 - tmp2
     tmp4 = tl.exp(tmp3)
     tmp5 = tmp0 * tmp4
     tl.store(out_ptr0 + (y0 + 64*x3 + 1024*y4), tmp0, xmask & ymask)
     tl.store(out_ptr1 + (x3 + 16*y5), tmp5, xmask & ymask)
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
