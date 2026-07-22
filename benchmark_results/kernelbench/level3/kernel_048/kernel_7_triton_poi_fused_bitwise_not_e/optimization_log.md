# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_bitwise_not_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_7` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2850 ns — **Best latency**: 2850 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_bitwise_not_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_7`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_048/kernel_7_triton_poi_fused_bitwise_not_e/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (8,1,1), Waves/SM: 0.00
- **Registers/thread**: 32
- **Duration**: 2850 ns
- **SM Throughput**: 0.23 % | **Memory**: 0.38 % | **DRAM**: 0.06 %
- **Achieved Occupancy**: 6.22 %
- **IPC Active**: 0.58 inst/cycle | **No Eligible**: 85.10 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_configuration | 0.993× | 0.956× | ✓ |  |
| B | expand_grid_and_tile_sizes_for_full_sm_utilization | 0.992× | 1.011× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_c... | Δ | expand_grid_and_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2850 ns | 2980 ns | +130 ns (+4.6%) | 2820 ns | -30 ns (-1.1%) |
| SM Throughput (%) | 0.23 % | 0.33 % | +0.10 pp | 0.33 % | +0.10 pp |
| Memory Throughput (%) | 0.38 % | 0.40 % | +0.02 pp | 0.45 % | +0.07 pp |
| DRAM Throughput (%) | 0.06 % | 0.06 % | +0.00 pp | 0.06 % | +0.00 pp |
| Achieved Occupancy (%) | 6.22 % | 6.22 % | +0.00 pp | 6.22 % | +0.00 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 85.10 % | 75.73 % | -9.37 pp | 77.17 % | -7.93 pp |
| IPC Active | 0.58 inst/cycle | 0.86 inst/cycle | +0.28 | 0.86 inst/cycle | +0.28 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | fix_grid_to_3d_with_correct_yblock_tiling | 0.946× | 0.956× | ✓ |  |
| E | add_triton_autotune | 0.945× | 0.908× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_to_3d_wi... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2850 ns | 2980 ns | +130 ns (+4.6%) | 3140 ns | +290 ns (+10.2%) |
| SM Throughput (%) | 0.23 % | 0.33 % | +0.10 pp | 0.32 % | +0.09 pp |
| Memory Throughput (%) | 0.38 % | 0.39 % | +0.01 pp | 0.37 % | -0.01 pp |
| DRAM Throughput (%) | 0.06 % | 0.06 % | +0.00 pp | 0.06 % | +0.00 pp |
| Achieved Occupancy (%) | 6.22 % | 6.22 % | +0.00 pp | 6.22 % | +0.00 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 85.10 % | 80.50 % | -4.60 pp | 77.46 % | -7.64 pp |
| IPC Active | 0.58 inst/cycle | 0.84 inst/cycle | +0.26 | 0.86 inst/cycle | +0.28 |

---

## Cumulative Progress

```
Baseline:  2850 ns  (1.000×)

Final best: 2850 ns  (1.0000×)
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
 def triton_poi_fused_bitwise_not_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_7(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
@@ -23,7 +24,7 @@

     y3 = yindex // 1024
     y7 = yindex // 512
     tmp0 = tl.load(in_ptr0 + (x4 + 64*y5), xmask & ymask, eviction_policy='evict_last')
-    tmp7 = tl.load(in_ptr1 + (y0 + 64*y2 + 128*y1 + 1024*y3), ymask, eviction_policy='evict_last')
+    tmp7 = tl.load(in_ptr1 + (y0 + 64*y2 + 128*y1 + 1024*y3), xmask & ymask, eviction_policy='evict_last')
     tmp8 = tl.load(in_ptr1 + (x4 + 64*y2 + 128*y1 + 1024*y3), xmask & ymask, eviction_policy='evict_last')
     tmp1 = x4 + ((-1)*y0)
     tmp2 = tl.full([1, 1], 0, tl.int64)
@@ -37,8 +38,6 @@

     tmp12 = tl.exp(tmp11)
     tmp13 = tmp0 * tmp12
     tl.store(out_ptr0 + (x4 + 64*y1 + 512*y0 + 32768*y7), tmp13, xmask & ymask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
