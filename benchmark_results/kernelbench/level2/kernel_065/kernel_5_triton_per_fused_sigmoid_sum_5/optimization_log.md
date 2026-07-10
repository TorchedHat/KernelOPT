# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused_sigmoid_sum_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2370 ns — **Best latency**: 2370 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused_sigmoid_sum_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_065/kernel_5_triton_per_fused_sigmoid_sum_5/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1,1,1), Waves/SM: 0.00
- **Registers/thread**: 27
- **Duration**: 2370 ns
- **SM Throughput**: 0.03 % | **Memory**: 0.44 % | **DRAM**: 0.06 %
- **Achieved Occupancy**: 6.23 %
- **IPC Active**: 0.42 inst/cycle | **No Eligible**: 88.88 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_xblock_to_increase_grid_parallelism | 0.969× | 0.915× | ✓ |  |
| D | increase_grid_parallelism_by_reducing_xblock | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_xblock_to_... | Δ | increase_grid_par... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2370 ns | 2590 ns | +220 ns (+9.3%) | 2370 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 0.03 % | 0.03 % | +0.00 pp | 0.03 % | +0.00 pp |
| Memory Throughput (%) | 0.44 % | 0.40 % | -0.04 pp | 0.57 % | +0.13 pp |
| DRAM Throughput (%) | 0.06 % | 0.05 % | -0.01 pp | 0.06 % | +0.00 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp | 6.23 % | +0.00 pp |
| Registers/Thread | 27.00 register/thread | 27.00 register/thread | +0 | 27.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 88.88 % | 90.24 % | +1.36 pp | 88.85 % | -0.03 pp |
| IPC Active | 0.42 inst/cycle | 0.33 inst/cycle | -0.09 | 0.34 inst/cycle | -0.08 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_with_autotune_for_warp_level_parallelism | 0.967× | 0.905× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2370 ns | 2620 ns | +250 ns (+10.5%) |
| SM Throughput (%) | 0.03 % | 0.03 % | +0.00 pp |
| Memory Throughput (%) | 0.44 % | 0.40 % | -0.04 pp |
| DRAM Throughput (%) | 0.06 % | 0.06 % | +0.00 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp |
| Registers/Thread | 27.00 register/thread | 27.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 88.88 % | 89.08 % | +0.20 pp |
| IPC Active | 0.42 inst/cycle | 0.35 inst/cycle | -0.07 |

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
 def triton_per_fused_sigmoid_sum_5(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
@@ -22,13 +23,11 @@

     rindex = r0_index
     r0_1 = r0_index
     x0 = xindex
-    tmp0 = tl.load(in_ptr0 + (r0_1 + 5*x0), r0_mask & xmask, other=0.0)
+    tmp0 = tl.load(in_ptr0 + (r0_1 + 5*x0), mask=r0_mask & xmask, other=0.0)
     tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
     tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
     tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
-    tl.store(out_ptr0 + (x0), tmp4, xmask)
-
-
+    tl.store(out_ptr0 + (x0), tmp4, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
