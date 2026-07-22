# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_leaky_relu_mul_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 324420 ns — **Best latency**: 324420 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_leaky_relu_mul_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_074/kernel_0_triton_poi_fused_convolution_l/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (524288,1,1), Waves/SM: 248.24
- **Registers/thread**: 16
- **Duration**: 324420 ns
- **SM Throughput**: 34.39 % | **Memory**: 32.02 % | **DRAM**: 32.02 %
- **Achieved Occupancy**: 45.35 %
- **IPC Active**: 1.38 inst/cycle | **No Eligible**: 65.41 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_for_pointwise_kernel | 0.880× | 0.892× | ✓ |  |
| B | increase_xblock_tile_size_for_latency_hiding | 0.998× | 0.287× | ✓ |  |
| C | increase_xblock_with_autotune_for_better_latency_hiding | 0.998× | 0.358× | ✓ |  |
| D | increase_xblock_for_vectorized_loads_and_latency_hiding | 0.880× | 0.891× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | increase_xblock_t... | Δ | increase_xblock_w... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 324420 ns | 363810 ns | +39390 ns (+12.1%) | 1130000 ns | +805580 ns (+248.3%) | 906460 ns | +582040 ns (+179.4%) | 364030 ns | +39610 ns (+12.2%) |
| SM Throughput (%) | 34.39 % | 68.53 % | +34.14 pp | 84.93 % | +50.54 pp | 88.07 % | +53.68 pp | 68.10 % | +33.71 pp |
| Memory Throughput (%) | 32.02 % | 37.05 % | +5.03 pp | 39.90 % | +7.88 pp | 49.84 % | +17.82 pp | 37.23 % | +5.21 pp |
| DRAM Throughput (%) | 32.02 % | 28.60 % | -3.42 pp | 9.19 % | -22.83 pp | 11.47 % | -20.55 pp | 28.59 % | -3.43 pp |
| Achieved Occupancy (%) | 45.35 % | 42.58 % | -2.77 pp | 82.19 % | +36.84 pp | 86.14 % | +40.79 pp | 42.59 % | -2.76 pp |
| Registers/Thread | 16.00 register/thread | 21.00 register/thread | +5 | 21.00 register/thread | +5 | 31.00 register/thread | +15 | 21.00 register/thread | +5 |
| Waves/SM | 248.24 | 248.24 | +0.00 | 992.97 | +744.73 | 496.48 | +248.24 | 248.24 | +0.00 |
| No Eligible (%) | 65.41 % | 31.05 % | -34.36 pp | 14.76 % | -50.65 pp | 11.65 % | -53.76 pp | 31.56 % | -33.85 pp |
| IPC Active | 1.38 inst/cycle | 2.75 inst/cycle | +1.37 | 3.41 inst/cycle | +2.03 | 3.53 inst/cycle | +2.15 | 2.72 inst/cycle | +1.34 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_fixed_autotune_grid | 0.795× | 0.804× | ✓ |  |
| B | persistent_tile_scheduling_with_correct_autotune_grid_lambda | 0.880× | 0.888× | ✓ |  |
| C | fix_autotune_grid_lambda_to_eliminate_thread_inflation | 0.998× | 0.358× | ✓ |  |
| D | fix_autotune_grid_mismatch_with_persistent_tile_scheduling | 0.998× | 0.287× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 324420 ns | 403490 ns | +79070 ns (+24.4%) | 365120 ns | +40700 ns (+12.5%) | 905920 ns | +581500 ns (+179.2%) | 1130000 ns | +805580 ns (+248.3%) |
| SM Throughput (%) | 34.39 % | 19.87 % | -14.52 pp | 67.52 % | +33.13 pp | 87.93 % | +53.54 pp | 85.28 % | +50.89 pp |
| Memory Throughput (%) | 32.02 % | 25.80 % | -6.22 pp | 36.95 % | +4.93 pp | 49.76 % | +17.74 pp | 40.06 % | +8.04 pp |
| DRAM Throughput (%) | 32.02 % | 25.80 % | -6.22 pp | 28.48 % | -3.54 pp | 11.48 % | -20.54 pp | 9.18 % | -22.84 pp |
| Achieved Occupancy (%) | 45.35 % | 35.70 % | -9.65 pp | 42.53 % | -2.82 pp | 85.89 % | +40.54 pp | 82.21 % | +36.86 pp |
| Registers/Thread | 16.00 register/thread | 32.00 register/thread | +16 | 21.00 register/thread | +5 | 31.00 register/thread | +15 | 21.00 register/thread | +5 |
| Waves/SM | 248.24 | 248.24 | +0.00 | 248.24 | +0.00 | 496.48 | +248.24 | 992.97 | +744.73 |
| No Eligible (%) | 65.41 % | 73.25 % | +7.84 pp | 32.09 % | -33.32 pp | 11.61 % | -53.80 pp | 14.73 % | -50.68 pp |
| IPC Active | 1.38 inst/cycle | 0.96 inst/cycle | -0.42 | 2.73 inst/cycle | +1.35 | 3.53 inst/cycle | +2.15 | 3.41 inst/cycle | +2.03 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling_with_fixed_autotune_grid` (0.804×)

Converting a single-tile-per-CTA Triton kernel to a persistent loop pattern (using tl.range over tiles) increased registers per thread from 21 to 32, which reduced Block Limit Registers from 21 to 16 blocks/SM and dropped achieved occupancy from 42.6% to 35.7%. Although total executed instructions dropped 3x (191M to 62M), the scheduler stall worsened dramatically: eligible warps per scheduler fell from 1.57 to 0.53 and warp cycles per issued instruction rose from 9.65 to 22.89, causing the kernel to run 11% slower. The persistent loop's loop-carried state and tl.range overhead inflated register usage, which is the dominant bottleneck when the kernel is already register-limited (Block Limit Registers < Block Limit Warps).

> **Applicable when**: Never apply this pattern when the kernel is already register-limited (Block Limit Registers < Block Limit Warps) and achieved occupancy is below theoretical due to register pressure.

> **Avoid when**: Kernel is register-limited (registers/thread close to or above 32 on SM90, Block Limit Registers < Block Limit Warps). Persistent loops add loop-carried variables that inflate register count, further reducing occupancy and worsening scheduler stall. Also avoid when the grid is not actually reduced (this diff kept grid=524288, negating the persistent-kernel benefit entirely).

---

## Cumulative Progress

```
Baseline:  324420 ns  (1.000×)

Final best: 324420 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,17 +5,28 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 2048}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 4096}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 4096}, num_warps=16, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_leaky_relu_mul_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
-    xnumel = 67108864
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x3 = xindex
     x1 = ((xindex // 131072) % 32)
-    tmp0 = tl.load(in_out_ptr0 + (x3), None)
-    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
-    tmp8 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
+    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
+    tmp8 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp3 = tl.full([1], 0.0, tl.float32)
     tmp4 = tmp2 > tmp3
@@ -26,9 +37,7 @@

     tmp10 = tmp9 > tmp3
     tmp11 = tmp9 * tmp5
     tmp12 = tl.where(tmp10, tmp9, tmp11)
-    tl.store(in_out_ptr0 + (x3), tmp12, None)
-
-
+    tl.store(in_out_ptr0 + (x3), tmp12, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -36,7 +45,7 @@

     arg_2 = inputs[2]
     xnumel = 67108864
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_convolution_leaky_relu_mul_0[grid](arg_0, arg_1, arg_2, 67108864, XBLOCK=128)
+    triton_poi_fused_convolution_leaky_relu_mul_0[grid](arg_0, arg_1, arg_2, 67108864)
     return arg_2
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
