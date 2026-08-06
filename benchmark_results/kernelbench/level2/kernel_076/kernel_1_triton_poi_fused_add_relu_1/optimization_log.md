# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_relu_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 42780 ns — **Best latency**: 42780 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_relu_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_076/kernel_1_triton_poi_fused_add_relu_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (65536,1,1), Waves/SM: 31.03
- **Registers/thread**: 16
- **Duration**: 42780 ns
- **SM Throughput**: 26.42 % | **Memory**: 20.14 % | **DRAM**: 19.39 %
- **Achieved Occupancy**: 37.42 %
- **IPC Active**: 1.12 inst/cycle | **No Eligible**: 71.98 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_for_latency_hiding | 0.900× | 0.944× | ✓ |  |
| B | autotune_xblock_num_warps_pointwise | 0.999× | 0.870× | ✓ |  |
| C | increase_xblock_vectorized_loads_with_autotune | 0.995× | 1.001× | ✓ |  |
| D | increase_xblock_with_autotune_for_latency_hiding | 0.902× | 0.938× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_n... | Δ | increase_xblock_v... | Δ | increase_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 42780 ns | 45310 ns | +2530 ns (+5.9%) | 49150 ns | +6370 ns (+14.9%) | 42750 ns | -30 ns (-0.1%) | 45630 ns | +2850 ns (+6.7%) |
| SM Throughput (%) | 26.42 % | 30.31 % | +3.89 pp | 42.82 % | +16.40 pp | 28.00 % | +1.58 pp | 30.06 % | +3.64 pp |
| Memory Throughput (%) | 20.14 % | 18.62 % | -1.52 pp | 24.72 % | +4.58 pp | 20.03 % | -0.11 pp | 18.40 % | -1.74 pp |
| DRAM Throughput (%) | 19.39 % | 18.57 % | -0.82 pp | 16.80 % | -2.59 pp | 19.56 % | +0.17 pp | 18.17 % | -1.22 pp |
| Achieved Occupancy (%) | 37.42 % | 34.53 % | -2.89 pp | 35.97 % | -1.45 pp | 27.13 % | -10.29 pp | 34.63 % | -2.79 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 24.00 register/thread | +8 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 71.98 % | 64.16 % | -7.82 pp | 52.75 % | -19.23 pp | 67.60 % | -4.38 pp | 64.25 % | -7.73 pp |
| IPC Active | 1.12 inst/cycle | 1.40 inst/cycle | +0.28 | 1.85 inst/cycle | +0.73 | 1.28 inst/cycle | +0.16 | 1.39 inst/cycle | +0.27 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_and_add_num_stages_pipelining | 0.993× | 1.001× | ✓ |  |
| B | fix_autotune_grid_and_add_software_pipelining | 0.905× | 0.940× | ✓ |  |
| C | fix_grid_launch_and_increase_num_stages_for_latency_hiding | 0.905× | 0.940× | ✓ |  |
| D | fix_grid_xblock_mismatch_to_enable_autotune | 0.999× | 0.384× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_a... | Δ | fix_autotune_grid... | Δ | fix_grid_launch_a... | Δ | fix_grid_xblock_m... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 42780 ns | 42720 ns | -60 ns (-0.1%) | 45500 ns | +2720 ns (+6.4%) | 45500 ns | +2720 ns (+6.4%) | 111360 ns | +68580 ns (+160.3%) |
| SM Throughput (%) | 26.42 % | 27.97 % | +1.55 pp | 30.23 % | +3.81 pp | 30.46 % | +4.04 pp | 73.56 % | +47.14 pp |
| Memory Throughput (%) | 20.14 % | 20.04 % | -0.10 pp | 18.56 % | -1.58 pp | 18.65 % | -1.49 pp | 31.89 % | +11.75 pp |
| DRAM Throughput (%) | 19.39 % | 19.38 % | -0.01 pp | 18.27 % | -1.12 pp | 18.35 % | -1.04 pp | 7.42 % | -11.97 pp |
| Achieved Occupancy (%) | 37.42 % | 27.18 % | -10.24 pp | 34.73 % | -2.69 pp | 34.82 % | -2.60 pp | 60.67 % | +23.25 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 34.00 register/thread | +18 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 | 82.75 | +51.72 |
| No Eligible (%) | 71.98 % | 67.41 % | -4.57 pp | 63.92 % | -8.06 pp | 63.99 % | -7.99 pp | 30.80 % | -41.18 pp |
| IPC Active | 1.12 inst/cycle | 1.27 inst/cycle | +0.15 | 1.38 inst/cycle | +0.26 | 1.39 inst/cycle | +0.27 | 2.76 inst/cycle | +1.64 |

---

## Cumulative Progress

```
Baseline:  42780 ns  (1.000×)

Final best: 42780 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,17 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_add_relu_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
@@ -19,13 +30,12 @@

     tmp4 = tl.maximum(tmp3, tmp2)
     tl.store(in_out_ptr0 + (x2), tmp4, xmask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 8388608
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_add_relu_1[grid](arg_0, arg_1, 8388608, XBLOCK=128)
+    triton_poi_fused_add_relu_1[grid](arg_0, arg_1, 8388608)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
