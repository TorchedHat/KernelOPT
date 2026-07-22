# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 21790 ns — **Best latency**: 21790 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_2_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (31360,1,1), Waves/SM: 14.85
- **Registers/thread**: 16
- **Duration**: 21790 ns
- **SM Throughput**: 35.52 % | **Memory**: 23.44 % | **DRAM**: 15.04 %
- **Achieved Occupancy**: 40.52 %
- **IPC Active**: 1.52 inst/cycle | **No Eligible**: 61.71 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_for_latency_hiding_and_occupancy | 0.914× | 0.945× | ✓ |  |
| B | autotune_xblock_to_hide_l1tex_latency | 0.913× | 0.946× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_ilp | 0.909× | 0.938× | ✓ |  |
| D | autotune_xblock_and_num_warps_for_latency_hiding | 0.996× | 0.322× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_f... | Δ | autotune_xblock_t... | Δ | increase_xblock_f... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21790 ns | 23070 ns | +1280 ns (+5.9%) | 23040 ns | +1250 ns (+5.7%) | 23230 ns | +1440 ns (+6.6%) | 67620 ns | +45830 ns (+210.3%) |
| SM Throughput (%) | 35.52 % | 58.71 % | +23.19 pp | 58.08 % | +22.56 pp | 58.67 % | +23.15 pp | 79.04 % | +43.52 pp |
| Memory Throughput (%) | 23.44 % | 34.74 % | +11.30 pp | 34.26 % | +10.82 pp | 34.78 % | +11.34 pp | 28.44 % | +5.00 pp |
| DRAM Throughput (%) | 15.04 % | 14.18 % | -0.86 pp | 14.21 % | -0.83 pp | 14.08 % | -0.96 pp | 4.84 % | -10.20 pp |
| Achieved Occupancy (%) | 40.52 % | 41.86 % | +1.34 pp | 42.30 % | +1.78 pp | 43.17 % | +2.65 pp | 76.96 % | +36.44 pp |
| Registers/Thread | 16.00 register/thread | 32.00 register/thread | +16 | 32.00 register/thread | +16 | 32.00 register/thread | +16 | 32.00 register/thread | +16 |
| Waves/SM | 14.85 | 14.85 | +0.00 | 14.85 | +0.00 | 14.85 | +0.00 | 59.39 | +44.54 |
| No Eligible (%) | 61.71 % | 33.00 % | -28.71 pp | 33.57 % | -28.14 pp | 33.90 % | -27.81 pp | 18.40 % | -43.31 pp |
| IPC Active | 1.52 inst/cycle | 2.66 inst/cycle | +1.14 | 2.63 inst/cycle | +1.11 | 2.64 inst/cycle | +1.12 | 3.26 inst/cycle | +1.74 |


### Key Learnings

#### ❌ Plan D: `autotune_xblock_and_num_warps_for_latency_hiding` (0.322×)

Increasing block size (XBLOCK) from 128 to 512 via autotune improved occupancy (40.5%→77%) and scheduler utilization (eligible warps 0.59→6.78), but caused a 7.3x explosion in executed instructions (5.77M→42.2M) and a 3.1x increase in kernel duration (21.8→67.6 µs). The autotune selected a large block config that dramatically increased waves_per_sm (14.85→59.39), meaning each SM processed far more work per launch with redundant computation. For small pointwise kernels where the original grid already provides sufficient parallelism (waves_per_sm ~15), increasing block size multiplies total work without proportional throughput gains, resulting in severe regression despite improved scheduler metrics.

> **Applicable when**: Never — this specific combination of autotune + mask introduction caused regression. The strategy of increasing block size to hide L1TEX latency is theoretically sound but failed here due to instruction count explosion and unnecessary mask overhead.

> **Avoid when**: Do not apply large XBLOCK autotune to pointwise kernels where: (1) the original grid already provides ~15+ waves_per_sm, (2) xnumel is exactly divisible by the original XBLOCK (making masks unnecessary overhead), and (3) the kernel body is simple (BN+ReLU) with low arithmetic intensity. Adding xmask to loads/stores that were previously mask-free introduces predication overhead that compounds with the instruction explosion from larger blocks.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_honor_autotuned_xblock | 0.911× | 0.940× | ✓ |  |
| B | fix_grid_lambda_and_add_small_xblock_autotune_configs | 0.996× | 1.004× | ✓ |  |
| C | reduce_register_pressure_to_double_sm_occupancy | 0.999× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | fix_grid_lambda_a... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21790 ns | 23170 ns | +1380 ns (+6.3%) | 21700 ns | -90 ns (-0.4%) | 21760 ns | -30 ns (-0.1%) |
| SM Throughput (%) | 35.52 % | 58.29 % | +22.77 pp | 42.63 % | +7.11 pp | 23.71 % | -11.81 pp |
| Memory Throughput (%) | 23.44 % | 34.41 % | +10.97 pp | 23.67 % | +0.23 pp | 21.44 % | -2.00 pp |
| DRAM Throughput (%) | 15.04 % | 14.13 % | -0.91 pp | 15.08 % | +0.04 pp | 15.05 % | +0.01 pp |
| Achieved Occupancy (%) | 40.52 % | 41.67 % | +1.15 pp | 42.91 % | +2.39 pp | 26.83 % | -13.69 pp |
| Registers/Thread | 16.00 register/thread | 32.00 register/thread | +16 | 17.00 register/thread | +1 | 22.00 register/thread | +6 |
| Waves/SM | 14.85 | 14.85 | +0.00 | 14.85 | +0.00 | 7.42 | -7.43 |
| No Eligible (%) | 61.71 % | 33.27 % | -28.44 pp | 51.74 % | -9.97 pp | 71.65 % | +9.94 pp |
| IPC Active | 1.52 inst/cycle | 2.68 inst/cycle | +1.16 | 1.93 inst/cycle | +0.41 | 1.13 inst/cycle | -0.39 |

---

## Cumulative Progress

```
Baseline:  21790 ns  (1.000×)

Final best: 21790 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,26 +5,38 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 1024}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=4),
+        triton.Config({'XBLOCK': 2048}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 2048}, num_warps=8,  num_stages=4),
+        triton.Config({'XBLOCK': 4096}, num_warps=8,  num_stages=4),
+        triton.Config({'XBLOCK': 4096}, num_warps=16, num_stages=4),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    # xnumel=3010560 is exactly divisible by XBLOCK=128 (3010560/128=23520) — no mask needed
-    x2 = xindex
+    xmask = xindex < xnumel
     x0 = (xindex % 32)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None, eviction_policy='evict_first')
-    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + xindex, xmask, eviction_policy='evict_first')
+    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
     # FMA-friendly BN normalization
     inv_std = tl.rsqrt(tmp3 + 1e-5)
     # tl.math.fma(a,b,c) = a*b+c as single FFMA PTX instruction
     norm   = tl.math.fma(-tmp1, inv_std, tmp0 * inv_std)  # (tmp0-tmp1)*inv_std
     out    = tl.math.fma(norm, tmp12, tmp14)               # norm*tmp12+tmp14
     result = tl.maximum(0.0, out)                          # ReLU
-    tl.store(in_out_ptr0 + (x2), result, None, eviction_policy='evict_first')
-
+    tl.store(in_out_ptr0 + xindex, result, xmask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -34,7 +46,7 @@

     arg_4 = inputs[4]
     xnumel = 4014080
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_relu_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 4014080, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_relu_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 4014080)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
