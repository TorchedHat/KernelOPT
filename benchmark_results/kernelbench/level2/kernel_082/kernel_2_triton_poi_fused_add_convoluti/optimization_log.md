# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_convolution_mul_tanh_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2540000 ns — **Best latency**: 2530000 ns (**1.0040×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`persistent_tile_scheduling_manual_sm_loop`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_convolution_mul_tanh_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_082/kernel_2_triton_poi_fused_add_convoluti/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4129024,1,1), Waves/SM: 1955.03
- **Registers/thread**: 16
- **Duration**: 2540000 ns
- **SM Throughput**: 44.43 % | **Memory**: 33.66 % | **DRAM**: 33.66 %
- **Achieved Occupancy**: 51.59 %
- **IPC Active**: 1.78 inst/cycle | **No Eligible**: 55.53 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_manual_sm_loop | 1.003× | 1.004× | ✓ | **✓ best** |
| B | manual_xblock_increase_to_hide_l1tex_latency | 0.999× | 1.000× | ✓ |  |
| D | vectorized_loads_stores_via_larger_xblock | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | manual_xblock_inc... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2540000 ns | 2530000 ns | -10000 ns (-0.4%) | 2540000 ns | +0 ns (+0.0%) | 2540000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 44.43 % | 15.67 % | -28.76 pp | 44.35 % | -0.08 pp | 44.35 % | -0.08 pp |
| Memory Throughput (%) | 33.66 % | 4.04 % | -29.62 pp | 33.66 % | +0.00 pp | 33.66 % | +0.00 pp |
| DRAM Throughput (%) | 33.66 % | 4.04 % | -29.62 pp | 33.66 % | +0.00 pp | 33.66 % | +0.00 pp |
| Achieved Occupancy (%) | 51.59 % | 14.36 % | -37.23 pp | 51.58 % | -0.01 pp | 51.66 % | +0.07 pp |
| Registers/Thread | 16.00 register/thread | 22.00 register/thread | +6 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 1955.03 | 1955.03 | +0.00 | 1955.03 | +0.00 | 1955.03 | +0.00 |
| No Eligible (%) | 55.53 % | 85.62 % | +30.09 pp | 55.61 % | +0.08 pp | 55.53 % | +0.00 pp |
| IPC Active | 1.78 inst/cycle | 0.46 inst/cycle | -1.32 | 1.78 inst/cycle | +0.00 | 1.78 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.004× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_persistent_kernel_num_tiles_and_increase_num_warps | 1.000× | 1.000× | ✓ |  |
| B | increase_num_warps_fixed_config_for_l1tex_latency_hiding | 1.000× | 1.000× | ✓ |  |
| C | increase_xblock_fixed_no_autotune_latency_hiding | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_persistent_ke... | Δ | increase_num_warp... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2540000 ns | 2540000 ns | +0 ns (+0.0%) | 2540000 ns | +0 ns (+0.0%) | 2540000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 44.43 % | 48.47 % | +4.04 pp | 44.34 % | -0.09 pp | 44.34 % | -0.09 pp |
| Memory Throughput (%) | 33.66 % | 33.67 % | +0.01 pp | 33.66 % | +0.00 pp | 33.66 % | +0.00 pp |
| DRAM Throughput (%) | 33.66 % | 33.67 % | +0.01 pp | 33.66 % | +0.00 pp | 33.66 % | +0.00 pp |
| Achieved Occupancy (%) | 51.59 % | 49.41 % | -2.18 pp | 51.64 % | +0.05 pp | 51.65 % | +0.06 pp |
| Registers/Thread | 16.00 register/thread | 22.00 register/thread | +6 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 1955.03 | 1955.03 | +0.00 | 1955.03 | +0.00 | 1955.03 | +0.00 |
| No Eligible (%) | 55.53 % | 51.40 % | -4.13 pp | 55.55 % | +0.02 pp | 55.62 % | +0.09 pp |
| IPC Active | 1.78 inst/cycle | 1.94 inst/cycle | +0.16 | 1.78 inst/cycle | +0.00 | 1.78 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.004× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_libdevice_tanh_with_fast_polynomial_approximation | 1.000× | 1.000× | ✓ |  |
| B | static_fixed_config_xblock_increase_no_autotune | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_libdevice... | Δ | static_fixed_conf... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2540000 ns | 2540000 ns | +0 ns (+0.0%) | 2540000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 44.43 % | 49.29 % | +4.86 pp | 44.34 % | -0.09 pp |
| Memory Throughput (%) | 33.66 % | 33.67 % | +0.01 pp | 33.66 % | +0.00 pp |
| DRAM Throughput (%) | 33.66 % | 33.67 % | +0.01 pp | 33.66 % | +0.00 pp |
| Achieved Occupancy (%) | 51.59 % | 49.45 % | -2.14 pp | 51.66 % | +0.07 pp |
| Registers/Thread | 16.00 register/thread | 20.00 register/thread | +4 | 16.00 register/thread | +0 |
| Waves/SM | 1955.03 | 1955.03 | +0.00 | 1955.03 | +0.00 |
| No Eligible (%) | 55.53 % | 50.67 % | -4.86 pp | 55.61 % | +0.08 pp |
| IPC Active | 1.78 inst/cycle | 1.98 inst/cycle | +0.20 | 1.78 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  2540000 ns  (1.000×)
Iter 2     2530000 ns  (1.004×)  — persistent_tile_scheduling_manual_sm_loop

Final best: 2530000 ns  (1.0040×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,25 +5,28 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
-def triton_poi_fused_add_convolution_mul_tanh_2(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
-    xnumel = 528515072
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
-    x2 = xindex
-    x0 = (xindex % 64)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None)
-    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp6 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp2 = tmp0 + tmp1
-    tmp3 = libdevice.tanh(tmp2)
-    tmp4 = tl.full([1], 2.0, tl.float32)
-    tmp5 = tmp3 * tmp4
-    tmp7 = tmp5 + tmp6
-    tl.store(in_out_ptr0 + (x2), tmp7, None)
-
-
+def triton_poi_fused_add_convolution_mul_tanh_2(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK: tl.constexpr):
+    # Persistent kernel: compute num_tiles from xnumel and XBLOCK at runtime
+    num_tiles = tl.cdiv(xnumel, XBLOCK)
+    pid = tl.program_id(0)
+    num_programs = tl.num_programs(0)
+    tile_id = pid
+    while tile_id < num_tiles:
+        xoffset = tile_id * XBLOCK
+        xindex = xoffset + tl.arange(0, XBLOCK)[:]
+        x0 = xindex % 64
+        tmp0 = tl.load(in_out_ptr0 + xindex, None)
+        tmp1 = tl.load(in_ptr0 + x0, None, eviction_policy='evict_last')
+        tmp6 = tl.load(in_ptr1 + x0, None, eviction_policy='evict_last')
+        tmp2 = tmp0 + tmp1
+        tmp3 = libdevice.tanh(tmp2)
+        tmp4 = tl.full([1], 2.0, tl.float32)
+        tmp5 = tmp3 * tmp4
+        tmp7 = tmp5 + tmp6
+        tl.store(in_out_ptr0 + xindex, tmp7, None)
+        tile_id += num_programs
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
