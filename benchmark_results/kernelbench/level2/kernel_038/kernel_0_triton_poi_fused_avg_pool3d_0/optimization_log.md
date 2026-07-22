# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_avg_pool3d_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 136900 ns — **Best latency**: 136640 ns (**1.0019×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`fix_eviction_policy_and_autotune_xblock_for_coalescing`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_avg_pool3d_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_038/kernel_0_triton_poi_fused_avg_pool3d_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (131072,1,1), Waves/SM: 62.06
- **Registers/thread**: 20
- **Duration**: 136900 ns
- **SM Throughput**: 36.79 % | **Memory**: 89.47 % | **DRAM**: 89.47 %
- **Achieved Occupancy**: 84.32 %
- **IPC Active**: 1.47 inst/cycle | **No Eligible**: 63.49 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguity_hints | 1.000× | 0.995× | ✓ |  |
| B | fix_eviction_policy_and_autotune_xblock_for_coalescing | 1.001× | 1.002× | ✓ | **✓ best** |
| C | memory_coalescing_via_autotune_xblock | 0.999× | 0.257× | ✓ |  |
| D | increase_xblock_and_fix_eviction_policy_for_coalescing | 0.465× | 0.406× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | fix_eviction_poli... | Δ | memory_coalescing... | Δ | increase_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 136900 ns | 137630 ns | +730 ns (+0.5%) | 136640 ns | -260 ns (-0.2%) | 533090 ns | +396190 ns (+289.4%) | 336990 ns | +200090 ns (+146.2%) |
| SM Throughput (%) | 36.79 % | 25.87 % | -10.92 pp | 48.30 % | +11.51 pp | 84.39 % | +47.60 pp | 60.27 % | +23.48 pp |
| Memory Throughput (%) | 89.47 % | 86.18 % | -3.29 pp | 86.80 % | -2.67 pp | 47.90 % | -41.57 pp | 72.92 % | -16.55 pp |
| DRAM Throughput (%) | 89.47 % | 86.18 % | -3.29 pp | 86.80 % | -2.67 pp | 22.96 % | -66.51 pp | 35.46 % | -54.01 pp |
| Achieved Occupancy (%) | 84.32 % | 83.91 % | -0.41 pp | 84.13 % | -0.19 pp | 85.38 % | +1.06 pp | 68.61 % | -15.71 pp |
| Registers/Thread | 20.00 register/thread | 16.00 register/thread | -4 | 22.00 register/thread | +2 | 32.00 register/thread | +12 | 40.00 register/thread | +20 |
| Waves/SM | 62.06 | 62.06 | +0.00 | 62.06 | +0.00 | 248.24 | +186.18 | 82.75 | +20.69 |
| No Eligible (%) | 63.49 % | 74.14 % | +10.65 pp | 50.90 % | -12.59 pp | 15.07 % | -48.42 pp | 39.36 % | -24.13 pp |
| IPC Active | 1.47 inst/cycle | 1.07 inst/cycle | -0.40 | 2.04 inst/cycle | +0.57 | 3.39 inst/cycle | +1.92 | 2.42 inst/cycle | +0.95 |

---

## Iteration 2

**Best so far**: 1.0019× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reindex_for_coalesced_input_loads_via_input_space_tiling | 0.999× | 1.004× | ✓ |  |
| C | restore_low_register_pressure_by_removing_large_xblock_autotune | 0.999× | 0.990× | ✓ |  |
| D | remove_large_xblock_autotune_restore_fixed_small_tile | 0.999× | 1.004× | ✓ |  |
| E | add_triton_autotune | 0.999× | 1.008× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reindex_for_coale... | Δ | restore_low_regis... | Δ | remove_large_xblo... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 136900 ns | 136420 ns | -480 ns (-0.4%) | 138240 ns | +1340 ns (+1.0%) | 136380 ns | -520 ns (-0.4%) | 135780 ns | -1120 ns (-0.8%) |
| SM Throughput (%) | 36.79 % | 25.05 % | -11.74 pp | 33.45 % | -3.34 pp | 37.00 % | +0.21 pp | 49.23 % | +12.44 pp |
| Memory Throughput (%) | 89.47 % | 89.76 % | +0.29 pp | 88.56 % | -0.91 pp | 89.78 % | +0.31 pp | 87.39 % | -2.08 pp |
| DRAM Throughput (%) | 89.47 % | 89.76 % | +0.29 pp | 88.56 % | -0.91 pp | 89.78 % | +0.31 pp | 87.39 % | -2.08 pp |
| Achieved Occupancy (%) | 84.32 % | 84.27 % | -0.05 pp | 84.57 % | +0.25 pp | 84.56 % | +0.24 pp | 83.89 % | -0.43 pp |
| Registers/Thread | 20.00 register/thread | 16.00 register/thread | -4 | 20.00 register/thread | +0 | 20.00 register/thread | +0 | 22.00 register/thread | +2 |
| Waves/SM | 62.06 | 62.06 | +0.00 | 62.06 | +0.00 | 62.06 | +0.00 | 62.06 | +0.00 |
| No Eligible (%) | 63.49 % | 74.67 % | +11.18 pp | 66.09 % | +2.60 pp | 62.03 % | -1.46 pp | 49.25 % | -14.24 pp |
| IPC Active | 1.47 inst/cycle | 0.99 inst/cycle | -0.48 | 1.33 inst/cycle | -0.14 | 1.47 inst/cycle | +0.00 | 2.03 inst/cycle | +0.56 |

---

## Iteration 3

**Best so far**: 1.0019× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_paired_loads_via_contiguity_hints_on_adjacent_pool_elements | 1.000× | 0.991× | ✓ |  |
| B | coalesced_load_via_paired_contiguous_loads_with_multiple_of_hint | 0.998× | 0.989× | ✓ |  |
| D | vectorized_loads_via_max_contiguous_and_multiple_of_hints | 0.998× | 1.004× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_paired... | Δ | coalesced_load_vi... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 136900 ns | 138110 ns | +1210 ns (+0.9%) | 138430 ns | +1530 ns (+1.1%) | 136290 ns | -610 ns (-0.4%) |
| SM Throughput (%) | 36.79 % | 35.86 % | -0.93 pp | 25.40 % | -11.39 pp | 32.50 % | -4.29 pp |
| Memory Throughput (%) | 89.47 % | 85.87 % | -3.60 pp | 88.47 % | -1.00 pp | 89.85 % | +0.38 pp |
| DRAM Throughput (%) | 89.47 % | 85.87 % | -3.60 pp | 88.47 % | -1.00 pp | 89.85 % | +0.38 pp |
| Achieved Occupancy (%) | 84.32 % | 83.95 % | -0.37 pp | 84.09 % | -0.23 pp | 84.45 % | +0.13 pp |
| Registers/Thread | 20.00 register/thread | 16.00 register/thread | -4 | 16.00 register/thread | -4 | 20.00 register/thread | +0 |
| Waves/SM | 62.06 | 62.06 | +0.00 | 62.06 | +0.00 | 62.06 | +0.00 |
| No Eligible (%) | 63.49 % | 63.25 % | -0.24 pp | 73.26 % | +9.77 pp | 67.66 % | +4.17 pp |
| IPC Active | 1.47 inst/cycle | 1.51 inst/cycle | +0.04 | 1.06 inst/cycle | -0.41 | 1.29 inst/cycle | -0.18 |

---

## Cumulative Progress

```
Baseline:  136900 ns  (1.000×)
Iter 1     136640 ns  (1.002×)  — fix_eviction_policy_and_autotune_xblock_for_coalescing

Final best: 136640 ns  (1.0019×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,31 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 128},  num_warps=4),
+    triton.Config({'XBLOCK': 256},  num_warps=4),
+    triton.Config({'XBLOCK': 512},  num_warps=8),
+    triton.Config({'XBLOCK': 1024}, num_warps=8),
+], key=['xnumel'])
 @triton.jit
 def triton_poi_fused_avg_pool3d_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 16777216
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 32)
     x1 = ((xindex // 32) % 32)
     x2 = xindex // 1024
     x3 = xindex
-    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
-    tmp7 = tl.load(in_ptr0 + (4096 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
-    tmp9 = tl.load(in_ptr0 + (4097 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
-    tmp11 = tl.load(in_ptr0 + (4160 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
-    tmp13 = tl.load(in_ptr0 + (4161 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
+    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
+    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
+    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
+    tmp7 = tl.load(in_ptr0 + (4096 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
+    tmp9 = tl.load(in_ptr0 + (4097 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
+    tmp11 = tl.load(in_ptr0 + (4160 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
+    tmp13 = tl.load(in_ptr0 + (4161 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
     tmp2 = tmp0 + tmp1
     tmp4 = tmp2 + tmp3
     tmp6 = tmp4 + tmp5
@@ -32,16 +39,14 @@

     tmp14 = tmp12 + tmp13
     tmp15 = tl.full([1], 8.0, tl.float32)
     tmp16 = (tmp14 / tmp15)
-    tl.store(out_ptr0 + (x3), tmp16, None)
-
-
+    tl.store(out_ptr0 + (x3), tmp16, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 16777216
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_avg_pool3d_0[grid](arg_0, arg_1, 16777216, XBLOCK=128)
+    triton_poi_fused_avg_pool3d_0[grid](arg_0, arg_1, 16777216)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
