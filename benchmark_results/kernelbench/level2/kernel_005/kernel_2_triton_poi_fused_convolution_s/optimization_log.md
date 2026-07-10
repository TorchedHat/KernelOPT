# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_sub_tanh_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1630000 ns — **Best latency**: 1610000 ns (**1.0124×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`remove_incorrect_hints_and_pipeline_latency_hiding`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_sub_tanh_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_005/kernel_2_triton_poi_fused_convolution_s/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (526592,1,1), Waves/SM: 249.33
- **Registers/thread**: 32
- **Duration**: 1630000 ns
- **SM Throughput**: 52.22 % | **Memory**: 69.27 % | **DRAM**: 65.68 %
- **Achieved Occupancy**: 95.89 %
- **IPC Active**: 2.08 inst/cycle | **No Eligible**: 47.86 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_tile_transpose_for_coalesced_access | 1.025× | 1.006× | ✓ | **✓ best** |
| B | shared_memory_tile_transpose_for_coalesced_io | 0.997× | 1.006× | ✓ |  |
| D | increase_tile_sizes_reduce_grid_overhead | 0.998× | 1.006× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_til... | Δ | shared_memory_til... | Δ | increase_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1630000 ns | 1620000 ns | -10000 ns (-0.6%) | 1620000 ns | -10000 ns (-0.6%) | 1620000 ns | -10000 ns (-0.6%) |
| SM Throughput (%) | 52.22 % | 52.06 % | -0.16 pp | 52.35 % | +0.13 pp | 52.32 % | +0.10 pp |
| Memory Throughput (%) | 69.27 % | 69.79 % | +0.52 pp | 70.40 % | +1.13 pp | 69.62 % | +0.35 pp |
| DRAM Throughput (%) | 65.68 % | 66.16 % | +0.48 pp | 66.05 % | +0.37 pp | 66.08 % | +0.40 pp |
| Achieved Occupancy (%) | 95.89 % | 95.86 % | -0.03 pp | 95.84 % | -0.05 pp | 95.85 % | -0.04 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 249.33 | 249.33 | +0.00 | 249.33 | +0.00 | 249.33 | +0.00 |
| No Eligible (%) | 47.86 % | 47.97 % | +0.11 pp | 47.30 % | -0.56 pp | 47.87 % | +0.01 pp |
| IPC Active | 2.08 inst/cycle | 2.10 inst/cycle | +0.02 | 2.10 inst/cycle | +0.02 | 2.10 inst/cycle | +0.02 |

---

## Iteration 3

**Best so far**: 1.0062× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesced_channel_inner_tiling_fixed_config | 1.025× | 1.006× | ✓ | **✓ best** |
| B | vectorized_loads_with_contiguity_hints | 0.996× | 1.000× | ✓ |  |
| C | vectorized_loads_contiguity_hints_fixed_config | 0.698× | 0.676× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesced_channel... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1630000 ns | 1620000 ns | -10000 ns (-0.6%) | 1630000 ns | +0 ns (+0.0%) | 2410000 ns | +780000 ns (+47.9%) |
| SM Throughput (%) | 52.22 % | 51.39 % | -0.83 pp | 52.12 % | -0.10 pp | 34.77 % | -17.45 pp |
| Memory Throughput (%) | 69.27 % | 69.54 % | +0.27 pp | 69.81 % | +0.54 pp | 54.44 % | -14.83 pp |
| DRAM Throughput (%) | 65.68 % | 65.94 % | +0.26 pp | 65.54 % | -0.14 pp | 54.44 % | -11.24 pp |
| Achieved Occupancy (%) | 95.89 % | 95.88 % | -0.01 pp | 95.85 % | -0.04 pp | 96.18 % | +0.29 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 249.33 | 249.33 | +0.00 | 249.33 | +0.00 | 249.33 | +0.00 |
| No Eligible (%) | 47.86 % | 48.27 % | +0.41 pp | 47.53 % | -0.33 pp | 65.32 % | +17.46 pp |
| IPC Active | 2.08 inst/cycle | 2.06 inst/cycle | -0.02 | 2.12 inst/cycle | +0.04 | 1.39 inst/cycle | -0.69 |

---

## Iteration 4

**Best so far**: 1.0062× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | vectorized_loads_channel_dim_contiguity_hints | 0.995× | 1.006× | ✓ |  |
| C | increase_num_warps_and_pipeline_stages_fixed_config | 0.698× | 0.674× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 1630000 ns | 1620000 ns | -10000 ns (-0.6%) | 2420000 ns | +790000 ns (+48.5%) |
| SM Throughput (%) | 52.22 % | 52.27 % | +0.05 pp | 34.76 % | -17.46 pp |
| Memory Throughput (%) | 69.27 % | 70.22 % | +0.95 pp | 54.32 % | -14.95 pp |
| DRAM Throughput (%) | 65.68 % | 65.87 % | +0.19 pp | 54.32 % | -11.36 pp |
| Achieved Occupancy (%) | 95.89 % | 95.86 % | -0.03 pp | 96.20 % | +0.31 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 249.33 | 249.33 | +0.00 | 249.33 | +0.00 |
| No Eligible (%) | 47.86 % | 47.35 % | -0.51 pp | 64.75 % | +16.89 pp |
| IPC Active | 2.08 inst/cycle | 2.11 inst/cycle | +0.03 | 1.39 inst/cycle | -0.69 |

---

## Iteration 5

**Best so far**: 1.0062× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | remove_incorrect_hints_and_pipeline_latency_hiding | 2.343× | 1.012× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | remove_incorrect_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 1630000 ns | 1610000 ns | -20000 ns (-1.2%) |
| SM Throughput (%) | 52.22 % | 52.65 % | +0.43 pp |
| Memory Throughput (%) | 69.27 % | 69.83 % | +0.56 pp |
| DRAM Throughput (%) | 65.68 % | 66.12 % | +0.44 pp |
| Achieved Occupancy (%) | 95.89 % | 95.87 % | -0.02 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 249.33 | 249.33 | +0.00 |
| No Eligible (%) | 47.86 % | 47.75 % | -0.11 pp |
| IPC Active | 2.08 inst/cycle | 2.07 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  1630000 ns  (1.000×)
Iter 2     1620000 ns  (1.006×)  — shared_memory_tile_transpose_for_coalesced_access
Iter 5     1610000 ns  (1.012×)  — remove_incorrect_hints_and_pipeline_latency_hiding

Final best: 1610000 ns  (1.0124×)
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
 def triton_poi_fused_convolution_sub_tanh_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
@@ -19,15 +20,13 @@

     y0 = (yindex % 64)
     y1 = yindex // 64
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 64*x2 + 16842816*y1), xmask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (y0 + 64*x2 + 16842816*y1), mask=xmask & ymask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr1 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr2 + (y0), mask=ymask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp4 = tmp2 - tmp3
     tmp5 = libdevice.tanh(tmp4)
-    tl.store(out_ptr0 + (x2 + 263169*y3), tmp5, xmask)
-
-
+    tl.store(out_ptr0 + (x2 + 263169*y3), tmp5, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
