# Kernel Optimization Log

## Summary

- **Kernel**: `max_pool1d_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4570000 ns — **Best latency**: 4160000 ns (**1.0986×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`add_triton_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `max_pool1d_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_041/synthesis/kernel_0_triton_poi_fused_max_pool2d_with_indices.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (786276,1,1), Waves/SM: 496.39
- **Registers/thread**: 40
- **Duration**: 4570000 ns
- **SM Throughput**: 88.59 % | **Memory**: 97.25 % | **DRAM**: 28.57 %
- **Achieved Occupancy**: 71.33 %
- **IPC Active**: 3.26 inst/cycle | **No Eligible**: 18.39 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_uncoalesced_memory_access_via_larger_block_size_autotune | 0.934× | 0.856× | ✓ |  |
| C | vectorized_loads_and_reordered_index_decomposition | 1.005× | 0.994× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_uncoalesced_m... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4570000 ns | 5340000 ns | +770000 ns (+16.8%) | 4600000 ns | +30000 ns (+0.7%) |
| SM Throughput (%) | 88.59 % | 76.37 % | -12.22 pp | 87.98 % | -0.61 pp |
| Memory Throughput (%) | 97.25 % | 83.02 % | -14.23 pp | 98.06 % | +0.81 pp |
| DRAM Throughput (%) | 28.57 % | 24.46 % | -4.11 pp | 28.42 % | -0.15 pp |
| Achieved Occupancy (%) | 71.33 % | 46.86 % | -24.47 pp | 57.83 % | -13.50 pp |
| Registers/Thread | 40.00 register/thread | 64.00 register/thread | +24 | 48.00 register/thread | +8 |
| Waves/SM | 496.39 | 744.58 | +248.19 | 595.66 | +99.27 |
| No Eligible (%) | 18.39 % | 27.74 % | +9.35 pp | 16.12 % | -2.27 pp |
| IPC Active | 3.26 inst/cycle | 2.89 inst/cycle | -0.37 | 3.36 inst/cycle | +0.10 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | reduce_register_pressure_by_replacing_static_range_with_dynamic_range | 1.005× | 1.020× | ✓ |  |
| E | add_triton_autotune | 1.009× | 1.099× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4570000 ns | 4480000 ns | -90000 ns (-2.0%) | 4160000 ns | -410000 ns (-9.0%) |
| SM Throughput (%) | 88.59 % | 89.02 % | +0.43 pp | 87.88 % | -0.71 pp |
| Memory Throughput (%) | 97.25 % | 55.86 % | -41.39 pp | 34.91 % | -62.34 pp |
| DRAM Throughput (%) | 28.57 % | 29.18 % | +0.61 pp | 31.42 % | +2.85 pp |
| Achieved Occupancy (%) | 71.33 % | 90.63 % | +19.30 pp | 84.67 % | +13.34 pp |
| Registers/Thread | 40.00 register/thread | 32.00 register/thread | -8 | 25.00 register/thread | -15 |
| Waves/SM | 496.39 | 1489.16 | +992.77 | 2978.32 | +2481.93 |
| No Eligible (%) | 18.39 % | 10.94 % | -7.45 pp | 12.10 % | -6.29 pp |
| IPC Active | 3.26 inst/cycle | 3.56 inst/cycle | +0.30 | 3.52 inst/cycle | +0.26 |

---

## Iteration 3

**Best so far**: 1.0986× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | add_triton_autotune | 1.005× | 1.099× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_triton_autotu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 4570000 ns | 4160000 ns | -410000 ns (-9.0%) |
| SM Throughput (%) | 88.59 % | 87.85 % | -0.74 pp |
| Memory Throughput (%) | 97.25 % | 34.90 % | -62.35 pp |
| DRAM Throughput (%) | 28.57 % | 31.41 % | +2.84 pp |
| Achieved Occupancy (%) | 71.33 % | 84.73 % | +13.40 pp |
| Registers/Thread | 40.00 register/thread | 25.00 register/thread | -15 |
| Waves/SM | 496.39 | 2978.32 | +2481.93 |
| No Eligible (%) | 18.39 % | 12.08 % | -6.31 pp |
| IPC Active | 3.26 inst/cycle | 3.52 inst/cycle | +0.26 |

---

## Iteration 4

**Best so far**: 1.0986× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  4570000 ns  (1.000×)
Iter 2     4160000 ns  (1.099×)  — add_triton_autotune

Final best: 4160000 ns  (1.0986×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,16 +3,13 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
+        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
         triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
+        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
         triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 128}, num_warps=8),
     ],
     key=['total_elements'],
 )
@@ -32,7 +29,7 @@

     BLOCK_SIZE: tl.constexpr,
 ):
     pid = tl.program_id(0)
-    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
+    offsets = tl.multiple_of(pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
     mask = offsets < total_elements
 
     # Decompose flat index into (batch, channel, out_pos)
@@ -55,15 +52,14 @@

     max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
 
     # Unroll over kernel positions
-    for k in range(kernel_size):
+    for k in tl.static_range(kernel_size):
         in_pos = in_start + k * dilation
         valid = (in_pos >= 0) & (in_pos < in_length) & mask
-        load_offset = base_offset + in_pos
+        load_offset = tl.multiple_of(base_offset + in_pos, 1)
         val = tl.load(in_ptr + load_offset, mask=valid, other=float('-inf'))
         max_val = tl.maximum(max_val, val)
 
     tl.store(out_ptr + offsets, max_val, mask=mask)
-
 
 def kernel_function(x):
     # Parameters matching the model
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
