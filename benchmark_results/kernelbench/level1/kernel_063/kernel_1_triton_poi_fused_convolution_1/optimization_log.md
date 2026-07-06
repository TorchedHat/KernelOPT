# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 285120 ns — **Best latency**: 285060 ns (**1.0002×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`memory_coalescing_swap_block_dimensions`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_063/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 285120 ns
- **SM Throughput**: 19.82 % | **Memory**: 79.05 % | **DRAM**: 72.37 %
- **Achieved Occupancy**: 94.06 %
- **IPC Active**: 0.79 inst/cycle | **No Eligible**: 80.11 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | memory_coalescing_swap_block_dimensions | 1.000× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 285120 ns | 285060 ns | -60 ns (-0.0%) |
| SM Throughput (%) | 19.82 % | 20.05 % | +0.23 pp |
| Memory Throughput (%) | 79.05 % | 79.05 % | +0.00 pp |
| DRAM Throughput (%) | 72.37 % | 72.38 % | +0.01 pp |
| Achieved Occupancy (%) | 94.06 % | 94.06 % | +0.00 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 80.11 % | 80.17 % | +0.06 pp |
| IPC Active | 0.79 inst/cycle | 0.79 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0002× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_stores_with_contiguous_hints | 0.985× | 0.985× | ✓ |  |
| B | shared_memory_staged_transpose_for_coalesced_access | 0.985× | 0.983× | ✓ |  |
| D | increase_yblock_reduce_xblock_for_coalesced_loads | 1.000× | 0.997× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_stores... | Δ | shared_memory_sta... | Δ | increase_yblock_r... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 285120 ns | 289540 ns | +4420 ns (+1.6%) | 290140 ns | +5020 ns (+1.8%) | 286080 ns | +960 ns (+0.3%) |
| SM Throughput (%) | 19.82 % | 18.02 % | -1.80 pp | 18.19 % | -1.63 pp | 20.06 % | +0.24 pp |
| Memory Throughput (%) | 79.05 % | 77.98 % | -1.07 pp | 77.79 % | -1.26 pp | 78.79 % | -0.26 pp |
| DRAM Throughput (%) | 72.37 % | 71.28 % | -1.09 pp | 71.14 % | -1.23 pp | 72.12 % | -0.25 pp |
| Achieved Occupancy (%) | 94.06 % | 93.90 % | -0.16 pp | 93.91 % | -0.15 pp | 94.02 % | -0.04 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.11 % | 81.46 % | +1.35 pp | 81.29 % | +1.18 pp | 80.25 % | +0.14 pp |
| IPC Active | 0.79 inst/cycle | 0.74 inst/cycle | -0.05 | 0.74 inst/cycle | -0.05 | 0.80 inst/cycle | +0.01 |

---

## Iteration 3

**Best so far**: 1.0002× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | vectorized_loads_with_correct_contiguous_hints | 0.987× | 0.994× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 285120 ns | 286750 ns | +1630 ns (+0.6%) |
| SM Throughput (%) | 19.82 % | 42.97 % | +23.15 pp |
| Memory Throughput (%) | 79.05 % | 78.14 % | -0.91 pp |
| DRAM Throughput (%) | 72.37 % | 17.13 % | -55.24 pp |
| Achieved Occupancy (%) | 94.06 % | 70.42 % | -23.64 pp |
| Registers/Thread | 25.00 register/thread | 40.00 register/thread | +15 |
| Waves/SM | 59.88 | 79.84 | +19.96 |
| No Eligible (%) | 80.11 % | 56.42 % | -23.69 pp |
| IPC Active | 0.79 inst/cycle | 1.75 inst/cycle | +0.96 |

---

## Cumulative Progress

```
Baseline:  285120 ns  (1.000×)
Iter 1     285060 ns  (1.000×)  — memory_coalescing_swap_block_dimensions

Final best: 285060 ns  (1.0002×)
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
 def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
@@ -19,10 +20,8 @@

     y0 = (yindex % 128)
     y1 = yindex // 128
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), mask=xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
