# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6020 ns — **Best latency**: 3300 ns (**1.8242×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`fix_autotune_bypass_and_eviction_policy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_082/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6020 ns
- **SM Throughput**: 16.06 % | **Memory**: 41.20 % | **DRAM**: 20.50 %
- **Achieved Occupancy**: 55.07 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 71.55 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.029× | 1.022× | ✓ |  |
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.452× | 1.824× | ✓ | **✓ best** |
| D | autotune_tile_sizes_for_occupancy_and_coalescing | 1.338× | 1.427× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 5890 ns | -130 ns (-2.2%) | 3300 ns | -2720 ns (-45.2%) | 4220 ns | -1800 ns (-29.9%) |
| SM Throughput (%) | 16.06 % | 16.79 % | +0.73 pp | 17.63 % | +1.57 pp | 26.90 % | +10.84 pp |
| Memory Throughput (%) | 41.20 % | 42.27 % | +1.07 pp | 20.71 % | -20.49 pp | 21.36 % | -19.84 pp |
| DRAM Throughput (%) | 20.50 % | 20.96 % | +0.46 pp | 5.01 % | -15.49 pp | 3.92 % | -16.58 pp |
| Achieved Occupancy (%) | 55.07 % | 55.40 % | +0.33 pp | 43.98 % | -11.09 pp | 69.13 % | +14.06 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 71.55 % | 70.91 % | -0.64 pp | 58.40 % | -13.15 pp | 37.40 % | -34.15 pp |
| IPC Active | 1.14 inst/cycle | 1.14 inst/cycle | +0.00 | 1.70 inst/cycle | +0.56 | 2.41 inst/cycle | +1.27 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_for_coalescing_and_occupancy` (1.824×)

When a Triton pointwise kernel shows high L1TEX stall cycles (CPIStall ~49% of warp cycles) and low eligible warps per scheduler (~0.91), adding a @triton.autotune decorator with a range of XBLOCK/YBLOCK/num_warps/num_stages configurations allows the runtime to discover tile shapes that reduce register pressure (26→18 registers/thread) and halve warp stall cycles (30→15.6 cycles/instruction). The winning configuration also improved L2 hit rate from 60.8% to 77.0% and cut elapsed cycles by 45%, yielding a 1.82x speedup despite slightly lower achieved occupancy. This strategy is especially effective when the kernel has a strided output layout causing uncoalesced stores, as smaller YBLOCK values can eliminate intra-block stride scatter by making the modular y-index constant per block.

> **Applicable when**: Triton pointwise kernel with: (1) CPIStall rule triggered with L1TEX stall >40% of warp cycles, (2) eligible warps per scheduler <1.0, (3) No Eligible scheduler cycles >60%, (4) hardcoded XBLOCK/YBLOCK tile sizes, (5) strided or non-unit-stride output layout causing UncoalescedGlobalAccess. Particularly effective when ynumel is small (e.g. <64) and the y-index modular arithmetic creates intra-block stride scatter.

> **Avoid when**: Kernel is already autotuned or tile sizes are already optimal. Kernel is compute-bound (Compute SM Throughput >70%) rather than memory-latency-bound. Grid is already large enough for >2 waves/SM and occupancy is near theoretical. Autotuning overhead is unacceptable (e.g. very short-lived kernels called once).

---

## Iteration 2

**Best so far**: 1.8242× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_bypass_and_eviction_policy | 1.337× | 1.824× | ✓ |  |
| C | software_pipelining_with_evict_first_and_yblock1 | 0.902× | 0.960× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_bypa... | Δ | software_pipelini... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 3300 ns | -2720 ns (-45.2%) | 6270 ns | +250 ns (+4.2%) |
| SM Throughput (%) | 16.06 % | 17.76 % | +1.70 pp | 16.82 % | +0.76 pp |
| Memory Throughput (%) | 41.20 % | 20.58 % | -20.62 pp | 40.32 % | -0.88 pp |
| DRAM Throughput (%) | 20.50 % | 5.01 % | -15.49 pp | 19.59 % | -0.91 pp |
| Achieved Occupancy (%) | 55.07 % | 43.93 % | -11.14 pp | 55.05 % | -0.02 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.55 % | 55.84 % | -15.71 pp | 70.36 % | -1.19 pp |
| IPC Active | 1.14 inst/cycle | 1.69 inst/cycle | +0.55 | 1.17 inst/cycle | +0.03 |

---

## Iteration 3

**Best so far**: 1.8242× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_yblock1_large_xblock_and_fix_grid_lambda | 1.399× | 1.808× | ✓ |  |
| B | flat_1d_output_indexing_to_eliminate_stride3_scatter | 1.225× | 1.636× | ✓ |  |
| C | yblock_divisor_of_3_autotune_to_eliminate_stride_scatter | 1.342× | 1.808× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_yblock1_... | Δ | flat_1d_output_in... | Δ | yblock_divisor_of... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 3330 ns | -2690 ns (-44.7%) | 3680 ns | -2340 ns (-38.9%) | 3330 ns | -2690 ns (-44.7%) |
| SM Throughput (%) | 16.06 % | 17.55 % | +1.49 pp | 17.31 % | +1.25 pp | 17.66 % | +1.60 pp |
| Memory Throughput (%) | 41.20 % | 20.60 % | -20.60 pp | 24.43 % | -16.77 pp | 20.67 % | -20.53 pp |
| DRAM Throughput (%) | 20.50 % | 4.97 % | -15.53 pp | 4.51 % | -15.99 pp | 4.98 % | -15.52 pp |
| Achieved Occupancy (%) | 55.07 % | 43.66 % | -11.41 pp | 55.40 % | +0.33 pp | 44.38 % | -10.69 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.55 % | 60.57 % | -10.98 pp | 60.43 % | -11.12 pp | 53.98 % | -17.57 pp |
| IPC Active | 1.14 inst/cycle | 1.68 inst/cycle | +0.54 | 1.52 inst/cycle | +0.38 | 1.71 inst/cycle | +0.57 |

---

## Cumulative Progress

```
Baseline:  6020 ns  (1.000×)
Iter 1     3300 ns  (1.824×)  — autotune_tile_sizes_for_coalescing_and_occupancy

Final best: 3300 ns  (1.8242×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 4}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2}, num_warps=8, num_stages=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -19,10 +35,8 @@

     y3 = yindex
     y0 = (yindex % 3)
     y1 = yindex // 3
-    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_first')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +44,7 @@

     ynumel = 30
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
