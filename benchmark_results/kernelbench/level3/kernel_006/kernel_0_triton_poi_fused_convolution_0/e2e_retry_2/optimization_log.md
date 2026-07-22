# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6180 ns — **Best latency**: 7330 ns (**0.8431×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_006/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6180 ns
- **SM Throughput**: 16.35 % | **Memory**: 41.21 % | **DRAM**: 19.90 %
- **Achieved Occupancy**: 54.61 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 71.12 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguity_hints | 1.253× | 1.039× | ✓ |  |
| B | autotune_tile_sizes_for_grid_coverage_and_coalescing | 1.385× | 0.843× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6180 ns | 5950 ns | -230 ns (-3.7%) | 7330 ns | +1150 ns (+18.6%) |
| SM Throughput (%) | 16.35 % | 16.00 % | -0.35 pp | 34.62 % | +18.27 pp |
| Memory Throughput (%) | 41.21 % | 42.77 % | +1.56 pp | 48.32 % | +7.11 pp |
| DRAM Throughput (%) | 19.90 % | 20.71 % | +0.81 pp | 9.00 % | -10.90 pp |
| Achieved Occupancy (%) | 54.61 % | 55.29 % | +0.68 pp | 61.33 % | +6.72 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 38.00 register/thread | +12 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.98 | +1.24 |
| No Eligible (%) | 71.12 % | 70.43 % | -0.69 pp | 43.35 % | -27.77 pp |
| IPC Active | 1.15 inst/cycle | 1.15 inst/cycle | +0.00 | 2.15 inst/cycle | +1.00 |


### Key Learnings

#### ❌ Plan B: `autotune_tile_sizes_for_grid_coverage_and_coalescing` (0.843×)

Adding @triton.autotune with larger XBLOCK (256–1024) and num_stages=3 increased register usage from 26 to 38 per thread and dynamic shared memory from 4 KB to 16 KB per block, capping theoretical occupancy at 75% (down from 100%) and limiting Block Limit Registers to 6 blocks/SM. Although waves_per_sm improved (0.74→1.98) and eligible warps per scheduler improved (0.93→2.47), the total executed instruction count nearly tripled (771K→2060K) due to the autotuner selecting a configuration with much larger tiles, causing net latency increase from 6.18 µs to 7.33 µs. For small, latency-bound pointwise kernels where the baseline already fits in 26 registers, adding autotune configs with large XBLOCK and num_stages>1 can cause register spill pressure and smem bloat that outweighs the grid-coverage benefit.

> **Applicable when**: Kernel has high register pressure from large tile configs; theoretical occupancy is already 100% at baseline; instruction count is low (<1M); kernel duration is short (<10 µs); adding autotune configs with num_stages>1 would increase register count significantly.

> **Avoid when**: Kernel is already register-efficient (<=26 registers/thread) and fits 100% theoretical occupancy; adding larger XBLOCK configs with num_stages=3 would push registers above 32/thread and cap theoretical occupancy below 100%; the grid is fixed and not updated to use a meta lambda (causing the autotuner to select configs but the grid to remain at the original fixed size).

---

## Cumulative Progress

```
Baseline:  6180 ns  (1.000×)

Final best: 7330 ns  (0.8431×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 4}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 8}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4, num_stages=1),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +40,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 30
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
