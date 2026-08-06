# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5820 ns — **Best latency**: 5730 ns (**1.0157×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_with_contiguous_hints`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_092/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5820 ns
- **SM Throughput**: 16.25 % | **Memory**: 42.59 % | **DRAM**: 21.13 %
- **Achieved Occupancy**: 54.94 %
- **IPC Active**: 1.16 inst/cycle | **No Eligible**: 70.93 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.192× | 1.016× | ✓ | **✓ best** |
| B | autotune_tile_sizes_and_pipeline_stages | 1.167× | 1.444× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5820 ns | 5730 ns | -90 ns (-1.5%) | 4030 ns | -1790 ns (-30.8%) |
| SM Throughput (%) | 16.25 % | 29.59 % | +13.34 pp | 17.03 % | +0.78 pp |
| Memory Throughput (%) | 42.59 % | 37.26 % | -5.33 pp | 34.35 % | -8.24 pp |
| DRAM Throughput (%) | 21.13 % | 11.50 % | -9.63 pp | 16.36 % | -4.77 pp |
| Achieved Occupancy (%) | 54.94 % | 52.27 % | -2.67 pp | 54.87 % | -0.07 pp |
| Registers/Thread | 26.00 register/thread | 38.00 register/thread | +12 | 21.00 register/thread | -5 |
| Waves/SM | 0.74 | 0.99 | +0.25 | 0.74 | +0.00 |
| No Eligible (%) | 70.93 % | 44.62 % | -26.31 pp | 64.31 % | -6.62 pp |
| IPC Active | 1.16 inst/cycle | 2.15 inst/cycle | +0.99 | 1.34 inst/cycle | +0.18 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_and_pipeline_stages` (1.444×)

When a Triton kernel shows high no-eligible-warp cycles (>65%), significant L1TEX scoreboard stalls (CPIStall), and a fixed launch configuration chosen without runtime tuning, adding a @triton.autotune decorator with configs spanning XBLOCK/YBLOCK tile sizes, num_warps, and num_stages allows the runtime to select the configuration that best hides memory latency. In this case, autotuning reduced warp cycles per issued instruction from 30.4 to 25.5, increased eligible warps per scheduler from 0.94 to 1.30, and cut kernel duration from 5.82 µs to 4.03 µs (1.44x speedup) with no changes to kernel arithmetic. The key mechanism is that larger tile sizes increase in-flight memory requests per CTA and higher num_stages enables software pipelining of global loads (cp.async on Ampere+), directly attacking L1TEX scoreboard stalls.

> **Applicable when**: Triton kernel has: (1) no-eligible-warp cycles >60%, (2) CPIStall from L1TEX scoreboard stalls >30% of warp cycles, (3) fixed tile sizes chosen statically (e.g., by torch.inductor) without runtime autotuning, (4) waves_per_sm < 1.0 suggesting the grid is too small to fully occupy the GPU, (5) kernel is memory-bound (layout permutation, elementwise, or similar) with no complex arithmetic that would be affected by tile size changes.

> **Avoid when**: Kernel has complex data-dependent control flow that makes correctness sensitive to tile boundaries; tile sizes are already autotuned; the kernel is compute-bound (high SM throughput >80%) rather than memory-latency-bound; the autotuning overhead (first-run compilation of N configs) is unacceptable in the deployment context; XBLOCK/YBLOCK must be specific values for correctness (e.g., exact divisibility required without masking).

---

## Cumulative Progress

```
Baseline:  5820 ns  (1.000×)
Iter 1     5730 ns  (1.016×)  — vectorized_loads_with_contiguous_hints

Final best: 5730 ns  (1.0157×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 8}, num_warps=8),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -14,6 +26,7 @@

     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = tl.max_contiguous(tl.multiple_of(xindex, [1, 16]), [1, 16])  # enables 128-bit vectorized loads
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +35,13 @@

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
