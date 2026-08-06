# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_59` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2560 ns — **Best latency**: 2820 ns (**0.9078×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_59`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_59_triton_poi_fused_convolution_5/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (256,1,1), Waves/SM: 0.12
- **Registers/thread**: 28
- **Duration**: 2560 ns
- **SM Throughput**: 4.68 % | **Memory**: 5.09 % | **DRAM**: 1.69 %
- **Achieved Occupancy**: 10.97 %
- **IPC Active**: 0.55 inst/cycle | **No Eligible**: 84.99 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_via_autotune_block_sizes | 1.232× | 0.908× | ✓ | **✓ best** |
| B | autotune_tile_sizes_to_fix_underutilization | 1.194× | 0.851× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2560 ns | 2820 ns | +260 ns (+10.2%) | 3010 ns | +450 ns (+17.6%) |
| SM Throughput (%) | 4.68 % | 0.79 % | -3.89 pp | 1.48 % | -3.20 pp |
| Memory Throughput (%) | 5.09 % | 7.72 % | +2.63 pp | 14.43 % | +9.34 pp |
| DRAM Throughput (%) | 1.69 % | 7.72 % | +6.03 pp | 14.43 % | +12.74 pp |
| Achieved Occupancy (%) | 10.97 % | 5.34 % | -5.63 pp | 10.22 % | -0.75 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.12 | 0.06 | -0.06 | 0.12 | +0.00 |
| No Eligible (%) | 84.99 % | 96.99 % | +12.00 pp | 95.93 % | +10.94 pp |
| IPC Active | 0.55 inst/cycle | 0.09 inst/cycle | -0.46 | 0.13 inst/cycle | -0.42 |


### Key Learnings

#### ❌ Plan B: `autotune_tile_sizes_to_fix_underutilization` (0.851×)

Adding @triton.autotune to select smaller XBLOCK values (e.g., XBLOCK=1 or 4) for a kernel that permutes data with a stride-based access pattern caused severe uncoalesced global memory access: L2 hit rate dropped from 60.6% to 24.1%, warp cycles per issued instruction tripled from 13 to 41, and a new CPIStall rule appeared citing 27.7 cycles stalled on L1TEX. The autotune also failed to fix the grid launch because the grid lambda was not updated — the kernel was still launched with the old fixed grid (256 CTAs, 0.12 waves), so occupancy did not improve. When the kernel performs a strided permutation (e.g., transposing dimensions), reducing XBLOCK below the natural coalescing width destroys memory access locality and introduces massive uncoalesced access overhead that outweighs any thread-waste savings.

> **Applicable when**: Never applicable as implemented here. The autotune approach for fixing thread waste is conceptually valid but requires: (1) the grid to be updated to a lambda using meta['XBLOCK']/meta['YBLOCK'], and (2) the access pattern to remain coalesced under the new tile sizes.

> **Avoid when**: Do not apply autotune with small XBLOCK values to kernels performing strided permutations where reducing XBLOCK breaks memory coalescing. Do not add @triton.autotune without simultaneously updating the grid to a lambda — leaving a fixed grid with autotune causes the kernel to be launched with mismatched tile sizes. Avoid XBLOCK < 16 when the load pattern is `in_ptr0 + (x2 + stride*y3)` with large stride, as this creates non-contiguous per-thread access.

---

## Cumulative Progress

```
Baseline:  2560 ns  (1.000×)

Final best: 2820 ns  (0.9078×)
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
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 64},  num_warps=2),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=8),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 16},  num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 32},  num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_59(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 2048
@@ -19,10 +35,8 @@

     y3 = yindex
     y0 = (yindex % 32)
     y1 = yindex // 32
-    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 32*x2 + 800*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), ymask & xmask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 32*x2 + 800*y1), tmp0, ymask & xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +44,7 @@

     ynumel = 2048
     xnumel = 25
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_59[grid](arg_0, arg_1, 2048, 25, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_59[grid](arg_0, arg_1, 2048, 25)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
