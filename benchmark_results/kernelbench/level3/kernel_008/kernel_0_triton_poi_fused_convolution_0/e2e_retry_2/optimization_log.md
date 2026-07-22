# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6180 ns — **Best latency**: 4160 ns (**1.4856×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_coalesced_stores`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_008/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6180 ns
- **SM Throughput**: 15.76 % | **Memory**: 41.22 % | **DRAM**: 20.00 %
- **Achieved Occupancy**: 55.07 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 70.32 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.265× | 1.341× | ✓ |  |
| B | autotune_tile_sizes_for_coalesced_stores | 1.417× | 1.486× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6180 ns | 4610 ns | -1570 ns (-25.4%) | 4160 ns | -2020 ns (-32.7%) |
| SM Throughput (%) | 15.76 % | 21.01 % | +5.25 pp | 24.98 % | +9.22 pp |
| Memory Throughput (%) | 41.22 % | 30.20 % | -11.02 pp | 19.85 % | -21.37 pp |
| DRAM Throughput (%) | 20.00 % | 14.32 % | -5.68 pp | 3.99 % | -16.01 pp |
| Achieved Occupancy (%) | 55.07 % | 46.79 % | -8.28 pp | 68.14 % | +13.07 pp |
| Registers/Thread | 26.00 register/thread | 27.00 register/thread | +1 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 70.32 % | 55.54 % | -14.78 pp | 37.06 % | -33.26 pp |
| IPC Active | 1.15 inst/cycle | 1.68 inst/cycle | +0.53 | 2.35 inst/cycle | +1.20 |


### Key Learnings

#### ✅ Plan A: `vectorized_loads_with_contiguous_hints` (1.341×)

When a Triton kernel's load address is provably contiguous and block-aligned in one dimension (evidenced by high L1TEX stall cycles ~46% and uncoalesced global access excess sectors ~23%), applying tl.max_contiguous and tl.multiple_of hints to the index arange before adding the block offset enables the compiler to emit 128-bit (LDG.128) vectorized loads instead of scalar LDG.32. This reduces memory transactions, cuts warp cycles per issued instruction from ~32 to ~17, and improves issue slot utilization from 29% to 43%. Pairing with @triton.autotune over XBLOCK/YBLOCK/num_warps configurations further explores the tile-size space to find the best occupancy-throughput tradeoff.

> **Applicable when**: Triton kernel has a load dimension that is contiguous and block-aligned (program_id * BLOCK stride); NCU reports CPIStall on L1TEX >30% of cycles and UncoalescedGlobalAccess with >15% excess sectors; bytes utilized per sector <24/32; kernel is memory-bound or latency-bound on L1TEX.

> **Avoid when**: Load addresses are not contiguous in the hinted dimension (e.g., strided or indirect accesses); the kernel is already compute-bound with high SM throughput; the multiple_of value does not match the actual alignment of the index (would produce incorrect vectorization hints and potentially wrong results).

#### ✅ Plan B: `autotune_tile_sizes_for_coalesced_stores` (1.486×)

When a Triton kernel performs a layout permutation with a stride-N store pattern (e.g., NCHW→NHWC), a YBLOCK>1 tile causes consecutive threads in the Y-dimension to write to addresses spaced N elements apart, producing uncoalesced scatter stores (low bytes-utilized-per-sector, high excess-sectors%). Adding a @triton.autotune decorator with configs that include YBLOCK=1 allows the runtime to select a configuration where each CTA writes a single contiguous row, eliminating intra-CTA Y-stride scatter. Simultaneously, larger XBLOCK and more num_warps increase waves-per-SM and eligible-warps-per-scheduler, reducing the CPIStall from L1TEX latency and improving latency hiding.

> **Applicable when**: Triton kernel performs a layout permutation (e.g., NCHW→NHWC or similar stride-N reindex) where the store address has a modulo/stride term in the Y-tile dimension; NCU reports UncoalescedGlobalAccess with >15% excess sectors, MemoryCacheAccessPattern bytes-utilized-per-sector <24, CPIStall from L1TEX >30% of cycles, IssueSlotUtilization speedup headroom >40%, and waves-per-SM <1.0.

> **Avoid when**: Kernel stores are already coalesced (bytes-utilized-per-sector near 32, excess sectors <5%); or the Y-dimension is the fast-varying output dimension (YBLOCK=1 would not help coalescing); or the grid is already large enough (waves-per-SM >2) that autotune overhead is not justified.

---

## Cumulative Progress

```
Baseline:  6180 ns  (1.000×)
Iter 1     4160 ns  (1.486×)  — autotune_tile_sizes_for_coalesced_stores

Final best: 4160 ns  (1.4856×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2}, num_warps=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 4}, num_warps=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +36,13 @@

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

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
