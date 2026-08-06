# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6080 ns — **Best latency**: 4290 ns (**1.4172×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_to_fix_underutilization_and_uncoalesced_stores`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_006/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6080 ns
- **SM Throughput**: 16.69 % | **Memory**: 41.99 % | **DRAM**: 20.30 %
- **Achieved Occupancy**: 55.30 %
- **IPC Active**: 1.11 inst/cycle | **No Eligible**: 70.22 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_and_coalesced_store_via_autotune | 1.333× | 1.483× | ✓ |  |
| B | autotune_tile_sizes_to_fix_underutilization_and_uncoalesced_stores | 1.367× | 1.417× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6080 ns | 4100 ns | -1980 ns (-32.6%) | 4290 ns | -1790 ns (-29.4%) |
| SM Throughput (%) | 16.69 % | 15.95 % | -0.74 pp | 27.04 % | +10.35 pp |
| Memory Throughput (%) | 41.99 % | 33.59 % | -8.40 pp | 21.44 % | -20.55 pp |
| DRAM Throughput (%) | 20.30 % | 16.08 % | -4.22 pp | 3.87 % | -16.43 pp |
| Achieved Occupancy (%) | 55.30 % | 54.39 % | -0.91 pp | 67.91 % | +12.61 pp |
| Registers/Thread | 26.00 register/thread | 21.00 register/thread | -5 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 70.22 % | 65.40 % | -4.82 pp | 37.14 % | -33.08 pp |
| IPC Active | 1.11 inst/cycle | 1.34 inst/cycle | +0.23 | 2.38 inst/cycle | +1.27 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_to_fix_underutilization_and_uncoalesced_stores` (1.417×)

When a Triton kernel launches with fewer than ~1 wave per SM (waves_per_sm < 1.0) and eligible warps per scheduler is near 1 or below, the GPU is severely underutilized due to insufficient parallelism and warp starvation from L1TEX scoreboard stalls. Adding @triton.autotune over a range of XBLOCK/YBLOCK tile sizes and num_warps values allows the runtime to select a configuration that doubles waves_per_sm, triples eligible warps per scheduler, and reduces warp cycles per issued instruction by ~40%, yielding a 1.4x speedup. Register pressure also drops (26→18 registers/thread) as a side effect of the larger block size selection, further improving occupancy.

> **Applicable when**: Triton kernel with waves_per_sm < 1.0, eligible_warps_per_scheduler < 1.5, no_eligible_cycles > 60%, and L1TEX scoreboard stalls dominating CPI (CPIStall rule triggered). Kernel body is a pure memory permutation or elementwise op with no algorithmic complexity. XBLOCK/YBLOCK are tl.constexpr tile parameters.

> **Avoid when**: Kernel already achieves waves_per_sm > 2 or eligible_warps_per_scheduler > 3. Kernel body has shared memory usage that scales with tile size and would exceed smem limits at larger XBLOCK. Autotuning overhead is unacceptable (e.g., very short-lived kernels called once).

---

## Cumulative Progress

```
Baseline:  6080 ns  (1.000×)
Iter 1     4290 ns  (1.417×)  — autotune_tile_sizes_to_fix_underutilization_and_uncoalesced_stores

Final best: 4290 ns  (1.4172×)
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
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4, num_stages=1),
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

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
