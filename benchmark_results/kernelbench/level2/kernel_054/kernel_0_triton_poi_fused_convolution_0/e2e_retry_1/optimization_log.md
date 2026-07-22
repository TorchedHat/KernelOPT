# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5890 ns — **Best latency**: 4260 ns (**1.3826×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_coalesced_stores`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_054/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5890 ns
- **SM Throughput**: 15.38 % | **Memory**: 41.49 % | **DRAM**: 20.87 %
- **Achieved Occupancy**: 54.99 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 70.41 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.170× | 0.920× | ✓ |  |
| B | autotune_tile_sizes_for_coalesced_stores | 1.359× | 1.383× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5890 ns | 6400 ns | +510 ns (+8.7%) | 4260 ns | -1630 ns (-27.7%) |
| SM Throughput (%) | 15.38 % | 16.52 % | +1.14 pp | 16.99 % | +1.61 pp |
| Memory Throughput (%) | 41.49 % | 38.31 % | -3.18 pp | 32.33 % | -9.16 pp |
| DRAM Throughput (%) | 20.87 % | 19.30 % | -1.57 pp | 15.48 % | -5.39 pp |
| Achieved Occupancy (%) | 54.99 % | 54.84 % | -0.15 pp | 54.58 % | -0.41 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 21.00 register/thread | -5 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.41 % | 70.43 % | +0.02 pp | 64.71 % | -5.70 pp |
| IPC Active | 1.14 inst/cycle | 1.16 inst/cycle | +0.02 | 1.31 inst/cycle | +0.17 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_for_coalesced_stores` (1.383×)

When a kernel writes to a strided output layout (e.g., NHWC with stride-3 channel dimension), the fixed tile shape may cause every store transaction to be uncoalesced, wasting 30-50% of memory bandwidth. Adding @triton.autotune with configs spanning multiple YBLOCK and XBLOCK values allows the runtime to select a tile shape that aligns the y-tile boundary to the stride period, reducing excess sectors and L1TEX scoreboard stalls. This also reduces register pressure (26→21 registers/thread) and shared memory usage, improving scheduler eligibility (eligible warps/scheduler: 0.95→1.29) and cutting kernel duration by ~28%.

> **Applicable when**: Triton kernel has UncoalescedGlobalAccess rule triggered with >15% excess sectors; store address contains a modulo/stride term (e.g., yindex % K) that creates non-unit stride scatter; CPIStall on L1TEX scoreboard is >30% of warp cycles; kernel uses fixed hardcoded tile sizes (XBLOCK, YBLOCK) without autotuning; IssueSlotUtilization shows >50% no-eligible cycles.

> **Avoid when**: Kernel is already autotuned or tile sizes are already optimal; the store pattern is naturally coalesced (unit stride); the grid is already large enough to saturate the device (waves_per_sm >> 1); autotuning overhead is unacceptable (e.g., very short-lived kernels called once); YBLOCK must remain a power of 2 and the stride period is not a power of 2 (alignment may not improve).

---

## Cumulative Progress

```
Baseline:  5890 ns  (1.000×)
Iter 1     4260 ns  (1.383×)  — autotune_tile_sizes_for_coalesced_stores

Final best: 4260 ns  (1.3826×)
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
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 4},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 8},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK':  64, 'YBLOCK': 4},  num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK':  64, 'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 16}, num_warps=8, num_stages=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +38,13 @@

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
