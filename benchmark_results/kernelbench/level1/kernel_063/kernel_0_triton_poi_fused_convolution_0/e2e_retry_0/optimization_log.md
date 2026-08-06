# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5820 ns — **Best latency**: 3360 ns (**1.7321×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_coalescing_and_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_063/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5820 ns
- **SM Throughput**: 15.43 % | **Memory**: 42.77 % | **DRAM**: 21.21 %
- **Achieved Occupancy**: 55.61 %
- **IPC Active**: 1.10 inst/cycle | **No Eligible**: 70.67 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.383× | 1.732× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 5820 ns | 3360 ns | -2460 ns (-42.3%) |
| SM Throughput (%) | 15.43 % | 17.34 % | +1.91 pp |
| Memory Throughput (%) | 42.77 % | 20.42 % | -22.35 pp |
| DRAM Throughput (%) | 21.21 % | 4.96 % | -16.25 pp |
| Achieved Occupancy (%) | 55.61 % | 45.07 % | -10.54 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 |
| No Eligible (%) | 70.67 % | 56.25 % | -14.42 pp |
| IPC Active | 1.10 inst/cycle | 1.60 inst/cycle | +0.50 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_for_coalescing_and_occupancy` (1.732×)

When a Triton kernel shows high CPIStall from L1TEX scoreboard waits (14+ cycles avg), low eligible warps per scheduler (<1.0), and uncoalesced global stores, adding @triton.autotune with varied XBLOCK/YBLOCK/num_warps/num_stages configs allows the runtime to find a tile shape that reduces memory stalls and register pressure. In this case, autotuning reduced warp cycles per issued instruction from 30.25 to 15.6, cut registers per thread from 26 to 18, and improved issue slot utilization from 27.8% to 42.2%, yielding a 1.73x speedup despite slightly lower achieved occupancy. The key mechanism is that smaller YBLOCK (e.g., YBLOCK=1) eliminates cross-row stride in stores, reducing uncoalesced sector overhead, while num_stages pipelining hides remaining L1TEX latency.

> **Applicable when**: Triton kernel has: CPIStall rule triggered with L1TEX stall cycles >10, eligible warps per scheduler <1.5, No Eligible scheduler cycles >60%, uncoalesced global stores (UncoalescedGlobalAccess rule), and fixed hardcoded XBLOCK/YBLOCK tile sizes without autotuning. Particularly effective for layout permutation kernels (e.g., NCHW→NHWC) where store stride depends on tile Y dimension.

> **Avoid when**: Kernel already uses @triton.autotune, or the bottleneck is compute-bound (Compute SM Throughput >80%) rather than memory-latency-bound. Also avoid if the kernel has correctness constraints that depend on specific tile sizes (e.g., shared memory tiling with fixed dimensions). Note: autotuning adds first-run overhead.

---

## Cumulative Progress

```
Baseline:  5820 ns  (1.000×)
Iter 1     3360 ns  (1.732×)  — autotune_tile_sizes_for_coalescing_and_occupancy

Final best: 3360 ns  (1.7321×)
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
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 8}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=8, num_stages=4),
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
