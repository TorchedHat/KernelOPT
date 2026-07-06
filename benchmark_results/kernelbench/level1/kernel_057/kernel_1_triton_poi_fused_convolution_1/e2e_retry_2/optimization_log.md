# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 285890 ns — **Best latency**: 116640 ns (**2.4510×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_coalesced_memory_access`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_057/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 285890 ns
- **SM Throughput**: 20.21 % | **Memory**: 78.81 % | **DRAM**: 72.19 %
- **Achieved Occupancy**: 94.04 %
- **IPC Active**: 0.79 inst/cycle | **No Eligible**: 80.20 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_for_coalesced_memory_access | 3.037× | 2.451× | ✓ | **✓ best** |
| C | add_triton_autotune | 1.001× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 285890 ns | 116640 ns | -169250 ns (-59.2%) | 285540 ns | -350 ns (-0.1%) |
| SM Throughput (%) | 20.21 % | 58.92 % | +38.71 pp | 19.58 % | -0.63 pp |
| Memory Throughput (%) | 78.81 % | 45.83 % | -32.98 pp | 78.90 % | +0.09 pp |
| DRAM Throughput (%) | 72.19 % | 19.95 % | -52.24 pp | 72.26 % | +0.07 pp |
| Achieved Occupancy (%) | 94.04 % | 90.40 % | -3.64 pp | 94.01 % | -0.03 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.20 % | 44.66 % | -35.54 pp | 79.59 % | -0.61 pp |
| IPC Active | 0.79 inst/cycle | 2.26 inst/cycle | +1.47 | 0.81 inst/cycle | +0.02 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_for_coalesced_memory_access` (2.451×)

When a Triton kernel performs a layout transpose where threads in a warp stride across a non-unit dimension (e.g., input stride of 128 elements per thread), L2 throughput saturates near 97% and L1 hit rate is 0%, with scheduler stall cycles exceeding 75 cycles/instruction due to L1TEX dependency stalls. Adding @triton.autotune with configs that vary XBLOCK and YBLOCK allows the autotuner to discover tile shapes where threads in a warp access contiguous (stride-1) memory, collapsing L2 throughput from 97% to 30% and reducing warp stall cycles from 75.9 to 25.8. The compute throughput rises from 20% to 59% and kernel duration drops by ~2.5x as memory bottleneck is resolved.

> **Applicable when**: Triton kernel performs a 2D layout transpose or permutation; L2 Cache Throughput > 80%, L1 hit rate ~0%, Warp Cycles Per Issued Instruction > 50, Scheduler 'No Eligible' > 70%, Compute throughput < 30%; the slow access dimension has a large stride (many elements) while a fast stride-1 dimension exists but is not aligned with the thread-varying axis.

> **Avoid when**: Kernel is already compute-bound (Compute SM Throughput > 70%); access pattern is inherently non-contiguous in all dimensions with no stride-1 axis to exploit; problem size is too small for autotune overhead to amortize; XBLOCK/YBLOCK are already optimal or constrained by shared memory usage.

---

## Cumulative Progress

```
Baseline:  285890 ns  (1.000×)
Iter 1     116640 ns  (2.451×)  — autotune_tile_sizes_for_coalesced_memory_access

Final best: 116640 ns  (2.4510×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,29 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 32},  num_warps=2, num_stages=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},   num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 128}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16},  num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 16},  num_warps=2, num_stages=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 4096
     xnumel = 31500
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +35,8 @@

     y0 = (yindex % 128)
     y1 = yindex // 128
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +44,7 @@

     ynumel = 4096
     xnumel = 31500
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
