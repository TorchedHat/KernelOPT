# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_sigmoid_sum_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 115970 ns — **Best latency**: 116420 ns (**0.9961×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_sigmoid_sum_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_065/kernel_4_triton_red_fused_sigmoid_sum_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (640,1,1), Waves/SM: 0.40
- **Registers/thread**: 40
- **Duration**: 115970 ns
- **SM Throughput**: 42.02 % | **Memory**: 52.31 % | **DRAM**: 52.31 %
- **Achieved Occupancy**: 29.40 %
- **IPC Active**: 1.70 inst/cycle | **No Eligible**: 57.31 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_and_autotune_tile_sizes | 1.000× | 0.998× | ✓ |  |
| B | memory_coalescing_via_autotune_tile_reorder | 1.001× | 0.996× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | memory_coalescing... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 115970 ns | 116260 ns | +290 ns (+0.3%) | 116420 ns | +450 ns (+0.4%) |
| SM Throughput (%) | 42.02 % | 41.84 % | -0.18 pp | 41.91 % | -0.11 pp |
| Memory Throughput (%) | 52.31 % | 52.18 % | -0.13 pp | 52.12 % | -0.19 pp |
| DRAM Throughput (%) | 52.31 % | 52.18 % | -0.13 pp | 52.12 % | -0.19 pp |
| Achieved Occupancy (%) | 29.40 % | 29.45 % | +0.05 pp | 29.45 % | +0.05 pp |
| Registers/Thread | 40.00 register/thread | 40.00 register/thread | +0 | 40.00 register/thread | +0 |
| Waves/SM | 0.40 | 0.40 | +0.00 | 0.40 | +0.00 |
| No Eligible (%) | 57.31 % | 57.05 % | -0.26 pp | 56.85 % | -0.46 pp |
| IPC Active | 1.70 inst/cycle | 1.71 inst/cycle | +0.01 | 1.71 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  115970 ns  (1.000×)

Final best: 116420 ns  (0.9961×)
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
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 2048}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 1024}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 2048}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 1024}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 256},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 256},  num_warps=8,  num_stages=1),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused_sigmoid_sum_4(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 640
@@ -24,15 +38,13 @@

         roffset = r0_offset
         rindex = r0_index
         r0_1 = r0_index
-        tmp0 = tl.load(in_ptr0 + (r0_1 + 115520*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp0 = tl.load(in_ptr0 + (r0_1 + 115520*x0), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
         tmp1 = tl.sigmoid(tmp0)
         tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
         tmp4 = _tmp3 + tmp2
         _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
     tmp3 = tl.sum(_tmp3, 1)[:, None]
-    tl.store(out_ptr0 + (x0), tmp3, xmask)
-
-
+    tl.store(out_ptr0 + (x0), tmp3, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -40,7 +52,7 @@

     xnumel = 640
     r0_numel = 115520
     grid = (triton.cdiv(xnumel, 1),)
-    triton_red_fused_sigmoid_sum_4[grid](arg_0, arg_1, 640, 115520, XBLOCK=1, R0_BLOCK=2048)
+    triton_red_fused_sigmoid_sum_4[grid](arg_0, arg_1, 640, 115520)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
