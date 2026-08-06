# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_tanh_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 22750 ns — **Best latency**: 22750 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_addmm_tanh_3`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_033/kernel_2_triton_poi_fused_addmm_tanh_3/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (32768,1,1), Waves/SM: 15.52
- **Registers/thread**: 16
- **Duration**: 22750 ns
- **SM Throughput**: 36.91 % | **Memory**: 19.89 % | **DRAM**: 15.08 %
- **Achieved Occupancy**: 45.05 %
- **IPC Active**: 1.66 inst/cycle | **No Eligible**: 58.18 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_increase_tile_size | 1.001× | 1.000× | ✓ | **✓ best** |
| B | increase_xblock_with_autotune_for_better_cache_reuse_and_parallelism | 0.897× | 0.921× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_i... | Δ | increase_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 22750 ns | 22750 ns | +0 ns (+0.0%) | 24700 ns | +1950 ns (+8.6%) |
| SM Throughput (%) | 36.91 % | 40.79 % | +3.88 pp | 65.03 % | +28.12 pp |
| Memory Throughput (%) | 19.89 % | 20.00 % | +0.11 pp | 17.90 % | -1.99 pp |
| DRAM Throughput (%) | 15.08 % | 15.12 % | +0.04 pp | 13.90 % | -1.18 pp |
| Achieved Occupancy (%) | 45.05 % | 46.19 % | +1.14 pp | 48.17 % | +3.12 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 18.00 register/thread | +2 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 58.18 % | 54.04 % | -4.14 pp | 26.18 % | -32.00 pp |
| IPC Active | 1.66 inst/cycle | 1.83 inst/cycle | +0.17 | 2.91 inst/cycle | +1.25 |

---

## Cumulative Progress

```
Baseline:  22750 ns  (1.000×)

Final best: 22750 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,28 +5,37 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 2048}, num_warps=8,  num_stages=4),
+        triton.Config({'XBLOCK': 4096}, num_warps=16, num_stages=4),
+        triton.Config({'XBLOCK': 4096}, num_warps=8,  num_stages=4),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_addmm_tanh_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
-    xnumel = 4194304
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 16384)
     x2 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_out_ptr0 + (x2), None)
+    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
     tmp2 = tmp0 + tmp1
     tmp3 = libdevice.tanh(tmp2)
-    tl.store(in_out_ptr0 + (x2), tmp3, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 4194304
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_addmm_tanh_3[grid](arg_0, arg_1, 4194304, XBLOCK=128)
+    triton_poi_fused_addmm_tanh_3[grid](arg_0, arg_1, 4194304)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
