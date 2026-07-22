# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6110 ns — **Best latency**: 3360 ns (**1.8185×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_with_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_021/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6110 ns
- **SM Throughput**: 17.39 % | **Memory**: 39.58 % | **DRAM**: 20.24 %
- **Achieved Occupancy**: 54.90 %
- **IPC Active**: 1.17 inst/cycle | **No Eligible**: 70.63 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.383× | 1.819× | ✓ | **✓ best** |
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.074× | 1.005× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6110 ns | 3360 ns | -2750 ns (-45.0%) | 6080 ns | -30 ns (-0.5%) |
| SM Throughput (%) | 17.39 % | 9.73 % | -7.66 pp | 17.10 % | -0.29 pp |
| Memory Throughput (%) | 39.58 % | 20.70 % | -18.88 pp | 40.20 % | +0.62 pp |
| DRAM Throughput (%) | 20.24 % | 4.96 % | -15.28 pp | 20.26 % | +0.02 pp |
| Achieved Occupancy (%) | 54.90 % | 52.00 % | -2.90 pp | 54.55 % | -0.35 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.63 % | 69.12 % | -1.51 pp | 69.61 % | -1.02 pp |
| IPC Active | 1.17 inst/cycle | 1.07 inst/cycle | -0.10 | 1.18 inst/cycle | +0.01 |


### Key Learnings

#### ✅ Plan A: `vectorized_loads_with_autotune` (1.819×)

Applying tl.max_contiguous and tl.multiple_of hints to the X-dimension index in a Triton kernel enables the compiler to emit 128-bit vectorized loads (LDG.128) when the access pattern is provably stride-1 contiguous. This dramatically reduces total executed instructions (771K → 238K) and elapsed cycles (9030 → 4958), cutting kernel duration nearly in half. Pairing these hints with @triton.autotune over XBLOCK/YBLOCK/num_warps configurations allows the runtime to select the tile shape that best amortizes memory latency, particularly effective when CPIStall (L1TEX scoreboard) and UncoalescedGlobalAccess rules are both triggered.

> **Applicable when**: Triton kernel with stride-1 contiguous X-dimension access pattern; CPIStall rule triggered with L1TEX scoreboard stalls >10 cycles/warp; UncoalescedGlobalAccess rule triggered with >15% excess sectors; IssueSlotUtilization rule triggered; kernel uses hardcoded XBLOCK/YBLOCK tile sizes without autotune; registers_per_thread is high (>20) suggesting spill or suboptimal config

> **Avoid when**: X-dimension access is not stride-1 contiguous (e.g., gather/scatter patterns); tl.multiple_of hint value does not match actual alignment (would produce incorrect vectorization); kernel is already autotuned; access pattern has non-trivial strides in the X dimension that would make the contiguity hint incorrect

---

## Cumulative Progress

```
Baseline:  6110 ns  (1.000×)
Iter 1     3360 ns  (1.819×)  — vectorized_loads_with_autotune

Final best: 3360 ns  (1.8185×)
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
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=4),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=8),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +27,8 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    # Apply hints to 1D arange before broadcasting to 2D to avoid shape mismatch
+    xindex = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +37,13 @@

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
