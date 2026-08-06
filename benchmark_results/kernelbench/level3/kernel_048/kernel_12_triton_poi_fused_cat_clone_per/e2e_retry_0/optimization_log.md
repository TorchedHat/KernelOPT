# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_cat_clone_permute_unsqueeze_view_zeros_like_12` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 242850 ns — **Best latency**: 312610 ns (**0.7768×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_cat_clone_permute_unsqueeze_view_zeros_like_12`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_048/kernel_12_triton_poi_fused_cat_clone_per/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (393216,1,1), Waves/SM: 186.18
- **Registers/thread**: 16
- **Duration**: 242850 ns
- **SM Throughput**: 39.46 % | **Memory**: 26.24 % | **DRAM**: 26.24 %
- **Achieved Occupancy**: 32.91 %
- **IPC Active**: 1.45 inst/cycle | **No Eligible**: 63.43 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | increase_xblock_tile_size_for_occupancy_and_latency_hiding | — | 0.777× | ✓ | **✓ best** |
| C | add_triton_autotune | — | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 242850 ns | 312610 ns | +69760 ns (+28.7%) | 242910 ns | +60 ns (+0.0%) |
| SM Throughput (%) | 39.46 % | 76.58 % | +37.12 pp | 42.50 % | +3.04 pp |
| Memory Throughput (%) | 26.24 % | 20.34 % | -5.90 pp | 26.24 % | +0.00 pp |
| DRAM Throughput (%) | 26.24 % | 20.34 % | -5.90 pp | 26.24 % | +0.00 pp |
| Achieved Occupancy (%) | 32.91 % | 72.78 % | +39.87 pp | 33.26 % | +0.35 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 186.18 | 372.36 | +186.18 | 186.18 | +0.00 |
| No Eligible (%) | 63.43 % | 35.47 % | -27.96 pp | 61.12 % | -2.31 pp |
| IPC Active | 1.45 inst/cycle | 2.57 inst/cycle | +1.12 | 1.55 inst/cycle | +0.10 |


### Key Learnings

#### ❌ Plan B: `increase_xblock_tile_size_for_occupancy_and_latency_hiding` (0.777×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) to a memory-bound pointwise kernel caused a 29% latency regression despite improving achieved occupancy from 32.9% to 72.8%. The executed instruction count more than doubled (69M → 152M), indicating the autotune selected a configuration that introduced significant overhead—likely from the added xmask guard on every load/store (previously xmask=True was elided). The kernel became compute-bound (SM throughput 39% → 76%) rather than memory-latency-bound, but the extra compute work outweighed the latency-hiding benefit, and DRAM throughput actually dropped (26% → 20%), suggesting the larger tiles did not improve memory access efficiency.

> **Applicable when**: Never: this specific combination of autotune + xmask introduction caused regression. Avoid applying this pattern to kernels where xnumel is exactly divisible by the original XBLOCK (making xmask=True valid and elided), as adding an explicit xmask reintroduces predication overhead across all memory ops.

> **Avoid when**: When the original kernel uses xmask=True (full tile, no boundary check needed) and xnumel is divisible by XBLOCK—adding an explicit xmask guard while also increasing XBLOCK doubles instruction count and causes compute-bound regression even if occupancy improves. Also avoid when the kernel is already latency-bound on L1TEX but the total work is small enough that instruction overhead dominates.

---

## Cumulative Progress

```
Baseline:  242850 ns  (1.000×)

Final best: 312610 ns  (0.7768×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,12 +5,23 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=3),
+        triton.Config({'XBLOCK': 2048}, num_warps=32, num_stages=3),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=4),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=4),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_cat_clone_permute_unsqueeze_view_zeros_like_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 50331648
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x1 = ((xindex // 1024) % 3)
     x0 = (xindex % 1024)
     x2 = ((xindex // 3072) % 8)
@@ -27,18 +38,16 @@

     tmp8 = tmp0 >= tmp3
     tmp9 = tl.full([1], 3, tl.int64)
     tmp10 = tmp0 < tmp9
-    tmp11 = tl.load(in_ptr0 + (x0 + 1024*x2 + 8192*((-1) + x1) + 16384*x3), tmp8, other=0.0)
+    tmp11 = tl.load(in_ptr0 + (x0 + 1024*x2 + 8192*((-1) + x1) + 16384*x3), mask=tmp8 & xmask, other=0.0)
     tmp12 = tl.where(tmp4, tmp7, tmp11)
-    tl.store(out_ptr0 + (x4), tmp12, None)
-
-
+    tl.store(out_ptr0 + (x4), tmp12, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 50331648
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_cat_clone_permute_unsqueeze_view_zeros_like_12[grid](arg_0, arg_1, 50331648, XBLOCK=128)
+    triton_poi_fused_cat_clone_permute_unsqueeze_view_zeros_like_12[grid](arg_0, arg_1, 50331648)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
