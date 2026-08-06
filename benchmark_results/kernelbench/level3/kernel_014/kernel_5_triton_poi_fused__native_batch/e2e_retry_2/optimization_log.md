# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 12320 ns — **Best latency**: 12320 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_014/kernel_5_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (15680,1,1), Waves/SM: 7.42
- **Registers/thread**: 17
- **Duration**: 12320 ns
- **SM Throughput**: 47.52 % | **Memory**: 20.95 % | **DRAM**: 13.33 %
- **Achieved Occupancy**: 56.04 %
- **IPC Active**: 2.35 inst/cycle | **No Eligible**: 42.29 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_for_latency_hiding | 0.778× | 0.731× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_latency_hiding | 0.998× | 0.987× | ✓ |  |
| C | add_triton_autotune | 1.011× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_a... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12320 ns | 16860 ns | +4540 ns (+36.9%) | 12480 ns | +160 ns (+1.3%) | 12320 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 47.52 % | 76.32 % | +28.80 pp | 56.10 % | +8.58 pp | 55.40 % | +7.88 pp |
| Memory Throughput (%) | 20.95 % | 21.87 % | +0.92 pp | 21.04 % | +0.09 pp | 20.88 % | -0.07 pp |
| DRAM Throughput (%) | 13.33 % | 9.71 % | -3.62 pp | 13.15 % | -0.18 pp | 13.31 % | -0.02 pp |
| Achieved Occupancy (%) | 56.04 % | 85.22 % | +29.18 pp | 60.11 % | +4.07 pp | 60.04 % | +4.00 pp |
| Registers/Thread | 17.00 register/thread | 32.00 register/thread | +15 | 18.00 register/thread | +1 | 18.00 register/thread | +1 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 |
| No Eligible (%) | 42.29 % | 13.51 % | -28.78 pp | 30.88 % | -11.41 pp | 31.41 % | -10.88 pp |
| IPC Active | 2.35 inst/cycle | 3.44 inst/cycle | +1.09 | 2.74 inst/cycle | +0.39 | 2.75 inst/cycle | +0.40 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune_for_latency_hiding` (0.731×)

Adding @triton.autotune with larger XBLOCK values (512–2048) caused a 2.19x increase in executed instructions (4.58M → 10.05M) and doubled registers per thread (17 → 32), despite improving occupancy (56% → 85%) and eligible warps (1.23 → 7.02). The instruction count explosion—likely from unrolled loop bodies at larger tile sizes combined with the xmask guard introduced for non-power-of-2 boundary safety—dominated any latency-hiding benefit, increasing duration from 12.32 µs to 16.86 µs. When the original kernel already has high L1 hit rate (83%) and the bottleneck is L1TEX scoreboard stalls, increasing tile size can backfire if it causes instruction count to grow faster than latency is hidden.

> **Applicable when**: Kernel has L1TEX scoreboard stalls >40% of cycles, low eligible warps (<2 per scheduler), and small tile sizes with repeated broadcast loads over a small channel dimension. Registers per thread are low (<20) leaving headroom for larger tiles.

> **Avoid when**: The kernel already has high L1 hit rate (>80%) and the xnumel is exactly divisible by the current XBLOCK (making xmask=True valid). Adding xmask guards and larger XBLOCK causes instruction count to explode (>2x) without proportional latency hiding. Also avoid when the grid lambda is not updated to match the autotuned XBLOCK—this is an implementation flaw that causes the grid to remain fixed at the old tile size while the kernel uses a different XBLOCK.

---

## Cumulative Progress

```
Baseline:  12320 ns  (1.000×)

Final best: 12320 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,19 +5,26 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=2),
+    triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=2),
+], key=['xnumel'])
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 2007040
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 64)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None)
-    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
@@ -31,9 +38,7 @@

     tmp15 = tmp13 + tmp14
     tmp16 = tl.full([1], 0, tl.int32)
     tmp17 = tl.maximum(tmp16, tmp15)
-    tl.store(in_out_ptr0 + (x2), tmp17, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -43,7 +48,7 @@

     arg_4 = inputs[4]
     xnumel = 2007040
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2007040, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2007040)
     return arg_4
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
