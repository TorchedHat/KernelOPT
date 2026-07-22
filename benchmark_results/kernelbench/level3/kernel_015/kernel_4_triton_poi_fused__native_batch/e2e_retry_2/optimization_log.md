# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 41020 ns — **Best latency**: 45570 ns (**0.9002×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_4_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (62720,1,1), Waves/SM: 29.70
- **Registers/thread**: 17
- **Duration**: 41020 ns
- **SM Throughput**: 56.57 % | **Memory**: 24.58 % | **DRAM**: 18.55 %
- **Achieved Occupancy**: 55.63 %
- **IPC Active**: 2.41 inst/cycle | **No Eligible**: 39.92 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_via_autotune_to_hide_l1tex_latency | 1.000× | 0.369× | ✓ |  |
| B | autotune_xblock_num_warps_for_latency_hiding | 1.003× | 0.900× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_v... | Δ | autotune_xblock_n... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 41020 ns | 111070 ns | +70050 ns (+170.8%) | 45570 ns | +4550 ns (+11.1%) |
| SM Throughput (%) | 56.57 % | 87.60 % | +31.03 pp | 78.16 % | +21.59 pp |
| Memory Throughput (%) | 24.58 % | 20.37 % | -4.21 pp | 29.84 % | +5.26 pp |
| DRAM Throughput (%) | 18.55 % | 6.89 % | -11.66 pp | 16.77 % | -1.78 pp |
| Achieved Occupancy (%) | 55.63 % | 83.60 % | +27.97 pp | 62.35 % | +6.72 pp |
| Registers/Thread | 17.00 register/thread | 32.00 register/thread | +15 | 24.00 register/thread | +7 |
| Waves/SM | 29.70 | 59.39 | +29.69 | 29.70 | +0.00 |
| No Eligible (%) | 39.92 % | 10.63 % | -29.29 pp | 16.53 % | -23.39 pp |
| IPC Active | 2.41 inst/cycle | 3.57 inst/cycle | +1.16 | 3.32 inst/cycle | +0.91 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_via_autotune_to_hide_l1tex_latency` (0.369×)

Adding @triton.autotune with larger XBLOCK configs (256–1024) improved occupancy (55.6%→83.6%) and eligible warps per scheduler (1.17→7.58), but caused a severe 2.7x regression. The root cause was a 4.2x explosion in executed instructions (18.3M→76.6M) combined with a doubling of registers per thread (17→32), which together overwhelmed any latency-hiding benefit. The autotune selected XBLOCK=256 with 32 registers/thread, indicating the larger tile size triggered a more complex code path or register-spill-adjacent behavior that dramatically increased per-element work.

> **Applicable when**: Never: this specific combination of autotune + mask change caused a 4x instruction explosion. Do not apply autotune with dynamic masking to kernels where xnumel is exactly divisible by the block size and the original kernel used tl.full([XBLOCK], True) masking.

> **Avoid when**: When the kernel already uses tl.full([XBLOCK], True, tl.int1) as a no-op mask (meaning xnumel is divisible by XBLOCK), switching to xmask = xindex < xnumel adds a runtime comparison and changes load/store behavior. When register count doubles after the change (17→32 regs/thread), the autotune is selecting a config that triggers heavier register pressure, reducing the number of concurrent blocks per SM despite higher nominal occupancy. When executed instruction count increases by more than 2x, the latency-hiding benefit is negated.

---

## Cumulative Progress

```
Baseline:  41020 ns  (1.000×)

Final best: 45570 ns  (0.9002×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,19 +5,31 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=16),
+        triton.Config({'XBLOCK': 2048}, num_warps=16),
+        triton.Config({'XBLOCK': 128},  num_warps=4),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 8028160
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
@@ -31,9 +43,7 @@

     tmp15 = tmp13 + tmp14
     tmp16 = tl.full([1], 0, tl.int32)
     tmp17 = tl.maximum(tmp16, tmp15)
-    tl.store(in_out_ptr0 + (x2), tmp17, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -43,7 +53,7 @@

     arg_4 = inputs[4]
     xnumel = 8028160
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 8028160, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 8028160)
     return arg_4
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
