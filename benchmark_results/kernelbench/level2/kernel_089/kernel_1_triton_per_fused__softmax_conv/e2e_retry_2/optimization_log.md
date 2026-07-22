# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused__softmax_convolution_exp_max_max_pool3d_with_indices_mul_prepare_softmax_online_sigmoid_sub_view_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 274080 ns — **Best latency**: 280350 ns (**0.9776×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused__softmax_convolution_exp_max_max_pool3d_with_indices_mul_prepare_softmax_online_sigmoid_sub_view_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_089/kernel_1_triton_per_fused__softmax_conv/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (16384,1,1), Waves/SM: 7.76
- **Registers/thread**: 32
- **Duration**: 274080 ns
- **SM Throughput**: 36.51 % | **Memory**: 82.59 % | **DRAM**: 82.59 %
- **Achieved Occupancy**: 93.32 %
- **IPC Active**: 1.49 inst/cycle | **No Eligible**: 62.26 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | memory_coalescing_fix_stride2_access | 1.239× | 0.978× | ✓ | **✓ best** |
| B | online_softmax_attention | 1.000× | 0.999× | ✓ |  |
| C | add_triton_autotune | 1.079× | 1.086× | ✓ |  |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ | online_softmax_at... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 274080 ns | 280350 ns | +6270 ns (+2.3%) | 274500 ns | +420 ns (+0.2%) | 252320 ns | -21760 ns (-7.9%) |
| SM Throughput (%) | 36.51 % | 34.86 % | -1.65 pp | 36.34 % | -0.17 pp | 55.82 % | +19.31 pp |
| Memory Throughput (%) | 82.59 % | 95.34 % | +12.75 pp | 82.41 % | -0.18 pp | 87.47 % | +4.88 pp |
| DRAM Throughput (%) | 82.59 % | 39.72 % | -42.87 pp | 82.41 % | -0.18 pp | 87.47 % | +4.88 pp |
| Achieved Occupancy (%) | 93.32 % | 60.75 % | -32.57 pp | 93.16 % | -0.16 pp | 18.22 % | -75.10 pp |
| Registers/Thread | 32.00 register/thread | 44.00 register/thread | +12 | 32.00 register/thread | +0 | 143.00 register/thread | +111 |
| Waves/SM | 7.76 | 12.41 | +4.65 | 7.76 | +0.00 | 41.37 | +33.61 |
| No Eligible (%) | 62.26 % | 64.40 % | +2.14 pp | 62.39 % | +0.13 pp | 42.93 % | -19.33 pp |
| IPC Active | 1.49 inst/cycle | 1.42 inst/cycle | -0.07 | 1.50 inst/cycle | +0.01 | 2.27 inst/cycle | +0.78 |

---

## Cumulative Progress

```
Baseline:  274080 ns  (1.000×)

Final best: 280350 ns  (0.9776×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_per_fused__softmax_convolution_exp_max_max_pool3d_with_indices_mul_prepare_softmax_online_sigmoid_sub_view_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
@@ -21,20 +22,20 @@

     roffset = r0_offset
     rindex = r0_index
     r0_4 = r0_index
-    x0 = (xindex % 32)
-    x1 = ((xindex // 32) % 32)
-    x2 = ((xindex // 1024) % 16)
-    x3 = xindex // 16384
+    x0 = (xindex % 64)
+    x1 = ((xindex // 64) % 32)
+    x2 = ((xindex // 2048) % 16)
+    x3 = xindex // 32768
     x5 = (xindex % 16384)
     x6 = xindex
-    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
-    tmp7 = tl.load(in_ptr0 + (4096 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
-    tmp9 = tl.load(in_ptr0 + (4097 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
-    tmp11 = tl.load(in_ptr0 + (4160 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
-    tmp13 = tl.load(in_ptr0 + (4161 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr0 + (1 + x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
+    tmp5 = tl.load(in_ptr0 + (65 + x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
+    tmp7 = tl.load(in_ptr0 + (4096 + x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
+    tmp9 = tl.load(in_ptr0 + (4097 + x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
+    tmp11 = tl.load(in_ptr0 + (4160 + x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
+    tmp13 = tl.load(in_ptr0 + (4161 + x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
     tmp28 = tl.load(in_ptr1 + (r0_4), None, eviction_policy='evict_last')
     tmp2 = tl.maximum(tmp0, tmp1)
     tmp4 = tl.maximum(tmp2, tmp3)
@@ -59,8 +60,6 @@

     tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
     tmp34 = triton_helpers.max2(tmp32, 1)[:, None].to(tl.float32)
     tl.store(in_out_ptr0 + (x6), tmp34, None)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
