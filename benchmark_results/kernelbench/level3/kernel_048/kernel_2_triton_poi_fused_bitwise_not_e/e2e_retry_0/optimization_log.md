# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3200 ns — **Best latency**: 3260 ns (**0.9816×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_048/kernel_2_triton_poi_fused_bitwise_not_e/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1152,1,1), Waves/SM: 0.55
- **Registers/thread**: 16
- **Duration**: 3200 ns
- **SM Throughput**: 8.44 % | **Memory**: 5.51 % | **DRAM**: 1.28 %
- **Achieved Occupancy**: 41.43 %
- **IPC Active**: 0.94 inst/cycle | **No Eligible**: 73.99 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_reduce_grid_overhead | 1.044× | 1.088× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_underutilized_small_grid | 1.045× | 0.982× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_r... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3200 ns | 2940 ns | -260 ns (-8.1%) | 3260 ns | +60 ns (+1.9%) |
| SM Throughput (%) | 8.44 % | 15.33 % | +6.89 pp | 9.28 % | +0.84 pp |
| Memory Throughput (%) | 5.51 % | 5.62 % | +0.11 pp | 5.53 % | +0.02 pp |
| DRAM Throughput (%) | 1.28 % | 1.40 % | +0.12 pp | 1.25 % | -0.03 pp |
| Achieved Occupancy (%) | 41.43 % | 38.95 % | -2.48 pp | 39.75 % | -1.68 pp |
| Registers/Thread | 16.00 register/thread | 22.00 register/thread | +6 | 16.00 register/thread | +0 |
| Waves/SM | 0.55 | 0.55 | +0.00 | 0.55 | +0.00 |
| No Eligible (%) | 73.99 % | 50.19 % | -23.80 pp | 69.19 % | -4.80 pp |
| IPC Active | 0.94 inst/cycle | 1.66 inst/cycle | +0.72 | 0.96 inst/cycle | +0.02 |

---

## Cumulative Progress

```
Baseline:  3200 ns  (1.000×)

Final best: 3260 ns  (0.9816×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,19 +5,32 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=8),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 147456
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 3)
     x1 = ((xindex // 3) % 3)
     x3 = xindex // 3
     x2 = xindex // 9
     x4 = xindex
-    tmp6 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
-    tmp7 = tl.load(in_ptr0 + (x0 + 3*x2), None, eviction_policy='evict_last')
+    tmp6 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
+    tmp7 = tl.load(in_ptr0 + (x0 + 3*x2), xmask, eviction_policy='evict_last')
     tmp0 = x0 + ((-1)*x1)
     tmp1 = tl.full([1], 0, tl.int64)
     tmp2 = tmp0 <= tmp1
@@ -28,16 +41,14 @@

     tmp9 = tl.full([1], float("-inf"), tl.float32)
     tmp10 = tl.where(tmp5, tmp9, tmp8)
     tmp11 = tl.exp(tmp10)
-    tl.store(out_ptr0 + (x4), tmp11, None)
-
-
+    tl.store(out_ptr0 + (x4), tmp11, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 147456
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2[grid](arg_0, arg_1, 147456, XBLOCK=128)
+    triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2[grid](arg_0, arg_1, 147456)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
