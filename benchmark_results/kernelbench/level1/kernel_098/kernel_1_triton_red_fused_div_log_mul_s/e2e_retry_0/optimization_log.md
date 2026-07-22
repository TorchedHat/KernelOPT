# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_div_log_mul_sub_sum_xlogy_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3550 ns — **Best latency**: 2660 ns (**1.3346×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_reduce_register_pressure`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_div_log_mul_sub_sum_xlogy_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_098/kernel_1_triton_red_fused_div_log_mul_s/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1,1,1), Waves/SM: 0.00
- **Registers/thread**: 135
- **Duration**: 3550 ns
- **SM Throughput**: 0.16 % | **Memory**: 0.41 % | **DRAM**: 0.10 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.48 inst/cycle | **No Eligible**: 87.50 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_reduce_register_pressure | 1.101× | 1.335× | ✓ | **✓ best** |
| B | fuse_pointwise_ops_into_reduction_and_autotune_tile_sizes | 1.101× | 1.403× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | fuse_pointwise_op... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3550 ns | 2660 ns | -890 ns (-25.1%) | 2530 ns | -1020 ns (-28.7%) |
| SM Throughput (%) | 0.16 % | 0.01 % | -0.15 pp | 0.02 % | -0.14 pp |
| Memory Throughput (%) | 0.41 % | 0.42 % | +0.01 pp | 0.44 % | +0.03 pp |
| DRAM Throughput (%) | 0.10 % | 0.11 % | +0.01 pp | 0.10 % | +0.00 pp |
| Achieved Occupancy (%) | 6.24 % | 1.56 % | -4.68 pp | 5.93 % | -0.31 pp |
| Registers/Thread | 135.00 register/thread | 66.00 register/thread | -69 | 24.00 register/thread | -111 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 87.50 % | 90.53 % | +3.03 pp | 93.15 % | +5.65 pp |
| IPC Active | 0.48 inst/cycle | 0.10 inst/cycle | -0.38 | 0.24 inst/cycle | -0.24 |

---

## Cumulative Progress

```
Baseline:  3550 ns  (1.000×)
Iter 1     2660 ns  (1.335×)  — autotune_tile_sizes_reduce_register_pressure

Final best: 2660 ns  (1.3346×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 2048}, num_warps=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 2048}, num_warps=2),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=2),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 512},  num_warps=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 256},  num_warps=1),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused_div_log_mul_sub_sum_xlogy_1(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 1
@@ -23,7 +35,7 @@

         roffset = r0_offset
         rindex = r0_index
         r0_0 = r0_index
-        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
+        tmp0 = tl.load(in_ptr0 + (r0_0), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
         tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
         tmp3 = _tmp2 + tmp1
         _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
@@ -31,9 +43,7 @@

     tmp4 = tl.full([1, 1], 6.103515625e-05, tl.float32)
     tmp5 = tmp2 * tmp4
     tl.debug_barrier()
-    tl.store(in_out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp5, None)
-
-
+    tl.store(in_out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp5, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -41,7 +51,7 @@

     xnumel = 1
     r0_numel = 2048
     grid = (triton.cdiv(xnumel, 128),)
-    triton_red_fused_div_log_mul_sub_sum_xlogy_1[grid](arg_0, arg_1, 1, 2048, XBLOCK=128, R0_BLOCK=512)
+    triton_red_fused_div_log_mul_sub_sum_xlogy_1[grid](arg_0, arg_1, 1, 2048)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
