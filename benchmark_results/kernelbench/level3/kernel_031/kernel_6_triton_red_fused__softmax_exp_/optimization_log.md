# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused__softmax_exp_prepare_softmax_online_sub_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6050000 ns — **Best latency**: 6050000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused__softmax_exp_prepare_softmax_online_sub_6`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_031/kernel_6_triton_red_fused__softmax_exp_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (131072,1,1), Waves/SM: 82.75
- **Registers/thread**: 40
- **Duration**: 6050000 ns
- **SM Throughput**: 63.89 % | **Memory**: 86.47 % | **DRAM**: 86.47 %
- **Achieved Occupancy**: 73.37 %
- **IPC Active**: 2.54 inst/cycle | **No Eligible**: 36.59 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | increase_xblock_autotune_tile_sizes_for_l1_reuse | 0.996× | 0.997× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_a... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 6050000 ns | 6070000 ns | +20000 ns (+0.3%) |
| SM Throughput (%) | 63.89 % | 65.62 % | +1.73 pp |
| Memory Throughput (%) | 86.47 % | 86.19 % | -0.28 pp |
| DRAM Throughput (%) | 86.47 % | 86.19 % | -0.28 pp |
| Achieved Occupancy (%) | 73.37 % | 73.42 % | +0.05 pp |
| Registers/Thread | 40.00 register/thread | 40.00 register/thread | +0 |
| Waves/SM | 82.75 | 82.75 | +0.00 |
| No Eligible (%) | 36.59 % | 34.32 % | -2.27 pp |
| IPC Active | 2.54 inst/cycle | 2.65 inst/cycle | +0.11 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_stages_to_hide_l1tex_latency_in_two_pass_softmax | 0.997× | 0.997× | ✓ |  |
| B | replace_division_with_reciprocal_multiply_and_fix_eviction_policies | 0.948× | 0.947× | ✓ |  |
| C | fix_eviction_policy_and_add_pipeline_stages_to_hide_l1tex_latency | 0.949× | 0.948× | ✓ |  |
| D | increase_num_stages_to_hide_l1tex_latency_via_software_pipelining | 0.996× | 0.995× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_stag... | Δ | replace_division_... | Δ | fix_eviction_poli... | Δ | increase_num_stag... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6050000 ns | 6070000 ns | +20000 ns (+0.3%) | 6390000 ns | +340000 ns (+5.6%) | 6380000 ns | +330000 ns (+5.5%) | 6080000 ns | +30000 ns (+0.5%) |
| SM Throughput (%) | 63.89 % | 65.64 % | +1.75 pp | 58.67 % | -5.22 pp | 62.39 % | -1.50 pp | 65.66 % | +1.77 pp |
| Memory Throughput (%) | 86.47 % | 86.20 % | -0.27 pp | 81.95 % | -4.52 pp | 82.09 % | -4.38 pp | 86.14 % | -0.33 pp |
| DRAM Throughput (%) | 86.47 % | 86.20 % | -0.27 pp | 81.95 % | -4.52 pp | 82.09 % | -4.38 pp | 86.14 % | -0.33 pp |
| Achieved Occupancy (%) | 73.37 % | 73.42 % | +0.05 pp | 72.73 % | -0.64 pp | 72.93 % | -0.44 pp | 73.42 % | +0.05 pp |
| Registers/Thread | 40.00 register/thread | 40.00 register/thread | +0 | 40.00 register/thread | +0 | 40.00 register/thread | +0 | 40.00 register/thread | +0 |
| Waves/SM | 82.75 | 82.75 | +0.00 | 82.75 | +0.00 | 82.75 | +0.00 | 82.75 | +0.00 |
| No Eligible (%) | 36.59 % | 34.30 % | -2.29 pp | 41.55 % | +4.96 pp | 37.50 % | +0.91 pp | 34.33 % | -2.26 pp |
| IPC Active | 2.54 inst/cycle | 2.64 inst/cycle | +0.10 | 2.33 inst/cycle | -0.21 | 2.51 inst/cycle | -0.03 | 2.65 inst/cycle | +0.11 |

---

## Cumulative Progress

```
Baseline:  6050000 ns  (1.000×)

Final best: 6050000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,21 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 1024}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 1024}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 8, 'R0_BLOCK': 1024}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 2048}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 8, 'R0_BLOCK': 2048}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 512}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 512}, num_warps=4, num_stages=3),
+    ],
+    key=['xnumel', 'r0_numel'],
+    restore_value=['in_out_ptr0'],
+)
 @triton.jit
 def triton_red_fused__softmax_exp_prepare_softmax_online_sub_6(in_out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 131072
@@ -13,7 +28,7 @@

     RBLOCK: tl.constexpr = R0_BLOCK
     xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
-    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
+    xmask = (xindex < xnumel)
     r0_base = tl.arange(0, R0_BLOCK)[None, :].to(tl.int64)
     rbase = r0_base
     x0 = xindex
@@ -25,7 +40,7 @@

         roffset = r0_offset
         rindex = r0_index
         r0_1 = r0_index
-        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 16384*x0), r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 16384*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
         tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
 
         _tmp2_max_next, _tmp2_sum_next = triton_helpers.online_softmax_combine(
@@ -45,20 +60,18 @@

         roffset = r0_offset
         rindex = r0_index
         r0_1 = r0_index
-        tmp4 = tl.load(in_out_ptr0 + (r0_1 + 16384*x0), r0_mask, eviction_policy='evict_first', other=0.0)
+        tmp4 = tl.load(in_out_ptr0 + (r0_1 + 16384*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
         tmp5 = tmp4 - tmp2
         tmp6 = tl.exp(tmp5)
         tmp7 = (tmp6 / tmp3)
-        tl.store(in_out_ptr0 + (r0_1 + 16384*x0), tmp7, r0_mask)
-
-
+        tl.store(in_out_ptr0 + (r0_1 + 16384*x0), tmp7, r0_mask & xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     xnumel = 131072
     r0_numel = 16384
     grid = (triton.cdiv(xnumel, 1),)
-    triton_red_fused__softmax_exp_prepare_softmax_online_sub_6[grid](arg_0, 131072, 16384, XBLOCK=1, R0_BLOCK=1024)
+    triton_red_fused__softmax_exp_prepare_softmax_online_sub_6[grid](arg_0, 131072, 16384)
     return arg_0
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
