# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_addmm_mul_native_group_norm_sigmoid_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 647550 ns — **Best latency**: 647550 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_addmm_mul_native_group_norm_sigmoid_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_037/kernel_2_triton_poi_fused_add_addmm_mul/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1048576,1,1), Waves/SM: 496.48
- **Registers/thread**: 24
- **Duration**: 647550 ns
- **SM Throughput**: 52.43 % | **Memory**: 33.46 % | **DRAM**: 33.46 %
- **Achieved Occupancy**: 53.52 %
- **IPC Active**: 2.10 inst/cycle | **No Eligible**: 47.37 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | increase_xblock_with_autotune_to_hide_l1tex_latency | 0.630× | 0.629× | ✓ |  |
| D | vectorized_loads_via_larger_xblock_autotune | 0.630× | 0.629× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 647550 ns | 1030000 ns | +382450 ns (+59.1%) | 1030000 ns | +382450 ns (+59.1%) |
| SM Throughput (%) | 52.43 % | 88.19 % | +35.76 pp | 88.20 % | +35.77 pp |
| Memory Throughput (%) | 33.46 % | 41.10 % | +7.64 pp | 41.11 % | +7.65 pp |
| DRAM Throughput (%) | 33.46 % | 21.01 % | -12.45 pp | 21.00 % | -12.46 pp |
| Achieved Occupancy (%) | 53.52 % | 87.77 % | +34.25 pp | 87.91 % | +34.39 pp |
| Registers/Thread | 24.00 register/thread | 32.00 register/thread | +8 | 32.00 register/thread | +8 |
| Waves/SM | 496.48 | 496.48 | +0.00 | 496.48 | +0.00 |
| No Eligible (%) | 47.37 % | 11.62 % | -35.75 pp | 11.64 % | -35.73 pp |
| IPC Active | 2.10 inst/cycle | 3.53 inst/cycle | +1.43 | 3.53 inst/cycle | +1.43 |


### Key Learnings

#### ❌ Plan C: `increase_xblock_with_autotune_to_hide_l1tex_latency` (0.629×)

Increasing XBLOCK from 128 to 512–2048 via @triton.autotune successfully raised achieved occupancy (53.5%→87.8%) and eligible warps/scheduler (0.87→7.73), resolving the CPIStall bottleneck. However, the total executed instruction count exploded by ~2.66x (268M→715M instructions), and registers per thread increased from 24 to 32, reducing the block limit from 21 to 16 blocks/SM. The net effect was a 59% longer kernel duration (648µs→1030µs) despite better scheduler utilization, because the extra instructions dominated. This pattern occurs when the kernel body contains predicated or loop-unrolled code that scales super-linearly with XBLOCK, or when the autotune grid lambda is not updated to match the new XBLOCK (leaving the grid fixed at cdiv(xnumel, 128) and causing each thread to process multiple elements with masking overhead).

> **Applicable when**: Kernel has CPIStall rule triggered with L1TEX stall fraction >50% of total stall cycles, low eligible warps/scheduler (<1.0), and achieved occupancy well below theoretical. The kernel body is simple (no inner loops) and instruction count should scale linearly with XBLOCK.

> **Avoid when**: The autotune grid lambda is not correctly updated to use meta['XBLOCK'] — this causes the grid to remain fixed at the old small-XBLOCK size, forcing each thread to process multiple elements with masking overhead, exploding instruction count. Also avoid when increasing XBLOCK causes registers/thread to increase (e.g. 24→32), which reduces block-limit-registers and can offset occupancy gains. Do not apply when xnumel is not evenly divisible by all candidate XBLOCK values and the original kernel used tl.full([XBLOCK], True) to skip masking.

#### ❌ Plan D: `vectorized_loads_via_larger_xblock_autotune` (0.629×)

Adding @triton.autotune with larger XBLOCK values (512–2048) increased register pressure from 24 to 32 registers/thread, reducing Block Limit Registers from 21 to 16 blocks/SM. More critically, executed instructions ballooned from 268M to 715M (2.66×) — likely due to the autotune selecting a config that unrolls more work per thread and/or the xmask guard (changed from tl.full([XBLOCK], True) to xindex < xnumel) adding predication overhead across all loads and stores. Despite occupancy improving from 53% to 88% and eligible warps rising from 0.87 to 7.69, the 2.66× instruction count increase dominated, causing a 59% duration increase. When the original kernel has xnumel exactly divisible by XBLOCK (no masking needed), introducing a dynamic xmask on every load/store adds unnecessary predication instructions that inflate instruction count and negate the benefits of larger tiles.

> **Applicable when**: Never apply this exact pattern. The combination of autotune XBLOCK expansion + replacing tl.full([XBLOCK], True) masks with dynamic xindex < xnumel masks causes instruction count explosion even when xnumel is perfectly divisible by all candidate XBLOCK values.

> **Avoid when**: Do not add dynamic xmask (xindex < xnumel) when xnumel is statically known to be divisible by all candidate XBLOCK values — this adds predication to every load/store and inflates instruction count by 2-3×. Do not apply autotune XBLOCK expansion when the kernel already has high register pressure (≥24 regs/thread) and the block limit is register-bound, as larger tiles increase register usage further and reduce occupancy headroom.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_xmask_predication_overhead_to_collapse_instruction_count | 0.630× | 0.629× | ✓ |  |
| B | reduce_non_fused_fp32_instructions_via_fma_friendly_arithmetic_rewrite | 1.000× | 0.999× | ✓ |  |
| C | fix_autotune_grid_lambda_and_fuse_fp32_instructions | 0.630× | 0.629× | ✓ |  |
| D | eliminate_dynamic_xmask_predication_to_restore_baseline_instruction_count | 0.630× | 0.629× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_xmask_p... | Δ | reduce_non_fused_... | Δ | fix_autotune_grid... | Δ | eliminate_dynamic... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 647550 ns | 1030000 ns | +382450 ns (+59.1%) | 648030 ns | +480 ns (+0.1%) | 1030000 ns | +382450 ns (+59.1%) | 1030000 ns | +382450 ns (+59.1%) |
| SM Throughput (%) | 52.43 % | 88.26 % | +35.83 pp | 52.43 % | +0.00 pp | 88.06 % | +35.63 pp | 88.16 % | +35.73 pp |
| Memory Throughput (%) | 33.46 % | 41.14 % | +7.68 pp | 33.44 % | -0.02 pp | 41.05 % | +7.59 pp | 41.09 % | +7.63 pp |
| DRAM Throughput (%) | 33.46 % | 20.94 % | -12.52 pp | 33.44 % | -0.02 pp | 21.01 % | -12.45 pp | 20.97 % | -12.49 pp |
| Achieved Occupancy (%) | 53.52 % | 87.82 % | +34.30 pp | 54.30 % | +0.78 pp | 87.82 % | +34.30 pp | 87.80 % | +34.28 pp |
| Registers/Thread | 24.00 register/thread | 32.00 register/thread | +8 | 22.00 register/thread | -2 | 32.00 register/thread | +8 | 32.00 register/thread | +8 |
| Waves/SM | 496.48 | 496.48 | +0.00 | 496.48 | +0.00 | 496.48 | +0.00 | 496.48 | +0.00 |
| No Eligible (%) | 47.37 % | 11.46 % | -35.91 pp | 47.37 % | +0.00 pp | 11.73 % | -35.64 pp | 11.77 % | -35.60 pp |
| IPC Active | 2.10 inst/cycle | 3.54 inst/cycle | +1.44 | 2.10 inst/cycle | +0.00 | 3.53 inst/cycle | +1.43 | 3.53 inst/cycle | +1.43 |

---

## Cumulative Progress

```
Baseline:  647550 ns  (1.000×)

Final best: 647550 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,21 +5,33 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=8,  num_stages=2),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_add_addmm_mul_native_group_norm_sigmoid_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
     xnumel = 134217728
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 4096)
     x2 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_out_ptr0 + (x2), None)
-    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp7 = tl.load(in_ptr2 + (x2 // 64), None, eviction_policy='evict_last')
-    tmp9 = tl.load(in_ptr3 + (x2 // 64), None, eviction_policy='evict_last')
-    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
-    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp7 = tl.load(in_ptr2 + (x2 // 64), xmask, eviction_policy='evict_last')
+    tmp9 = tl.load(in_ptr3 + (x2 // 64), xmask, eviction_policy='evict_last')
+    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
+    tmp18 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp3 = tl.sigmoid(tmp2)
     tmp4 = tmp3 * tmp2
@@ -33,9 +45,7 @@

     tmp15 = tmp8 * tmp14
     tmp17 = tmp15 * tmp16
     tmp19 = tmp17 + tmp18
-    tl.store(in_out_ptr0 + (x2), tmp19, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -47,7 +57,7 @@

     arg_6 = inputs[6]
     xnumel = 134217728
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_add_addmm_mul_native_group_norm_sigmoid_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 134217728, XBLOCK=128)
+    triton_poi_fused_add_addmm_mul_native_group_norm_sigmoid_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 134217728)
     return arg_6
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
