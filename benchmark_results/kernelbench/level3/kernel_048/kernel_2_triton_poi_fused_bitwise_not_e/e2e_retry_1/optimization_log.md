# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3200 ns — **Best latency**: 3040 ns (**1.0526×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_xblock_and_num_warps_for_better_occupancy`** beats baseline latency.

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
- **SM Throughput**: 8.71 % | **Memory**: 5.42 % | **DRAM**: 1.29 %
- **Achieved Occupancy**: 41.90 %
- **IPC Active**: 0.96 inst/cycle | **No Eligible**: 72.63 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_and_num_warps_for_better_occupancy | 0.993× | 1.053× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_underutilized_pointwise_kernel | 1.039× | 1.053× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_xblock_a... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3200 ns | 3040 ns | -160 ns (-5.0%) | 3040 ns | -160 ns (-5.0%) |
| SM Throughput (%) | 8.71 % | 8.80 % | +0.09 pp | 7.66 % | -1.05 pp |
| Memory Throughput (%) | 5.42 % | 6.02 % | +0.60 pp | 5.68 % | +0.26 pp |
| DRAM Throughput (%) | 1.29 % | 1.36 % | +0.07 pp | 1.37 % | +0.08 pp |
| Achieved Occupancy (%) | 41.90 % | 45.60 % | +3.70 pp | 21.57 % | -20.33 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 22.00 register/thread | +6 |
| Waves/SM | 0.55 | 0.55 | +0.00 | 0.27 | -0.28 |
| No Eligible (%) | 72.63 % | 76.72 % | +4.09 pp | 70.61 % | -2.02 pp |
| IPC Active | 0.96 inst/cycle | 1.08 inst/cycle | +0.12 | 0.99 inst/cycle | +0.03 |


### Key Learnings

#### ✅ Plan A: `autotune_xblock_and_num_warps_for_better_occupancy` (1.053×)

When a Triton pointwise kernel launches with a very small XBLOCK (e.g., 128), it creates many tiny CTAs with few warps each, leaving warp schedulers idle most cycles (No Eligible >70%). Adding a @triton.autotune decorator that sweeps XBLOCK across {128, 512, 1024, 2048} and num_warps across {4, 8, 16} allows the runtime to select a tile size that packs more work per CTA, raising achieved occupancy and reducing scheduler idle time. The grid lambda must use meta['XBLOCK'] so it scales correctly with the autotuned block size. Note: in this case the speedup was modest (1.05x) because the kernel remained wave-limited (0.55 waves/SM) and the dominant bottleneck (L1TEX latency stalls, ~43% of cycles) was not addressed by tile size alone.

> **Applicable when**: Triton pointwise kernel with small fixed XBLOCK (<=128), achieved occupancy <60%, scheduler No Eligible >60%, waves_per_sm <1.0, and the kernel is not already memory-bandwidth saturated. NCU rules SOLBottleneck, IssueSlotUtilization, and AchievedOccupancy all triggered.

> **Avoid when**: Kernel is already bandwidth-bound (DRAM throughput >80%) or compute-bound (SM throughput >80%); adding autotune overhead would not help and the autotuning cost may exceed savings for very short-lived kernels. Also avoid when xnumel is not evenly divisible by all candidate XBLOCK values (requires careful config selection).

#### ✅ Plan B: `autotune_xblock_and_num_warps_for_underutilized_pointwise_kernel` (1.053×)

For small pointwise Triton kernels with low achieved occupancy (<50%), high no-eligible-warp fraction (>70%), and significant L1TEX stall cycles (>40%), replacing a hardcoded XBLOCK with @triton.autotune over multiple XBLOCK/num_warps/num_stages configurations allows the runtime to select the tile size that best hides memory latency. The key causal improvement here was a reduction in Warp Cycles Per Issued Instruction from 24.0 to 16.6 and an increase in L1/TEX Hit Rate from 33.6% to 49.5%, driven by the autotuner selecting a smaller block size (64 threads) with better cache reuse. Note that achieved occupancy dropped (41.9% → 21.6%) because fewer total threads were launched, but the per-warp efficiency gain dominated.

> **Applicable when**: Triton pointwise kernel with: hardcoded XBLOCK (no autotuning), achieved occupancy <50%, no-eligible-warp fraction >65%, L1TEX stall contributing >30% of warp cycles, and grid size that produces <1 full wave across SMs. Kernel body has no shared memory or barriers (safe for num_stages>1).

> **Avoid when**: Kernel already uses @triton.autotune; kernel has shared memory allocations or inter-warp barriers (num_stages>1 unsafe); xnumel is not divisible by all candidate XBLOCK values (requires careful masking); kernel is already compute-bound (high SM throughput >60%); grid is already large enough to saturate SMs (waves_per_sm >2).

---

## Cumulative Progress

```
Baseline:  3200 ns  (1.000×)
Iter 1     3040 ns  (1.053×)  — autotune_xblock_and_num_warps_for_underutilized_pointwise_kernel

Final best: 3040 ns  (1.0526×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,19 +5,33 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 128},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 512},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=2),
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
@@ -28,16 +42,14 @@

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
