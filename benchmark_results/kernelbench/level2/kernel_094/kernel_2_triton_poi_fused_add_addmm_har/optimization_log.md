# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_addmm_hardtanh_mish_native_group_norm_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 46430 ns — **Best latency**: 43200 ns (**1.0748×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`precompute_group_norm_row_index_to_eliminate_per_element_integer_division`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_addmm_hardtanh_mish_native_group_norm_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_094/kernel_2_triton_poi_fused_add_addmm_har/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (65536,1,1), Waves/SM: 31.03
- **Registers/thread**: 22
- **Duration**: 46430 ns
- **SM Throughput**: 78.56 % | **Memory**: 28.74 % | **DRAM**: 18.81 %
- **Achieved Occupancy**: 83.95 %
- **IPC Active**: 3.30 inst/cycle | **No Eligible**: 17.25 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune | 0.489× | 0.394× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_fused_pointwise | 0.997× | 0.904× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_fused_fp32 | 0.787× | 0.638× | ✓ |  |
| D | increase_xblock_and_autotune_for_fp32_fusion_and_l1_reuse | 0.998× | 0.394× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_a... | Δ | increase_xblock_f... | Δ | increase_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 46430 ns | 117890 ns | +71460 ns (+153.9%) | 51360 ns | +4930 ns (+10.6%) | 72770 ns | +26340 ns (+56.7%) | 117890 ns | +71460 ns (+153.9%) |
| SM Throughput (%) | 78.56 % | 89.51 % | +10.95 pp | 80.79 % | +2.23 pp | 86.58 % | +8.02 pp | 89.57 % | +11.01 pp |
| Memory Throughput (%) | 28.74 % | 22.00 % | -6.74 pp | 25.36 % | -3.38 pp | 26.23 % | -2.51 pp | 22.02 % | -6.72 pp |
| DRAM Throughput (%) | 18.81 % | 7.39 % | -11.42 pp | 17.02 % | -1.79 pp | 12.00 % | -6.81 pp | 7.40 % | -11.41 pp |
| Achieved Occupancy (%) | 83.95 % | 68.74 % | -15.21 pp | 85.34 % | +1.39 pp | 90.03 % | +6.08 pp | 68.80 % | -15.15 pp |
| Registers/Thread | 22.00 register/thread | 40.00 register/thread | +18 | 23.00 register/thread | +1 | 30.00 register/thread | +8 | 40.00 register/thread | +18 |
| Waves/SM | 31.03 | 41.37 | +10.34 | 31.03 | +0.00 | 31.03 | +0.00 | 41.37 | +10.34 |
| No Eligible (%) | 17.25 % | 8.83 % | -8.42 pp | 15.18 % | -2.07 pp | 10.83 % | -6.42 pp | 8.82 % | -8.43 pp |
| IPC Active | 3.30 inst/cycle | 3.65 inst/cycle | +0.35 | 3.38 inst/cycle | +0.08 | 3.56 inst/cycle | +0.26 | 3.64 inst/cycle | +0.34 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune` (0.394×)

Adding @triton.autotune with larger XBLOCK values (512–2048) caused the compiler to allocate significantly more registers per thread (22 → 40), which reduced theoretical occupancy from 100% to 75% and achieved occupancy from 83.95% to 68.74%. Despite higher compute SM throughput (78.56% → 89.51%), the 2.9x increase in executed instructions and 2.5x longer kernel duration indicate the autotune selected a config that dramatically increased per-thread work without proportional ILP gains, while the register pressure increase was the primary occupancy killer. When a kernel is already near 100% theoretical occupancy with low register count, increasing tile size via autotune risks register pressure that collapses occupancy and worsens overall throughput.

> **Applicable when**: Never — this is a negative result. Avoid increasing XBLOCK via autotune when: registers_per_thread is already low (≤24), theoretical occupancy is at or near 100%, and the kernel is compute-bound with high SM busy (>75%).

> **Avoid when**: Do not apply larger XBLOCK autotune when the kernel already achieves 100% theoretical occupancy with low register count. Increasing tile size in this regime causes register pressure to grow, reducing Block Limit Registers (21→12 blocks/SM), collapsing theoretical occupancy (100%→75%), and increasing total instruction count without proportional latency reduction.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_increase_num_warps_for_transcendental_latency_hiding | 1.003× | 0.906× | ✓ |  |
| B | replace_integer_division_with_bitshift_to_reduce_non_fused_fp_instructions | 0.756× | 0.611× | ✓ |  |
| C | reduce_register_pressure_via_maxnreg_to_restore_occupancy | 0.998× | 0.902× | ✓ |  |
| D | fix_broken_autotune_grid_and_restore_occupancy_via_num_warps_sweep | 1.009× | 1.075× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | replace_integer_d... | Δ | reduce_register_p... | Δ | fix_broken_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 46430 ns | 51230 ns | +4800 ns (+10.3%) | 76030 ns | +29600 ns (+63.8%) | 51490 ns | +5060 ns (+10.9%) | 43200 ns | -3230 ns (-7.0%) |
| SM Throughput (%) | 78.56 % | 81.03 % | +2.47 pp | 87.09 % | +8.53 pp | 80.79 % | +2.23 pp | 76.79 % | -1.77 pp |
| Memory Throughput (%) | 28.74 % | 25.43 % | -3.31 pp | 36.04 % | +7.30 pp | 25.36 % | -3.38 pp | 28.73 % | -0.01 pp |
| DRAM Throughput (%) | 18.81 % | 17.06 % | -1.75 pp | 11.48 % | -7.33 pp | 16.96 % | -1.85 pp | 20.33 % | +1.52 pp |
| Achieved Occupancy (%) | 83.95 % | 85.52 % | +1.57 pp | 90.54 % | +6.59 pp | 85.40 % | +1.45 pp | 63.80 % | -20.15 pp |
| Registers/Thread | 22.00 register/thread | 23.00 register/thread | +1 | 28.00 register/thread | +6 | 23.00 register/thread | +1 | 30.00 register/thread | +8 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 | 15.52 | -15.51 |
| No Eligible (%) | 17.25 % | 15.43 % | -1.82 pp | 10.47 % | -6.78 pp | 15.20 % | -2.05 pp | 15.75 % | -1.50 pp |
| IPC Active | 3.30 inst/cycle | 3.39 inst/cycle | +0.09 | 3.58 inst/cycle | +0.28 | 3.39 inst/cycle | +0.09 | 3.33 inst/cycle | +0.03 |


### Key Learnings

#### ❌ Plan B: `replace_integer_division_with_bitshift_to_reduce_non_fused_fp_instructions` (0.611×)

Replacing integer division with bitshifts (x2 // 32 → x2 >> 5, xindex % 8192 → xindex & 8191) while simultaneously changing the autotune configs caused executed instructions to balloon by ~60% (32.5M → 51.9M), registers per thread to increase from 23 to 28, and memory throughput to drop from 836 GB/s to 564 GB/s, resulting in a 48% longer kernel duration (51 µs → 76 µs). The bitshift substitution itself is a no-op at PTX level because LLVM/Triton already optimizes power-of-2 integer division to shifts for compile-time constants; the regression was caused by the simultaneous autotune config change (removing small XBLOCK configs and adding XBLOCK=2048) which selected a larger tile size, inflating instruction count and register pressure. The UncoalescedGlobalAccess rule newly appeared after the change, indicating the larger XBLOCK worsened memory access coalescing.

> **Applicable when**: Never — this is a negative result. Do not apply this combination.

> **Avoid when**: Do not replace power-of-2 integer division/modulo with bitshifts in Triton kernels targeting LLVM-based backends: the compiler already performs this optimization for compile-time-constant power-of-2 divisors, making the change a no-op at PTX level. More critically, do not simultaneously change autotune configs (especially removing num_stages or adding very large XBLOCK values like 2048) alongside arithmetic substitutions, as the autotune config change is the actual driver of regression — it selects larger tiles that increase register pressure (23→28 regs/thread), inflate executed instruction count (~60%), and worsen memory coalescing (UncoalescedGlobalAccess rule newly triggered, MemoryCacheAccessPattern worsened from 29.5 to 17.3 bytes/sector utilized).

---

## Iteration 3

**Best so far**: 1.0748× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_add_high_warp_configs_for_transcendental_latency_hiding | 1.004× | 1.075× | ✓ |  |
| B | change_eviction_policy_streaming_vs_reuse_tensors | 0.991× | 0.880× | ✓ |  |
| C | change_eviction_policy_on_strided_non_reused_loads_to_evict_first | 1.009× | 1.075× | ✓ |  |
| D | restore_small_xblock_configs_and_fix_eviction_policy_for_streaming_access | 1.005× | 0.902× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | change_eviction_p... | Δ | change_eviction_p... | Δ | restore_small_xbl... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 46430 ns | 43170 ns | -3260 ns (-7.0%) | 52740 ns | +6310 ns (+13.6%) | 43170 ns | -3260 ns (-7.0%) | 51490 ns | +5060 ns (+10.9%) |
| SM Throughput (%) | 78.56 % | 77.50 % | -1.06 pp | 81.99 % | +3.43 pp | 76.48 % | -2.08 pp | 81.55 % | +2.99 pp |
| Memory Throughput (%) | 28.74 % | 28.99 % | +0.25 pp | 24.68 % | -4.06 pp | 28.57 % | -0.17 pp | 25.76 % | -2.98 pp |
| DRAM Throughput (%) | 18.81 % | 20.33 % | +1.52 pp | 16.56 % | -2.25 pp | 20.28 % | +1.47 pp | 17.05 % | -1.76 pp |
| Achieved Occupancy (%) | 83.95 % | 61.03 % | -22.92 pp | 86.31 % | +2.36 pp | 59.83 % | -24.12 pp | 85.69 % | +1.74 pp |
| Registers/Thread | 22.00 register/thread | 30.00 register/thread | +8 | 23.00 register/thread | +1 | 30.00 register/thread | +8 | 22.00 register/thread | +0 |
| Waves/SM | 31.03 | 15.52 | -15.51 | 31.03 | +0.00 | 15.52 | -15.51 | 31.03 | +0.00 |
| No Eligible (%) | 17.25 % | 16.84 % | -0.41 pp | 14.34 % | -2.91 pp | 16.95 % | -0.30 pp | 14.64 % | -2.61 pp |
| IPC Active | 3.30 inst/cycle | 3.31 inst/cycle | +0.01 | 3.41 inst/cycle | +0.11 | 3.29 inst/cycle | -0.01 | 3.41 inst/cycle | +0.11 |


### Key Learnings

#### ✅ Plan C: `change_eviction_policy_on_strided_non_reused_loads_to_evict_first` (1.075×)

When a kernel contains loads indexed by a strided expression (e.g., xindex // 32) that produces a unique address per thread with no intra-block reuse, marking those loads with eviction_policy='evict_first' prevents them from polluting L1/TEX cache and competing with truly reused tensors (indexed by xindex % N, with high reuse factor). In this case, changing two strided group-norm stat loads from evict_last to evict_first improved L1/TEX hit rate from 69.8% to 71.53% and yielded a modest ~7.5% end-to-end speedup. The strategy is most effective when the kernel has a mix of high-reuse (evict_last) and streaming (evict_first) loads, and the streaming loads are large enough to evict the reused data from L1.

> **Applicable when**: Kernel has a mix of (a) high-reuse loads indexed by xindex % N (same address accessed by many blocks) and (b) streaming/strided loads indexed by xindex // K or similar expressions that produce unique addresses per thread with no intra-block reuse. L1/TEX hit rate is below ~75% and MemoryCacheAccessPattern rule is triggered. The streaming tensors are large enough to evict reused data from L1 when marked evict_last.

> **Avoid when**: Do not change eviction policy to evict_first when the load IS reused within a block or across nearby blocks (e.g., indexed by xindex % N where N is small relative to block size). Also avoid when L1 hit rate is already high (>85%) or when the kernel is purely memory-bound with no reuse — the benefit is marginal in those cases. The speedup here was modest (~7.5%), so this is a low-priority optimization when larger bottlenecks (e.g., FPInstructions, occupancy) remain unaddressed.

---

## Iteration 4

**Best so far**: 1.0748× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | precompute_group_norm_row_index_to_eliminate_per_element_integer_division | 0.998× | 1.075× | ✓ |  |
| B | add_num_stages3_and_num_warps16_configs_to_close_occupancy_gap | 1.004× | 1.073× | ✓ |  |
| C | eliminate_dead_mish_softplus_branch_after_hardtanh_clamp | 0.998× | 0.911× | ✓ |  |
| D | fix_grid_lambda_to_enable_correct_xblock256_autotune | 0.998× | 0.879× | ✓ |  |


### Metric Comparison

| Metric | Baseline | precompute_group_... | Δ | add_num_stages3_a... | Δ | eliminate_dead_mi... | Δ | fix_grid_lambda_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 46430 ns | 43200 ns | -3230 ns (-7.0%) | 43260 ns | -3170 ns (-6.8%) | 50980 ns | +4550 ns (+9.8%) | 52800 ns | +6370 ns (+13.7%) |
| SM Throughput (%) | 78.56 % | 77.45 % | -1.11 pp | 76.71 % | -1.85 pp | 81.37 % | +2.81 pp | 82.00 % | +3.44 pp |
| Memory Throughput (%) | 28.74 % | 28.64 % | -0.10 pp | 28.66 % | -0.08 pp | 25.71 % | -3.03 pp | 24.69 % | -4.05 pp |
| DRAM Throughput (%) | 18.81 % | 20.24 % | +1.43 pp | 20.15 % | +1.34 pp | 17.22 % | -1.59 pp | 16.49 % | -2.32 pp |
| Achieved Occupancy (%) | 83.95 % | 61.64 % | -22.31 pp | 59.71 % | -24.24 pp | 86.07 % | +2.12 pp | 86.24 % | +2.29 pp |
| Registers/Thread | 22.00 register/thread | 30.00 register/thread | +8 | 30.00 register/thread | +8 | 22.00 register/thread | +0 | 23.00 register/thread | +1 |
| Waves/SM | 31.03 | 15.52 | -15.51 | 15.52 | -15.51 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 17.25 % | 15.30 % | -1.95 pp | 17.03 % | -0.22 pp | 14.64 % | -2.61 pp | 14.37 % | -2.88 pp |
| IPC Active | 3.30 inst/cycle | 3.35 inst/cycle | +0.05 | 3.31 inst/cycle | +0.01 | 3.40 inst/cycle | +0.10 | 3.41 inst/cycle | +0.11 |

---

## Cumulative Progress

```
Baseline:  46430 ns  (1.000×)
Iter 2     43200 ns  (1.075×)  — fix_broken_autotune_grid_and_restore_occupancy_via_num_warps_sweep

Final best: 43200 ns  (1.0748×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,21 +5,37 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=2,  num_stages=3),
+        triton.Config({'XBLOCK': 128}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 128}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 128}, num_warps=16, num_stages=3),
+        triton.Config({'XBLOCK': 128}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 128}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 128}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 128}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=16, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_add_addmm_hardtanh_mish_native_group_norm_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
     xnumel = 8388608
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 8192)
     x2 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_out_ptr0 + (x2), None)
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp16 = tl.load(in_ptr2 + (x2 // 32), None, eviction_policy='evict_last')
-    tmp18 = tl.load(in_ptr3 + (x2 // 32), None, eviction_policy='evict_last')
-    tmp25 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
-    tmp27 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp16 = tl.load(in_ptr2 + (x2 // 32), xmask, eviction_policy='evict_last')
+    tmp18 = tl.load(in_ptr3 + (x2 // 32), xmask, eviction_policy='evict_last')
+    tmp25 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
+    tmp27 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp4 = tmp2 + tmp3
     tmp5 = tl.full([1], -1.0, tl.float32)
@@ -42,9 +58,7 @@

     tmp24 = tmp17 * tmp23
     tmp26 = tmp24 * tmp25
     tmp28 = tmp26 + tmp27
-    tl.store(in_out_ptr0 + (x2), tmp28, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp28, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -56,7 +70,7 @@

     arg_6 = inputs[6]
     xnumel = 8388608
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_add_addmm_hardtanh_mish_native_group_norm_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 8388608, XBLOCK=128)
+    triton_poi_fused_add_addmm_hardtanh_mish_native_group_norm_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 8388608)
     return arg_6
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
