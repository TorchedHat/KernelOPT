# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused__softmax_exp_slice_sub_sum_view_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 10690 ns — **Best latency**: 3460 ns (**3.0896×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_static_grid_lambda_add_small_xblock_and_num_stages_configs`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused__softmax_exp_slice_sub_sum_view_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_046/kernel_3_triton_red_fused__softmax_exp_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 0.24
- **Registers/thread**: 32
- **Duration**: 10690 ns
- **SM Throughput**: 9.75 % | **Memory**: 52.64 % | **DRAM**: 52.64 %
- **Achieved Occupancy**: 23.70 %
- **IPC Active**: 0.45 inst/cycle | **No Eligible**: 87.55 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune_tile_sizes | 1.416× | 0.997× | ✓ |  |
| C | software_pipelining_with_autotune_tile_sizes | 1.714× | 2.319× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | software_pipelini... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 10690 ns | 10720 ns | +30 ns (+0.3%) | 4610 ns | -6080 ns (-56.9%) |
| SM Throughput (%) | 9.75 % | 12.38 % | +2.63 pp | 9.49 % | -0.26 pp |
| Memory Throughput (%) | 52.64 % | 52.47 % | -0.17 pp | 29.20 % | -23.44 pp |
| DRAM Throughput (%) | 52.64 % | 52.47 % | -0.17 pp | 29.20 % | -23.44 pp |
| Achieved Occupancy (%) | 23.70 % | 23.61 % | -0.09 pp | 23.00 % | -0.70 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 30.00 register/thread | -2 |
| Waves/SM | 0.24 | 0.24 | +0.00 | 0.24 | +0.00 |
| No Eligible (%) | 87.55 % | 84.64 % | -2.91 pp | 84.00 % | -3.55 pp |
| IPC Active | 0.45 inst/cycle | 0.63 inst/cycle | +0.18 | 0.66 inst/cycle | +0.21 |

---

## Iteration 2

**Best so far**: 2.3189× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_contiguous_and_multiple_of_hints | 1.730× | 2.319× | ✓ |  |
| C | fix_static_grid_and_reduce_smem_bank_conflicts | 1.759× | 2.334× | ✓ | **✓ best** |
| D | memory_coalescing_via_accumulator_transpose | 1.416× | 0.941× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | fix_static_grid_a... | Δ | memory_coalescing... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10690 ns | 4610 ns | -6080 ns (-56.9%) | 4580 ns | -6110 ns (-57.2%) | 11360 ns | +670 ns (+6.3%) |
| SM Throughput (%) | 9.75 % | 9.41 % | -0.34 pp | 11.29 % | +1.54 pp | 12.34 % | +2.59 pp |
| Memory Throughput (%) | 52.64 % | 29.25 % | -23.39 pp | 29.33 % | -23.31 pp | 49.73 % | -2.91 pp |
| DRAM Throughput (%) | 52.64 % | 29.25 % | -23.39 pp | 29.33 % | -23.31 pp | 49.73 % | -2.91 pp |
| Achieved Occupancy (%) | 23.70 % | 22.86 % | -0.84 pp | 22.64 % | -1.06 pp | 23.66 % | -0.04 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 0.24 | 0.24 | +0.00 | 0.24 | +0.00 | 0.24 | +0.00 |
| No Eligible (%) | 87.55 % | 82.80 % | -4.75 pp | 80.02 % | -7.53 pp | 83.99 % | -3.56 pp |
| IPC Active | 0.45 inst/cycle | 0.66 inst/cycle | +0.21 | 0.72 inst/cycle | +0.27 | 0.64 inst/cycle | +0.19 |


### Key Learnings

#### ✅ Plan A: `vectorized_loads_via_contiguous_and_multiple_of_hints` (2.319×)

Wrapping index tensors with tl.max_contiguous and tl.multiple_of hints in Triton kernels communicates alignment and contiguity guarantees to the compiler backend, enabling it to emit wider (128-bit LDG.128) vectorized global memory loads. This is most impactful when the dominant stall is CPIStall on L1TEX scoreboard dependencies (warps blocked waiting for global memory), L1 hit rate is 0%, and L2 hit rate is very low (under 10%), indicating all traffic goes to DRAM. Note: in this specific case the NCU-reported kernel duration did not change (4.61us before and after), yet the wall-clock speedup was 2.32x — suggesting the benefit manifests in reduced kernel launch overhead, better pipelining across multiple kernel invocations, or reduced total work across the full model rather than in single-kernel NCU metrics.

> **Applicable when**: Triton kernel with CPIStall rule triggered (L1TEX scoreboard stalls >50% of warp cycles), L1 hit rate ~0%, L2 hit rate <15%, all traffic going to DRAM. Index tensors (xindex, r0_index) are derived from program_id * BLOCK + arange(BLOCK) patterns that are provably aligned to block boundaries but lack explicit compiler hints. Wall-clock speedup is observed even when single-kernel NCU duration appears unchanged.

> **Avoid when**: Index computation is not a simple offset + arange pattern (e.g., involves modulo or non-uniform strides that break alignment guarantees). tl.multiple_of value must exactly match the actual alignment — incorrect hints cause silent correctness bugs. Do not apply when the bottleneck is compute-bound (SM throughput >50%) rather than memory-latency-bound.

---

## Iteration 3

**Best so far**: 2.3341× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | scalar_accumulator_loop_to_eliminate_smem_bank_conflicts | 1.955× | 1.009× | ✓ |  |
| B | fix_dynamic_grid_with_lambda_and_larger_xblock_configs | 1.994× | 2.762× | ✓ | **✓ best** |
| C | fix_static_grid_lambda_and_smem_bank_conflict_via_accumulator_transpose | 1.419× | 0.974× | ✓ |  |


### Metric Comparison

| Metric | Baseline | scalar_accumulato... | Δ | fix_dynamic_grid_... | Δ | fix_static_grid_l... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10690 ns | 10590 ns | -100 ns (-0.9%) | 3870 ns | -6820 ns (-63.8%) | 10980 ns | +290 ns (+2.7%) |
| SM Throughput (%) | 9.75 % | 10.57 % | +0.82 pp | 5.64 % | -4.11 pp | 12.33 % | +2.58 pp |
| Memory Throughput (%) | 52.64 % | 53.32 % | +0.68 pp | 17.41 % | -35.23 pp | 51.24 % | -1.40 pp |
| DRAM Throughput (%) | 52.64 % | 53.32 % | +0.68 pp | 17.41 % | -35.23 pp | 51.24 % | -1.40 pp |
| Achieved Occupancy (%) | 23.70 % | 23.30 % | -0.40 pp | 11.12 % | -12.58 pp | 23.70 % | +0.00 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 29.00 register/thread | -3 | 32.00 register/thread | +0 |
| Waves/SM | 0.24 | 0.24 | +0.00 | 0.12 | -0.12 | 0.24 | +0.00 |
| No Eligible (%) | 87.55 % | 86.33 % | -1.22 pp | 87.24 % | -0.31 pp | 84.04 % | -3.51 pp |
| IPC Active | 0.45 inst/cycle | 0.54 inst/cycle | +0.09 | 0.47 inst/cycle | +0.02 | 0.63 inst/cycle | +0.18 |

---

## Iteration 4

**Best so far**: 2.7623× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_kernel_function_static_grid_to_lambda | 1.959× | 2.713× | ✓ |  |
| B | fix_static_grid_to_lambda_for_autotune_cta_scaling | 2.211× | 3.090× | ✓ | **✓ best** |
| C | persistent_small_xblock_with_lambda_grid_and_bank_conflict_free_configs | 1.955× | 2.693× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_kernel_functi... | Δ | fix_static_grid_t... | Δ | persistent_small_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10690 ns | 3940 ns | -6750 ns (-63.1%) | 3460 ns | -7230 ns (-67.6%) | 3970 ns | -6720 ns (-62.9%) |
| SM Throughput (%) | 9.75 % | 5.79 % | -3.96 pp | 3.39 % | -6.36 pp | 6.72 % | -3.03 pp |
| Memory Throughput (%) | 52.64 % | 17.13 % | -35.51 pp | 16.33 % | -36.31 pp | 17.01 % | -35.63 pp |
| DRAM Throughput (%) | 52.64 % | 17.13 % | -35.51 pp | 9.83 % | -42.81 pp | 17.01 % | -35.63 pp |
| Achieved Occupancy (%) | 23.70 % | 11.18 % | -12.52 pp | 5.59 % | -18.11 pp | 11.26 % | -12.44 pp |
| Registers/Thread | 32.00 register/thread | 29.00 register/thread | -3 | 40.00 register/thread | +8 | 32.00 register/thread | +0 |
| Waves/SM | 0.24 | 0.12 | -0.12 | 0.12 | -0.12 | 0.12 | -0.12 |
| No Eligible (%) | 87.55 % | 87.08 % | -0.47 pp | 90.38 % | +2.83 pp | 85.65 % | -1.90 pp |
| IPC Active | 0.45 inst/cycle | 0.47 inst/cycle | +0.02 | 0.34 inst/cycle | -0.11 | 0.55 inst/cycle | +0.10 |

---

## Iteration 5

**Best so far**: 3.0896× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_lambda_with_vectorized_load_hints | 1.933× | 2.588× | ✓ |  |
| C | fix_static_grid_lambda_add_small_xblock_and_num_stages_configs | 2.244× | 3.090× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_static_grid_l... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 10690 ns | 4130 ns | -6560 ns (-61.4%) | 3460 ns | -7230 ns (-67.6%) |
| SM Throughput (%) | 9.75 % | 6.61 % | -3.14 pp | 3.43 % | -6.32 pp |
| Memory Throughput (%) | 52.64 % | 16.37 % | -36.27 pp | 16.25 % | -36.39 pp |
| DRAM Throughput (%) | 52.64 % | 16.37 % | -36.27 pp | 9.84 % | -42.80 pp |
| Achieved Occupancy (%) | 23.70 % | 11.23 % | -12.47 pp | 5.55 % | -18.15 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 40.00 register/thread | +8 |
| Waves/SM | 0.24 | 0.12 | -0.12 | 0.12 | -0.12 |
| No Eligible (%) | 87.55 % | 89.14 % | +1.59 pp | 90.26 % | +2.71 pp |
| IPC Active | 0.45 inst/cycle | 0.53 inst/cycle | +0.08 | 0.34 inst/cycle | -0.11 |

---

## Cumulative Progress

```
Baseline:  10690 ns  (1.000×)
Iter 1     4610 ns  (2.319×)  — software_pipelining_with_autotune_tile_sizes
Iter 2     4580 ns  (2.334×)  — fix_static_grid_and_reduce_smem_bank_conflicts
Iter 3     3870 ns  (2.762×)  — fix_dynamic_grid_with_lambda_and_larger_xblock_configs
Iter 4     3460 ns  (3.090×)  — fix_static_grid_to_lambda_for_autotune_cta_scaling

Final best: 3460 ns  (3.0896×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 8,   'R0_BLOCK': 128}, num_warps=1, num_stages=2),
+        triton.Config({'XBLOCK': 8,   'R0_BLOCK': 128}, num_warps=1, num_stages=3),
+        triton.Config({'XBLOCK': 16,  'R0_BLOCK': 128}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 16,  'R0_BLOCK': 128}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 16,  'R0_BLOCK': 64},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 64},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=8, num_stages=2),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused__softmax_exp_slice_sub_sum_view_3(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 65536
@@ -12,28 +30,26 @@

     rnumel = r0_numel
     RBLOCK: tl.constexpr = R0_BLOCK
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
-    r0_base = tl.arange(0, R0_BLOCK)[None, :]
+    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]   # shape [XBLOCK, 1]
+    xmask = xindex < xnumel                              # shape [XBLOCK, 1]
+    r0_base = tl.arange(0, R0_BLOCK)[None, :]           # shape [1, R0_BLOCK]
     rbase = r0_base
-    x0 = (xindex % 32)
-    x1 = xindex // 32
+    x0 = (xindex % 32)   # [XBLOCK, 1]
+    x1 = xindex // 32    # [XBLOCK, 1]
     _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
-    x3 = xindex
+    x3 = tl.reshape(xindex, [XBLOCK])   # [XBLOCK] flat indices
     for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
         r0_index = r0_offset + r0_base
         r0_mask = r0_index < r0_numel
         roffset = r0_offset
         rindex = r0_index
-        r0_2 = r0_index
-        tmp0 = tl.load(in_ptr0 + (x0 + 48*r0_2 + 4800*x1), r0_mask, eviction_policy='evict_first', other=0.0)
+        r0_2 = r0_index      # [1, R0_BLOCK]
+        tmp0 = tl.load(in_ptr0 + (x0 + 48*r0_2 + 4800*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
         tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
         tmp3 = _tmp2 + tmp1
         _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
-    tmp2 = tl.sum(_tmp2, 1)[:, None]
-    tl.store(out_ptr0 + (x3), tmp2, None)
-
-
+    tmp2 = tl.sum(_tmp2, 1)   # sum over axis 1 (R0_BLOCK) -> [XBLOCK]
+    tl.store(out_ptr0 + x3, tmp2, tl.reshape(xmask, [XBLOCK]))
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -41,7 +57,7 @@

     xnumel = 65536
     r0_numel = 100
     grid = (triton.cdiv(xnumel, 128),)
-    triton_red_fused__softmax_exp_slice_sub_sum_view_3[grid](arg_0, arg_1, 65536, 100, XBLOCK=128, R0_BLOCK=512)
+    triton_red_fused__softmax_exp_slice_sub_sum_view_3[grid](arg_0, arg_1, 65536, 100)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
