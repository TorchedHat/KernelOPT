# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_smooth_l1_loss_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3940 ns — **Best latency**: 3740 ns (**1.0535×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`increase_r0_block_and_autotune_reduction_tile`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_smooth_l1_loss_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_096/kernel_1_triton_red_fused_smooth_l1_los/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1,1,1), Waves/SM: 0.00
- **Registers/thread**: 135
- **Duration**: 3940 ns
- **SM Throughput**: 0.17 % | **Memory**: 0.41 % | **DRAM**: 0.22 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.47 inst/cycle | **No Eligible**: 86.58 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_r0_block_and_autotune_reduction_tile | 1.192× | 1.054× | ✓ | **✓ best** |
| B | add_autotune_reduce_register_pressure_increase_parallelism | 1.068× | 1.152× | ✓ |  |
| C | vectorized_loads_and_autotune_for_underutilized_reduction | 1.085× | 1.101× | ✓ |  |
| D | autotune_tile_sizes_and_expand_grid_for_underutilized_reduction | 1.089× | 0.985× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_r0_block... | Δ | add_autotune_redu... | Δ | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3940 ns | 3740 ns | -200 ns (-5.1%) | 3420 ns | -520 ns (-13.2%) | 3580 ns | -360 ns (-9.1%) | 4000 ns | +60 ns (+1.5%) |
| SM Throughput (%) | 0.17 % | 0.03 % | -0.14 pp | 0.03 % | -0.14 pp | 0.03 % | -0.14 pp | 0.04 % | -0.13 pp |
| Memory Throughput (%) | 0.41 % | 0.39 % | -0.02 pp | 0.55 % | +0.14 pp | 0.47 % | +0.06 pp | 0.38 % | -0.03 pp |
| DRAM Throughput (%) | 0.22 % | 0.22 % | +0.00 pp | 0.23 % | +0.01 pp | 0.22 % | +0.00 pp | 0.20 % | -0.02 pp |
| Achieved Occupancy (%) | 6.24 % | 6.09 % | -0.15 pp | 6.08 % | -0.16 pp | 6.08 % | -0.16 pp | 6.10 % | -0.14 pp |
| Registers/Thread | 135.00 register/thread | 32.00 register/thread | -103 | 32.00 register/thread | -103 | 32.00 register/thread | -103 | 32.00 register/thread | -103 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 86.58 % | 93.71 % | +7.13 pp | 92.75 % | +6.17 pp | 93.88 % | +7.30 pp | 91.80 % | +5.22 pp |
| IPC Active | 0.47 inst/cycle | 0.25 inst/cycle | -0.22 | 0.24 inst/cycle | -0.23 | 0.24 inst/cycle | -0.23 | 0.31 inst/cycle | -0.16 |

---

## Iteration 2

**Best so far**: 1.0535× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | collapse_2d_accumulator_to_1d_and_fix_grid_lambda | 1.094× | 1.119× | ✓ |  |
| B | add_num_stages_pipelining_and_fix_grid_lambda_for_l1tex_latency_hiding | 1.151× | 1.139× | ✓ |  |
| C | collapse_2d_accumulator_to_1d_and_scalar_store_for_scalar_reduction | 1.065× | 1.139× | ✓ |  |
| D | collapse_2d_accumulator_to_1d_scalar_reduction_with_autotune | 1.032× | 1.119× | ✓ |  |


### Metric Comparison

| Metric | Baseline | collapse_2d_accum... | Δ | add_num_stages_pi... | Δ | collapse_2d_accum... | Δ | collapse_2d_accum... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3940 ns | 3520 ns | -420 ns (-10.7%) | 3460 ns | -480 ns (-12.2%) | 3460 ns | -480 ns (-12.2%) | 3520 ns | -420 ns (-10.7%) |
| SM Throughput (%) | 0.17 % | 0.03 % | -0.14 pp | 0.03 % | -0.14 pp | 0.03 % | -0.14 pp | 0.03 % | -0.14 pp |
| Memory Throughput (%) | 0.41 % | 0.46 % | +0.05 pp | 0.47 % | +0.06 pp | 0.48 % | +0.07 pp | 0.42 % | +0.01 pp |
| DRAM Throughput (%) | 0.22 % | 0.23 % | +0.01 pp | 0.33 % | +0.11 pp | 0.23 % | +0.01 pp | 0.23 % | +0.01 pp |
| Achieved Occupancy (%) | 6.24 % | 6.08 % | -0.16 pp | 6.08 % | -0.16 pp | 6.06 % | -0.18 pp | 6.08 % | -0.16 pp |
| Registers/Thread | 135.00 register/thread | 32.00 register/thread | -103 | 32.00 register/thread | -103 | 32.00 register/thread | -103 | 32.00 register/thread | -103 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 86.58 % | 93.57 % | +6.99 pp | 92.93 % | +6.35 pp | 92.81 % | +6.23 pp | 94.05 % | +7.47 pp |
| IPC Active | 0.47 inst/cycle | 0.25 inst/cycle | -0.22 | 0.25 inst/cycle | -0.22 | 0.27 inst/cycle | -0.20 | 0.25 inst/cycle | -0.22 |

---

## Iteration 3

**Best so far**: 1.0535× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | two_pass_split_reduction_to_utilize_multiple_sms | 1.081× | 1.088× | ✓ |  |
| C | split_reduction_across_multiple_blocks_with_atomic_add | 1.168× | 1.220× | ✓ |  |


### Metric Comparison

| Metric | Baseline | two_pass_split_re... | Δ | split_reduction_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3940 ns | 3620 ns | -320 ns (-8.1%) | 3230 ns | -710 ns (-18.0%) |
| SM Throughput (%) | 0.17 % | 0.02 % | -0.15 pp | 0.01 % | -0.16 pp |
| Memory Throughput (%) | 0.41 % | 0.34 % | -0.07 pp | 0.31 % | -0.10 pp |
| DRAM Throughput (%) | 0.22 % | 0.05 % | -0.17 pp | 0.03 % | -0.19 pp |
| Achieved Occupancy (%) | 6.24 % | 4.42 % | -1.82 pp | 4.28 % | -1.96 pp |
| Registers/Thread | 135.00 register/thread | 31.00 register/thread | -104 | 24.00 register/thread | -111 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 86.58 % | 93.29 % | +6.71 pp | 95.48 % | +8.90 pp |
| IPC Active | 0.47 inst/cycle | 0.17 inst/cycle | -0.30 | 0.12 inst/cycle | -0.35 |


### Key Learnings

#### ✅ Plan C: `split_reduction_across_multiple_blocks_with_atomic_add` (1.220×)

When a Triton reduction kernel operates on a single block (grid_size=1) over a moderate-sized reduction dimension (e.g., 8192 elements) on a many-SM GPU, the LaunchConfiguration bottleneck leaves nearly all SMs idle. Repurposing the XBLOCK autotune parameter as NUM_BLOCKS to partition the reduction dimension across multiple blocks, with each block computing a partial sum and accumulating via tl.atomic_add into the scalar output, reduces elapsed cycles (5122→4824) and executed instructions (793→302) by spreading work across SMs. The L1/TEX hit rate improved dramatically (0%→47%) and L2 hit rate improved (38.81%→58.59%) because each block now accesses a smaller, more cache-friendly slice of the input. The output tensor must be zeroed before launch since tl.atomic_add accumulates rather than overwrites.

> **Applicable when**: Triton reduction kernel with grid_size=1 on a GPU with many SMs (waves_per_sm=0.0, achieved_occupancy<10%), LaunchConfiguration NCU rule triggered with high speedup_pct (>90%), moderate reduction dimension (r0_numel in range 1K–64K), all other single-block optimizations (XBLOCK=1, 1D accumulator, evict_last, num_stages) already applied and the kernel remains bottlenecked by under-utilization of SMs.

> **Avoid when**: The reduction dimension is very small (r0_numel < 512) where block launch overhead dominates; the output is not a scalar (xnumel > 1) requiring a different parallelism strategy; tl.atomic_add on float32 is not supported or causes correctness issues due to aliasing; the kernel is already memory-bandwidth-bound across multiple blocks; the output tensor cannot be safely zeroed before launch.

---

## Cumulative Progress

```
Baseline:  3940 ns  (1.000×)
Iter 1     3740 ns  (1.054×)  — increase_r0_block_and_autotune_reduction_tile

Final best: 3740 ns  (1.0535×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,35 +5,36 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'R0_BLOCK': 8192, 'XBLOCK': 1}, num_warps=4, num_stages=1),
+        triton.Config({'R0_BLOCK': 8192, 'XBLOCK': 1}, num_warps=8, num_stages=1),
+        triton.Config({'R0_BLOCK': 4096, 'XBLOCK': 1}, num_warps=4, num_stages=2),
+        triton.Config({'R0_BLOCK': 4096, 'XBLOCK': 1}, num_warps=8, num_stages=2),
+        triton.Config({'R0_BLOCK': 2048, 'XBLOCK': 1}, num_warps=4, num_stages=2),
+        triton.Config({'R0_BLOCK': 2048, 'XBLOCK': 1}, num_warps=8, num_stages=2),
+        triton.Config({'R0_BLOCK': 1024, 'XBLOCK': 1}, num_warps=4, num_stages=4),
+        triton.Config({'R0_BLOCK': 1024, 'XBLOCK': 1}, num_warps=8, num_stages=4),
+        triton.Config({'R0_BLOCK': 512,  'XBLOCK': 1}, num_warps=2, num_stages=4),
+        triton.Config({'R0_BLOCK': 512,  'XBLOCK': 1}, num_warps=4, num_stages=4),
+    ],
+    key=['r0_numel'],
+)
 @triton.jit
-def triton_red_fused_smooth_l1_loss_1(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
-    xnumel = 1
+def triton_red_fused_smooth_l1_loss_1(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
     r0_numel = 8192
-    rnumel = r0_numel
-    RBLOCK: tl.constexpr = R0_BLOCK
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
-    r0_base = tl.arange(0, R0_BLOCK)[None, :]
-    rbase = r0_base
-    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
+    r0_base = tl.arange(0, R0_BLOCK)
+    _acc = tl.zeros([R0_BLOCK], dtype=tl.float32)
     for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
         r0_index = r0_offset + r0_base
         r0_mask = r0_index < r0_numel
-        roffset = r0_offset
-        rindex = r0_index
-        r0_0 = r0_index
-        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
-        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
-        tmp3 = _tmp2 + tmp1
-        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
-    tmp2 = tl.sum(_tmp2, 1)[:, None]
-    tmp4 = tl.full([1, 1], 1073741824.0, tl.float32)
-    tmp5 = (tmp2 / tmp4)
+        tmp0 = tl.load(in_ptr0 + r0_index, mask=r0_mask, eviction_policy='evict_last', other=0.0)
+        _acc = tl.where(r0_mask, _acc + tmp0, _acc)
+    result = tl.sum(_acc, axis=0)
+    result = result / 1073741824.0
     tl.debug_barrier()
-    tl.store(in_out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp5, None)
-
-
+    tl.store(in_out_ptr0, result)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -41,7 +42,7 @@

     xnumel = 1
     r0_numel = 8192
     grid = (triton.cdiv(xnumel, 128),)
-    triton_red_fused_smooth_l1_loss_1[grid](arg_0, arg_1, 1, 8192, XBLOCK=128, R0_BLOCK=512)
+    triton_red_fused_smooth_l1_loss_1[grid](arg_0, arg_1, 1, 8192)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
