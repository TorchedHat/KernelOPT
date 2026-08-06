# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused_native_group_norm_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7420 ns — **Best latency**: 7550 ns (**0.9828×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused_native_group_norm_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_075/kernel_1_triton_per_fused_native_group_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2048,1,1), Waves/SM: 0.97
- **Registers/thread**: 32
- **Duration**: 7420 ns
- **SM Throughput**: 21.51 % | **Memory**: 46.26 % | **DRAM**: 46.26 %
- **Achieved Occupancy**: 68.29 %
- **IPC Active**: 1.22 inst/cycle | **No Eligible**: 68.73 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | memory_coalescing_transpose_tile_layout | 1.015× | 0.995× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 7420 ns | 7460 ns | +40 ns (+0.5%) |
| SM Throughput (%) | 21.51 % | 21.33 % | -0.18 pp |
| Memory Throughput (%) | 46.26 % | 46.16 % | -0.10 pp |
| DRAM Throughput (%) | 46.26 % | 46.16 % | -0.10 pp |
| Achieved Occupancy (%) | 68.29 % | 67.94 % | -0.35 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 0.97 | 0.97 | +0.00 |
| No Eligible (%) | 68.73 % | 69.22 % | +0.49 pp |
| IPC Active | 1.22 inst/cycle | 1.22 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | memory_coalescing_swap_reduction_and_batch_axes | 1.015× | 0.995× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 7420 ns | 7460 ns | +40 ns (+0.5%) |
| SM Throughput (%) | 21.51 % | 21.17 % | -0.34 pp |
| Memory Throughput (%) | 46.26 % | 45.97 % | -0.29 pp |
| DRAM Throughput (%) | 46.26 % | 45.97 % | -0.29 pp |
| Achieved Occupancy (%) | 68.29 % | 68.27 % | -0.02 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 0.97 | 0.97 | +0.00 |
| No Eligible (%) | 68.73 % | 68.85 % | +0.12 pp |
| IPC Active | 1.22 inst/cycle | 1.22 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_correct_1d_hints | 0.996× | 0.987× | ✓ |  |
| C | software_pipelining_num_stages_without_stride_change | 1.026× | 1.000× | ✓ | **✓ best** |
| D | increase_xblock_only_for_better_sm_utilization | 0.994× | 0.991× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | software_pipelini... | Δ | increase_xblock_o... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7420 ns | 7520 ns | +100 ns (+1.3%) | 7420 ns | +0 ns (+0.0%) | 7490 ns | +70 ns (+0.9%) |
| SM Throughput (%) | 21.51 % | 21.08 % | -0.43 pp | 21.18 % | -0.33 pp | 21.40 % | -0.11 pp |
| Memory Throughput (%) | 46.26 % | 45.65 % | -0.61 pp | 46.17 % | -0.09 pp | 45.88 % | -0.38 pp |
| DRAM Throughput (%) | 46.26 % | 45.65 % | -0.61 pp | 46.17 % | -0.09 pp | 45.88 % | -0.38 pp |
| Achieved Occupancy (%) | 68.29 % | 67.97 % | -0.32 pp | 68.17 % | -0.12 pp | 68.39 % | +0.10 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 68.73 % | 68.89 % | +0.16 pp | 68.88 % | +0.15 pp | 68.75 % | +0.02 pp |
| IPC Active | 1.22 inst/cycle | 1.22 inst/cycle | +0.00 | 1.20 inst/cycle | -0.02 | 1.17 inst/cycle | -0.05 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_r0block_stride_normalization_with_coalesced_xblock_autotune | 1.028× | 0.995× | ✓ |  |
| D | expand_autotune_configs_larger_xblock_and_num_warps_for_sm_utilization | 1.050× | 0.983× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_r0block_strid... | Δ | expand_autotune_c... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7420 ns | 7460 ns | +40 ns (+0.5%) | 7550 ns | +130 ns (+1.8%) |
| SM Throughput (%) | 21.51 % | 21.21 % | -0.30 pp | 21.07 % | -0.44 pp |
| Memory Throughput (%) | 46.26 % | 46.11 % | -0.15 pp | 45.41 % | -0.85 pp |
| DRAM Throughput (%) | 46.26 % | 46.11 % | -0.15 pp | 45.41 % | -0.85 pp |
| Achieved Occupancy (%) | 68.29 % | 68.03 % | -0.26 pp | 68.17 % | -0.12 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 68.73 % | 68.97 % | +0.24 pp | 69.01 % | +0.28 pp |
| IPC Active | 1.22 inst/cycle | 1.22 inst/cycle | +0.00 | 1.19 inst/cycle | -0.03 |

---

## Cumulative Progress

```
Baseline:  7420 ns  (1.000×)

Final best: 7550 ns  (0.9828×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,35 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # Original XBLOCK=128 configs (preserved)
+        triton.Config({'XBLOCK': 128}, num_stages=1, num_warps=4),
+        triton.Config({'XBLOCK': 128}, num_stages=2, num_warps=4),
+        triton.Config({'XBLOCK': 128}, num_stages=3, num_warps=4),
+        triton.Config({'XBLOCK': 128}, num_stages=4, num_warps=4),
+        triton.Config({'XBLOCK': 128}, num_stages=2, num_warps=8),
+        triton.Config({'XBLOCK': 128}, num_stages=3, num_warps=8),
+        triton.Config({'XBLOCK': 128}, num_stages=4, num_warps=8),
+        # New: num_warps=16 for XBLOCK=128 — more in-flight requests hide L1TEX latency
+        triton.Config({'XBLOCK': 128}, num_stages=2, num_warps=16),
+        triton.Config({'XBLOCK': 128}, num_stages=3, num_warps=16),
+        # New: XBLOCK=256 — larger tiles, better L2 reuse, grid=1024 CTAs
+        triton.Config({'XBLOCK': 256}, num_stages=1, num_warps=4),
+        triton.Config({'XBLOCK': 256}, num_stages=2, num_warps=4),
+        triton.Config({'XBLOCK': 256}, num_stages=2, num_warps=8),
+        triton.Config({'XBLOCK': 256}, num_stages=3, num_warps=8),
+        triton.Config({'XBLOCK': 256}, num_stages=2, num_warps=16),
+        triton.Config({'XBLOCK': 256}, num_stages=3, num_warps=16),
+        # New: XBLOCK=512 — maximum tile size, grid=512 CTAs, best L2 reuse
+        triton.Config({'XBLOCK': 512}, num_stages=1, num_warps=8),
+        triton.Config({'XBLOCK': 512}, num_stages=2, num_warps=8),
+        triton.Config({'XBLOCK': 512}, num_stages=2, num_warps=16),
+        triton.Config({'XBLOCK': 512}, num_stages=3, num_warps=16),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_per_fused_native_group_norm_1(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 262144
@@ -13,27 +42,22 @@

     rnumel = r0_numel
     RBLOCK: tl.constexpr = R0_BLOCK
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
-    xmask = xindex < xnumel
-    r0_index = tl.arange(0, R0_BLOCK)[None, :]
-    r0_offset = 0
-    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
-    roffset = r0_offset
-    rindex = r0_index
+    x0 = xoffset + tl.arange(0, XBLOCK)[:, None]      # shape [XBLOCK, 1]
+    r0_index = tl.arange(0, R0_BLOCK)[None, :]         # shape [1, R0_BLOCK]
+    xmask = x0 < xnumel
     r0_1 = r0_index
-    x0 = xindex
     tmp0 = tl.load(in_ptr0 + (r0_1 + 16*x0), xmask, other=0.0, eviction_policy='evict_first')
+    # tmp0 shape: [XBLOCK, R0_BLOCK] — r0 varies fastest (stride-1), coalesced!
     tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
     tmp3 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
-    tmp5 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
+    tmp5 = tl.sum(tmp3, 1)[:, None].to(tl.float32)    # sum over axis=1 (R0_BLOCK)
     tmp8 = tmp5 * (1.0 / 16.0)
     tmp9 = tmp1 - tmp8
     tmp10 = tmp9 * tmp9
     tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
-    tmp13 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
+    tmp13 = tl.sum(tmp11, 1)[:, None].to(tl.float32)  # sum over axis=1 (R0_BLOCK)
     tl.store(out_ptr0 + (x0), tmp8, xmask, eviction_policy='evict_last')
     tl.store(out_ptr1 + (x0), tmp13, xmask, eviction_policy='evict_last')
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -42,7 +66,7 @@

     xnumel = 262144
     r0_numel = 32
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused_native_group_norm_1[grid](arg_0, arg_1, arg_2, 262144, 32, XBLOCK=128)
+    triton_per_fused_native_group_norm_1[grid](arg_0, arg_1, arg_2, 262144, 32)
     return arg_2
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
