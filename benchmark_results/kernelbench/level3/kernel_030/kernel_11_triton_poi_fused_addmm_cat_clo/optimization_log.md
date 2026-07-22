# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_cat_clone_permute_select_view_zeros_like_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 17310 ns — **Best latency**: 16960 ns (**1.0206×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`eliminate_dead_conditional_branches_in_index_arithmetic`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_addmm_cat_clone_permute_select_view_zeros_like_11`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_030/kernel_11_triton_poi_fused_addmm_cat_clo/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (23520,1,1), Waves/SM: 11.14
- **Registers/thread**: 16
- **Duration**: 17310 ns
- **SM Throughput**: 36.81 % | **Memory**: 17.82 % | **DRAM**: 14.27 %
- **Achieved Occupancy**: 42.46 %
- **IPC Active**: 1.73 inst/cycle | **No Eligible**: 56.43 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_for_better_occupancy_and_latency_hiding | 1.015× | 1.021× | ✓ | **✓ best** |
| C | autotune_xblock_to_hide_l1tex_latency | 1.005× | 1.017× | ✓ |  |
| D | increase_xblock_for_vectorized_loads_and_better_l1_reuse | 0.993× | 0.458× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_f... | Δ | autotune_xblock_t... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17310 ns | 16960 ns | -350 ns (-2.0%) | 17020 ns | -290 ns (-1.7%) | 37760 ns | +20450 ns (+118.1%) |
| SM Throughput (%) | 36.81 % | 38.75 % | +1.94 pp | 39.51 % | +2.70 pp | 69.92 % | +33.11 pp |
| Memory Throughput (%) | 17.82 % | 18.18 % | +0.36 pp | 18.12 % | +0.30 pp | 19.27 % | +1.45 pp |
| DRAM Throughput (%) | 14.27 % | 14.58 % | +0.31 pp | 14.53 % | +0.26 pp | 6.55 % | -7.72 pp |
| Achieved Occupancy (%) | 42.46 % | 42.28 % | -0.18 pp | 42.83 % | +0.37 pp | 86.29 % | +43.83 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 28.00 register/thread | +12 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 22.27 | +11.13 |
| No Eligible (%) | 56.43 % | 54.70 % | -1.73 pp | 53.08 % | -3.35 pp | 30.31 % | -26.12 pp |
| IPC Active | 1.73 inst/cycle | 1.81 inst/cycle | +0.08 | 1.86 inst/cycle | +0.13 | 2.79 inst/cycle | +1.06 |

---

## Iteration 2

**Best so far**: 1.0206× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.046× | 0.039× | ✓ |  |
| B | persistent_tile_scheduling | 1.008× | 1.022× | ✓ |  |
| C | eliminate_dead_conditional_branches_in_index_arithmetic | 1.007× | 1.021× | ✓ |  |
| D | add_triton_autotune | 1.014× | 1.011× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | eliminate_dead_co... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17310 ns | 440350 ns | +423040 ns (+2443.9%) | 16930 ns | -380 ns (-2.2%) | 16960 ns | -350 ns (-2.0%) | 17120 ns | -190 ns (-1.1%) |
| SM Throughput (%) | 36.81 % | 82.94 % | +46.13 pp | 40.97 % | +4.16 pp | 36.71 % | -0.10 pp | 38.74 % | +1.93 pp |
| Memory Throughput (%) | 17.82 % | 56.44 % | +38.62 pp | 18.24 % | +0.42 pp | 18.22 % | +0.40 pp | 18.00 % | +0.18 pp |
| DRAM Throughput (%) | 14.27 % | 0.66 % | -13.61 pp | 14.62 % | +0.35 pp | 14.59 % | +0.32 pp | 14.45 % | +0.18 pp |
| Achieved Occupancy (%) | 42.46 % | 98.67 % | +56.21 pp | 43.99 % | +1.53 pp | 41.74 % | -0.72 pp | 42.30 % | -0.16 pp |
| Registers/Thread | 16.00 register/thread | 19.00 register/thread | +3 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 56.43 % | 19.85 % | -36.58 pp | 51.79 % | -4.64 pp | 57.01 % | +0.58 pp | 54.63 % | -1.80 pp |
| IPC Active | 1.73 inst/cycle | 3.20 inst/cycle | +1.47 | 1.89 inst/cycle | +0.16 | 1.70 inst/cycle | -0.03 | 1.80 inst/cycle | +0.07 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.039×)

Converting a flat one-CTA-per-tile Triton kernel (grid=23520) to a persistent kernel (grid=132, while-loop over tiles) caused a catastrophic 25x slowdown (16.96 us → 440.35 us). Although occupancy improved dramatically (42% → 99%) and L2 hit rate rose from 52% to 99% (confirming bias vector caching worked), executed instructions exploded from 5.17M to 277M — a 53x increase. The while-loop in Triton generates a massive instruction overhead per iteration (branch evaluation, tile_id arithmetic, loop-carried dependencies), and the Triton compiler appears unable to optimize the loop body as efficiently as the flat kernel. The net effect is that the compute pipeline became the bottleneck (83% SM throughput) but with far more total work, making the kernel 25x slower despite better cache behavior. The grid size in the launch call was also not updated to 132 (still used cdiv(xnumel, 128)=23520), meaning 23520 CTAs each ran the full while-loop — processing every tile 23520/132 ≈ 178 times redundantly.

> **Applicable when**: Never for this kernel pattern. Persistent while-loop kernels in Triton are not beneficial when: (1) the grid is not actually reduced (launch bug), or (2) the Triton compiler cannot efficiently lower the while-loop, causing instruction count explosion. Even if the grid were correctly set to 132, the 53x instruction increase suggests the loop overhead dominates.

> **Avoid when**: Triton kernels where the launch grid is not simultaneously reduced to match the persistent CTA count — the while-loop will cause each CTA to redundantly process all tiles, multiplying instruction count by num_tiles. Also avoid when the kernel body is simple enough that the flat launch already achieves near-optimal throughput; the while-loop overhead will dominate. Do not use persistent kernels in Triton without verifying the grid lambda is updated to (NUM_SMS,) at the call site.

#### ❌ Plan B: `persistent_tile_scheduling` (0.039×)

Converting a flat one-CTA-per-tile Triton kernel (grid=23520) to a persistent kernel (grid=132, while-loop over tiles) caused a catastrophic 25x slowdown (16.96 us → 440.35 us). Although occupancy improved dramatically (42% → 99%) and L2 hit rate rose from 52% to 99% (confirming bias vector caching worked), executed instructions exploded from 5.17M to 277M — a 53x increase. The while-loop in Triton generates a massive instruction overhead per iteration (branch evaluation, tile_id arithmetic, loop-carried dependencies), and the Triton compiler appears unable to optimize the loop body as efficiently as the flat kernel. The net effect is that the compute pipeline became the bottleneck (83% SM throughput) but with far more total work, making the kernel 25x slower despite better cache behavior. The grid size in the launch call was also not updated to 132 (still used cdiv(xnumel, 128)=23520), meaning 23520 CTAs each ran the full while-loop — processing every tile 23520/132 ≈ 178 times redundantly.

> **Applicable when**: Never for this kernel pattern. Persistent while-loop kernels in Triton are not beneficial when: (1) the grid is not actually reduced (launch bug), or (2) the Triton compiler cannot efficiently lower the while-loop, causing instruction count explosion. Even if the grid were correctly set to 132, the 53x instruction increase suggests the loop overhead dominates.

> **Avoid when**: Triton kernels where the launch grid is not simultaneously reduced to match the persistent CTA count — the while-loop will cause each CTA to redundantly process all tiles, multiplying instruction count by num_tiles. Also avoid when the kernel body is simple enough that the flat launch already achieves near-optimal throughput; the while-loop overhead will dominate. Do not use persistent kernels in Triton without verifying the grid lambda is updated to (NUM_SMS,) at the call site.

---

## Iteration 3

**Best so far**: 1.0206× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  17310 ns  (1.000×)
Iter 1     16960 ns  (1.021×)  — autotune_xblock_for_better_occupancy_and_latency_hiding

Final best: 16960 ns  (1.0206×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,35 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
+    triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=3),
+    triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=1),
+    triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=3),
+    triton.Config({'XBLOCK': 512}, num_warps=4, num_stages=1),
+    triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=3),
+    triton.Config({'XBLOCK': 1024}, num_warps=4, num_stages=1),
+    triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=3),
+], key=['xnumel'])
 @triton.jit
 def triton_poi_fused_addmm_cat_clone_permute_select_view_zeros_like_11(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 3010560
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 32)
     x2 = ((xindex // 1568) % 3)
     x1 = ((xindex // 32) % 49)
     x3 = xindex // 4704
     x4 = xindex
-    tmp19 = tl.load(in_ptr2 + (192 + x0 + 32*x2 + 288*x1 + 14112*x3), None)
+    tmp19 = tl.load(in_ptr2 + (192 + x0 + 32*x2 + 288*x1 + 14112*x3), xmask)
     tmp0 = 192 + x0 + 32*x2
     tmp1 = tl.full([1], 0, tl.int64)
     tmp2 = tmp0 >= tmp1
     tmp3 = tl.full([1], 96, tl.int64)
     tmp4 = tmp0 < tmp3
-    tmp5 = tl.load(in_ptr0 + (192 + x0 + 32*x2), tmp4, eviction_policy='evict_last', other=0.0)
+    tmp5 = tl.load(in_ptr0 + (192 + x0 + 32*x2), mask=xmask, eviction_policy='evict_last', other=0.0)
     tmp6 = tmp0 >= tmp3
     tmp7 = tl.full([1], 192, tl.int64)
     tmp8 = tmp0 < tmp7
@@ -33,13 +44,11 @@

     tmp13 = tmp0 >= tmp7
     tmp14 = tl.full([1], 288, tl.int64)
     tmp15 = tmp0 < tmp14
-    tmp16 = tl.load(in_ptr1 + (x0 + 32*x2), tmp13, eviction_policy='evict_last', other=0.0)
+    tmp16 = tl.load(in_ptr1 + (x0 + 32*x2), mask=xmask, eviction_policy='evict_last', other=0.0)
     tmp17 = tl.where(tmp9, tmp12, tmp16)
     tmp18 = tl.where(tmp4, tmp5, tmp17)
     tmp20 = tmp18 + tmp19
-    tl.store(out_ptr0 + (x4), tmp20, None)
-
-
+    tl.store(out_ptr0 + (x4), tmp20, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -48,7 +57,7 @@

     arg_3 = inputs[3]
     xnumel = 3010560
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_addmm_cat_clone_permute_select_view_zeros_like_11[grid](arg_0, arg_1, arg_2, arg_3, 3010560, XBLOCK=128)
+    triton_poi_fused_addmm_cat_clone_permute_select_view_zeros_like_11[grid](arg_0, arg_1, arg_2, arg_3, 3010560)
     return arg_3
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
