# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_23` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 20580 ns — **Best latency**: 20580 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_23`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_23_triton_poi_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (18375,1,1), Waves/SM: 8.70
- **Registers/thread**: 24
- **Duration**: 20580 ns
- **SM Throughput**: 61.85 % | **Memory**: 13.21 % | **DRAM**: 6.04 %
- **Achieved Occupancy**: 84.43 %
- **IPC Active**: 2.79 inst/cycle | **No Eligible**: 30.17 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_tile_size_for_better_sm_utilization | 0.967× | 0.227× | ✓ |  |
| B | increase_xblock_tile_size_with_autotune | 0.990× | 0.444× | ✓ |  |
| C | autotune_xblock_reduce_scheduling_overhead | 0.985× | 0.446× | ✓ |  |
| D | increase_xblock_for_vectorized_loads_stores | 0.991× | 0.445× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_t... | Δ | increase_xblock_t... | Δ | autotune_xblock_r... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 20580 ns | 90820 ns | +70240 ns (+341.3%) | 46400 ns | +25820 ns (+125.5%) | 46180 ns | +25600 ns (+124.4%) | 46300 ns | +25720 ns (+125.0%) |
| SM Throughput (%) | 61.85 % | 74.28 % | +12.43 pp | 73.17 % | +11.32 pp | 73.11 % | +11.26 pp | 73.16 % | +11.31 pp |
| Memory Throughput (%) | 13.21 % | 20.95 % | +7.74 pp | 21.82 % | +8.61 pp | 21.81 % | +8.60 pp | 21.83 % | +8.62 pp |
| DRAM Throughput (%) | 6.04 % | 1.37 % | -4.67 pp | 2.67 % | -3.37 pp | 2.69 % | -3.35 pp | 2.68 % | -3.36 pp |
| Achieved Occupancy (%) | 84.43 % | 89.50 % | +5.07 pp | 89.55 % | +5.12 pp | 89.57 % | +5.14 pp | 89.53 % | +5.10 pp |
| Registers/Thread | 24.00 register/thread | 31.00 register/thread | +7 | 31.00 register/thread | +7 | 31.00 register/thread | +7 | 31.00 register/thread | +7 |
| Waves/SM | 8.70 | 34.80 | +26.10 | 17.40 | +8.70 | 17.40 | +8.70 | 17.40 | +8.70 |
| No Eligible (%) | 30.17 % | 23.76 % | -6.41 pp | 23.64 % | -6.53 pp | 23.65 % | -6.52 pp | 23.35 % | -6.82 pp |
| IPC Active | 2.79 inst/cycle | 3.05 inst/cycle | +0.26 | 3.06 inst/cycle | +0.27 | 3.06 inst/cycle | +0.27 | 3.06 inst/cycle | +0.27 |


### Key Learnings

#### ❌ Plan B: `increase_xblock_tile_size_with_autotune` (0.444×)

Adding @triton.autotune with larger XBLOCK configs (256, 512, 1024) caused the autotuner to select XBLOCK=256, which doubled registers per thread (24→31) and caused a 2.67x explosion in executed instructions (9.99M→26.64M), more than doubling kernel duration (20.58→46.4 µs). Despite slightly improved occupancy (84.4%→89.6%) and compute throughput (61.9%→73.2%), the massive instruction count increase dominated, resulting in 2.25x slower execution. The CPIStall fraction actually worsened (33.4%→43.9%), and waves per SM doubled (8.7→17.4) rather than decreasing, indicating the autotuner selected a suboptimal config that increased rather than decreased grid pressure.

> **Applicable when**: Never: this specific pattern (autotune with larger XBLOCK on a kernel with complex multi-dimensional index arithmetic) caused regression. Avoid when the kernel has heavy integer ALU index computation that scales with XBLOCK.

> **Avoid when**: Kernels with complex multi-dimensional index arithmetic (many integer mod/div ops per element) where increasing XBLOCK causes superlinear instruction count growth. When the baseline already has 8+ waves/SM and the bottleneck is ALU-bound integer indexing rather than memory latency. When register pressure increases significantly with larger block sizes (24→31 registers/thread), as this can reduce occupancy headroom and cause the autotuner to pick a config that increases total work.

#### ❌ Plan C: `autotune_xblock_reduce_scheduling_overhead` (0.446×)

Adding @triton.autotune to a pointwise kernel with a CPIStall 'not selected' bottleneck caused a 2.24x regression (20.58 µs → 46.18 µs). Despite the autotune selecting XBLOCK=256 (block_size doubled), executed instructions exploded from ~10M to ~26.6M — a 2.66x increase — indicating the autotune selected a configuration that dramatically increased per-thread work or unrolling overhead. Registers per thread also rose from 24 to 31, reducing the effective block limit from 21 to 8 blocks/SM, which doubled waves_per_SM from 8.7 to 17.4 and worsened scheduling pressure rather than alleviating it. The grid was not updated to use the lambda form, so the static grid=(cdiv(xnumel,128),) was passed to the autotuned kernel, causing a mismatch between the grid and the selected XBLOCK — the kernel ran with a fixed 128-element grid stride but a 256-element XBLOCK, causing each thread block to process twice the intended work and inflating instruction count.

> **Applicable when**: Never apply this specific pattern (static grid with autotuned XBLOCK). The implementation flaw is the root cause of regression.

> **Avoid when**: Do not use a static hardcoded grid size when adding @triton.autotune that varies XBLOCK. The grid must always be a lambda using meta['XBLOCK'] to match the selected block size. If the grid remains static at the old XBLOCK value, each block processes more elements than the grid covers, causing 2-8x instruction inflation and severe register pressure increase.

#### ❌ Plan D: `increase_xblock_for_vectorized_loads_stores` (0.445×)

Adding @triton.autotune with larger XBLOCK configs (256–1024) caused the autotuner to select XBLOCK=256, which increased registers per thread from 24 to 31. This reduced Block Limit Registers from 21 to 8 blocks per SM, cutting the effective number of concurrent blocks and forcing 2x more waves (8.7→17.4 waves/SM), more than doubling kernel duration (20.58→46.3 µs). The instruction count exploded from ~10M to ~26.6M, indicating the larger tile size introduced significant overhead (likely from predicated/masked instructions over a non-power-of-2 problem size) rather than the expected vectorization benefit.

> **Applicable when**: Never apply this specific pattern as implemented. The grid was not updated to use meta['XBLOCK'], so the autotuner selected a larger XBLOCK but the grid remained sized for XBLOCK=128, causing each thread to process more elements than the grid covers and massively inflating instruction count.

> **Avoid when**: Do not add @triton.autotune with larger XBLOCK configs when: (1) the grid launch is not simultaneously updated to lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),); (2) the kernel has complex non-stride-1 index expressions that increase register pressure with larger tiles; (3) registers per thread are already near the block limit boundary (e.g., 24 regs/thread with Block Limit Registers=21 — adding even a few registers drops to 8 blocks/SM). Also avoid when the problem size is not a clean multiple of the new XBLOCK, as masked tail handling inflates instruction count.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.984× | 0.994× | ✓ |  |
| B | persistent_tile_scheduling | 0.974× | 0.966× | ✓ |  |
| C | persistent_tile_scheduling | 0.025× | 0.022× | ✓ |  |
| D | persistent_tile_scheduling_with_fixed_grid | 0.943× | 0.945× | ✓ |  |
| E | add_triton_autotune | 0.982× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 20580 ns | 20700 ns | +120 ns (+0.6%) | 21310 ns | +730 ns (+3.5%) | 928830 ns | +908250 ns (+4413.3%) | 21790 ns | +1210 ns (+5.9%) | 20540 ns | -40 ns (-0.2%) |
| SM Throughput (%) | 61.85 % | 63.57 % | +1.72 pp | 64.61 % | +2.76 pp | 81.50 % | +19.65 pp | 63.62 % | +1.77 pp | 61.82 % | -0.03 pp |
| Memory Throughput (%) | 13.21 % | 13.25 % | +0.04 pp | 12.88 % | -0.33 pp | 17.33 % | +4.12 pp | 12.48 % | -0.73 pp | 13.21 % | +0.00 pp |
| DRAM Throughput (%) | 6.04 % | 5.99 % | -0.05 pp | 5.83 % | -0.21 pp | 0.17 % | -5.87 pp | 5.70 % | -0.34 pp | 6.04 % | +0.00 pp |
| Achieved Occupancy (%) | 84.43 % | 84.15 % | -0.28 pp | 84.76 % | +0.33 pp | 99.19 % | +14.76 pp | 84.91 % | +0.48 pp | 83.97 % | -0.46 pp |
| Registers/Thread | 24.00 register/thread | 28.00 register/thread | +4 | 28.00 register/thread | +4 | 28.00 register/thread | +4 | 28.00 register/thread | +4 | 24.00 register/thread | +0 |
| Waves/SM | 8.70 | 8.70 | +0.00 | 8.70 | +0.00 | 8.70 | +0.00 | 8.70 | +0.00 | 8.70 | +0.00 |
| No Eligible (%) | 30.17 % | 28.84 % | -1.33 pp | 27.38 % | -2.79 pp | 18.25 % | -11.92 pp | 28.97 % | -1.20 pp | 29.75 % | -0.42 pp |
| IPC Active | 2.79 inst/cycle | 2.83 inst/cycle | +0.04 | 2.86 inst/cycle | +0.07 | 3.27 inst/cycle | +0.48 | 2.83 inst/cycle | +0.04 | 2.77 inst/cycle | -0.02 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.022×)

Converting a standard grid-stride pointwise kernel to a persistent kernel with a Triton while-loop (132 CTAs each iterating ~139 tiles) caused a 45x regression (46.18 µs → 928.83 µs). Executed instructions exploded from ~26.6M to ~599.5M — a 22.5x increase — despite the same total work. The while-loop in Triton introduces massive per-iteration overhead: dynamic loop control, repeated index recomputation, and the inability for the compiler to statically unroll or optimize the loop body. Although L2 hit rate improved dramatically (64.6% → 98.5%) and occupancy reached 99.2%, the instruction count explosion completely dominated, making the kernel ~45x slower. The persistent kernel pattern is fundamentally incompatible with Triton's compilation model for kernels with complex multi-dimensional integer index arithmetic, where the compiler relies on static tile counts for optimization.

> **Applicable when**: Never: persistent while-loop kernels in Triton cause catastrophic instruction explosion for kernels with complex multi-dimensional index arithmetic. The pattern may theoretically work for simple memory-bound kernels with trivial index computation.

> **Avoid when**: Do not apply persistent while-loop kernels in Triton when: (1) the kernel body contains complex multi-dimensional integer index arithmetic (mod/div chains over 4+ dimensions); (2) the kernel is compute-bound or ALU-bound rather than purely memory-latency-bound; (3) the number of tiles per SM is large (~100+), as the while-loop overhead compounds multiplicatively. The 22.5x instruction explosion (26.6M → 599.5M) demonstrates that Triton's compiler cannot optimize a dynamic while-loop over tile indices the same way it handles static grid-stride patterns. Even though L2 hit rate improved to 98.5%, the instruction overhead completely dominated.

#### ❌ Plan B: `persistent_tile_scheduling` (0.022×)

Converting a standard grid-stride pointwise kernel to a persistent kernel with a Triton while-loop (132 CTAs each iterating ~139 tiles) caused a 45x regression (46.18 µs → 928.83 µs). Executed instructions exploded from ~26.6M to ~599.5M — a 22.5x increase — despite the same total work. The while-loop in Triton introduces massive per-iteration overhead: dynamic loop control, repeated index recomputation, and the inability for the compiler to statically unroll or optimize the loop body. Although L2 hit rate improved dramatically (64.6% → 98.5%) and occupancy reached 99.2%, the instruction count explosion completely dominated, making the kernel ~45x slower. The persistent kernel pattern is fundamentally incompatible with Triton's compilation model for kernels with complex multi-dimensional integer index arithmetic, where the compiler relies on static tile counts for optimization.

> **Applicable when**: Never: persistent while-loop kernels in Triton cause catastrophic instruction explosion for kernels with complex multi-dimensional index arithmetic. The pattern may theoretically work for simple memory-bound kernels with trivial index computation.

> **Avoid when**: Do not apply persistent while-loop kernels in Triton when: (1) the kernel body contains complex multi-dimensional integer index arithmetic (mod/div chains over 4+ dimensions); (2) the kernel is compute-bound or ALU-bound rather than purely memory-latency-bound; (3) the number of tiles per SM is large (~100+), as the while-loop overhead compounds multiplicatively. The 22.5x instruction explosion (26.6M → 599.5M) demonstrates that Triton's compiler cannot optimize a dynamic while-loop over tile indices the same way it handles static grid-stride patterns. Even though L2 hit rate improved to 98.5%, the instruction overhead completely dominated.

#### ❌ Plan C: `persistent_tile_scheduling` (0.022×)

Converting a standard grid-stride pointwise kernel to a persistent kernel with a Triton while-loop (132 CTAs each iterating ~139 tiles) caused a 45x regression (46.18 µs → 928.83 µs). Executed instructions exploded from ~26.6M to ~599.5M — a 22.5x increase — despite the same total work. The while-loop in Triton introduces massive per-iteration overhead: dynamic loop control, repeated index recomputation, and the inability for the compiler to statically unroll or optimize the loop body. Although L2 hit rate improved dramatically (64.6% → 98.5%) and occupancy reached 99.2%, the instruction count explosion completely dominated, making the kernel ~45x slower. The persistent kernel pattern is fundamentally incompatible with Triton's compilation model for kernels with complex multi-dimensional integer index arithmetic, where the compiler relies on static tile counts for optimization.

> **Applicable when**: Never: persistent while-loop kernels in Triton cause catastrophic instruction explosion for kernels with complex multi-dimensional index arithmetic. The pattern may theoretically work for simple memory-bound kernels with trivial index computation.

> **Avoid when**: Do not apply persistent while-loop kernels in Triton when: (1) the kernel body contains complex multi-dimensional integer index arithmetic (mod/div chains over 4+ dimensions); (2) the kernel is compute-bound or ALU-bound rather than purely memory-latency-bound; (3) the number of tiles per SM is large (~100+), as the while-loop overhead compounds multiplicatively. The 22.5x instruction explosion (26.6M → 599.5M) demonstrates that Triton's compiler cannot optimize a dynamic while-loop over tile indices the same way it handles static grid-stride patterns. Even though L2 hit rate improved to 98.5%, the instruction overhead completely dominated.

---

## Cumulative Progress

```
Baseline:  20580 ns  (1.000×)

Final best: 20580 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 512},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 256},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128},  num_warps=4, num_stages=2),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 2352000
@@ -26,25 +40,23 @@

     tmp8 = tmp2 & tmp4
     tmp9 = tmp8 & tmp6
     tmp10 = tmp9 & tmp7
-    tmp11 = tl.load(in_ptr0 + ((-22272) + x0 + 32*(x1 // 49) + 192*((((x1 % 49)) % 7)) + 1344*((x2 % 5)) + 5376*(((x1 % 49)) // 7) + 37632*(((x2 // 5) % 5)) + 150528*(x2 // 25)), tmp10 & xmask, other=0.0)
-    tmp12 = tl.load(in_ptr1 + ((-116) + 7*((x2 % 5)) + 28*(((x1 % 49)) // 7) + 196*(((x2 // 5) % 5)) + 784*(x2 // 25) + ((((x1 % 49)) % 7))), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp11 = tl.load(in_ptr0 + ((-22272) + x0 + 32*(x1 // 49) + 192*((((x1 % 49)) % 7)) + 1344*((x2 % 5)) + 5376*(((x1 % 49)) // 7) + 37632*(((x2 // 5) % 5)) + 150528*(x2 // 25)), mask=tmp10 & xmask, other=0.0)
+    tmp12 = tl.load(in_ptr1 + ((-116) + 7*((x2 % 5)) + 28*(((x1 % 49)) // 7) + 196*(((x2 // 5) % 5)) + 784*(x2 // 25) + ((((x1 % 49)) % 7))), mask=tmp10 & xmask, eviction_policy='evict_last', other=0.0)
     tmp13 = tmp11 - tmp12
-    tmp14 = tl.load(in_ptr2 + ((-116) + 7*((x2 % 5)) + 28*(((x1 % 49)) // 7) + 196*(((x2 // 5) % 5)) + 784*(x2 // 25) + ((((x1 % 49)) % 7))), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp14 = tl.load(in_ptr2 + ((-116) + 7*((x2 % 5)) + 28*(((x1 % 49)) // 7) + 196*(((x2 // 5) % 5)) + 784*(x2 // 25) + ((((x1 % 49)) % 7))), mask=tmp10 & xmask, eviction_policy='evict_last', other=0.0)
     tmp15 = tl.full([1], 192.0, tl.float32)
     tmp16 = (tmp14 / tmp15)
     tmp17 = tl.full([1], 1e-05, tl.float32)
     tmp18 = tmp16 + tmp17
     tmp19 = libdevice.rsqrt(tmp18)
     tmp20 = tmp13 * tmp19
-    tmp21 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp21 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), mask=tmp10 & xmask, eviction_policy='evict_last', other=0.0)
     tmp22 = tmp20 * tmp21
-    tmp23 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp23 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), mask=tmp10 & xmask, eviction_policy='evict_last', other=0.0)
     tmp24 = tmp22 + tmp23
     tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
     tmp26 = tl.where(tmp10, tmp24, tmp25)
-    tl.store(out_ptr0 + (x3), tmp26, xmask)
-
-
+    tl.store(out_ptr0 + (x3), tmp26, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -55,7 +67,7 @@

     arg_5 = inputs[5]
     xnumel = 2352000
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_23[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 2352000, XBLOCK=128)
+    triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_23[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 2352000)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
