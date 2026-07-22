# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_25` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 12160 ns — **Best latency**: 12160 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_25`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_25_triton_poi_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (11760,1,1), Waves/SM: 5.57
- **Registers/thread**: 18
- **Duration**: 12160 ns
- **SM Throughput**: 60.83 % | **Memory**: 30.30 % | **DRAM**: 30.30 %
- **Achieved Occupancy**: 80.68 %
- **IPC Active**: 2.98 inst/cycle | **No Eligible**: 24.28 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_for_better_sm_utilization | 0.975× | 0.138× | ✓ |  |
| B | autotune_xblock_tile_size_for_better_occupancy_and_ipc | 0.572× | 0.518× | ✓ |  |
| C | autotune_xblock_for_pointwise_kernel | 0.979× | 1.002× | ✓ |  |
| D | add_autotune_larger_xblock_vectorized_loads | 0.751× | 0.706× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_t... | Δ | autotune_xblock_f... | Δ | add_autotune_larg... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12160 ns | 87870 ns | +75710 ns (+622.6%) | 23490 ns | +11330 ns (+93.2%) | 12130 ns | -30 ns (-0.2%) | 17220 ns | +5060 ns (+41.6%) |
| SM Throughput (%) | 60.83 % | 78.13 % | +17.30 pp | 73.35 % | +12.52 pp | 62.11 % | +1.28 pp | 71.77 % | +10.94 pp |
| Memory Throughput (%) | 30.30 % | 15.28 % | -15.02 pp | 20.31 % | -9.99 pp | 30.41 % | +0.11 pp | 21.39 % | -8.91 pp |
| DRAM Throughput (%) | 30.30 % | 4.19 % | -26.11 pp | 15.68 % | -14.62 pp | 30.41 % | +0.11 pp | 21.39 % | -8.91 pp |
| Achieved Occupancy (%) | 80.68 % | 92.11 % | +11.43 pp | 87.83 % | +7.15 pp | 81.31 % | +0.63 pp | 85.75 % | +5.07 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 18.00 register/thread | +0 | 26.00 register/thread | +8 |
| Waves/SM | 5.57 | 22.27 | +16.70 | 5.57 | +0.00 | 5.57 | +0.00 | 5.57 | +0.00 |
| No Eligible (%) | 24.28 % | 21.00 % | -3.28 pp | 20.84 % | -3.44 pp | 24.58 % | +0.30 pp | 20.78 % | -3.50 pp |
| IPC Active | 2.98 inst/cycle | 3.16 inst/cycle | +0.18 | 3.13 inst/cycle | +0.15 | 2.98 inst/cycle | +0.00 | 3.13 inst/cycle | +0.15 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune_for_better_sm_utilization` (0.138×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) to a pointwise kernel that had low achieved occupancy (80.7%) and low L1 hit rate (42%) caused a severe 7.2x regression. The autotune selected XBLOCK=512 with 16 warps, which increased registers per thread from 18 to 32, reducing blocks per SM from 21 to 4 (Block Limit Registers dropped from 21 to 4). This caused waves per SM to balloon from 5.57 to 22.27 and elapsed cycles to jump from 18,092 to 131,638, despite L1 hit rate improving to 66% and DRAM throughput dropping. The dominant effect was the register pressure increase from larger XBLOCK causing severe occupancy-limiting register spill pressure and massively more total instruction count (5.9M → 53.3M), overwhelming any cache reuse benefit.

> **Applicable when**: Never: this specific strategy caused severe regression. Do not apply autotune with large XBLOCK configs when the kernel has low register count (≤18 regs/thread) but the larger block sizes would increase register pressure significantly (e.g., from 18 to 32 regs/thread), collapsing Block Limit Registers from 21 to 4.

> **Avoid when**: When increasing XBLOCK causes registers per thread to increase substantially (e.g., >50% increase), which collapses Block Limit Registers and forces many more waves per SM. Also do not apply when the grid lambda is not updated to use meta['XBLOCK'] — in this diff the grid remained fixed at cdiv(xnumel, 128) while autotune varied XBLOCK, causing each block to process 4x more elements than the grid accounts for (implementation flaw compounding the regression). Do not apply when total instruction count would increase by >5x.

#### ❌ Plan D: `add_autotune_larger_xblock_vectorized_loads` (0.706×)

Adding @triton.autotune with larger XBLOCK values (256–1024) to a compute-bound pointwise kernel caused a 41% slowdown (12.16 µs → 17.22 µs). Registers per thread jumped from 18 to 26, and total executed instructions increased from 5.88M to 9.31M, indicating the autotuner selected a configuration that increased register pressure and instruction count without a compensating reduction in memory traffic. The ALU pipeline became oversubscribed (80.7% vs 67.9%), and two new CPIStall rules appeared (pipe not available 32.4%, not selected 39.9%), showing that larger tiles created more ALU contention rather than better vectorization or L2 reuse.

> **Applicable when**: Never: this specific change caused regression. In general, larger XBLOCK autotune is only beneficial when the kernel is memory-bandwidth-bound (DRAM Throughput > 70%), registers/thread are very low (<12), and the original XBLOCK is so small that vectorized 128-bit loads cannot be emitted.

> **Avoid when**: Kernel is already compute-bound (Compute SOL > 55%) with ALU pipeline utilization > 60%; registers/thread are moderate (≥16); increasing XBLOCK raises register pressure and instruction count without reducing memory traffic; CPIStall (pipe not available) is absent before the change but appears after.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.996× | 0.992× | ✓ |  |
| B | persistent_tile_scheduling | 0.996× | 0.998× | ✓ |  |
| C | persistent_tile_scheduling_with_fixed_sm_grid | 0.045× | 0.036× | ✓ |  |
| D | persistent_tile_scheduling_with_corrected_grid | 0.045× | 0.036× | ✓ |  |
| E | add_triton_autotune | 0.982× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12160 ns | 12260 ns | +100 ns (+0.8%) | 12190 ns | +30 ns (+0.2%) | 332700 ns | +320540 ns (+2636.0%) | 332700 ns | +320540 ns (+2636.0%) | 12160 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 60.83 % | 64.94 % | +4.11 pp | 63.96 % | +3.13 pp | 90.68 % | +29.85 pp | 90.65 % | +29.82 pp | 62.23 % | +1.40 pp |
| Memory Throughput (%) | 30.30 % | 30.13 % | -0.17 pp | 30.27 % | -0.03 pp | 25.24 % | -5.06 pp | 25.25 % | -5.05 pp | 30.33 % | +0.03 pp |
| DRAM Throughput (%) | 30.30 % | 30.13 % | -0.17 pp | 30.27 % | -0.03 pp | 1.19 % | -29.11 pp | 1.19 % | -29.11 pp | 30.33 % | +0.03 pp |
| Achieved Occupancy (%) | 80.68 % | 81.03 % | +0.35 pp | 81.60 % | +0.92 pp | 98.63 % | +17.95 pp | 98.62 % | +17.94 pp | 80.78 % | +0.10 pp |
| Registers/Thread | 18.00 register/thread | 24.00 register/thread | +6 | 24.00 register/thread | +6 | 25.00 register/thread | +7 | 25.00 register/thread | +7 | 18.00 register/thread | +0 |
| Waves/SM | 5.57 | 5.57 | +0.00 | 5.57 | +0.00 | 5.57 | +0.00 | 5.57 | +0.00 | 5.57 | +0.00 |
| No Eligible (%) | 24.28 % | 21.65 % | -2.63 pp | 21.63 % | -2.65 pp | 8.59 % | -15.69 pp | 8.58 % | -15.70 pp | 24.07 % | -0.21 pp |
| IPC Active | 2.98 inst/cycle | 3.09 inst/cycle | +0.11 | 3.10 inst/cycle | +0.12 | 3.65 inst/cycle | +0.67 | 3.65 inst/cycle | +0.67 | 2.96 inst/cycle | -0.02 |


### Key Learnings

#### ❌ Plan C: `persistent_tile_scheduling_with_fixed_sm_grid` (0.036×)

Converting a standard pointwise kernel (grid=11760 CTAs) to a persistent kernel (grid=132 CTAs, each iterating ~89 tiles in a while-loop) caused a catastrophic 27x slowdown (12.13 µs → 332.7 µs). The while-loop body added loop-control overhead (branch instructions jumped from 94K to 2.2M) and the total executed instruction count exploded from 5.88M to 238.7M (40.6x more instructions), despite the L2 hit rate improving dramatically from 28% to 91% and DRAM throughput dropping from 30% to 1.2%. The root cause is that the persistent loop's per-tile overhead (loop counter increment, comparison, branch) dominates when each tile is tiny (128 threads × ~5 loads + 1 store), and the while-loop inside a Triton kernel is compiled to scalar loop-control instructions that are not amortized efficiently — the ALU pipeline became a bottleneck at 83% utilization. Additionally, the grid in the launcher was NOT updated from `(triton.cdiv(xnumel, 128),)` to `(132,)`, so the kernel was still launched with 11760 CTAs each running the while-loop, multiplying the instruction count by ~89x.

> **Applicable when**: Never apply persistent kernel scheduling to tiny pointwise kernels where each tile does only O(XBLOCK) loads and stores. Persistent kernels are only beneficial when: (1) the grid is correctly updated to NUM_SMS, (2) each tile does substantial compute (e.g., matmul, reduction), and (3) the per-tile loop overhead is negligible relative to tile compute time.

> **Avoid when**: When the kernel body is a simple pointwise operation (few loads, arithmetic, one store per element) — the while-loop overhead dominates. When the grid lambda is not simultaneously updated from `(cdiv(N, XBLOCK),)` to `(NUM_SMS,)` — this implementation flaw causes each of the original 11760 CTAs to run the full while-loop, multiplying instruction count by ~89x. When registers per thread would increase (18 → 25) due to loop counter variables, reducing Block Limit Registers.

#### ❌ Plan D: `persistent_tile_scheduling_with_corrected_grid` (0.036×)

Converting a standard pointwise Triton kernel to a persistent tile scheduling pattern (one CTA per SM, each looping over ~89 tiles via `tl.range(sm_id, num_tiles, NUM_SMS)`) caused a catastrophic 27x slowdown (12.13 µs → 332.7 µs). The executed instruction count exploded from 5.88M to 238.7M (40.6x), while registers per thread increased from 18 to 25, indicating the persistent loop body was not being compiled efficiently — the loop overhead and index recomputation per tile dominated. Despite the L2 hit rate improving dramatically (28% → 91%) and DRAM throughput dropping to near zero (30.4% → 1.19%), the ALU pipeline became severely oversubscribed (62% → 90.65%) and elapsed cycles grew 27.7x. The persistent scheduling pattern is inappropriate for this workload: with only 5.57 waves/SM in the original, the kernel was already well-parallelized and did not suffer from launch overhead; the persistent loop added massive per-tile index recomputation overhead that dwarfed any launch savings.

> **Applicable when**: Never for this pattern as implemented. Persistent tile scheduling is theoretically applicable when: (1) waves_per_sm >> 1 (e.g., >20), (2) kernel launch overhead is measurable relative to kernel duration, (3) the tile body is very short and loop overhead is negligible, and (4) the grid is correctly reduced to (NUM_SMS,).

> **Avoid when**: Do not apply persistent tile scheduling when: (1) the original kernel already has low waves_per_sm (~5-6), meaning launch overhead is not the bottleneck; (2) the grid is NOT updated from (cdiv(xnumel, XBLOCK),) to (NUM_SMS,) — leaving the grid at 11760 while adding a persistent loop causes each of the 11760 CTAs to independently loop over all 89 tiles, inflating instruction count by ~89x; (3) the kernel is already compute-bound (Compute SOL > 60%) with ALU as the bottleneck pipeline, as persistent loops add more ALU index computation overhead.

---

## Cumulative Progress

```
Baseline:  12160 ns  (1.000×)

Final best: 12160 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,19 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=8),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 1505280
@@ -26,8 +39,6 @@

     tmp8 = tmp4 + tmp7
     tl.store(in_out_ptr0 + (x3), tmp8, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -36,7 +47,7 @@

     arg_4 = inputs[4]
     xnumel = 1505280
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_25[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1505280, XBLOCK=128)
+    triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_25[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1505280)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
