# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 174780 ns — **Best latency**: 99420 ns (**1.7580×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fuse_bias_broadcast_with_evict_first_and_expand_autotune_configs`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_055/kernel_2_triton_poi_fused_convolution_2/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (73728,1,1), Waves/SM: 34.91
- **Registers/thread**: 32
- **Duration**: 174780 ns
- **SM Throughput**: 28.01 % | **Memory**: 81.46 % | **DRAM**: 67.19 %
- **Achieved Occupancy**: 94.18 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 71.35 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.000× | 0.997× | ✓ |  |
| B | autotune_tile_sizes_to_reduce_l2_pressure_and_uncoalesced_stores | 1.813× | 1.796× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 174780 ns | 175330 ns | +550 ns (+0.3%) | 97340 ns | -77440 ns (-44.3%) |
| SM Throughput (%) | 28.01 % | 27.83 % | -0.18 pp | 48.49 % | +20.48 pp |
| Memory Throughput (%) | 81.46 % | 81.18 % | -0.28 pp | 53.36 % | -28.10 pp |
| DRAM Throughput (%) | 67.19 % | 66.97 % | -0.22 pp | 28.73 % | -38.46 pp |
| Achieved Occupancy (%) | 94.18 % | 94.20 % | +0.02 pp | 93.00 % | -1.18 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 30.00 register/thread | -2 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.35 % | 70.85 % | -0.50 pp | 51.14 % | -20.21 pp |
| IPC Active | 1.15 inst/cycle | 1.15 inst/cycle | +0.00 | 2.00 inst/cycle | +0.85 |

---

## Iteration 2

**Best so far**: 1.7956× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_eviction_policy_for_large_tensor_loads_and_stores | 1.787× | 1.728× | ✓ |  |
| B | fuse_bias_broadcast_with_evict_first_and_expand_autotune_configs | 1.945× | 1.758× | ✓ | **✓ best** |
| C | autotune_tile_sizes_for_strided_nhwc_to_nchw_transpose | 1.817× | 1.748× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_evictio... | Δ | fuse_bias_broadca... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 174780 ns | 101150 ns | -73630 ns (-42.1%) | 99420 ns | -75360 ns (-43.1%) | 99970 ns | -74810 ns (-42.8%) |
| SM Throughput (%) | 28.01 % | 49.90 % | +21.89 pp | 49.23 % | +21.22 pp | 48.59 % | +20.58 pp |
| Memory Throughput (%) | 81.46 % | 53.15 % | -28.31 pp | 54.14 % | -27.32 pp | 53.41 % | -28.05 pp |
| DRAM Throughput (%) | 67.19 % | 27.13 % | -40.06 pp | 28.29 % | -38.90 pp | 28.11 % | -39.08 pp |
| Achieved Occupancy (%) | 94.18 % | 93.05 % | -1.13 pp | 92.99 % | -1.19 pp | 93.01 % | -1.17 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 | 30.00 register/thread | -2 | 30.00 register/thread | -2 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.35 % | 48.66 % | -22.69 pp | 50.64 % | -20.71 pp | 51.08 % | -20.27 pp |
| IPC Active | 1.15 inst/cycle | 2.04 inst/cycle | +0.89 | 1.99 inst/cycle | +0.84 | 1.99 inst/cycle | +0.84 |


### Key Learnings

#### ❌ Plan A: `streaming_eviction_policy_for_large_tensor_loads_and_stores` (1.728×)

Switching large streaming tensor loads/stores from evict_last to evict_first in Triton did not reduce kernel duration (97.34us → 101.15us, a 4% regression at the NCU level despite a reported 1.73x end-to-end speedup that likely reflects measurement noise or autotuning re-selection). The key memory metrics barely moved: L2 Hit Rate went from 55.47% to 56.48%, No-Eligible cycles dropped only from 51.14% to 48.66%, and eligible warps per scheduler improved only marginally (3.46→3.66). This indicates that when L2 is already operating at moderate utilization (~42%) rather than saturation (>80%), eviction policy hints provide negligible benefit because L2 pressure is not the binding bottleneck — the remaining stalls are driven by DRAM latency and uncoalesced access patterns that eviction hints cannot address.

> **Applicable when**: L2 Cache Throughput > 80% (L2 is saturated), large tensor (>>L2 size) accessed in a single streaming pass with evict_last, and the primary bottleneck is L2 thrashing rather than DRAM latency or uncoalesced access

> **Avoid when**: L2 Cache Throughput is already at moderate levels (<60%), the primary stall source is uncoalesced access patterns (MemoryCacheAccessPattern rule active with <27 bytes/sector utilization), or eligible warps per scheduler is already low (<4) due to DRAM latency — in these cases eviction policy changes provide negligible benefit and may slightly increase duration

---

## Iteration 3

**Best so far**: 1.7956× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fma_fusion_and_bias_broadcast_contiguous_hints | 1.771× | 1.749× | ✓ |  |
| B | increase_num_stages_to_hide_memory_latency | 1.806× | 1.747× | ✓ |  |
| C | fix_autotune_grid_lambda_to_activate_autotuner | 1.786× | 1.713× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fma_fusion_and_bi... | Δ | increase_num_stag... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 174780 ns | 99940 ns | -74840 ns (-42.8%) | 100030 ns | -74750 ns (-42.8%) | 102020 ns | -72760 ns (-41.6%) |
| SM Throughput (%) | 28.01 % | 49.03 % | +21.02 pp | 48.72 % | +20.71 pp | 49.23 % | +21.22 pp |
| Memory Throughput (%) | 81.46 % | 52.26 % | -29.20 pp | 53.56 % | -27.90 pp | 52.38 % | -29.08 pp |
| DRAM Throughput (%) | 67.19 % | 27.50 % | -39.69 pp | 28.07 % | -39.12 pp | 27.08 % | -40.11 pp |
| Achieved Occupancy (%) | 94.18 % | 93.04 % | -1.14 pp | 92.98 % | -1.20 pp | 93.07 % | -1.11 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 | 30.00 register/thread | -2 | 30.00 register/thread | -2 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.35 % | 48.56 % | -22.79 pp | 50.73 % | -20.62 pp | 49.85 % | -21.50 pp |
| IPC Active | 1.15 inst/cycle | 2.02 inst/cycle | +0.87 | 1.99 inst/cycle | +0.84 | 2.01 inst/cycle | +0.86 |


### Key Learnings

#### ❌ Plan A: `fma_fusion_and_bias_broadcast_contiguous_hints` (1.749×)

Replacing FADD with libdevice.fma() and adding tl.max_contiguous/tl.multiple_of hints on the bias index produced no measurable kernel duration improvement (99.42us → 99.94us, essentially flat). Despite the FPInstructions NCU rule reporting 2.36M non-fused FP32 ops, the actual bottleneck is memory latency (IssueSlotUtilization: ~50% no-eligible cycles driven by DRAM/L2 latency), not ALU throughput. The instruction count actually increased slightly (37.3M → 38.5M), and all key scheduler and memory metrics were essentially unchanged, confirming that FMA fusion provides no benefit when the kernel is memory-latency-bound rather than compute-bound.

> **Applicable when**: Kernel is compute-bound (Compute SM Throughput > 70%, No-Eligible cycles < 30%), FPInstructions rule reports large non-fused FP32 count, and memory metrics are not the primary bottleneck

> **Avoid when**: Kernel is memory-latency-bound (No-Eligible cycles > 45%, Eligible Warps Per Scheduler < 5, DRAM or L2 throughput is the binding constraint). In this regime, FMA fusion and contiguous load hints do not reduce stall cycles because the scheduler is waiting on memory, not on ALU instruction throughput. The FPInstructions NCU rule can be misleading when IssueSlotUtilization is the dominant bottleneck.

#### ❌ Plan B: `increase_num_stages_to_hide_memory_latency` (1.747×)

Expanding the Triton autotune config list with num_stages=5 entries (and num_warps=16) to hide DRAM latency produced no measurable improvement: kernel duration increased slightly from 99.42us to 100.03us, and all scheduler metrics were essentially unchanged (No-Eligible 50.64%→50.73%, Eligible Warps 3.47→3.45, Warp Cycles Per Issued Instruction 29.86→29.92). The FMA rewrite (tl.fma with a ones tensor) also had no effect on FP instruction fusion. When the kernel is already autotuned with num_stages=3-4 and the bottleneck is a strided NHWC→NCHW transpose pattern where DRAM latency is partially structural (uncoalesced access, not pipeline depth), adding deeper pipeline stages does not reduce the no-eligible-warp fraction because the stalls are access-pattern-driven rather than pipeline-depth-driven.

> **Applicable when**: Never: this strategy did not help even when IssueSlotUtilization rule showed 46% potential speedup and No-Eligible was 51%.

> **Avoid when**: When the kernel is a strided format-transpose (e.g., NHWC→NCHW) that is already autotuned with num_stages=3-4 and the scheduler stalls (No-Eligible ~50%, Eligible Warps ~3.5) persist despite high occupancy (93%). In this regime, the stalls are caused by uncoalesced access patterns (MemoryCacheAccessPattern rule: only 26.3/32 bytes utilized per sector), not insufficient pipeline depth. Adding num_stages=5 configs does not reduce stall cycles because the autotuner already selected the best available tile shape, and deeper pipelining cannot compensate for fundamentally strided memory access. Also avoid tl.fma(x, ones, y) as a substitute for x+y — the Triton compiler does not emit FFMA from this pattern.

---

## Iteration 4

**Best so far**: 1.7956× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | store_coalescing_via_multiple_of_hints_and_xblock_alignment | 1.795× | 1.727× | ✓ |  |
| C | shared_memory_tiled_transpose_to_coalesce_stores | 1.629× | 1.602× | ✓ |  |


### Metric Comparison

| Metric | Baseline | store_coalescing_... | Δ | shared_memory_til... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 174780 ns | 101180 ns | -73600 ns (-42.1%) | 109120 ns | -65660 ns (-37.6%) |
| SM Throughput (%) | 28.01 % | 50.26 % | +22.25 pp | 50.48 % | +22.47 pp |
| Memory Throughput (%) | 81.46 % | 53.60 % | -27.86 pp | 49.36 % | -32.10 pp |
| DRAM Throughput (%) | 67.19 % | 27.15 % | -40.04 pp | 25.12 % | -42.07 pp |
| Achieved Occupancy (%) | 94.18 % | 93.09 % | -1.09 pp | 93.49 % | -0.69 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 | 32.00 register/thread | +0 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.35 % | 48.69 % | -22.66 pp | 48.60 % | -22.75 pp |
| IPC Active | 1.15 inst/cycle | 2.02 inst/cycle | +0.87 | 2.04 inst/cycle | +0.89 |


### Key Learnings

#### ❌ Plan B: `store_coalescing_via_multiple_of_hints_and_xblock_alignment` (1.727×)

Adding tl.max_contiguous and tl.multiple_of hints on the store index (x2) to signal XBLOCK-alignment to the Triton compiler produced no measurable improvement: kernel duration increased slightly from 99.94us to 101.18us (~1.2% regression at NCU level). All key metrics were essentially unchanged — No-Eligible cycles (48.56% → 48.69%), Eligible Warps Per Scheduler (3.67 → 3.68), L2 Hit Rate (55.91% → 55.38%), and Warp Cycles Per Issued Instruction (29.1 → 29.18) — confirming the hints had no effect on actual memory transaction efficiency. When MemoryCacheAccessPattern reports only 26.3/32 bytes utilized per sector for global stores, the root cause is the strided output layout (x2 + 3025*y3 with large y-stride), not compiler alignment uncertainty; compiler hints cannot fix structural stride-induced sector underutilization.

> **Applicable when**: Never demonstrated effective: tl.multiple_of/tl.max_contiguous hints on store indices do not help when the sector underutilization is caused by a large stride in the non-x dimension (y-stride >> cache line size), because the hints only inform the compiler about x-dimension contiguity, not y-dimension spacing.

> **Avoid when**: When MemoryCacheAccessPattern reports <27 bytes/sector for global stores and the store index has a large stride in the y-dimension (e.g., out[x + LARGE_STRIDE * y]). In this pattern, consecutive y-threads write to addresses LARGE_STRIDE elements apart, causing sector underutilization that is structural and cannot be fixed by alignment hints on x. The hints are only useful when the compiler genuinely cannot prove x-alignment, which is rarely the binding issue when y-stride dominates. Also avoid combining this with evict_first on stores when L2 throughput is already moderate (<60%) — it provides no benefit.

---

## Cumulative Progress

```
Baseline:  174780 ns  (1.000×)
Iter 1     97340 ns  (1.796×)  — autotune_tile_sizes_to_reduce_l2_pressure_and_uncoalesced_stores

Final best: 99420 ns  (1.7580×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,26 +5,39 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=4, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 24576
     xnumel = 3025
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y0 = (yindex % 96)
+    y0_bias = tl.multiple_of(tl.max_contiguous(y0, [16, 1]), [16, 1])  # hint: 16-aligned contiguous run
     y1 = yindex // 96
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 290400*y1), xmask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
-    tmp2 = tmp0 + tmp1
-    tl.store(out_ptr0 + (x2 + 3025*y3), tmp2, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 290400*y1), xmask & ymask, eviction_policy='evict_first')  # 290MB streaming — do NOT pollute L2
+    tmp1 = tl.load(in_ptr1 + y0_bias, xmask & ymask, eviction_policy='evict_last')                     # 96-element bias — keep in L2, reused by all CTAs
+    tmp2 = libdevice.fma(tmp0, 1.0, tmp1)  # forces FFMA instead of FADD
+    tl.store(out_ptr0 + (x2 + 3025*y3), tmp2, xmask & ymask, eviction_policy='evict_first')            # 290MB write-once output — do NOT pollute L2
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -33,7 +46,7 @@

     ynumel = 24576
     xnumel = 3025
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025)
     return arg_2
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
