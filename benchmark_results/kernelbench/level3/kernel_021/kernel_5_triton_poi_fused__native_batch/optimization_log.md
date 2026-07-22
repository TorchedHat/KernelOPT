# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 54880 ns — **Best latency**: 65340 ns (**0.8399×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_021/kernel_5_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (23520,1,1), Waves/SM: 14.85
- **Registers/thread**: 36
- **Duration**: 54880 ns
- **SM Throughput**: 59.69 % | **Memory**: 74.04 % | **DRAM**: 64.60 %
- **Achieved Occupancy**: 69.71 %
- **IPC Active**: 2.46 inst/cycle | **No Eligible**: 39.43 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_and_fix_eviction_policy | 1.085× | 0.807× | ✓ |  |
| D | autotune_tile_sizes_and_warps_for_memory_bound_pointwise | 1.517× | 0.510× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 54880 ns | 68030 ns | +13150 ns (+24.0%) | 107550 ns | +52670 ns (+96.0%) |
| SM Throughput (%) | 59.69 % | 71.17 % | +11.48 pp | 72.83 % | +13.14 pp |
| Memory Throughput (%) | 74.04 % | 70.28 % | -3.76 pp | 50.06 % | -23.98 pp |
| DRAM Throughput (%) | 64.60 % | 50.87 % | -13.73 pp | 31.65 % | -32.95 pp |
| Achieved Occupancy (%) | 69.71 % | 58.48 % | -11.23 pp | 70.41 % | +0.70 pp |
| Registers/Thread | 36.00 register/thread | 48.00 register/thread | +12 | 40.00 register/thread | +4 |
| Waves/SM | 14.85 | 17.82 | +2.97 | 29.70 | +14.85 |
| No Eligible (%) | 39.43 % | 25.70 % | -13.73 pp | 25.15 % | -14.28 pp |
| IPC Active | 2.46 inst/cycle | 2.98 inst/cycle | +0.52 | 2.97 inst/cycle | +0.51 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_contiguous_hints_and_rsqrt_simplification | 1.593× | 0.851× | ✓ |  |
| C | fuse_reciprocal_sqrt_to_rsqrt_reduce_register_pressure | 1.725× | 0.580× | ✓ | **✓ best** |
| D | fuse_reciprocal_sqrt_to_reduce_instruction_count_and_register_pressure | 1.717× | 0.569× | ✓ |  |
| E | add_triton_autotune | 1.085× | 0.808× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | fuse_reciprocal_s... | Δ | fuse_reciprocal_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 54880 ns | 64480 ns | +9600 ns (+17.5%) | 94690 ns | +39810 ns (+72.5%) | 96450 ns | +41570 ns (+75.7%) | 67940 ns | +13060 ns (+23.8%) |
| SM Throughput (%) | 59.69 % | 33.98 % | -25.71 pp | 61.31 % | +1.62 pp | 60.92 % | +1.23 pp | 70.83 % | +11.14 pp |
| Memory Throughput (%) | 74.04 % | 72.31 % | -1.73 pp | 55.83 % | -18.21 pp | 55.33 % | -18.71 pp | 69.39 % | -4.65 pp |
| DRAM Throughput (%) | 64.60 % | 55.47 % | -9.13 pp | 35.90 % | -28.70 pp | 35.27 % | -29.33 pp | 50.92 % | -13.68 pp |
| Achieved Occupancy (%) | 69.71 % | 93.26 % | +23.55 pp | 69.96 % | +0.25 pp | 69.97 % | +0.26 pp | 58.51 % | -11.20 pp |
| Registers/Thread | 36.00 register/thread | 32.00 register/thread | -4 | 34.00 register/thread | -2 | 34.00 register/thread | -2 | 48.00 register/thread | +12 |
| Waves/SM | 14.85 | 11.14 | -3.71 | 29.70 | +14.85 | 29.70 | +14.85 | 17.82 | +2.97 |
| No Eligible (%) | 39.43 % | 64.61 % | +25.18 pp | 37.98 % | -1.45 pp | 37.79 % | -1.64 pp | 26.24 % | -13.19 pp |
| IPC Active | 2.46 inst/cycle | 1.43 inst/cycle | -1.03 | 2.49 inst/cycle | +0.03 | 2.47 inst/cycle | +0.01 | 2.95 inst/cycle | +0.49 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fma_fusion_to_reduce_nonfused_fp32_instructions_and_register_pressure | 1.729× | 0.584× | ✓ | **✓ best** |
| C | vectorized_loads_via_correct_scalar_contiguous_hints | 1.084× | 0.791× | ✓ |  |
| D | precompute_batchnorm_scale_bias_to_enable_fma_fusion | 1.725× | 0.621× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fma_fusion_to_red... | Δ | vectorized_loads_... | Δ | precompute_batchn... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 54880 ns | 94020 ns | +39140 ns (+71.3%) | 69380 ns | +14500 ns (+26.4%) | 88420 ns | +33540 ns (+61.1%) |
| SM Throughput (%) | 59.69 % | 60.94 % | +1.25 pp | 69.76 % | +10.07 pp | 58.53 % | -1.16 pp |
| Memory Throughput (%) | 74.04 % | 55.37 % | -18.67 pp | 69.00 % | -5.04 pp | 60.16 % | -13.88 pp |
| DRAM Throughput (%) | 64.60 % | 36.19 % | -28.41 pp | 50.01 % | -14.59 pp | 38.51 % | -26.09 pp |
| Achieved Occupancy (%) | 69.71 % | 69.91 % | +0.20 pp | 58.53 % | -11.18 pp | 69.69 % | -0.02 pp |
| Registers/Thread | 36.00 register/thread | 34.00 register/thread | -2 | 48.00 register/thread | +12 | 40.00 register/thread | +4 |
| Waves/SM | 14.85 | 29.70 | +14.85 | 17.82 | +2.97 | 29.70 | +14.85 |
| No Eligible (%) | 39.43 % | 37.23 % | -2.20 pp | 26.74 % | -12.69 pp | 39.39 % | -0.04 pp |
| IPC Active | 2.46 inst/cycle | 2.52 inst/cycle | +0.06 | 2.88 inst/cycle | +0.42 | 2.37 inst/cycle | -0.09 |


### Key Learnings

#### ❌ Plan C: `vectorized_loads_via_correct_scalar_contiguous_hints` (0.791×)

Applying tl.max_contiguous(tl.multiple_of(xrange, XBLOCK), XBLOCK) with correct scalar arguments on the 1D arange before broadcasting, combined with adding larger autotune configs (XBLOCK=512/1024) and replacing rsqrt with a sqrt_rn+int_div chain, increased registers per thread from 32 to 48 and reduced theoretical occupancy from 100% to 62.5%. The net effect was a 7.5% duration increase (64.48µs → 69.38µs), with total executed instructions more than doubling (17.2M → 37.3M) and divergent branches exploding from 0 to 757 average per warp. The contiguous hints did not produce vectorization benefit sufficient to offset the register pressure increase and instruction count growth introduced by the arithmetic regression (sqrt_rn+int_div chain instead of rsqrt) and the larger autotune tile configs.

> **Applicable when**: Never apply this combined change (scalar contiguous hints + reverting rsqrt to multi-op chain + adding larger autotune configs) simultaneously. Each change must be isolated to attribute its effect.

> **Avoid when**: Do not revert a working rsqrt simplification back to a multi-op sqrt_rn+int_div chain while simultaneously adding contiguous hints and expanding autotune configs. The multi-op chain increases register pressure (32→48 regs/thread), which drops theoretical occupancy (100%→62.5%) and causes the TheoreticalOccupancy NCU rule to trigger. Even if scalar contiguous hints are syntactically correct, they cannot overcome the register pressure regression introduced by the arithmetic revert. Also do not add XBLOCK=512/1024 configs when the kernel is already register-limited at 48 regs/thread — the autotuner will select configs that increase divergent branches without improving throughput.

#### ❌ Plan D: `precompute_batchnorm_scale_bias_to_enable_fma_fusion` (0.592×)

Hoisting channel-only batch-norm scalar loads before the large strided input load and collapsing the 5-op BN chain into two precomputed scalars (combined_scale, combined_bias) plus a single FMA is algebraically correct and did reduce warp stall cycles (41.9→18.7 cycles/instruction) and improve scheduler eligibility (0.81→3.28 eligible warps). However, the simultaneously applied autotune config change (replacing the original config set with 8 new configs including num_stages=2-3) caused the autotuner to select a 256-thread block that raised registers per thread (32→40), cut theoretical occupancy (100%→75%), doubled waves per SM (11.14→29.7), degraded L2 hit rate (64.79%→51.25%), and increased total executed instructions by 2.4x (17.2M→40.7M), causing a 46% duration increase (63.39µs→92.67µs). This is now the fifth consecutive iteration where changing autotune configs alongside an arithmetic optimization caused the autotuner to select a larger block size that overwhelmed the intended benefit — the autotune config change is the dominant regression driver, not the arithmetic restructuring.

> **Applicable when**: Never apply this arithmetic restructuring simultaneously with autotune config set changes. The FMA precomputation pattern (hoisting broadcast scalars, collapsing multi-op BN chains to a single multiply-add) is algebraically valid and may help in isolation when the kernel is compute-bound with many non-fused FP32 instructions. Apply only when autotune configs are held exactly constant.

> **Avoid when**: Do not combine batch-norm FMA precomputation with autotune config set changes (adding/removing configs, adding num_stages) in the same iteration. This is the fifth consecutive iteration where changing autotune configs caused the autotuner to select a 256-thread block that doubled waves_per_sm (11.14→29.7), raised registers per thread (32→40), dropped theoretical occupancy (100%→75%), degraded L2 hit rate, and increased total instruction count by 2-3x, overwhelming any arithmetic savings. Also do not apply when the kernel is already memory-latency-bound (high warp stall cycles, low eligible warps per scheduler) — FMA precomputation does not address the root cause of L1TEX stall latency.

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_reciprocal_sqrt_to_rsqrt_in_isolation_no_autotune_change | 1.563× | 0.866× | ✓ |  |
| B | store_evict_first_to_free_l2_bandwidth_for_input_reads | 1.625× | 0.562× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_reciprocal_s... | Δ | store_evict_first... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 54880 ns | 63390 ns | +8510 ns (+15.5%) | 97700 ns | +42820 ns (+78.0%) |
| SM Throughput (%) | 59.69 % | 34.28 % | -25.41 pp | 61.32 % | +1.63 pp |
| Memory Throughput (%) | 74.04 % | 73.03 % | -1.01 pp | 54.82 % | -19.22 pp |
| DRAM Throughput (%) | 64.60 % | 56.22 % | -8.38 pp | 35.03 % | -29.57 pp |
| Achieved Occupancy (%) | 69.71 % | 93.69 % | +23.98 pp | 69.84 % | +0.13 pp |
| Registers/Thread | 36.00 register/thread | 32.00 register/thread | -4 | 34.00 register/thread | -2 |
| Waves/SM | 14.85 | 11.14 | -3.71 | 29.70 | +14.85 |
| No Eligible (%) | 39.43 % | 63.85 % | +24.42 pp | 37.79 % | -1.64 pp |
| IPC Active | 2.46 inst/cycle | 1.42 inst/cycle | -1.04 | 2.49 inst/cycle | +0.03 |


### Key Learnings

#### ❌ Plan A: `fuse_reciprocal_sqrt_to_rsqrt_in_isolation_no_autotune_change` (0.866×)

Replacing the multi-op reciprocal-sqrt chain (sqrt_rn + int-div + float-mul) with libdevice.rsqrt() in isolation — with no autotune config changes — successfully reduced registers per thread (48→32), cut total executed instructions by 54% (37.3M→17.2M), and raised theoretical occupancy from 62.5% to 100% and achieved occupancy from 58.5% to 93.7%. However, the higher occupancy dramatically worsened L1TEX scoreboard stall cycles per issued instruction (13.0→41.9 cycles), causing eligible warps per scheduler to collapse from 2.15 to 0.81 and No-Eligible cycles to jump from 26.7% to 63.9%. The net effect was a 13.4% regression (69.38µs→63.39µs wall time is actually faster by clock, but the speedup ratio is 0.8658 indicating the overall benchmark regressed), because packing more warps onto each SM amplified L1TEX memory latency contention rather than hiding it — the kernel is fundamentally L1TEX-stall-bound, not occupancy-limited.

> **Applicable when**: Never: even when applied in strict isolation (no autotune config changes), replacing a multi-op sqrt+div chain with libdevice.rsqrt() on a kernel that is L1TEX-stall-dominated (CPIStall rule triggered, stall fraction >35% of CPI, eligible warps per scheduler <2.5) will increase occupancy but worsen stall contention. The register reduction is real but counterproductive when the bottleneck is memory latency, not occupancy.

> **Avoid when**: Do not apply libdevice.rsqrt() substitution (or any register-reducing transformation) to kernels where the dominant bottleneck is L1TEX scoreboard stall latency (CPIStall rule triggered, warp cycles per issued instruction >12, stall fraction >35%). Reducing register pressure increases occupancy, which packs more memory-latency-bound warps onto each SM, amplifying contention and collapsing eligible warps per scheduler (2.15→0.81 observed). The kernel must first have its memory access pattern improved (coalescing, cache reuse) before occupancy increases can help. Also confirmed: even with autotune configs held constant, the autotuner selects the same tile config with fewer registers, so the occupancy increase is unavoidable once registers drop.

#### ❌ Plan B: `store_evict_first_to_free_l2_bandwidth_for_input_reads` (0.562×)

Adding eviction_policy='evict_first' to large streaming input/output loads and stores is a valid L2 cache pressure reduction technique, but the simultaneously applied autotune config change (replacing the original 14-config set with 8 configs including num_stages=2-3) caused the autotuner to select a 256-thread block that doubled waves per SM (17.82→29.7), increased total executed instructions by 21% (37.3M→45.3M), degraded L2 hit rate (65.53%→51.54%), and dropped memory throughput from 2.46 TB/s to 1.72 TB/s. The net effect was a 41% duration increase (69.38µs→97.7µs). This is the fourth consecutive iteration where conflating a targeted optimization with autotune config changes caused the autotuner to select a larger block size that overwhelmed the intended benefit.

> **Applicable when**: evict_first on large streaming write-once tensors (output >> L2 cache size) may help when L2 hit rate is below 70% and the MemoryL2Compression rule is triggered, BUT ONLY when autotune configs are held constant to isolate the eviction policy change's effect.

> **Avoid when**: Do not combine eviction_policy changes with autotune config set changes (adding/removing configs, adding num_stages) in the same iteration. This is the fourth consecutive iteration where changing autotune configs caused the autotuner to select a 256-thread block that doubled waves_per_sm (17.82→29.7), degraded L2 hit rate, and increased instruction count, overwhelming any targeted optimization. Never add num_stages to autotune configs for pointwise kernels without verifying the selected config does not increase waves_per_sm beyond the original baseline.

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | precompute_batchnorm_scale_bias_fma_isolated_arithmetic_only | 1.882× | 0.840× | ✓ | **✓ best** |
| D | reduce_register_pressure_by_hoisting_scalar_precomputation_fixed_autotune | 1.617× | 0.592× | ✓ |  |


### Metric Comparison

| Metric | Baseline | precompute_batchn... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 54880 ns | 65340 ns | +10460 ns (+19.1%) | 92670 ns | +37790 ns (+68.9%) |
| SM Throughput (%) | 59.69 % | 26.80 % | -32.89 pp | 57.51 % | -2.18 pp |
| Memory Throughput (%) | 74.04 % | 64.33 % | -9.71 pp | 58.40 % | -15.64 pp |
| DRAM Throughput (%) | 64.60 % | 56.82 % | -7.78 pp | 37.01 % | -27.59 pp |
| Achieved Occupancy (%) | 69.71 % | 93.48 % | +23.77 pp | 69.75 % | +0.04 pp |
| Registers/Thread | 36.00 register/thread | 32.00 register/thread | -4 | 40.00 register/thread | +4 |
| Waves/SM | 14.85 | 11.14 | -3.71 | 29.70 | +14.85 |
| No Eligible (%) | 39.43 % | 73.70 % | +34.27 pp | 40.64 % | +1.21 pp |
| IPC Active | 2.46 inst/cycle | 1.10 inst/cycle | -1.36 | 2.40 inst/cycle | -0.06 |


### Key Learnings

#### ❌ Plan A: `precompute_batchnorm_scale_bias_fma_isolated_arithmetic_only` (0.840×)

Precomputing batch-norm combined_scale and combined_bias by hoisting the 4 channel-only scalar loads (mean, var, weight, bias) before the main element load, and collapsing the 6-op chain into a single FMA, successfully reduced total executed instructions by 21% (17.2M→13.5M) while holding autotune configs, block size, and register count constant. However, this worsened L1TEX scoreboard stall cycles per issued instruction from 41.9 to 53.6 cycles (+28%), collapsed eligible warps per scheduler from 0.81 to 0.54, and increased No-Eligible cycles from 63.9% to 73.7%, causing a 3% duration increase. The L1/TEX hit rate also dropped from 9.92% to 4.93%, indicating that hoisting the scalar loads before the main element load disrupted the cache access ordering that previously allowed some L1 reuse of the channel-dimension values, and the earlier issuance of the scalar loads extended the dependency chain stall window for the FMA that follows.

> **Applicable when**: Never: even when applied in strict isolation with autotune configs held constant, hoisting channel-only scalar loads before the main element load and collapsing BN arithmetic into a single FMA on a kernel that is already L1TEX-stall-dominated (CPIStall rule triggered, warp cycles per issued instruction >40, eligible warps <1.0) worsens stall depth and reduces L1 hit rate. Fewer instructions do not translate to lower latency when the bottleneck is memory dependency stalls, not instruction throughput.

> **Avoid when**: Do not hoist scalar (channel-only) loads before the main element load in a memory-latency-bound Triton kernel where L1TEX scoreboard stalls already dominate (CPIStall rule triggered, stall fraction >50% of CPI, eligible warps per scheduler <1.0). Reordering loads to issue scalar loads earlier extends the dependency chain window for the compute that follows, worsening stall depth (41.9→53.6 cycles/warp observed) and reducing L1 hit rate (9.92%→4.93%). The instruction count reduction from FMA fusion does not compensate when the kernel is stall-bound rather than instruction-throughput-bound. This anti-pattern is confirmed across iterations 2, 3, 4, and 5 for this kernel class.

---

## Cumulative Progress

```
Baseline:  54880 ns  (1.000×)

Final best: 65340 ns  (0.8399×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,26 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=8),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=8),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=8),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 4}, num_warps=8),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=8),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2}, num_warps=8),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 1920
@@ -13,31 +33,29 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    # Apply scalar hints to 1D arange before broadcasting to 2D
+    xrange = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
+    xindex = xrange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y0 = (yindex % 192)
     y1 = yindex // 192
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 192*x2 + 2408448*y1), xmask & ymask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
-    tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1, 1], 1e-05, tl.float32)
-    tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1, 1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1, 1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
+    # Hoist channel-only scalar loads BEFORE the main element load (ymask only, no xmask needed)
+    tmp_mean    = tl.load(in_ptr1 + (y0), mask=ymask, eviction_policy='evict_last')   # [YBLOCK, 1]
+    tmp_var     = tl.load(in_ptr2 + (y0), mask=ymask, eviction_policy='evict_last')   # [YBLOCK, 1]
+    tmp_weight  = tl.load(in_ptr3 + (y0), mask=ymask, eviction_policy='evict_last')   # [YBLOCK, 1]
+    tmp_bias    = tl.load(in_ptr4 + (y0), mask=ymask, eviction_policy='evict_last')   # [YBLOCK, 1]
+    # Precompute per-channel scale and bias (amortized over XBLOCK elements)
+    tmp_eps     = tl.full([1, 1], 1e-05, tl.float32)
+    tmp_inv_std = libdevice.rsqrt(tmp_var + tmp_eps)        # rsqrt(var + eps)
+    combined_scale = tmp_weight * tmp_inv_std               # weight / sqrt(var+eps)  [YBLOCK, 1]
+    combined_bias  = tmp_bias - tmp_mean * combined_scale   # bias - mean*scale        [YBLOCK, 1]
+    # Load main input element
+    tmp0 = tl.load(in_ptr0 + (y0 + 192*x2 + 2408448*y1), xmask & ymask, eviction_policy='evict_first')
+    # Single FMA: x * combined_scale + combined_bias  (broadcasts [YBLOCK,1] over XBLOCK)
+    tmp15 = tmp0 * combined_scale + combined_bias
     tl.store(out_ptr0 + (x2 + 12544*y3), tmp15, xmask & ymask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -49,7 +67,7 @@

     ynumel = 1920
     xnumel = 12544
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1920, 12544, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1920, 12544)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
