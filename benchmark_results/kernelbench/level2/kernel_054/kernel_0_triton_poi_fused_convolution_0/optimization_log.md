# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5820 ns — **Best latency**: 5310 ns (**1.0960×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`restore_winning_autotune_configs_and_evict_first_store`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_054/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5820 ns
- **SM Throughput**: 16.24 % | **Memory**: 42.51 % | **DRAM**: 21.21 %
- **Achieved Occupancy**: 55.40 %
- **IPC Active**: 1.10 inst/cycle | **No Eligible**: 70.54 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.095× | 0.948× | ✓ |  |
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.315× | 1.096× | ✓ | **✓ best** |
| D | add_autotune_increase_tile_sizes | 0.970× | 0.485× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | add_autotune_incr... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5820 ns | 6140 ns | +320 ns (+5.5%) | 5310 ns | -510 ns (-8.8%) | 12000 ns | +6180 ns (+106.2%) |
| SM Throughput (%) | 16.24 % | 24.16 % | +7.92 pp | 35.98 % | +19.74 pp | 28.28 % | +12.04 pp |
| Memory Throughput (%) | 42.51 % | 35.67 % | -6.84 pp | 36.74 % | -5.77 pp | 40.34 % | -2.17 pp |
| DRAM Throughput (%) | 21.21 % | 10.76 % | -10.45 pp | 12.42 % | -8.79 pp | 10.27 % | -10.94 pp |
| Achieved Occupancy (%) | 55.40 % | 50.66 % | -4.74 pp | 74.74 % | +19.34 pp | 55.57 % | +0.17 pp |
| Registers/Thread | 26.00 register/thread | 38.00 register/thread | +12 | 26.00 register/thread | +0 | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 0.99 | +0.25 | 1.48 | +0.74 | 0.91 | +0.17 |
| No Eligible (%) | 70.54 % | 53.68 % | -16.86 pp | 37.63 % | -32.91 pp | 57.04 % | -13.50 pp |
| IPC Active | 1.10 inst/cycle | 1.77 inst/cycle | +0.67 | 2.44 inst/cycle | +1.34 | 1.72 inst/cycle | +0.62 |


### Key Learnings

#### ❌ Plan D: `add_autotune_increase_tile_sizes` (0.485×)

Adding @triton.autotune with larger XBLOCK/YBLOCK configs caused a 2x slowdown despite the kernel having low wave count (0.74) and high L1TEX stall rate. The larger tiles increased shared memory per block from 4 KB to 16 KB, reducing theoretical occupancy from 100% to 81%, and tripled the executed instruction count (771K→2.19M) without improving achieved occupancy (55.4%→55.6%). The dominant stall shifted from L1TEX dependency (14.9 cycles/warp) to MIO queue saturation (6.9 cycles/warp), indicating the shared memory staging introduced by larger tiles created new pipeline pressure that outweighed any latency-hiding benefit. The grid remained fixed at the original hardcoded values rather than using the autotune lambda, so the larger tile configs ran with mismatched grid dimensions.

> **Applicable when**: Never applicable in this form. The implementation flaw (hardcoded grid not updated to lambda) means autotune selects larger tile configs but the grid remains sized for the original small tiles, causing out-of-bounds or redundant work.

> **Avoid when**: Do not add @triton.autotune with larger tile configs when: (1) the launch grid is not simultaneously updated to a lambda using meta['XBLOCK']/meta['YBLOCK']; (2) the kernel uses shared memory staging that grows with tile size and would reduce theoretical occupancy below achieved occupancy; (3) the kernel is already near the minimum achievable duration given its wave count — adding autotune overhead (compilation of N configs) dominates for short-running kernels.

---

## Iteration 2

**Best so far**: 1.096× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_vectorized_loads | 1.317× | 1.096× | ✓ | **✓ best** |
| B | fix_autotune_grid_lambda_and_reduce_register_pressure | 1.096× | 1.039× | ✓ |  |
| C | vectorized_loads_with_contiguity_hints | 0.810× | 0.510× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5820 ns | 5310 ns | -510 ns (-8.8%) | 5600 ns | -220 ns (-3.8%) | 11420 ns | +5600 ns (+96.2%) |
| SM Throughput (%) | 16.24 % | 35.52 % | +19.28 pp | 24.47 % | +8.23 pp | 30.99 % | +14.75 pp |
| Memory Throughput (%) | 42.51 % | 36.40 % | -6.11 pp | 36.07 % | -6.44 pp | 44.21 % | +1.70 pp |
| DRAM Throughput (%) | 21.21 % | 12.42 % | -8.79 pp | 11.77 % | -9.44 pp | 10.74 % | -10.47 pp |
| Achieved Occupancy (%) | 55.40 % | 74.63 % | +19.23 pp | 50.87 % | -4.53 pp | 55.43 % | +0.03 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 38.00 register/thread | +12 | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.99 | +0.25 | 0.91 | +0.17 |
| No Eligible (%) | 70.54 % | 39.02 % | -31.52 pp | 56.70 % | -13.84 pp | 52.64 % | -17.90 pp |
| IPC Active | 1.10 inst/cycle | 2.45 inst/cycle | +1.35 | 1.78 inst/cycle | +0.68 | 1.88 inst/cycle | +0.78 |


### Key Learnings

#### ❌ Plan C: `vectorized_loads_with_contiguity_hints` (0.510×)

Adding tl.max_contiguous and tl.multiple_of hints to the load index caused a 2x slowdown, but the root cause was not the hints themselves — the autotune config list was simultaneously replaced with a smaller set of configs that dropped num_stages, which caused the autotuner to select a smaller block size (128 vs 256) and fewer waves_per_sm (0.91 vs 1.48). This reduced theoretical occupancy from 100% to 81.25% and achieved occupancy from 74.7% to 55.4%, while also increasing executed instructions by 61% (1.48M → 2.38M) and worsening uncoalesced access from 20% to 35% excess sectors. The L1/TEX hit rate improved (26% → 50%) suggesting the hints did help cache reuse, but the occupancy collapse and instruction count increase dominated. When applying compiler hints, do not simultaneously change the autotune config space — isolate the hint change to avoid confounding the autotune selection.

> **Applicable when**: Never apply in this combined form. The confounding of autotune config changes with compiler hint additions makes it impossible to attribute the regression to either change alone.

> **Avoid when**: Do not simultaneously replace the autotune config list AND add tl.max_contiguous/tl.multiple_of hints in the same iteration. If the autotune config change selects a smaller block size (reducing waves_per_sm below 1.0), the occupancy collapse will dominate any benefit from vectorized loads. Additionally, do not remove num_stages from autotune configs when the kernel is latency-bound — num_stages controls software pipelining and its removal can increase MIO stalls (CPIStall MIO queue full rose to 31.1% of warp cycles after).

---

## Iteration 3

**Best so far**: 1.096× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_partial_wave_via_grid_reshape | 1.311× | 1.096× | ✓ |  |
| B | add_num_stages_software_pipelining_to_autotune_configs | 1.103× | 1.021× | ✓ |  |
| C | add_num_stages_to_autotune_configs_for_software_pipelining | 0.978× | 0.498× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_partial... | Δ | add_num_stages_so... | Δ | add_num_stages_to... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5820 ns | 5310 ns | -510 ns (-8.8%) | 5700 ns | -120 ns (-2.1%) | 11680 ns | +5860 ns (+100.7%) |
| SM Throughput (%) | 16.24 % | 34.29 % | +18.05 pp | 24.00 % | +7.76 pp | 30.15 % | +13.91 pp |
| Memory Throughput (%) | 42.51 % | 35.02 % | -7.49 pp | 35.34 % | -7.17 pp | 43.05 % | +0.54 pp |
| DRAM Throughput (%) | 21.21 % | 12.40 % | -8.81 pp | 11.58 % | -9.63 pp | 10.55 % | -10.66 pp |
| Achieved Occupancy (%) | 55.40 % | 76.29 % | +20.89 pp | 51.24 % | -4.16 pp | 55.62 % | +0.22 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 38.00 register/thread | +12 | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.99 | +0.25 | 0.91 | +0.17 |
| No Eligible (%) | 70.54 % | 38.37 % | -32.17 pp | 54.13 % | -16.41 pp | 53.26 % | -17.28 pp |
| IPC Active | 1.10 inst/cycle | 2.28 inst/cycle | +1.18 | 1.80 inst/cycle | +0.70 | 1.88 inst/cycle | +0.78 |


### Key Learnings

#### ❌ Plan C: `add_num_stages_to_autotune_configs_for_software_pipelining` (0.498×)

Replacing a working autotune config set (12 configs, num_stages=2/3, best selects XBLOCK=256/block_size=256, waves=1.48) with a different set of 8 configs (all num_stages=3 but different tile shapes) caused the autotuner to select a smaller block size (128 vs 256), collapsing waves_per_sm from 1.48 to 0.91, achieved occupancy from 74.6% to 55.6%, and increasing executed instructions by 61% (1.48M→2.38M). The regression is not caused by num_stages itself but by the replacement config set lacking the tile shape (XBLOCK=256, YBLOCK=16 with num_warps=4) that the previous autotuner had already converged on. Additionally, the kernel body change (replacing tl.max_contiguous/tl.multiple_of on xindex with a different form) worsened uncoalesced access from 20% to 35% excess sectors. When the autotuner selects a smaller block size, the grid becomes under-filled (waves<1), dominating all other optimizations.

> **Applicable when**: Never apply in this form. Replacing a converged autotune config set with a different set risks the autotuner selecting a suboptimal tile shape, especially when the new set has fewer configs or different tile shape coverage.

> **Avoid when**: Do not replace an existing autotune config set that has already converged to a good tile shape (waves_per_sm >= 1.0, achieved occupancy > 70%). If the goal is to add num_stages, ADD it to the existing configs rather than replacing them. Replacing configs can cause the autotuner to select a smaller block size, collapsing occupancy and waves_per_sm. Also do not simultaneously change the kernel body (index computation hints) when changing the autotune config set — isolate changes to avoid confounding.

---

## Iteration 4

**Best so far**: 1.096× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_staging_for_coalesced_stores | 1.314× | 1.082× | ✓ |  |
| B | streaming_store_eviction_policy_for_strided_writes | 1.131× | 1.058× | ✓ |  |
| C | restore_winning_autotune_configs_and_evict_first_store | 1.222× | 1.096× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_sta... | Δ | streaming_store_e... | Δ | restore_winning_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5820 ns | 5380 ns | -440 ns (-7.6%) | 5500 ns | -320 ns (-5.5%) | 5310 ns | -510 ns (-8.8%) |
| SM Throughput (%) | 16.24 % | 35.53 % | +19.29 pp | 25.04 % | +8.80 pp | 36.61 % | +20.37 pp |
| Memory Throughput (%) | 42.51 % | 35.21 % | -7.30 pp | 36.26 % | -6.25 pp | 36.23 % | -6.28 pp |
| DRAM Throughput (%) | 21.21 % | 12.24 % | -8.97 pp | 12.02 % | -9.19 pp | 12.42 % | -8.79 pp |
| Achieved Occupancy (%) | 55.40 % | 74.27 % | +18.87 pp | 50.80 % | -4.60 pp | 74.39 % | +18.99 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 38.00 register/thread | +12 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.99 | +0.25 | 1.48 | +0.74 |
| No Eligible (%) | 70.54 % | 36.19 % | -34.35 pp | 53.54 % | -17.00 pp | 37.78 % | -32.76 pp |
| IPC Active | 1.10 inst/cycle | 2.48 inst/cycle | +1.38 | 1.84 inst/cycle | +0.74 | 2.48 inst/cycle | +1.38 |


### Key Learnings

#### ✅ Plan C: `restore_winning_autotune_configs_and_evict_first_store` (1.096×)

When prior iterations have replaced a converged autotune config set and caused occupancy collapse, restoring the previously-winning config set (including the specific XBLOCK/YBLOCK/num_warps/num_stages combination that achieved the best waves_per_sm) recovers the original performance. In this case, restoring the 12-config set with XBLOCK=256/YBLOCK=8/num_warps=4/num_stages=3 maintained waves_per_sm=1.48 and achieved occupancy ~74-76%, while adding eviction_policy='evict_first' to the streaming store (each output element written exactly once) marginally improved IPC active from 2.28 to 2.48 and Issue Slots Busy from 57.9% to 62.9%. The speedup is modest (1.096x) because the kernel was already near its performance ceiling given the structural stride-3 store pattern causing persistent uncoalesced access.

> **Applicable when**: Triton kernel has been regressed by prior autotune config replacement (waves_per_sm dropped from >=1.4 to <1.0, occupancy collapsed from >70% to <60%). The fix is to restore the previously-winning config set. Additionally, apply eviction_policy='evict_first' to tl.store when each output element is written exactly once (streaming write with no reuse), to reduce L1 cache pollution.

> **Avoid when**: Do not apply evict_first to stores where the written data may be read back in the same kernel (e.g., reduction accumulators, in-place updates). Do not restore old autotune configs if the kernel body has changed in a way that makes the old tile shapes incorrect (e.g., different index computation). The speedup ceiling is limited by structural access pattern issues (e.g., stride-3 stores) that cannot be fixed by tile tuning alone.

---

## Cumulative Progress

```
Baseline:  5820 ns  (1.000×)
Iter 1     5310 ns  (1.096×)  — autotune_tile_sizes_for_coalescing_and_occupancy

Final best: 5310 ns  (1.0960×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'YBLOCK': 4, 'XBLOCK': 512}, num_warps=8, num_stages=3),
+    triton.Config({'YBLOCK': 4, 'XBLOCK': 256}, num_warps=4, num_stages=3),
+    triton.Config({'YBLOCK': 4, 'XBLOCK': 128}, num_warps=4, num_stages=2),
+    triton.Config({'YBLOCK': 4, 'XBLOCK': 64}, num_warps=2, num_stages=2),
+    triton.Config({'YBLOCK': 8, 'XBLOCK': 512}, num_warps=8, num_stages=3),
+    triton.Config({'YBLOCK': 8, 'XBLOCK': 256}, num_warps=4, num_stages=3),
+    triton.Config({'YBLOCK': 8, 'XBLOCK': 128}, num_warps=4, num_stages=2),
+    triton.Config({'YBLOCK': 8, 'XBLOCK': 64}, num_warps=2, num_stages=2),
+    triton.Config({'YBLOCK': 16, 'XBLOCK': 256}, num_warps=4, num_stages=3),
+    triton.Config({'YBLOCK': 16, 'XBLOCK': 128}, num_warps=4, num_stages=2),
+    triton.Config({'YBLOCK': 32, 'XBLOCK': 128}, num_warps=4, num_stages=2),
+    triton.Config({'YBLOCK': 32, 'XBLOCK': 256}, num_warps=8, num_stages=3),
+    triton.Config({'YBLOCK': 32, 'XBLOCK': 512}, num_warps=8, num_stages=3),
+    triton.Config({'YBLOCK': 32, 'XBLOCK': 64}, num_warps=4, num_stages=3),
+    triton.Config({'YBLOCK': 32, 'XBLOCK': 64}, num_warps=2, num_stages=2),
+], key=['ynumel', 'xnumel'])
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +31,8 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xarange = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
+    xindex = xarange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +41,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 30
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
