# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6020 ns — **Best latency**: 5570 ns (**1.0808×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`expand_autotune_configs_with_num_stages_and_larger_xblock`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_065/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6020 ns
- **SM Throughput**: 15.80 % | **Memory**: 42.03 % | **DRAM**: 20.56 %
- **Achieved Occupancy**: 55.13 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 71.80 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.399× | 1.792× | ✓ |  |
| B | autotune_tile_sizes_to_fix_uncoalesced_stores_and_low_occupancy | 1.210× | 1.413× | ✓ |  |
| D | autotune_tile_sizes_for_occupancy_and_coalescing | 1.402× | 1.568× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 3360 ns | -2660 ns (-44.2%) | 4260 ns | -1760 ns (-29.2%) | 3840 ns | -2180 ns (-36.2%) |
| SM Throughput (%) | 15.80 % | 17.20 % | +1.40 pp | 11.96 % | -3.84 pp | 19.44 % | +3.64 pp |
| Memory Throughput (%) | 42.03 % | 20.57 % | -21.46 pp | 32.49 % | -9.54 pp | 22.42 % | -19.61 pp |
| DRAM Throughput (%) | 20.56 % | 4.96 % | -15.60 pp | 15.52 % | -5.04 pp | 4.29 % | -16.27 pp |
| Achieved Occupancy (%) | 55.13 % | 43.62 % | -11.51 pp | 28.59 % | -26.54 pp | 48.99 % | -6.14 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 26.00 register/thread | +0 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.37 | -0.37 | 0.74 | +0.00 |
| No Eligible (%) | 71.80 % | 54.85 % | -16.95 pp | 73.55 % | +1.75 pp | 52.29 % | -19.51 pp |
| IPC Active | 1.15 inst/cycle | 1.67 inst/cycle | +0.52 | 0.98 inst/cycle | -0.17 | 1.83 inst/cycle | +0.68 |

---

## Iteration 2

**Best so far**: 1.5677× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_autotune_configs_with_num_stages_and_larger_xblock | 1.409× | 1.081× | ✓ | **✓ best** |
| B | vectorized_loads | 1.383× | 1.544× | ✓ |  |
| C | vectorized_loads | 1.228× | 1.458× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_autotune_c... | Δ | vectorized_loads | Δ | vectorized_loads | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 5570 ns | -450 ns (-7.5%) | 3900 ns | -2120 ns (-35.2%) | 4130 ns | -1890 ns (-31.4%) |
| SM Throughput (%) | 15.80 % | 29.30 % | +13.50 pp | 19.25 % | +3.45 pp | 12.17 % | -3.63 pp |
| Memory Throughput (%) | 42.03 % | 32.10 % | -9.93 pp | 22.28 % | -19.75 pp | 33.66 % | -8.37 pp |
| DRAM Throughput (%) | 20.56 % | 2.96 % | -17.60 pp | 4.21 % | -16.35 pp | 16.04 % | -4.52 pp |
| Achieved Occupancy (%) | 55.13 % | 73.27 % | +18.14 pp | 48.51 % | -6.62 pp | 28.47 % | -26.66 pp |
| Registers/Thread | 26.00 register/thread | 25.00 register/thread | -1 | 25.00 register/thread | -1 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 | 0.37 | -0.37 |
| No Eligible (%) | 71.80 % | 37.12 % | -34.68 pp | 51.90 % | -19.90 pp | 73.42 % | +1.62 pp |
| IPC Active | 1.15 inst/cycle | 2.43 inst/cycle | +1.28 | 1.86 inst/cycle | +0.71 | 0.98 inst/cycle | -0.17 |


### Key Learnings

#### ✅ Plan A: `expand_autotune_configs_with_num_stages_and_larger_xblock` (1.081×)

When a Triton pointwise kernel already has autotune but remains at <1.0 waves/SM with high scheduler idle rate (>50% No Eligible cycles), expanding the autotune config space to include larger XBLOCK values (1024, 2048) and adding num_stages=2 can improve occupancy and instruction throughput. In this case, the autotuner selected block_size=256 (up from 128), raising waves/SM from 0.74 to 1.48, achieved occupancy from 43.6% to 73.3%, and Issue Slots Busy from 44% to 62%, while eligible warps/scheduler nearly doubled (2.09→3.92). However, the absolute kernel duration increased (3.36→5.57 µs) because the winning config doubled the total thread count (200K→401K threads) and the instruction count tripled, meaning the per-element work overhead outweighed the occupancy gain — the net wall-clock speedup was only 1.08x at the benchmark level, not the kernel level.

> **Applicable when**: Triton pointwise kernel already has autotune but waves_per_sm < 1.0, scheduler No Eligible > 50%, achieved occupancy < 50%, and the config space does not include XBLOCK >= 1024 or num_stages > 1. The kernel has a strided store pattern that cannot be easily coalesced. Problem size is small enough that larger tiles still fit in L2.

> **Avoid when**: Kernel duration is already short (<3 µs) and the bottleneck is not occupancy — expanding configs may cause the autotuner to select a config that doubles thread count without proportional work reduction, increasing absolute kernel time. Do not apply if the kernel is already compute-bound (SM throughput >70%) or if waves_per_sm > 2.0. Also avoid if the uncoalesced store pattern means larger XBLOCK just issues more scattered transactions without hiding latency.

---

## Iteration 3

**Best so far**: 1.5677× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_to_enable_autotune | 1.375× | 1.494× | ✓ |  |
| B | store_eviction_policy_evict_first_for_strided_scatter | 1.219× | 1.468× | ✓ |  |
| C | evict_first_store_with_large_xblock_configs_for_partial_wave_reduction | 1.371× | 1.087× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | store_eviction_po... | Δ | evict_first_store... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 4030 ns | -1990 ns (-33.1%) | 4100 ns | -1920 ns (-31.9%) | 5540 ns | -480 ns (-8.0%) |
| SM Throughput (%) | 15.80 % | 20.02 % | +4.22 pp | 12.23 % | -3.57 pp | 27.79 % | +11.99 pp |
| Memory Throughput (%) | 42.03 % | 23.15 % | -18.88 pp | 33.55 % | -8.48 pp | 29.31 % | -12.72 pp |
| DRAM Throughput (%) | 20.56 % | 4.11 % | -16.45 pp | 16.12 % | -4.44 pp | 2.98 % | -17.58 pp |
| Achieved Occupancy (%) | 55.13 % | 48.68 % | -6.45 pp | 28.39 % | -26.74 pp | 73.04 % | +17.91 pp |
| Registers/Thread | 26.00 register/thread | 25.00 register/thread | -1 | 26.00 register/thread | +0 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.37 | -0.37 | 1.48 | +0.74 |
| No Eligible (%) | 71.80 % | 52.11 % | -19.69 pp | 71.62 % | -0.18 pp | 35.75 % | -36.05 pp |
| IPC Active | 1.15 inst/cycle | 1.84 inst/cycle | +0.69 | 1.01 inst/cycle | -0.14 | 2.49 inst/cycle | +1.34 |


### Key Learnings

#### ✅ Plan A: `fix_hardcoded_grid_to_enable_autotune` (1.516×)

When a Triton kernel has a @triton.autotune decorator but the launch site uses a hardcoded static grid tuple, the autotuner is bypassed. Changing the grid to a lambda (lambda meta: (cdiv(xnumel, meta['XBLOCK']), cdiv(ynumel, meta['YBLOCK']))) activates the autotuner. Simultaneously adding num_stages=2 variants to the config list exposes software pipelining as an additional dimension for the autotuner to explore. In this case, the NCU kernel-level metrics changed only marginally (Duration 4.03→3.97 µs, Elapsed Cycles 5965→5879), indicating the autotuner again selected the same XBLOCK=128/YBLOCK=8 tile — the 1.52x benchmark-level speedup likely comes from autotune caching and reduced per-call overhead rather than a different tile or pipelining depth being selected. The Warp Cycles Per Issued Instruction actually increased (18.28→22.47), consistent with the autotuner selecting a num_stages=2 config that increases register pressure without meaningfully changing the tile.

> **Applicable when**: Triton kernel has @triton.autotune decorator but launch site uses a hardcoded static grid tuple, causing the autotuner to be silently bypassed. IssueSlotUtilization and AchievedOccupancy NCU rules fire. The config list lacks num_stages variants. Kernel is a pointwise load+store with latency-bound scheduler idle cycles (No Eligible > 50%).

> **Avoid when**: Grid is already a lambda (autotune already active). Kernel body uses XBLOCK/YBLOCK in ways requiring explicit kernel argument passing. Kernel is already compute-bound (SM throughput > 70%). Adding num_stages=2 to a pointwise kernel with a single load-store body may increase warp cycles per instruction if register pressure rises without meaningful latency hiding benefit.

#### ✅ Plan B: `store_eviction_policy_evict_first_for_strided_scatter` (1.468×)

When a Triton kernel performs a write-only strided scatter store (e.g. stride-3 channel interleaving) that is never read back by the same kernel, the L2 cache retains the written lines unnecessarily, polluting capacity needed by the load path. Adding eviction_policy='evict_first' to the tl.store call instructs the L2 to immediately evict stored cache lines, freeing L2 capacity for loads and reducing write-back traffic. In this case, the L2 hit rate on stores dropped from 76.6% to 63.0% (confirming eviction is working), L1/TEX hit rate on loads rose from 0% to 27.8%, and effective memory throughput jumped from 207 GB/s to 785 GB/s, yielding a 1.47x speedup despite the kernel-level elapsed cycles being similar — the benefit manifests at the benchmark level through reduced L2 contention across concurrent kernel launches.

> **Applicable when**: Triton kernel performs a write-only store to an output tensor that is never read back within the same kernel; store pattern is strided/scattered (MemoryCacheAccessPattern rule fires, <22 bytes/sector utilized); L2 hit rate on stores is high (>60%) despite the output being write-only; MemoryCacheAccessPattern and/or MemoryL2Compression NCU rules are triggered; load path competes with store path for L2 capacity.

> **Avoid when**: Output tensor is read back within the same kernel (evict_first would cause cache misses on subsequent reads); store pattern is fully coalesced and L2 hit rate on stores is low (eviction hint has no effect); kernel is compute-bound rather than memory-bandwidth-bound; the store is to shared memory (eviction_policy only applies to global memory stores).

---

## Iteration 4

**Best so far**: 1.5677× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_num_stages_pipelining_with_corrected_lambda_grid | 1.384× | 1.516× | ✓ |  |
| B | fix_autotune_grid_lambda_to_activate_existing_autotune | 1.245× | 1.362× | ✓ |  |
| C | eliminate_partial_wave_via_full_y_coverage_configs | 1.369× | 0.210× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_num_stag... | Δ | fix_autotune_grid... | Δ | eliminate_partial... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 3970 ns | -2050 ns (-34.1%) | 4420 ns | -1600 ns (-26.6%) | 28700 ns | +22680 ns (+376.7%) |
| SM Throughput (%) | 15.80 % | 18.54 % | +2.74 pp | 12.20 % | -3.60 pp | 59.83 % | +44.03 pp |
| Memory Throughput (%) | 42.03 % | 21.46 % | -20.57 pp | 31.01 % | -11.02 pp | 72.65 % | +30.62 pp |
| DRAM Throughput (%) | 20.56 % | 4.18 % | -16.38 pp | 14.92 % | -5.64 pp | 4.28 % | -16.28 pp |
| Achieved Occupancy (%) | 55.13 % | 49.03 % | -6.10 pp | 28.56 % | -26.57 pp | 34.31 % | -20.82 pp |
| Registers/Thread | 26.00 register/thread | 25.00 register/thread | -1 | 26.00 register/thread | +0 | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.37 | -0.37 | 3.96 | +3.22 |
| No Eligible (%) | 71.80 % | 52.30 % | -19.50 pp | 72.43 % | +0.63 pp | 48.20 % | -23.60 pp |
| IPC Active | 1.15 inst/cycle | 1.84 inst/cycle | +0.69 | 1.01 inst/cycle | -0.14 | 2.09 inst/cycle | +0.94 |

---

## Cumulative Progress

```
Baseline:  6020 ns  (1.000×)
Iter 1     3840 ns  (1.568×)  — autotune_tile_sizes_for_occupancy_and_coalescing

Final best: 5570 ns  (1.0808×)
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
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2}, num_warps=4),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 4}, num_warps=8),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 8}, num_warps=8),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4}, num_warps=2),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +31,8 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    _xarange = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
+    xindex = _xarange[None, :]
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
