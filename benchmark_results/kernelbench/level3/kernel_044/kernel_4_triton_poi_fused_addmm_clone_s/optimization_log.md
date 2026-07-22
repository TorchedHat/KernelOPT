# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_clone_split_transpose_view_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 244100 ns — **Best latency**: 243940 ns (**1.0007×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`persistent_tile_scheduling`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_addmm_clone_split_transpose_view_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_044/kernel_4_triton_poi_fused_addmm_clone_s/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (393216,1,1), Waves/SM: 186.18
- **Registers/thread**: 16
- **Duration**: 244100 ns
- **SM Throughput**: 35.02 % | **Memory**: 32.01 % | **DRAM**: 32.01 %
- **Achieved Occupancy**: 43.82 %
- **IPC Active**: 1.42 inst/cycle | **No Eligible**: 64.62 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_reduce_wave_overhead | 0.877× | 0.311× | ✓ |  |
| B | increase_xblock_tile_size_with_autotune | 0.877× | 0.548× | ✓ |  |
| C | increase_xblock_with_autotune_to_hide_l1tex_latency | 0.877× | 0.886× | ✓ |  |
| D | vectorized_loads_stores_via_larger_xblock_autotune | 0.998× | 0.549× | ✓ |  |
| E | add_triton_autotune | 1.000× | 1.001× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | increase_xblock_t... | Δ | increase_xblock_w... | Δ | vectorized_loads_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 244100 ns | 785340 ns | +541240 ns (+221.7%) | 445150 ns | +201050 ns (+82.4%) | 275490 ns | +31390 ns (+12.9%) | 444770 ns | +200670 ns (+82.2%) | 243940 ns | -160 ns (-0.1%) |
| SM Throughput (%) | 35.02 % | 72.81 % | +37.79 pp | 72.52 % | +37.50 pp | 38.05 % | +3.03 pp | 72.50 % | +37.48 pp | 37.70 % | +2.68 pp |
| Memory Throughput (%) | 32.01 % | 23.52 % | -8.49 pp | 27.03 % | -4.98 pp | 28.09 % | -3.92 pp | 27.01 % | -5.00 pp | 32.03 % | +0.02 pp |
| DRAM Throughput (%) | 32.01 % | 9.81 % | -22.20 pp | 17.36 % | -14.65 pp | 28.09 % | -3.92 pp | 17.37 % | -14.64 pp | 32.03 % | +0.02 pp |
| Achieved Occupancy (%) | 43.82 % | 68.29 % | +24.47 pp | 76.05 % | +32.23 pp | 35.17 % | -8.65 pp | 76.42 % | +32.60 pp | 44.41 % | +0.59 pp |
| Registers/Thread | 16.00 register/thread | 32.00 register/thread | +16 | 28.00 register/thread | +12 | 18.00 register/thread | +2 | 28.00 register/thread | +12 | 16.00 register/thread | +0 |
| Waves/SM | 186.18 | 372.36 | +186.18 | 372.36 | +186.18 | 186.18 | +0.00 | 372.36 | +186.18 | 186.18 | +0.00 |
| No Eligible (%) | 64.62 % | 26.92 % | -37.70 pp | 26.77 % | -37.85 pp | 59.35 % | -5.27 pp | 26.80 % | -37.82 pp | 62.20 % | -2.42 pp |
| IPC Active | 1.42 inst/cycle | 2.91 inst/cycle | +1.49 | 2.94 inst/cycle | +1.52 | 1.57 inst/cycle | +0.15 | 2.92 inst/cycle | +1.50 | 1.52 inst/cycle | +0.10 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune_to_reduce_wave_overhead` (0.311×)

Adding @triton.autotune with larger XBLOCK configs (512–4096) to a memory-bound elementwise kernel caused a 3.2x slowdown despite improving scheduler utilization (No Eligible dropped from 64.6% to 26.9%). The autotuner selected a 256-thread block config that doubled registers per thread (16→32) and increased executed instructions by 6.6x (67M→447M), indicating the autotune configs introduced excessive per-thread computation or unrolling overhead. When the grid size is already large (393K CTAs) and the kernel is genuinely memory-latency-bound with L1TEX stalls, increasing XBLOCK via autotune can backfire by selecting configs that inflate register pressure and instruction count without proportionally improving memory throughput, resulting in a compute-bound regression.

> **Applicable when**: Kernel has very few waves per SM (< 5), small block sizes (< 64 threads), and is genuinely compute-starved with near-zero memory throughput. Autotune XBLOCK increase is appropriate when the bottleneck is truly grid fragmentation with tiny per-CTA work.

> **Avoid when**: The kernel already has high wave count (>100 waves/SM) and the primary stall is L1TEX memory latency — in this case the grid is not fragmented, and increasing XBLOCK inflates register pressure and instruction count without improving memory throughput. Also avoid when the kernel is a simple elementwise load-add-store where larger tiles just serialize more work per thread without exposing new parallelism.

#### ❌ Plan B: `increase_xblock_tile_size_with_autotune` (0.548×)

Doubling the block size from 128 to 256 (via autotune selecting a larger XBLOCK config) caused registers per thread to jump from 16 to 28 and executed instructions to balloon from 67M to 252M — a 3.7x increase — despite the same total work. The root cause is that adding `xmask = xindex < xnumel` (replacing `tl.full([XBLOCK], True)`) introduced predicated masking on every load and store, generating substantially more instructions per thread. Although occupancy improved (43.8% → 76%) and eligible warps per scheduler rose (0.46 → 5.49), the instruction count explosion dominated, causing a 1.82x slowdown. For pointwise kernels where xnumel is exactly divisible by all candidate XBLOCK values, replacing `tl.full([XBLOCK], True)` with a runtime bounds check is an anti-pattern that adds unnecessary predication overhead.

> **Applicable when**: Kernel is latency-bound with low eligible warps per scheduler (<1.0) and high L1TEX stall fraction (>70% of CPI stall), and xnumel is exactly divisible by all candidate XBLOCK values so masking is provably unnecessary.

> **Avoid when**: The autotune configs cause the compiler to introduce runtime bounds-check masking (xindex < xnumel) where tl.full([XBLOCK], True) was previously used — this inflates instruction count by 3-4x and causes severe regression even if occupancy improves. Also avoid when register pressure increases significantly (>16 regs/thread baseline) as larger blocks may spill.

---

## Iteration 2

**Best so far**: 1.0007× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 1.001× | 1.001× | ✓ | **✓ best** |
| B | persistent_tile_scheduling | 0.999× | 1.000× | ✓ |  |
| C | persistent_tile_scheduling | 1.000× | 1.000× | ✓ |  |
| D | persistent_tile_scheduling | 1.000× | 1.000× | ✓ |  |
| E | add_triton_autotune | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 244100 ns | 243940 ns | -160 ns (-0.1%) | 244130 ns | +30 ns (+0.0%) | 244060 ns | -40 ns (-0.0%) | 244130 ns | +30 ns (+0.0%) | 244060 ns | -40 ns (-0.0%) |
| SM Throughput (%) | 35.02 % | 48.84 % | +13.82 pp | 43.94 % | +8.92 pp | 40.94 % | +5.92 pp | 45.90 % | +10.88 pp | 37.27 % | +2.25 pp |
| Memory Throughput (%) | 32.01 % | 31.46 % | -0.55 pp | 32.02 % | +0.01 pp | 32.04 % | +0.03 pp | 32.03 % | +0.02 pp | 32.02 % | +0.01 pp |
| DRAM Throughput (%) | 32.01 % | 31.46 % | -0.55 pp | 32.02 % | +0.01 pp | 32.04 % | +0.03 pp | 32.03 % | +0.02 pp | 32.02 % | +0.01 pp |
| Achieved Occupancy (%) | 43.82 % | 46.95 % | +3.13 pp | 46.89 % | +3.07 pp | 46.41 % | +2.59 pp | 47.76 % | +3.94 pp | 44.47 % | +0.65 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 |
| No Eligible (%) | 64.62 % | 50.69 % | -13.93 pp | 55.37 % | -9.25 pp | 58.88 % | -5.74 pp | 54.05 % | -10.57 pp | 62.24 % | -2.38 pp |
| IPC Active | 1.42 inst/cycle | 1.99 inst/cycle | +0.57 | 1.79 inst/cycle | +0.37 | 1.64 inst/cycle | +0.22 | 1.84 inst/cycle | +0.42 | 1.52 inst/cycle | +0.10 |

---

## Iteration 3

**Best so far**: 1.0007× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesced_memory_access_via_index_reorder | 0.998× | 0.999× | ✓ |  |
| B | cache_eviction_policy_streaming_load_to_improve_bias_reuse | 1.000× | 1.000× | ✓ |  |
| C | streaming_eviction_policy_for_non_reused_large_tensor_load | 1.000× | 0.999× | ✓ |  |
| D | streaming_eviction_policy_for_large_matrix_load | 0.999× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesced_memory_... | Δ | cache_eviction_po... | Δ | streaming_evictio... | Δ | streaming_evictio... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 244100 ns | 244350 ns | +250 ns (+0.1%) | 244030 ns | -70 ns (-0.0%) | 244260 ns | +160 ns (+0.1%) | 244350 ns | +250 ns (+0.1%) |
| SM Throughput (%) | 35.02 % | 60.03 % | +25.01 pp | 44.76 % | +9.74 pp | 48.82 % | +13.80 pp | 49.62 % | +14.60 pp |
| Memory Throughput (%) | 32.01 % | 31.39 % | -0.62 pp | 31.45 % | -0.56 pp | 31.44 % | -0.57 pp | 31.40 % | -0.61 pp |
| DRAM Throughput (%) | 32.01 % | 31.39 % | -0.62 pp | 31.45 % | -0.56 pp | 31.44 % | -0.57 pp | 31.40 % | -0.61 pp |
| Achieved Occupancy (%) | 43.82 % | 52.42 % | +8.60 pp | 46.10 % | +2.28 pp | 46.88 % | +3.06 pp | 47.80 % | +3.98 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 |
| No Eligible (%) | 64.62 % | 39.38 % | -25.24 pp | 54.55 % | -10.07 pp | 50.34 % | -14.28 pp | 49.56 % | -15.06 pp |
| IPC Active | 1.42 inst/cycle | 2.41 inst/cycle | +0.99 | 1.81 inst/cycle | +0.39 | 1.98 inst/cycle | +0.56 | 2.02 inst/cycle | +0.60 |

---

## Iteration 4

**Best so far**: 1.0007× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_fp32_add_to_fma_instruction | 1.000× | 1.000× | ✓ |  |
| B | increase_num_warps_to_hide_l1tex_latency | 1.000× | 1.000× | ✓ |  |
| C | shared_memory_preload_bias_to_eliminate_l1tex_stalls | 0.999× | 0.999× | ✓ |  |
| D | shared_memory_preload_bias_to_eliminate_l1tex_stalls | 0.997× | 0.997× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_fp32_add_to_... | Δ | increase_num_warp... | Δ | shared_memory_pre... | Δ | shared_memory_pre... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 244100 ns | 243970 ns | -130 ns (-0.1%) | 244130 ns | +30 ns (+0.0%) | 244320 ns | +220 ns (+0.1%) | 244770 ns | +670 ns (+0.3%) |
| SM Throughput (%) | 35.02 % | 45.53 % | +10.51 pp | 49.14 % | +14.12 pp | 59.66 % | +24.64 pp | 59.69 % | +24.67 pp |
| Memory Throughput (%) | 32.01 % | 31.45 % | -0.56 pp | 31.42 % | -0.59 pp | 31.40 % | -0.61 pp | 67.36 % | +35.35 pp |
| DRAM Throughput (%) | 32.01 % | 31.45 % | -0.56 pp | 31.42 % | -0.59 pp | 31.40 % | -0.61 pp | 31.36 % | -0.65 pp |
| Achieved Occupancy (%) | 43.82 % | 46.21 % | +2.39 pp | 46.80 % | +2.98 pp | 52.31 % | +8.49 pp | 54.21 % | +10.39 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 22.00 register/thread | +6 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 |
| No Eligible (%) | 64.62 % | 53.68 % | -10.94 pp | 50.35 % | -14.27 pp | 39.36 % | -25.26 pp | 39.68 % | -24.94 pp |
| IPC Active | 1.42 inst/cycle | 1.85 inst/cycle | +0.43 | 1.97 inst/cycle | +0.55 | 2.41 inst/cycle | +0.99 | 2.41 inst/cycle | +0.99 |

---

## Cumulative Progress

```
Baseline:  244100 ns  (1.000×)
Iter 1     243940 ns  (1.001×)  — add_triton_autotune

Final best: 243940 ns  (1.0007×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,23 +5,26 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
-def triton_poi_fused_addmm_clone_split_transpose_view_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
+def triton_poi_fused_addmm_clone_split_transpose_view_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
     xnumel = 50331648
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
-    x0 = (xindex % 96)
-    x2 = ((xindex // 49152) % 8)
-    x1 = ((xindex // 96) % 512)
-    x3 = xindex // 393216
-    x4 = xindex
-    tmp0 = tl.load(in_ptr0 + (1536 + x0 + 96*x2), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (1536 + x0 + 96*x2 + 2304*x1 + 1179648*x3), None)
-    tmp2 = tmp0 + tmp1
-    tl.store(out_ptr0 + (x4), tmp2, None)
-
-
+    num_tiles = tl.cdiv(xnumel, XBLOCK)
+    num_sms = tl.num_programs(0)
+    tile_id = tl.program_id(0)
+    while tile_id < num_tiles:
+        xoffset = tile_id * XBLOCK
+        xindex = xoffset + tl.arange(0, XBLOCK)[:]
+        x0 = (xindex % 96)
+        x2 = ((xindex // 49152) % 8)
+        x1 = ((xindex // 96) % 512)
+        x3 = xindex // 393216
+        x4 = xindex
+        tmp0 = tl.load(in_ptr0 + (1536 + x0 + 96*x2), eviction_policy='evict_last')
+        tmp1 = tl.load(in_ptr1 + (1536 + x0 + 96*x2 + 2304*x1 + 1179648*x3), eviction_policy='evict_first')
+        tmp2 = tmp0 + tmp1
+        tl.store(out_ptr0 + x4, tmp2)
+        tile_id += num_sms
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
