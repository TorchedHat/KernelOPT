# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_gelu_view_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 60100 ns — **Best latency**: 60100 ns (**1.0000×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_addmm_gelu_view_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_4_triton_poi_fused_addmm_gelu_vi/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (94080,1,1), Waves/SM: 44.55
- **Registers/thread**: 16
- **Duration**: 60100 ns
- **SM Throughput**: 47.03 % | **Memory**: 23.53 % | **DRAM**: 23.53 %
- **Achieved Occupancy**: 45.88 %
- **IPC Active**: 1.94 inst/cycle | **No Eligible**: 51.26 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_increase_tile_size | 0.647× | 0.085× | ✓ |  |
| B | increase_xblock_tile_size_with_autotune | 0.998× | 0.164× | ✓ |  |
| C | increase_xblock_with_autotune_to_hide_l1tex_latency | 0.648× | 0.642× | ✓ |  |
| D | increase_xblock_for_vectorized_loads_and_latency_hiding | 0.648× | 0.637× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_i... | Δ | increase_xblock_t... | Δ | increase_xblock_w... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 60100 ns | 707840 ns | +647740 ns (+1077.8%) | 367170 ns | +307070 ns (+510.9%) | 93660 ns | +33560 ns (+55.8%) | 94300 ns | +34200 ns (+56.9%) |
| SM Throughput (%) | 47.03 % | 83.72 % | +36.69 pp | 84.75 % | +37.72 pp | 82.95 % | +35.92 pp | 83.77 % | +36.74 pp |
| Memory Throughput (%) | 23.53 % | 7.23 % | -16.30 pp | 7.77 % | -15.76 pp | 15.09 % | -8.44 pp | 15.01 % | -8.52 pp |
| DRAM Throughput (%) | 23.53 % | 2.00 % | -21.53 pp | 3.87 % | -19.66 pp | 15.09 % | -8.44 pp | 15.01 % | -8.52 pp |
| Achieved Occupancy (%) | 45.88 % | 82.50 % | +36.62 pp | 81.42 % | +35.54 pp | 85.78 % | +39.90 pp | 85.77 % | +39.89 pp |
| Registers/Thread | 16.00 register/thread | 32.00 register/thread | +16 | 28.00 register/thread | +12 | 28.00 register/thread | +12 | 28.00 register/thread | +12 |
| Waves/SM | 44.55 | 178.18 | +133.63 | 178.18 | +133.63 | 44.55 | +0.00 | 44.55 | +0.00 |
| No Eligible (%) | 51.26 % | 15.89 % | -35.37 pp | 14.91 % | -36.35 pp | 15.01 % | -36.25 pp | 14.36 % | -36.90 pp |
| IPC Active | 1.94 inst/cycle | 3.36 inst/cycle | +1.42 | 3.41 inst/cycle | +1.47 | 3.40 inst/cycle | +1.46 | 3.42 inst/cycle | +1.48 |


### Key Learnings

#### ❌ Plan C: `increase_xblock_with_autotune_to_hide_l1tex_latency` (0.642×)

Adding @triton.autotune with larger XBLOCK configs (512, 1024) increased registers per thread from 16 to 28, which halved the number of blocks that can reside per SM and inflated executed instructions by 2.77x (22M → 61.5M). Although occupancy improved (45.9% → 85.8%) and scheduler eligibility improved (51% no-eligible → 15%), the massive instruction count increase dominated, causing duration to increase from 60.1 µs to 93.7 µs. The strategy of using larger tiles to hide L1TEX latency is counterproductive when the autotune-selected config causes significant register pressure growth, as the extra instructions and register spill overhead outweigh the latency-hiding benefit.

> **Applicable when**: Never apply this specific pattern (autotune XBLOCK 512/1024 on a kernel with 16 regs/thread that is L1TEX stall-bound) — it causes register pressure growth and instruction bloat that outweigh latency-hiding gains.

> **Avoid when**: Do not increase XBLOCK via autotune when: (1) the kernel already has low register count (≤16/thread) but autotune causes it to jump significantly (e.g., to 28+), (2) the instruction count grows proportionally with XBLOCK without a corresponding compute throughput gain, or (3) the kernel is a simple pointwise op where the compute-to-memory ratio is already low — larger tiles just serialize more work per block without hiding more latency.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 1.001× | 1.000× | ✓ |  |
| B | persistent_tile_scheduling | 1.002× | 0.998× | ✓ |  |
| C | persistent_tile_scheduling_fix_broken_autotune_grid | 1.000× | 0.995× | ✓ |  |
| D | fix_autotune_grid_lambda_to_eliminate_redundant_cta_work | 0.647× | 0.085× | ✓ |  |
| E | add_triton_autotune | 1.002× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | fix_autotune_grid... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 60100 ns | 60130 ns | +30 ns (+0.0%) | 60220 ns | +120 ns (+0.2%) | 60380 ns | +280 ns (+0.5%) | 707650 ns | +647550 ns (+1077.5%) | 60100 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 47.03 % | 59.47 % | +12.44 pp | 59.25 % | +12.22 pp | 53.96 % | +6.93 pp | 83.76 % | +36.73 pp | 50.27 % | +3.24 pp |
| Memory Throughput (%) | 23.53 % | 23.57 % | +0.04 pp | 23.49 % | -0.04 pp | 23.41 % | -0.12 pp | 7.23 % | -16.30 pp | 23.56 % | +0.03 pp |
| DRAM Throughput (%) | 23.53 % | 23.57 % | +0.04 pp | 23.49 % | -0.04 pp | 23.41 % | -0.12 pp | 2.00 % | -21.53 pp | 23.56 % | +0.03 pp |
| Achieved Occupancy (%) | 45.88 % | 50.77 % | +4.89 pp | 50.56 % | +4.68 pp | 48.24 % | +2.36 pp | 82.31 % | +36.43 pp | 46.26 % | +0.38 pp |
| Registers/Thread | 16.00 register/thread | 17.00 register/thread | +1 | 18.00 register/thread | +2 | 16.00 register/thread | +0 | 32.00 register/thread | +16 | 16.00 register/thread | +0 |
| Waves/SM | 44.55 | 44.55 | +0.00 | 44.55 | +0.00 | 44.55 | +0.00 | 178.18 | +133.63 | 44.55 | +0.00 |
| No Eligible (%) | 51.26 % | 37.78 % | -13.48 pp | 38.16 % | -13.10 pp | 43.85 % | -7.41 pp | 15.89 % | -35.37 pp | 48.21 % | -3.05 pp |
| IPC Active | 1.94 inst/cycle | 2.47 inst/cycle | +0.53 | 2.47 inst/cycle | +0.53 | 2.24 inst/cycle | +0.30 | 3.36 inst/cycle | +1.42 | 2.07 inst/cycle | +0.13 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | preload_bias_into_shared_memory_to_eliminate_l1tex_stalls | 0.996× | 0.993× | ✓ |  |
| B | increase_num_warps_to_improve_scheduler_eligibility | 0.999× | 0.999× | ✓ |  |
| C | increase_num_warps_to_hide_l1tex_stall_latency | 1.001× | 0.998× | ✓ |  |
| D | rewrite_gelu_arithmetic_to_maximize_fma_utilization | 0.637× | 0.086× | ✓ |  |


### Metric Comparison

| Metric | Baseline | preload_bias_into... | Δ | increase_num_warp... | Δ | increase_num_warp... | Δ | rewrite_gelu_arit... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 60100 ns | 60540 ns | +440 ns (+0.7%) | 60190 ns | +90 ns (+0.1%) | 60220 ns | +120 ns (+0.2%) | 696740 ns | +636640 ns (+1059.3%) |
| SM Throughput (%) | 47.03 % | 53.72 % | +6.69 pp | 59.43 % | +12.40 pp | 54.26 % | +7.23 pp | 84.90 % | +37.87 pp |
| Memory Throughput (%) | 23.53 % | 39.50 % | +15.97 pp | 23.54 % | +0.01 pp | 23.55 % | +0.02 pp | 7.33 % | -16.20 pp |
| DRAM Throughput (%) | 23.53 % | 23.39 % | -0.14 pp | 23.54 % | +0.01 pp | 23.55 % | +0.02 pp | 2.04 % | -21.49 pp |
| Achieved Occupancy (%) | 45.88 % | 49.31 % | +3.43 pp | 50.93 % | +5.05 pp | 48.31 % | +2.43 pp | 87.13 % | +41.25 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 17.00 register/thread | +1 | 16.00 register/thread | +0 | 32.00 register/thread | +16 |
| Waves/SM | 44.55 | 44.55 | +0.00 | 44.55 | +0.00 | 44.55 | +0.00 | 178.18 | +133.63 |
| No Eligible (%) | 51.26 % | 44.04 % | -7.22 pp | 37.78 % | -13.48 pp | 43.88 % | -7.38 pp | 14.78 % | -36.48 pp |
| IPC Active | 1.94 inst/cycle | 2.23 inst/cycle | +0.29 | 2.48 inst/cycle | +0.54 | 2.24 inst/cycle | +0.30 | 3.41 inst/cycle | +1.47 |


### Key Learnings

#### ❌ Plan D: `rewrite_gelu_arithmetic_to_maximize_fma_utilization` (0.086×)

Rewriting GELU arithmetic with libdevice.fma and simultaneously expanding autotune configs to very large XBLOCK values (2048, 4096) caused executed instructions to explode from 23.7M to 468M (19.7x increase), and duration to increase from 60.1µs to 696.7µs (11.6x slower). Although occupancy improved (46% → 87%) and compute SM throughput rose (50% → 85%), the instruction count growth completely dominated, as the autotune selected a large XBLOCK that serialized enormous amounts of work per block. The FMA arithmetic rewrite itself may be neutral or slightly positive, but it is inseparable from the catastrophic XBLOCK inflation introduced simultaneously in the same diff.

> **Applicable when**: Never apply this combined pattern (FMA rewrite + very large XBLOCK autotune configs of 2048/4096) to a pointwise elementwise kernel that is already fast at small XBLOCK. The FMA rewrite alone might be evaluated in isolation with the original XBLOCK configs.

> **Avoid when**: Do not introduce XBLOCK configs of 2048 or 4096 for pointwise kernels where the original XBLOCK was 128-512 and registers/thread were ≤16. The autotune will select the largest config, doubling register pressure and inflating instruction count by 10-20x, causing catastrophic slowdown regardless of any arithmetic-level improvements made simultaneously.

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_store_evict_first_to_free_l1_for_bias_broadcast | 0.998× | 0.999× | ✓ |  |
| B | add_evict_first_to_streaming_main_tensor_load_and_store | 1.000× | 1.000× | ✓ |  |
| C | eliminate_bias_preload_gather_use_direct_l1_cached_load | 1.002× | 1.001× | ✓ |  |
| D | revert_oversized_xblock_and_restore_clean_gelu_arithmetic | 1.002× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_store_e... | Δ | add_evict_first_t... | Δ | eliminate_bias_pr... | Δ | revert_oversized_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 60100 ns | 60160 ns | +60 ns (+0.1%) | 60100 ns | +0 ns (+0.0%) | 60060 ns | -40 ns (-0.1%) | 60100 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 47.03 % | 64.06 % | +17.03 pp | 57.75 % | +10.72 pp | 45.17 % | -1.86 pp | 49.85 % | +2.82 pp |
| Memory Throughput (%) | 23.53 % | 23.56 % | +0.03 pp | 23.60 % | +0.07 pp | 23.54 % | +0.01 pp | 23.59 % | +0.06 pp |
| DRAM Throughput (%) | 23.53 % | 23.56 % | +0.03 pp | 23.60 % | +0.07 pp | 23.54 % | +0.01 pp | 23.59 % | +0.06 pp |
| Achieved Occupancy (%) | 45.88 % | 51.92 % | +6.04 pp | 49.21 % | +3.33 pp | 46.06 % | +0.18 pp | 46.57 % | +0.69 pp |
| Registers/Thread | 16.00 register/thread | 17.00 register/thread | +1 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 44.55 | 44.55 | +0.00 | 44.55 | +0.00 | 44.55 | +0.00 | 44.55 | +0.00 |
| No Eligible (%) | 51.26 % | 33.18 % | -18.08 pp | 39.77 % | -11.49 pp | 52.65 % | +1.39 pp | 48.09 % | -3.17 pp |
| IPC Active | 1.94 inst/cycle | 2.68 inst/cycle | +0.74 | 2.41 inst/cycle | +0.47 | 1.88 inst/cycle | -0.06 | 2.09 inst/cycle | +0.15 |

---

## Cumulative Progress

```
Baseline:  60100 ns  (1.000×)

Final best: 60100 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,28 +5,29 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
-def triton_poi_fused_addmm_gelu_view_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
-    xnumel = 12042240
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
-    x0 = (xindex % 384)
-    x2 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_out_ptr0 + (x2), None)
-    tmp2 = tmp0 + tmp1
-    tmp3 = tl.full([1], 0.5, tl.float32)
-    tmp4 = tmp2 * tmp3
-    tmp5 = tl.full([1], 0.7071067811865476, tl.float32)
-    tmp6 = tmp2 * tmp5
-    tmp7 = libdevice.erf(tmp6)
-    tmp8 = tl.full([1], 1.0, tl.float32)
-    tmp9 = tmp7 + tmp8
-    tmp10 = tmp4 * tmp9
-    tl.store(in_out_ptr0 + (x2), tmp10, None)
-
-
+def triton_poi_fused_addmm_gelu_view_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
+    pid = tl.program_id(0)
+    num_programs = tl.num_programs(0)
+    num_tiles = tl.cdiv(xnumel, XBLOCK)
+    tile_id = pid
+    while tile_id < num_tiles:
+        xoffset = tile_id * XBLOCK
+        xindex = xoffset + tl.arange(0, XBLOCK)[:]
+        xmask = xindex < xnumel
+        x0 = xindex % 384
+        x2 = xindex
+        tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
+        tmp1 = tl.load(in_out_ptr0 + x2, xmask)
+        tmp2 = tmp0 + tmp1
+        tmp6 = tmp2 * 0.7071067811865476
+        tmp7 = libdevice.erf(tmp6)
+        tmp8 = tmp7 + 1.0
+        tmp9 = tmp2 * 0.5
+        tmp10 = tmp9 * tmp8
+        tl.store(in_out_ptr0 + x2, tmp10, xmask)
+        tile_id += num_programs
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
