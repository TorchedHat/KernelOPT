# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_convolution_native_group_norm_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 160480 ns — **Best latency**: 171390 ns (**0.9363×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_convolution_native_group_norm_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_083/kernel_0_triton_red_fused_convolution_n/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1024,1,1), Waves/SM: 1.11
- **Registers/thread**: 68
- **Duration**: 160480 ns
- **SM Throughput**: 64.29 % | **Memory**: 31.18 % | **DRAM**: 31.18 %
- **Achieved Occupancy**: 31.10 %
- **IPC Active**: 2.72 inst/cycle | **No Eligible**: 32.12 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_and_autotune | 0.261× | 0.245× | ✓ |  |
| B | increase_xblock_reduce_partial_wave_and_improve_coalescing | 1.000× | 0.992× | ✓ |  |
| C | increase_xblock_for_coalesced_access_and_eliminate_partial_wave | 0.261× | 0.124× | ✓ |  |
| D | autotune_xblock_r0block_reduce_register_pressure | 1.166× | 0.916× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | increase_xblock_r... | Δ | increase_xblock_f... | Δ | autotune_xblock_r... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 160480 ns | 654180 ns | +493700 ns (+307.6%) | 161730 ns | +1250 ns (+0.8%) | 1290000 ns | +1129520 ns (+703.8%) | 175200 ns | +14720 ns (+9.2%) |
| SM Throughput (%) | 64.29 % | 67.42 % | +3.13 pp | 64.42 % | +0.13 pp | 66.94 % | +2.65 pp | 61.95 % | -2.34 pp |
| Memory Throughput (%) | 31.18 % | 21.47 % | -9.71 pp | 30.89 % | -0.29 pp | 20.56 % | -10.62 pp | 28.51 % | -2.67 pp |
| DRAM Throughput (%) | 31.18 % | 7.64 % | -23.54 pp | 30.89 % | -0.29 pp | 3.88 % | -27.30 pp | 28.51 % | -2.67 pp |
| Achieved Occupancy (%) | 31.10 % | 11.89 % | -19.21 pp | 31.02 % | -0.08 pp | 11.90 % | -19.20 pp | 42.95 % | +11.85 pp |
| Registers/Thread | 68.00 register/thread | 255.00 register/thread | +187 | 68.00 register/thread | +0 | 255.00 register/thread | +187 | 29.00 register/thread | -39 |
| Waves/SM | 1.11 | 3.88 | +2.77 | 1.11 | +0.00 | 3.88 | +2.77 | 0.48 | -0.63 |
| No Eligible (%) | 32.12 % | 30.19 % | -1.93 pp | 32.08 % | -0.04 pp | 31.20 % | -0.92 pp | 34.23 % | +2.11 pp |
| IPC Active | 2.72 inst/cycle | 2.79 inst/cycle | +0.07 | 2.71 inst/cycle | -0.01 | 2.75 inst/cycle | +0.03 | 2.63 inst/cycle | -0.09 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_and_autotune` (0.245×)

Adding @triton.autotune with larger XBLOCK values (4–32) to a Triton reduction kernel caused registers_per_thread to spike from 68 to 255 (the hardware maximum, indicating register spilling), collapsing theoretical occupancy from 43.75% to 12.5% and achieved occupancy from 31.1% to 11.89%. This caused executed instructions to increase 4.3x (81M → 348M) and kernel duration to increase 4x (160 µs → 654 µs), despite warp cycles per instruction improving. The autotuner selected a configuration that required far more registers than the hardware can provide without spilling, negating any benefit from better tile coalescing or reduced grid size.

> **Applicable when**: Never applicable as described — this is a negative result. The pattern of adding autotune with large XBLOCK×R0_BLOCK product configs to a Welford reduction kernel that already uses 68 registers/thread is dangerous.

> **Avoid when**: Do NOT add autotune configs where XBLOCK * R0_BLOCK product is large (e.g., XBLOCK=8, R0_BLOCK=2048 = 16384 elements per block) for kernels that already use ≥60 registers/thread. The combined register pressure from holding XBLOCK×R0_BLOCK accumulator arrays (mean, m2, weight) will cause register spilling (registers_per_thread → 255), collapsing occupancy to <15% and increasing runtime 4x. Avoid when: kernel already has registers_per_thread ≥ 60, kernel uses multi-element reduction accumulators (Welford or similar), or R0_BLOCK is large (≥512).

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cap_registers_to_eliminate_partial_wave_and_improve_occupancy | 1.033× | 1.000× | ✓ |  |
| B | fix_dynamic_grid_and_increase_num_warps_for_l1tex_latency_hiding | 1.182× | 0.936× | ✓ | **✓ best** |
| C | remove_large_xblock_autotune_configs_to_restore_register_budget | 0.998× | 0.990× | ✓ |  |
| D | revert_autotune_register_spill_restore_fixed_small_tile_config | 1.000× | 0.986× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_registers_to_... | Δ | fix_dynamic_grid_... | Δ | remove_large_xblo... | Δ | revert_autotune_r... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 160480 ns | 160510 ns | +30 ns (+0.0%) | 171390 ns | +10910 ns (+6.8%) | 162180 ns | +1700 ns (+1.1%) | 162750 ns | +2270 ns (+1.4%) |
| SM Throughput (%) | 64.29 % | 64.50 % | +0.21 pp | 62.05 % | -2.24 pp | 64.45 % | +0.16 pp | 64.42 % | +0.13 pp |
| Memory Throughput (%) | 31.18 % | 31.14 % | -0.04 pp | 29.17 % | -2.01 pp | 30.84 % | -0.34 pp | 30.68 % | -0.50 pp |
| DRAM Throughput (%) | 31.18 % | 31.14 % | -0.04 pp | 29.17 % | -2.01 pp | 30.84 % | -0.34 pp | 30.68 % | -0.50 pp |
| Achieved Occupancy (%) | 31.10 % | 33.05 % | +1.95 pp | 42.97 % | +11.87 pp | 31.01 % | -0.09 pp | 31.04 % | -0.06 pp |
| Registers/Thread | 68.00 register/thread | 64.00 register/thread | -4 | 29.00 register/thread | -39 | 68.00 register/thread | +0 | 68.00 register/thread | +0 |
| Waves/SM | 1.11 | 0.97 | -0.14 | 0.48 | -0.63 | 1.11 | +0.00 | 1.11 | +0.00 |
| No Eligible (%) | 32.12 % | 31.90 % | -0.22 pp | 34.30 % | +2.18 pp | 31.92 % | -0.20 pp | 32.01 % | -0.11 pp |
| IPC Active | 2.72 inst/cycle | 2.73 inst/cycle | +0.01 | 2.62 inst/cycle | -0.10 | 2.72 inst/cycle | +0.00 | 2.71 inst/cycle | -0.01 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_alu_latency_within_register_budget | 1.157× | 1.145× | ✓ |  |
| B | reduce_num_warps_to_eliminate_partial_wave_and_improve_occupancy | 0.995× | 0.993× | ✓ |  |
| C | add_maxnreg_to_eliminate_partial_wave_and_improve_occupancy | 0.998× | 0.993× | ✓ |  |
| D | fix_grid_lambda_and_store_eviction_to_reduce_l1tex_stall | 1.181× | 0.934× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | reduce_num_warps_... | Δ | add_maxnreg_to_el... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 160480 ns | 140220 ns | -20260 ns (-12.6%) | 161600 ns | +1120 ns (+0.7%) | 161700 ns | +1220 ns (+0.8%) | 171840 ns | +11360 ns (+7.1%) |
| SM Throughput (%) | 64.29 % | 62.36 % | -1.93 pp | 64.36 % | +0.07 pp | 64.92 % | +0.63 pp | 50.30 % | -13.99 pp |
| Memory Throughput (%) | 31.18 % | 35.65 % | +4.47 pp | 30.94 % | -0.24 pp | 30.91 % | -0.27 pp | 29.10 % | -2.08 pp |
| DRAM Throughput (%) | 31.18 % | 35.65 % | +4.47 pp | 30.94 % | -0.24 pp | 30.91 % | -0.27 pp | 29.10 % | -2.08 pp |
| Achieved Occupancy (%) | 31.10 % | 50.05 % | +18.95 pp | 31.11 % | +0.01 pp | 31.01 % | -0.09 pp | 23.25 % | -7.85 pp |
| Registers/Thread | 68.00 register/thread | 45.00 register/thread | -23 | 68.00 register/thread | +0 | 68.00 register/thread | +0 | 32.00 register/thread | -36 |
| Waves/SM | 1.11 | 1.55 | +0.44 | 1.11 | +0.00 | 1.11 | +0.00 | 0.24 | -0.87 |
| No Eligible (%) | 32.12 % | 33.71 % | +1.59 pp | 32.10 % | -0.02 pp | 31.98 % | -0.14 pp | 47.47 % | +15.35 pp |
| IPC Active | 2.72 inst/cycle | 2.65 inst/cycle | -0.07 | 2.72 inst/cycle | +0.00 | 2.71 inst/cycle | -0.01 | 2.07 inst/cycle | -0.65 |


### Key Learnings

#### ✅ Plan A: `increase_num_warps_to_hide_alu_latency_within_register_budget` (1.145×)

When a Triton reduction kernel shows a large gap between theoretical and achieved occupancy (e.g., 50% theoretical vs 33% achieved) with low eligible warps per scheduler (~2.27) and high no-eligible-warp stall rate (~32%), increasing num_warps from 4 to 8 (block_size 128→256) while keeping XBLOCK=1 raises achieved occupancy from 33% to 50% and reduces kernel duration by ~13%. The key insight is that XBLOCK=1 prevents Welford accumulator register pressure from multiplying by XBLOCK, so the compiler actually allocates fewer registers per thread (64→45) at the larger block size, improving both theoretical occupancy (50%→62.5%) and waves per SM (0.97→1.55). The instruction count also drops (~16%) because the autotuner selects a more efficient config from the expanded search space.

> **Applicable when**: Triton reduction kernel with: achieved_occupancy < 40%, theoretical_occupancy > achieved by >10 points, no_eligible_warp_pct > 25%, eligible_warps_per_scheduler < 3, registers_per_thread <= 64, XBLOCK=1 is the dominant config (Welford or similar reduction accumulators that scale with XBLOCK). The kernel must use XBLOCK=1 to avoid register pressure multiplication from accumulator arrays.

> **Avoid when**: Kernel uses XBLOCK>1 in its reduction body (accumulator arrays scale with XBLOCK, so doubling num_warps with XBLOCK>1 will multiply register pressure). Do not apply if registers_per_thread is already near the hardware limit (>100) — increasing num_warps will not reduce register count and may cause spilling. Also avoid if the kernel is already memory-bandwidth bound (DRAM throughput >80%) rather than compute/latency bound.

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | deepen_reduction_loop_pipeline_with_num_stages_3_4 | 1.159× | 1.149× | ✓ |  |
| B | pipeline_reduction_loop_and_evict_stores_to_reduce_cache_pollution | 1.164× | 0.998× | ✓ |  |
| C | enable_num_stages2_software_pipelining_on_reduction_loop | 1.000× | 0.986× | ✓ |  |
| D | enable_num_stages2_software_pipelining_to_hide_l1tex_latency | 1.180× | 0.936× | ✓ |  |


### Metric Comparison

| Metric | Baseline | deepen_reduction_... | Δ | pipeline_reductio... | Δ | enable_num_stages... | Δ | enable_num_stages... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 160480 ns | 139710 ns | -20770 ns (-12.9%) | 160860 ns | +380 ns (+0.2%) | 162780 ns | +2300 ns (+1.4%) | 171520 ns | +11040 ns (+6.9%) |
| SM Throughput (%) | 64.29 % | 62.08 % | -2.21 pp | 64.47 % | +0.18 pp | 64.26 % | -0.03 pp | 49.79 % | -14.50 pp |
| Memory Throughput (%) | 31.18 % | 35.76 % | +4.58 pp | 31.08 % | -0.10 pp | 30.69 % | -0.49 pp | 29.14 % | -2.04 pp |
| DRAM Throughput (%) | 31.18 % | 35.76 % | +4.58 pp | 31.08 % | -0.10 pp | 30.69 % | -0.49 pp | 29.14 % | -2.04 pp |
| Achieved Occupancy (%) | 31.10 % | 50.20 % | +19.10 pp | 31.15 % | +0.05 pp | 31.03 % | -0.07 pp | 23.25 % | -7.85 pp |
| Registers/Thread | 68.00 register/thread | 45.00 register/thread | -23 | 68.00 register/thread | +0 | 68.00 register/thread | +0 | 32.00 register/thread | -36 |
| Waves/SM | 1.11 | 1.55 | +0.44 | 1.11 | +0.00 | 1.11 | +0.00 | 0.24 | -0.87 |
| No Eligible (%) | 32.12 % | 33.60 % | +1.48 pp | 31.98 % | -0.14 pp | 32.19 % | +0.07 pp | 47.68 % | +15.56 pp |
| IPC Active | 2.72 inst/cycle | 2.64 inst/cycle | -0.08 | 2.72 inst/cycle | +0.00 | 2.72 inst/cycle | +0.00 | 2.07 inst/cycle | -0.65 |


### Key Learnings

#### ✅ Plan A: `deepen_reduction_loop_pipeline_with_num_stages_3_4` (1.149×)

Adding num_stages=3 and num_stages=4 autotune configs to a Triton reduction kernel with very low L2 hit rate (~2%) and many reduction loop iterations (~58) produced a modest 14.9% speedup, but the profiling metrics before and after are nearly identical — no measurable change in scheduler stall rate, warp cycles per instruction, DRAM throughput, or occupancy. The actual speedup likely comes from the autotuner selecting a marginally better config from the expanded search space rather than from deeper software pipelining hiding DRAM latency. For Welford-style reductions with stateful inter-iteration data dependencies, num_stages>2 provides minimal benefit because the compiler cannot fully decouple loads from dependent accumulator updates, limiting the effective pipeline depth.

> **Applicable when**: Triton reduction kernel with very low L2 hit rate (<5%), many reduction loop iterations (>30), DRAM-bound loads, and existing autotune configs capped at num_stages<=2. Safe to try when XBLOCK=1 and maxnreg is capped (e.g., maxnreg=64) to prevent register spill from deeper pipeline staging.

> **Avoid when**: Kernel has stateful inter-iteration data dependencies (e.g., Welford online reduction where each iteration's accumulator feeds the next) — the compiler cannot fully pipeline such loops regardless of num_stages, so gains will be minimal. Do not expect large speedups (>20%) from this change alone when scheduler stall rates and warp cycles per instruction are unchanged before/after. Also avoid if register budget is already tight and maxnreg is not capped.

---

## Cumulative Progress

```
Baseline:  160480 ns  (1.000×)

Final best: 171390 ns  (0.9363×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    # XBLOCK=1 + num_warps=8: safe because Welford accumulators are [1, R0_BLOCK] (no XBLOCK multiplication)
+    triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=8, num_stages=1, maxnreg=64),
+    triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=8, num_stages=2, maxnreg=64),
+    triton.Config({'XBLOCK': 1, 'R0_BLOCK': 512},  num_warps=8, num_stages=2, maxnreg=64),
+    triton.Config({'XBLOCK': 1, 'R0_BLOCK': 256},  num_warps=8, num_stages=2, maxnreg=64),
+    # XBLOCK=1 + num_warps=4 with pipelining
+    triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=4, num_stages=2, maxnreg=64),
+    triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=4, num_stages=1, maxnreg=64),
+    # Small-XBLOCK fallbacks (safe: XBLOCK*R0_BLOCK product is small)
+    triton.Config({'XBLOCK': 4, 'R0_BLOCK': 256},  num_warps=4, num_stages=1, maxnreg=64),
+    triton.Config({'XBLOCK': 4, 'R0_BLOCK': 128},  num_warps=4, num_stages=2, maxnreg=64),
+], key=['xnumel', 'r0_numel'])
 @triton.jit
 def triton_red_fused_convolution_native_group_norm_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 1024
@@ -28,8 +42,8 @@

         rindex = r0_index
         r0_5 = r0_index
         r0_3 = r0_index // 19800
-        tmp0 = tl.load(in_ptr0 + (r0_5 + 59400*x4), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
-        tmp1 = tl.load(in_ptr1 + (r0_3 + 3*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp0 = tl.load(in_ptr0 + (r0_5 + 59400*x4), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp1 = tl.load(in_ptr1 + (r0_3 + 3*x0), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
         tmp2 = tmp0 + tmp1
         tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
         tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
@@ -42,10 +56,8 @@

     tmp4 = tmp5[:, None]
     tmp8 = tmp6[:, None]
     tmp9 = tmp7[:, None]
-    tl.store(out_ptr0 + (x4), tmp4, xmask)
-    tl.store(out_ptr1 + (x4), tmp8, xmask)
-
-
+    tl.store(out_ptr0 + (x4), tmp4, mask=xmask)
+    tl.store(out_ptr1 + (x4), tmp8, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -55,7 +67,7 @@

     xnumel = 1024
     r0_numel = 59400
     grid = (triton.cdiv(xnumel, 1),)
-    triton_red_fused_convolution_native_group_norm_0[grid](arg_0, arg_1, arg_2, arg_3, 1024, 59400, XBLOCK=1, R0_BLOCK=1024)
+    triton_red_fused_convolution_native_group_norm_0[grid](arg_0, arg_1, arg_2, arg_3, 1024, 59400)
     return arg_3
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
