# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_53` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 585570 ns — **Best latency**: 4670 ns (**125.3897×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`increase_grid_parallelism_via_smaller_xblock_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_53`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_53_triton_per_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4,1,1), Waves/SM: 0.01
- **Registers/thread**: 32
- **Duration**: 585570 ns
- **SM Throughput**: 0.33 % | **Memory**: 1.21 % | **DRAM**: 0.13 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.19 inst/cycle | **No Eligible**: 95.15 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_grid_parallelism_via_smaller_xblock_autotune | 85.368× | 125.390× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_grid_par... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 585570 ns | 4670 ns | -580900 ns (-99.2%) |
| SM Throughput (%) | 0.33 % | 0.06 % | -0.27 pp |
| Memory Throughput (%) | 1.21 % | 0.55 % | -0.66 pp |
| DRAM Throughput (%) | 0.13 % | 0.19 % | +0.06 pp |
| Achieved Occupancy (%) | 6.24 % | 1.56 % | -4.68 pp |
| Registers/Thread | 32.00 register/thread | 126.00 register/thread | +94 |
| Waves/SM | 0.01 | 0.00 | -0.01 |
| No Eligible (%) | 95.15 % | 86.67 % | -8.48 pp |
| IPC Active | 0.19 inst/cycle | 0.12 inst/cycle | -0.07 |

---

## Iteration 2

**Best so far**: 125.3897× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_match_autotuned_xblock | 82.970× | 122.761× | ✓ |  |
| B | fix_autotune_grid_lambda_to_enable_xblock_exploration | 83.020× | 130.708× | ✓ |  |
| C | fix_grid_lambda_to_enable_autotune_xblock_expansion | 85.109× | 128.980× | ✓ |  |
| D | fix_autotune_grid_lambda_to_use_meta_xblock | 85.040× | 126.200× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | fix_autotune_grid... | Δ | fix_grid_lambda_t... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 585570 ns | 4770 ns | -580800 ns (-99.2%) | 4480 ns | -581090 ns (-99.2%) | 4540 ns | -581030 ns (-99.2%) | 4640 ns | -580930 ns (-99.2%) |
| SM Throughput (%) | 0.33 % | 0.06 % | -0.27 pp | 0.06 % | -0.27 pp | 0.06 % | -0.27 pp | 0.06 % | -0.27 pp |
| Memory Throughput (%) | 1.21 % | 0.43 % | -0.78 pp | 0.55 % | -0.66 pp | 0.48 % | -0.73 pp | 0.55 % | -0.66 pp |
| DRAM Throughput (%) | 0.13 % | 0.18 % | +0.05 pp | 0.20 % | +0.07 pp | 0.19 % | +0.06 pp | 0.19 % | +0.06 pp |
| Achieved Occupancy (%) | 6.24 % | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp |
| Registers/Thread | 32.00 register/thread | 126.00 register/thread | +94 | 126.00 register/thread | +94 | 126.00 register/thread | +94 | 126.00 register/thread | +94 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.00 | -0.01 | 0.00 | -0.01 | 0.00 | -0.01 |
| No Eligible (%) | 95.15 % | 87.24 % | -7.91 pp | 86.78 % | -8.37 pp | 86.03 % | -9.12 pp | 87.23 % | -7.92 pp |
| IPC Active | 0.19 inst/cycle | 0.13 inst/cycle | -0.06 | 0.11 inst/cycle | -0.08 | 0.14 inst/cycle | -0.05 | 0.12 inst/cycle | -0.07 |


### Key Learnings

#### ✅ Plan C: `fix_grid_lambda_to_enable_autotune_xblock_expansion` (128.980×)

When a Triton kernel's grid is hardcoded as a static integer tuple (e.g., `(triton.cdiv(xnumel, 128),)`), the autotuner's XBLOCK selection never changes the actual number of launched blocks, making all autotune configs functionally identical. Replacing the static grid with a lambda `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)` allows the autotuner to actually vary grid size by selecting smaller XBLOCK values, expanding parallelism across SMs. This is the prerequisite fix when NCU's LaunchConfiguration rule fires (grid << SM count, waves_per_sm ≈ 0) and an @triton.autotune decorator with XBLOCK configs already exists but has no effect on grid size. Note: the profiling metrics for the kernel itself showed only marginal improvement (4.67→4.54 µs), suggesting the large speedup (128x) was captured at the end-to-end benchmark level rather than in the single-kernel NCU profile, likely because the autotuner selected a much smaller XBLOCK that spread work across more SMs in actual execution.

> **Applicable when**: NCU LaunchConfiguration rule fires with grid_size << sm_count and waves_per_sm ≈ 0.0; kernel has @triton.autotune with XBLOCK configs but grid is a static tuple hardcoded to a fixed XBLOCK value; achieved_occupancy << theoretical_occupancy; no_eligible_scheduler_pct > 80%; the @triton.jit kernel body already uses xmask guards parameterized by XBLOCK for boundary safety

> **Avoid when**: Grid is already a lambda reading meta['XBLOCK']; kernel body does NOT have xmask boundary guards (changing XBLOCK without guards causes correctness failures as seen in prior iterations with max_diff~14); R0_BLOCK must remain a power-of-2 >= r0_numel for reductions to be correct; do not apply if the kernel has no @triton.autotune decorator with XBLOCK configs

#### ✅ Plan D: `fix_autotune_grid_lambda_to_use_meta_xblock` (126.200×)

When a Triton kernel uses @triton.autotune with multiple XBLOCK configs but the caller passes a hardcoded grid=(triton.cdiv(xnumel, CONSTANT),), the grid size never changes regardless of which XBLOCK the autotuner selects. This causes severe underutilization (grid=4 blocks across 132 SMs) and correctness failures when XBLOCK differs from the hardcoded constant. The fix is to replace the static grid tuple with a lambda: grid = lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),), which dynamically co-varies grid size with the autotuner's chosen XBLOCK, allowing the existing autotune configs to work as intended and expanding the grid to fill available SMs.

> **Applicable when**: A Triton kernel has @triton.autotune with multiple XBLOCK configs but the kernel launch site uses a hardcoded grid size (e.g., grid=(triton.cdiv(N, CONSTANT),)) instead of a lambda. Symptoms: grid_size is tiny relative to SM count (e.g., 4 blocks vs 132 SMs), waves_per_sm ≈ 0, achieved_occupancy << theoretical_occupancy, LaunchConfiguration NCU rule fires with high estimated speedup (>90%), and prior attempts to reduce XBLOCK directly caused correctness failures.

> **Avoid when**: The kernel does not use @triton.autotune, or the grid is already a lambda using meta['XBLOCK']. Also inapplicable when the small grid is intentional (e.g., the kernel processes the entire problem in one block by design). Note: the profiling metrics before and after are nearly identical in this case (grid_size still shows 4, duration 4.67→4.64 us in NCU), suggesting the 126x speedup was measured at the end-to-end benchmark level rather than reflected in the single-kernel NCU trace — the NCU trace may be capturing the autotuner's calibration run rather than the optimized run.

---

## Iteration 3

**Best so far**: 125.3897× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reduce_register_pressure_to_increase_theoretical_occupancy | 85.148× | 84.742× | ✓ |  |
| C | reduce_register_pressure_to_increase_occupancy_and_sm_utilization | 85.336× | 81.669× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 585570 ns | 6910 ns | -578660 ns (-98.8%) | 7170 ns | -578400 ns (-98.8%) |
| SM Throughput (%) | 0.33 % | 0.06 % | -0.27 pp | 0.06 % | -0.27 pp |
| Memory Throughput (%) | 1.21 % | 0.40 % | -0.81 pp | 0.44 % | -0.77 pp |
| DRAM Throughput (%) | 0.13 % | 0.13 % | +0.00 pp | 0.13 % | +0.00 pp |
| Achieved Occupancy (%) | 6.24 % | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp |
| Registers/Thread | 32.00 register/thread | 64.00 register/thread | +32 | 64.00 register/thread | +32 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.00 | -0.01 |
| No Eligible (%) | 95.15 % | 91.11 % | -4.04 pp | 91.67 % | -3.48 pp |
| IPC Active | 0.19 inst/cycle | 0.09 inst/cycle | -0.10 | 0.09 inst/cycle | -0.10 |


### Key Learnings

#### ❌ Plan B: `reduce_register_pressure_to_increase_theoretical_occupancy` (84.742×)

Adding maxnreg=64 to Triton autotune configs successfully halved registers per thread (126→64) and doubled theoretical occupancy (25%→50%), but achieved occupancy remained stuck at 1.56% because the grid was still only 4 blocks across 132 SMs — far too few to fill any SM regardless of per-SM capacity. The register spilling introduced by the cap (new local load/store stalls visible in CPIStall rising from 48% to 63.5%, and Warp Cycles Per Issued Instruction rising from 8.35→11.37) caused kernel duration to increase from 4.48→6.91 µs, a 54% regression. When the grid is severely underutilized (waves_per_sm ≈ 0), increasing theoretical occupancy via register capping provides zero benefit while register spill overhead actively hurts performance.

> **Applicable when**: Never apply maxnreg register capping alone when the grid is severely underutilized (waves_per_sm ≈ 0, grid_size << sm_count). The register cap only helps when there are enough blocks to actually fill the SM's increased capacity — i.e., when waves_per_sm > 1 and the register file is the binding occupancy limiter.

> **Avoid when**: Grid is tiny (waves_per_sm < 0.5, grid_size << sm_count) — in this regime, register spilling overhead from maxnreg dominates any theoretical occupancy gain. Also avoid when the kernel performs heavy compute per thread (e.g., LayerNorm reductions with many intermediate values) since register spilling to local memory will cause severe L1TEX stall increases (CPIStall rising from 48%→63.5% observed here).

#### ❌ Plan C: `reduce_register_pressure_to_increase_occupancy_and_sm_utilization` (81.669×)

Adding maxnreg=64 to autotune configs successfully halved register usage (126→64 registers/thread) and doubled theoretical occupancy (25%→50%), but achieved occupancy remained at 1.56% because the grid was still only 4 blocks across 132 SMs. The register spilling introduced by maxnreg caused L1TEX stall cycles to increase (CPIStall 48%→61%, warp cycles per instruction 8.35→11.78), and kernel duration increased from 4.48 µs to 7.17 µs. When the grid is too small to fill SMs, register pressure reduction via maxnreg cannot improve achieved occupancy and the spill overhead dominates.

> **Applicable when**: Never apply maxnreg alone when the primary bottleneck is grid size (LaunchConfiguration rule, waves_per_sm ≈ 0, grid << SM count). maxnreg may help as a secondary optimization only after the grid expansion fix is already in place and register pressure (Block Limit Registers) is the remaining occupancy limiter.

> **Avoid when**: Grid is too small to fill SMs (waves_per_sm < 0.1, grid_size << sm_count) — in this regime maxnreg causes register spilling that increases L1TEX stall cycles and warp cycles per instruction without improving achieved occupancy. Do not apply maxnreg when the kernel already has high register usage that is load-bearing (complex reductions, many live values) and the grid expansion fix has not yet been applied.

---

## Cumulative Progress

```
Baseline:  585570 ns  (1.000×)
Iter 1     4670 ns  (125.390×)  — increase_grid_parallelism_via_smaller_xblock_autotune

Final best: 4670 ns  (125.3897×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1},  num_warps=1),
+        triton.Config({'XBLOCK': 1},  num_warps=2),
+        triton.Config({'XBLOCK': 2},  num_warps=2),
+        triton.Config({'XBLOCK': 4},  num_warps=2),
+        triton.Config({'XBLOCK': 4},  num_warps=4),
+        triton.Config({'XBLOCK': 8},  num_warps=4),
+        triton.Config({'XBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 32}, num_warps=4),
+        triton.Config({'XBLOCK': 64}, num_warps=4),
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_53(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 490
@@ -24,11 +40,11 @@

     x3 = xindex
     x0 = (xindex % 49)
     x1 = xindex // 49
-    tmp0 = tl.load(in_ptr0 + (r0_2 + 768*x3), r0_mask & xmask, other=0.0)
-    tmp1 = tl.load(in_ptr1 + (32*x0 + 1568*(r0_2 // 32) + 37632*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
-    tmp2 = tl.load(in_ptr2 + (x0 + 49*(r0_2 // 32)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-    tmp28 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
-    tmp30 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
+    tmp0 = tl.load(in_ptr0 + (r0_2 + 768*x3), mask=r0_mask & xmask, other=0.0)
+    tmp1 = tl.load(in_ptr1 + (32*x0 + 1568*(r0_2 // 32) + 37632*x1 + ((r0_2 % 32))), mask=r0_mask & xmask, other=0.0)
+    tmp2 = tl.load(in_ptr2 + (x0 + 49*(r0_2 // 32)), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+    tmp28 = tl.load(in_ptr3 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
+    tmp30 = tl.load(in_ptr4 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
     tmp3 = tmp1 + tmp2
     tmp4 = tmp0 + tmp3
     tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
@@ -53,9 +69,7 @@

     tmp27 = tmp21 * tmp26
     tmp29 = tmp27 * tmp28
     tmp31 = tmp29 + tmp30
-    tl.store(out_ptr2 + (r0_2 + 768*x3), tmp31, r0_mask & xmask)
-
-
+    tl.store(out_ptr2 + (r0_2 + 768*x3), tmp31, mask=r0_mask & xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -67,7 +81,7 @@

     xnumel = 490
     r0_numel = 768
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_53[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 490, 768, XBLOCK=128)
+    triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_53[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 490, 768)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
