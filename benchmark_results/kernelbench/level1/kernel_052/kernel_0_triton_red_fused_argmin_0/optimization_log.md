# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_argmin_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 59030000 ns — **Best latency**: 96740 ns (**610.1923×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`add_triton_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_argmin_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_052/kernel_0_triton_red_fused_argmin_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4095,1,1), Waves/SM: 1.94
- **Registers/thread**: 32
- **Duration**: 59030000 ns
- **SM Throughput**: 13.01 % | **Memory**: 70.21 % | **DRAM**: 70.21 %
- **Achieved Occupancy**: 93.33 %
- **IPC Active**: 0.32 inst/cycle | **No Eligible**: 91.98 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 464.832× | 487.771× | ✓ | **✓ best** |
| E | add_triton_autotune | 3.945× | 3.962× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 59030000 ns | 121020 ns | -58908980 ns (-99.8%) | 14900000 ns | -44130000 ns (-74.8%) |
| SM Throughput (%) | 13.01 % | 40.80 % | +27.79 pp | 19.16 % | +6.15 pp |
| Memory Throughput (%) | 70.21 % | 82.04 % | +11.83 pp | 66.62 % | -3.59 pp |
| DRAM Throughput (%) | 70.21 % | 11.69 % | -58.52 pp | 66.62 % | -3.59 pp |
| Achieved Occupancy (%) | 93.33 % | 95.76 % | +2.43 pp | 94.74 % | +1.41 pp |
| Registers/Thread | 32.00 register/thread | 16.00 register/thread | -16 | 32.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 91.98 % | 69.48 % | -22.50 pp | 84.56 % | -7.42 pp |
| IPC Active | 0.32 inst/cycle | 1.22 inst/cycle | +0.90 | 0.62 inst/cycle | +0.30 |


### Key Learnings

#### ✅ Plan E: `add_triton_autotune` (610.192×)

Adding four new autotune configs with large XBLOCK values (256, 512, 1024) to an already-tuned Triton reduction kernel produced only a marginal improvement: duration decreased from 97.63µs to 96.74µs (~0.9% gain), with memory throughput improving slightly from 712 to 718 GB/s and L2 hit rate from 84.25% to 85.34%. The kernel was already near its memory-bandwidth ceiling (91-92% Mem Busy, 95% achieved occupancy, 16 registers/thread) with the existing small-XBLOCK configs, so the new large-XBLOCK configs either were not selected by the autotuner or provided negligible benefit. The reported 610x speedup is relative to a much earlier register-spill baseline, not this specific change; the incremental gain from this step alone is ~1%.

> **Applicable when**: Triton reduction kernel already at near-peak memory throughput (Mem Busy >90%, achieved occupancy >95%, registers_per_thread ~16) where the existing autotune config set covers only small XBLOCK values and the problem may benefit from wider XBLOCK tiling. Marginal gains (~1%) are possible.

> **Avoid when**: When the kernel is already at its memory-bandwidth ceiling with the existing autotune configs (Mem Busy >90%, Duration stable). Adding more autotune configs at this point yields diminishing returns (<1% improvement). Do not apply when the kernel is register-limited (registers_per_thread >128) — large XBLOCK configs will worsen register spill. Do not apply when large XBLOCK × R0_BLOCK products would exceed register budget.

---

## Iteration 2

**Best so far**: 487.7706× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_misapplied_vectorization_hints_for_strided_argmin | 464.531× | 478.402× | ✓ |  |
| B | memory_coalescing_reorder_tile_dimensions | 518.923× | 3.593× | ✓ | **✓ best** |
| C | vectorized_loads_with_smaller_r0_block_configs | 3.577× | 3.524× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_misapplied_ve... | Δ | memory_coalescing... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 59030000 ns | 123390 ns | -58906610 ns (-99.8%) | 16430000 ns | -42600000 ns (-72.2%) | 16750000 ns | -42280000 ns (-71.6%) |
| SM Throughput (%) | 13.01 % | 40.38 % | +27.37 pp | 16.89 % | +3.88 pp | 19.51 % | +6.50 pp |
| Memory Throughput (%) | 70.21 % | 81.03 % | +10.82 pp | 10.65 % | -59.56 pp | 10.45 % | -59.76 pp |
| DRAM Throughput (%) | 70.21 % | 11.47 % | -58.74 pp | 10.65 % | -59.56 pp | 10.45 % | -59.76 pp |
| Achieved Occupancy (%) | 93.33 % | 95.82 % | +2.49 pp | 18.65 % | -74.68 pp | 18.44 % | -74.89 pp |
| Registers/Thread | 32.00 register/thread | 16.00 register/thread | -16 | 155.00 register/thread | +123 | 138.00 register/thread | +106 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 10.34 | +8.40 | 10.34 | +8.40 |
| No Eligible (%) | 91.98 % | 69.65 % | -22.33 pp | 88.03 % | -3.95 pp | 85.34 % | -6.64 pp |
| IPC Active | 0.32 inst/cycle | 1.24 inst/cycle | +0.92 | 0.48 inst/cycle | +0.16 | 0.59 inst/cycle | +0.27 |

---

## Iteration 3

**Best so far**: 487.7706× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | improve_l1_reuse_with_evict_last_policy | 518.977× | 3.588× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | improve_l1_reuse_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 59030000 ns | 16450000 ns | -42580000 ns (-72.1%) |
| SM Throughput (%) | 13.01 % | 16.95 % | +3.94 pp |
| Memory Throughput (%) | 70.21 % | 10.64 % | -59.57 pp |
| DRAM Throughput (%) | 70.21 % | 10.64 % | -59.57 pp |
| Achieved Occupancy (%) | 93.33 % | 18.68 % | -74.65 pp |
| Registers/Thread | 32.00 register/thread | 155.00 register/thread | +123 |
| Waves/SM | 1.94 | 10.34 | +8.40 |
| No Eligible (%) | 91.98 % | 88.06 % | -3.92 pp |
| IPC Active | 0.32 inst/cycle | 0.48 inst/cycle | +0.16 |

---

## Iteration 4

**Best so far**: 487.7706× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_to_use_dynamic_xblock | 586.200× | 604.630× | ✓ | **✓ best** |
| B | constrain_autotune_to_small_r0_block_avoid_register_spill | 377.435× | 365.149× | ✓ |  |
| C | software_pipelining_increase_num_stages | 520.547× | 3.591× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | constrain_autotun... | Δ | software_pipelini... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 59030000 ns | 97630 ns | -58932370 ns (-99.8%) | 161660 ns | -58868340 ns (-99.7%) | 16440000 ns | -42590000 ns (-72.1%) |
| SM Throughput (%) | 13.01 % | 24.65 % | +11.64 pp | 60.49 % | +47.48 pp | 16.95 % | +3.94 pp |
| Memory Throughput (%) | 70.21 % | 91.07 % | +20.86 pp | 85.22 % | +15.01 pp | 10.65 % | -59.56 pp |
| DRAM Throughput (%) | 70.21 % | 14.49 % | -55.72 pp | 34.16 % | -36.05 pp | 10.65 % | -59.56 pp |
| Achieved Occupancy (%) | 93.33 % | 95.71 % | +2.38 pp | 94.37 % | +1.04 pp | 18.66 % | -74.67 pp |
| Registers/Thread | 32.00 register/thread | 16.00 register/thread | -16 | 18.00 register/thread | -14 | 155.00 register/thread | +123 |
| Waves/SM | 1.94 | 0.97 | -0.97 | 1.94 | +0.00 | 10.34 | +8.40 |
| No Eligible (%) | 91.98 % | 80.59 % | -11.39 pp | 56.30 % | -35.68 pp | 88.05 % | -3.93 pp |
| IPC Active | 0.32 inst/cycle | 0.75 inst/cycle | +0.43 | 1.78 inst/cycle | +1.46 | 0.48 inst/cycle | +0.16 |


### Key Learnings

#### ✅ Plan A: `fix_autotune_grid_to_use_dynamic_xblock` (604.630×)

When a Triton kernel uses @triton.autotune with multiple XBLOCK configs but the grid is hardcoded as `(cdiv(xnumel, 128),)`, only the XBLOCK=128 configs produce correct results — all other XBLOCK values launch with the wrong grid size, causing correctness failures. This prevents the autotuner from exploring small-XBLOCK configs that would eliminate register spill. Changing the grid to a lambda `lambda meta: (cdiv(xnumel, meta['XBLOCK']),)` allows the autotuner to safely select small XBLOCK values (e.g., XBLOCK=4, R0_BLOCK=64), which drops registers/thread from 155 to 16, raises theoretical occupancy from 18.75% to 100%, and reduces kernel duration by ~168x (16.45ms → 97.63µs). The root cause is that a hardcoded grid silently invalidates all autotune configs with XBLOCK ≠ the hardcoded divisor, masking the register-spill fix.

> **Applicable when**: Triton kernel with @triton.autotune over XBLOCK configs where the grid call site uses a hardcoded divisor (e.g., cdiv(xnumel, 128)) instead of meta['XBLOCK']. Symptoms: registers_per_thread near hardware max (>128), Block Limit Registers << Block Limit Warps (e.g., 3 vs 16), theoretical occupancy <25%, scheduler No Eligible >85%, and prior autotune attempts with small XBLOCK configs failed correctness (max_diff proportional to the reduction dimension size, e.g., max_diff=4095 for r0_numel=4096).

> **Avoid when**: The grid is already a lambda using meta['XBLOCK']. Also do not add XBLOCK=1 with R0_BLOCK equal to the full reduction dimension (e.g., XBLOCK=1, R0_BLOCK=4096 when r0_numel=4096) — this is a confirmed correctness failure pattern (max_diff=4095) likely due to boundary/masking issues at the full reduction size.

#### ✅ Plan B: `constrain_autotune_to_small_r0_block_avoid_register_spill` (365.149×)

When a Triton reduction kernel's autotune selects a large XBLOCK×R0_BLOCK product (e.g., 128×256=32768 accumulator pairs), threads must maintain too many live values simultaneously, causing massive register spill to local memory. This manifests as registers_per_thread near the hardware maximum (155), Block Limit Registers far below other limits (3 vs 32), theoretical occupancy ~18%, and >88% scheduler idle cycles. Replacing all autotune configs with ones where XBLOCK×R0_BLOCK ≤ 512 and XBLOCK ≥ 4 (to avoid known correctness failures at XBLOCK=1/2) drops registers/thread from 155 to 18, raises theoretical occupancy from 18.75% to 100%, achieved occupancy from 18.68% to 94.37%, and reduces kernel duration by ~100× (16.45ms → 161.66µs).

> **Applicable when**: Triton reduction kernel (argmin/argmax/sum with index tracking) where NCU shows: registers_per_thread near hardware max (>128), Block Limit Registers << Block Limit Warps (e.g., 3 vs 16), theoretical occupancy <25%, scheduler No Eligible >85%, and CPIStall rule triggered with L1TEX scoreboard stalls >85% of warp cycles. The kernel maintains [XBLOCK, R0_BLOCK] accumulator arrays (value + index pairs) that exceed register file capacity.

> **Avoid when**: XBLOCK=1 and XBLOCK=2 configs must be excluded for this specific argmin kernel pattern (known correctness failures with max_diff=4095). Do not apply if register pressure is already low (registers_per_thread < 32, Block Limit Registers near Block Limit Warps). Do not apply if the kernel is already memory-bandwidth bound rather than register-spill bound.

#### ❌ Plan C: `software_pipelining_increase_num_stages` (3.591×)

Adding num_stages=2 and num_stages=3 autotune configs to a Triton reduction kernel that is already register-limited (155 registers/thread, Block Limit Registers=3, 18.75% theoretical occupancy) produced zero measurable improvement: duration changed by only 0.01ms (16.45→16.44ms), all scheduler metrics (No Eligible 88.06%→88.05%, Eligible Warps 0.13→0.13), warp stall cycles (25.02→25.01), and memory metrics remained statistically identical. The autotuner selected the same winning config as before (the new num_stages=2/3 configs were not chosen, or if chosen, provided no benefit). When a kernel is register-limited to 3 blocks/SM and the CPIStall is caused by strided global memory access (not sequential loads amenable to async prefetch), software pipelining via num_stages cannot hide the latency because the hardware lacks the warp-level parallelism to overlap prefetch with compute.

> **Applicable when**: Never effective in this configuration — do not apply when the kernel is register-limited (Block Limit Registers << Block Limit Warps) and the memory access pattern is fundamentally strided (non-sequential loads that cannot benefit from async prefetch coalescing)

> **Avoid when**: When the kernel is register-limited (registers_per_thread > 128, Block Limit Registers <= 4, theoretical occupancy < 25%) AND the CPIStall is caused by strided/non-coalesced global memory access rather than sequential streaming loads. In this regime, adding num_stages=2/3 to autotune configs produces zero measurable improvement because: (1) the autotuner selects the same winning config, (2) even if a pipelined config is selected, the register pressure from the pipeline buffer further reduces occupancy, and (3) strided loads cannot be efficiently prefetched via async copy. The reported 3.59x speedup is relative to a much earlier baseline, not this specific change.

---

## Iteration 5

**Best so far**: 604.6297× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_tiling_for_strided_reduction | 584.772× | 6.618× | ✓ |  |
| B | fix_hardcoded_grid_enable_dynamic_xblock_autotune | 582.392× | 604.444× | ✓ |  |
| C | constrain_autotune_configs_to_eliminate_register_spill | 377.127× | 367.697× | ✓ |  |
| D | add_triton_autotune | 587.097× | 610.192× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | shared_memory_til... | Δ | fix_hardcoded_gri... | Δ | constrain_autotun... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 59030000 ns | 8920000 ns | -50110000 ns (-84.9%) | 97660 ns | -58932340 ns (-99.8%) | 160540 ns | -58869460 ns (-99.7%) | 96740 ns | -58933260 ns (-99.8%) |
| SM Throughput (%) | 13.01 % | 15.57 % | +2.56 pp | 48.32 % | +35.31 pp | 61.08 % | +48.07 pp | 25.69 % | +12.68 pp |
| Memory Throughput (%) | 70.21 % | 9.82 % | -60.39 pp | 88.79 % | +18.58 pp | 85.73 % | +15.52 pp | 92.45 % | +22.24 pp |
| DRAM Throughput (%) | 70.21 % | 9.82 % | -60.39 pp | 14.50 % | -55.71 pp | 34.42 % | -35.79 pp | 14.62 % | -55.59 pp |
| Achieved Occupancy (%) | 93.33 % | 18.38 % | -74.95 pp | 95.77 % | +2.44 pp | 94.86 % | +1.53 pp | 95.47 % | +2.14 pp |
| Registers/Thread | 32.00 register/thread | 155.00 register/thread | +123 | 16.00 register/thread | -16 | 18.00 register/thread | -14 | 16.00 register/thread | -16 |
| Waves/SM | 1.94 | 5.17 | +3.23 | 0.97 | -0.97 | 1.94 | +0.00 | 0.97 | -0.97 |
| No Eligible (%) | 91.98 % | 88.13 % | -3.85 pp | 63.88 % | -28.10 pp | 55.23 % | -36.75 pp | 81.32 % | -10.66 pp |
| IPC Active | 0.32 inst/cycle | 0.47 inst/cycle | +0.15 | 1.45 inst/cycle | +1.13 | 1.76 inst/cycle | +1.44 | 0.77 inst/cycle | +0.45 |


### Key Learnings

#### ⚠️ Plan C: `constrain_autotune_configs_to_eliminate_register_spill` (0.000×)

Plan 'constrain_autotune_configs_to_eliminate_register_spill' was applied but failed correctness. Error: Correctness FAILED:
seed=0 output[0]: max_diff=4.095e+03 > atol=0.001
seed=42 output[0]: max_diff=4.095e+03 > atol=0.001
seed=1337 output[0]: max_diff=4.095e+03 > atol=0.001

> **Applicable when**: The kernel currently has @triton.autotune but the winning config is selecting a large XBLOCK x R0_BLOCK tile (e.g., XBLOCK=128, R0_BLOCK=256 = 32,768 accumulator pairs per block). Each thread in the argmin kernel must maintain two live accumulator arrays (_tmp2 float32 and _tmp2_index int32), both of shape [XBLOCK, R0_BLOCK]. At XBLOCK=128, R0_BLOCK=256, each thread holds 256 float+int pairs = 512 values, far exceeding register file capacity and causing massive spilling to local memory. This manifests as 155 registers/thread, only 3 blocks/SM (Block Limit Registers=3), 18.75% occupancy, and 88% scheduler idle time due to L1TEX scoreboard stalls on local memory loads. The fix is to remove all configs with XBLOCK x R0_BLOCK > 8192 from the autotune list, forcing the autotuner to select small-tile configs (e.g., XBLOCK=1, R0_BLOCK=64/128) that keep accumulators in registers, restore occupancy, and eliminate the local-memory spill bottleneck.

> **Avoid when**: Previous implementation failed: Correctness FAILED:
seed=0 output[0]: max_diff=4.095e+03 > atol=0.001
seed=42 output[0]: max_diff=4.095e+03 > atol=0.001
seed=1337 output[0]: max_diff=4.095e+03 > atol=0.001

#### ✅ Plan D: `add_triton_autotune` (610.192×)

Adding four new autotune configs with large XBLOCK values (256, 512, 1024) to an already-tuned Triton reduction kernel produced only a marginal improvement: duration decreased from 97.63µs to 96.74µs (~0.9% gain), with memory throughput improving slightly from 712 to 718 GB/s and L2 hit rate from 84.25% to 85.34%. The kernel was already near its memory-bandwidth ceiling (91-92% Mem Busy, 95% achieved occupancy, 16 registers/thread) with the existing small-XBLOCK configs, so the new large-XBLOCK configs either were not selected by the autotuner or provided negligible benefit. The reported 610x speedup is relative to a much earlier register-spill baseline, not this specific change; the incremental gain from this step alone is ~1%.

> **Applicable when**: Triton reduction kernel already at near-peak memory throughput (Mem Busy >90%, achieved occupancy >95%, registers_per_thread ~16) where the existing autotune config set covers only small XBLOCK values and the problem may benefit from wider XBLOCK tiling. Marginal gains (~1%) are possible.

> **Avoid when**: When the kernel is already at its memory-bandwidth ceiling with the existing autotune configs (Mem Busy >90%, Duration stable). Adding more autotune configs at this point yields diminishing returns (<1% improvement). Do not apply when the kernel is register-limited (registers_per_thread >128) — large XBLOCK configs will worsen register spill. Do not apply when large XBLOCK × R0_BLOCK products would exceed register budget.

---

## Cumulative Progress

```
Baseline:  59030000 ns  (1.000×)
Iter 1     121020 ns  (487.771×)  — vectorized_loads_with_autotune
Iter 4     97630 ns  (604.630×)  — fix_autotune_grid_to_use_dynamic_xblock
Iter 5     96740 ns  (610.192×)  — add_triton_autotune

Final best: 96740 ns  (610.1923×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,34 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 64},  num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 4,   'R0_BLOCK': 64},  num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 4,   'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 8,   'R0_BLOCK': 64},  num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 8,   'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 64},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 256}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 256}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 256}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 64},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 256}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 512}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 4,   'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 64},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 512, 'R0_BLOCK': 64},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 1024,'R0_BLOCK': 64},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused_argmin_0(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 524160
@@ -26,8 +54,8 @@

         r0_mask = r0_index < r0_numel
         roffset = r0_offset
         rindex = r0_index
-        r0_2 = r0_index
-        tmp0 = tl.load(in_ptr0 + (x0 + 4095*r0_2 + 16773120*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        load_idx = x0 + 4095 * r0_index + 16773120 * x1
+        tmp0 = tl.load(in_ptr0 + load_idx, mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
         tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
         _tmp2_next, _tmp2_index_next = triton_helpers.minimum_with_index(
             _tmp2, _tmp2_index, tmp1, rindex
@@ -38,15 +66,13 @@

     tmp2 = tmp2_idx[:, None]
     tl.store(out_ptr0 + (x3), tmp2, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 524160
     r0_numel = 4096
     grid = (triton.cdiv(xnumel, 128),)
-    triton_red_fused_argmin_0[grid](arg_0, arg_1, 524160, 4096, XBLOCK=128, R0_BLOCK=512)
+    triton_red_fused_argmin_0[grid](arg_0, arg_1, 524160, 4096)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
