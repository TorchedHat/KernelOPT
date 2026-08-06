# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_clone_exp_mul_permute_unsqueeze_view_16` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 647140 ns — **Best latency**: 2080000 ns (**0.3111×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_clone_exp_mul_permute_unsqueeze_view_16`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_048/kernel_16_triton_poi_fused_add_clone_exp/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1048576,1,1), Waves/SM: 496.48
- **Registers/thread**: 16
- **Duration**: 647140 ns
- **SM Throughput**: 43.55 % | **Memory**: 50.59 % | **DRAM**: 50.59 %
- **Achieved Occupancy**: 51.70 %
- **IPC Active**: 1.75 inst/cycle | **No Eligible**: 56.41 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_larger_xblock | — | 0.311× | ✓ | **✓ best** |
| B | autotune_xblock_tile_size | — | 0.765× | ✓ |  |
| D | increase_xblock_tile_size_with_autotune | — | 0.767× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_xblock_t... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 647140 ns | 2080000 ns | +1432860 ns (+221.4%) | 845790 ns | +198650 ns (+30.7%) | 844220 ns | +197080 ns (+30.5%) |
| SM Throughput (%) | 43.55 % | 81.95 % | +38.40 pp | 56.24 % | +12.69 pp | 55.75 % | +12.20 pp |
| Memory Throughput (%) | 50.59 % | 31.61 % | -18.98 pp | 38.59 % | -12.00 pp | 38.65 % | -11.94 pp |
| DRAM Throughput (%) | 50.59 % | 15.65 % | -34.94 pp | 38.59 % | -12.00 pp | 38.65 % | -11.94 pp |
| Achieved Occupancy (%) | 51.70 % | 87.56 % | +35.86 pp | 46.31 % | -5.39 pp | 46.15 % | -5.55 pp |
| Registers/Thread | 16.00 register/thread | 32.00 register/thread | +16 | 25.00 register/thread | +9 | 25.00 register/thread | +9 |
| Waves/SM | 496.48 | 992.97 | +496.49 | 496.48 | +0.00 | 496.48 | +0.00 |
| No Eligible (%) | 56.41 % | 17.95 % | -38.46 pp | 44.44 % | -11.97 pp | 42.59 % | -13.82 pp |
| IPC Active | 1.75 inst/cycle | 3.28 inst/cycle | +1.53 | 2.24 inst/cycle | +0.49 | 2.24 inst/cycle | +0.49 |


### Key Learnings

#### ❌ Plan D: `increase_xblock_tile_size_with_autotune` (0.767×)

Adding @triton.autotune with larger XBLOCK values (512–2048) to a pointwise kernel that was stall-bound on L1TEX scoreboard waits (77% of stall cycles, L1 hit rate 5.77%) caused a 30% regression. Despite the stall diagnosis suggesting larger tiles would improve L1 reuse, the autotune selected a config that increased registers per thread from 16 to 25, which reduced achieved occupancy from 51.7% to 46.1% and increased total executed instructions by 65% (222M → 367M). The net effect was a 30% longer kernel duration (647 µs → 844 µs), because the instruction count explosion from larger tiles outweighed any latency hiding benefit, and the L1 hit rate barely improved (5.77% → 7.21%). When a kernel already has very high wave count (496 waves/SM) and the bottleneck is L1TEX stalls, increasing tile size inflates instruction count and register pressure without meaningfully improving cache hit rates, resulting in regression.

> **Applicable when**: Kernel has L1TEX scoreboard stalls >70% of stall cycles, low L1 hit rate (<10%), and small tile size relative to working set. Registers per thread are low (<20) leaving headroom for larger tiles.

> **Avoid when**: Kernel already has very high wave count (>400 waves/SM) and the broadcast load pattern does not have strong spatial locality within a tile. Increasing XBLOCK causes register pressure to rise (e.g., 16→25 registers/thread), which reduces occupancy and inflates instruction count without proportional cache hit improvement. Also avoid when xnumel is exactly divisible by the original XBLOCK (no masking needed), as adding xmask introduces unnecessary predication overhead.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_multiply_add_to_fma_reduce_instruction_count | — | 0.765× | ✓ |  |
| B | hoist_broadcast_load_to_scalar_reduce_l1tex_stalls | — | 1.000× | ✓ |  |
| C | fix_multiple_of_alignment_hint_to_enable_vectorized_loads_and_fma | — | 0.335× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_multiply_add... | Δ | hoist_broadcast_l... | Δ | fix_multiple_of_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 647140 ns | 845760 ns | +198620 ns (+30.7%) | 647420 ns | +280 ns (+0.0%) | 1930000 ns | +1282860 ns (+198.2%) |
| SM Throughput (%) | 43.55 % | 55.81 % | +12.26 pp | 45.05 % | +1.50 pp | 79.39 % | +35.84 pp |
| Memory Throughput (%) | 50.59 % | 38.59 % | -12.00 pp | 50.54 % | -0.05 pp | 34.26 % | -16.33 pp |
| DRAM Throughput (%) | 50.59 % | 38.59 % | -12.00 pp | 50.54 % | -0.05 pp | 16.94 % | -33.65 pp |
| Achieved Occupancy (%) | 51.70 % | 46.48 % | -5.22 pp | 53.36 % | +1.66 pp | 86.23 % | +34.53 pp |
| Registers/Thread | 16.00 register/thread | 25.00 register/thread | +9 | 16.00 register/thread | +0 | 32.00 register/thread | +16 |
| Waves/SM | 496.48 | 496.48 | +0.00 | 496.48 | +0.00 | 992.97 | +496.49 |
| No Eligible (%) | 56.41 % | 43.46 % | -12.95 pp | 54.79 % | -1.62 pp | 20.68 % | -35.73 pp |
| IPC Active | 1.75 inst/cycle | 2.26 inst/cycle | +0.51 | 1.81 inst/cycle | +0.06 | 3.18 inst/cycle | +1.43 |


### Key Learnings

#### ❌ Plan A: `fuse_multiply_add_to_fma_reduce_instruction_count` (0.765×)

Replacing a separate MUL+ADD pair with `libdevice.fma()` in a Triton kernel produced zero measurable improvement (0.77x speedup, effectively neutral to slightly regressed). The profiling data shows that total executed instructions remained identical (367,001,600 before and after), meaning the compiler already fused the operations or the FMA call did not reduce PTX instruction count. All key metrics — duration, compute throughput, DRAM throughput, warp cycles per instruction, eligible warps per scheduler — were statistically unchanged. When a kernel is bottlenecked by L1TEX scoreboard stalls (~70% of stall cycles, 9.1 cycles avg stall per warp), reducing ALU instruction count via FMA does not help because the critical path is memory latency, not compute throughput.

> **Applicable when**: Never effective in this configuration. FMA fusion via libdevice.fma() may help only when the kernel is compute-bound (ALU utilization >80%) and the compiler has not already fused the operations.

> **Avoid when**: Kernel is bottlenecked by L1TEX scoreboard stalls (CPIStall rule triggered, stall fraction >60% of warp cycles). In this regime, reducing ALU instruction count via FMA does not shorten the critical path. Also avoid when the NCU FPInstructions rule reports a potential gain but total executed instruction count is already high (367M+) and warp cycles per instruction are dominated by memory latency (>9 cycles avg L1TEX stall) — the compiler may already be fusing what it can, and the remaining non-fused instructions are not on the critical path.

#### ❌ Plan C: `fix_multiple_of_alignment_hint_to_enable_vectorized_loads_and_fma` (0.335×)

Changing tl.multiple_of(xindex, 4) to tl.multiple_of(xindex, XBLOCK) and collapsing the multiply-add into a single FMA expression did not fix the instruction explosion — executed instructions increased ~3.3x (367M → 1201M) and kernel duration more than doubled (844 µs → 1930 µs). The autotune selected block_size=256 which pushed registers per thread from 25 to 32, halving Block Limit Registers (16→8) and Block Limit Warps (16→8), doubling waves_per_SM (497→993). Despite higher achieved occupancy (46% → 86%) and compute throughput (56% → 79%), the dominant effect was the massive instruction count increase driven by the larger XBLOCK tile size combined with higher register pressure. The stall pattern also shifted from L1TEX scoreboard stalls (before) to 'not selected' warp stalls (after), indicating the kernel became over-subscribed with warps rather than latency-bound.

> **Applicable when**: Never — this combination of tl.max_contiguous + tl.multiple_of(xindex, XBLOCK) with autotune configs selecting larger block sizes causes register pressure to spike (25→32 regs/thread), which doubles waves_per_SM and inflates instruction count by 3x even with the corrected alignment hint.

> **Avoid when**: Do not apply tl.multiple_of(xindex, XBLOCK) with autotune when the kernel already has moderate-to-high register pressure (≥20 regs/thread). The autotune will select larger XBLOCK values that push registers per thread to 32+, halving Block Limit Registers and Warps, and causing instruction count explosion. The alignment hint fix alone does not overcome the register pressure regression from larger tile sizes. Also avoid when the kernel is already running at high wave count (>400 waves/SM) — doubling waves_per_SM via larger block sizes does not improve throughput and inflates total work.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | hoist_broadcast_scalar_load_eliminate_tl_where_overhead | — | 0.978× | ✓ |  |
| B | scalar_hoist_broadcast_load_split_tile_halves | — | 0.998× | ✓ |  |
| C | cap_register_pressure_with_maxnreg_to_reduce_instruction_explosion | — | 0.510× | ✓ |  |
| D | add_triton_autotune | — | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | hoist_broadcast_s... | Δ | scalar_hoist_broa... | Δ | cap_register_pres... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 647140 ns | 661730 ns | +14590 ns (+2.3%) | 648450 ns | +1310 ns (+0.2%) | 1270000 ns | +622860 ns (+96.2%) | 647710 ns | +570 ns (+0.1%) |
| SM Throughput (%) | 43.55 % | 49.02 % | +5.47 pp | 41.05 % | -2.50 pp | 73.33 % | +29.78 pp | 45.37 % | +1.82 pp |
| Memory Throughput (%) | 50.59 % | 49.47 % | -1.12 pp | 50.45 % | -0.14 pp | 28.71 % | -21.88 pp | 50.52 % | -0.07 pp |
| DRAM Throughput (%) | 50.59 % | 49.47 % | -1.12 pp | 50.45 % | -0.14 pp | 25.64 % | -24.95 pp | 50.52 % | -0.07 pp |
| Achieved Occupancy (%) | 51.70 % | 80.47 % | +28.77 pp | 53.96 % | +2.26 pp | 79.14 % | +27.44 pp | 53.48 % | +1.78 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 23.00 register/thread | +7 | 16.00 register/thread | +0 |
| Waves/SM | 496.48 | 496.48 | +0.00 | 496.48 | +0.00 | 992.97 | +496.49 | 496.48 | +0.00 |
| No Eligible (%) | 56.41 % | 50.84 % | -5.57 pp | 58.90 % | +2.49 pp | 26.10 % | -30.31 pp | 54.41 % | -2.00 pp |
| IPC Active | 1.75 inst/cycle | 1.97 inst/cycle | +0.22 | 1.64 inst/cycle | -0.11 | 2.95 inst/cycle | +1.20 | 1.82 inst/cycle | +0.07 |


### Key Learnings

#### ❌ Plan C: `cap_register_pressure_with_maxnreg_to_reduce_instruction_explosion` (0.510×)

Adding maxnreg=24 to autotune configs with XBLOCK values of 256–1024 caused a 2x slowdown (647 µs → 1270 µs) despite the intent to reduce register pressure. The autotune selected block_size=256 with 23 registers/thread, which doubled waves_per_SM (497→993) and tripled total executed instructions (231M→732M). Although scheduler eligibility improved dramatically (0.75→5.41 eligible warps/scheduler) and compute throughput rose (45%→73%), the dominant effect was the instruction count explosion from the larger tile size — the kernel did 3x more total work. The stall pattern shifted from L1TEX scoreboard stalls (before) to 'not selected' warp stalls (after), confirming the kernel became over-subscribed rather than latency-bound. maxnreg capping does not prevent instruction explosion when the autotune selects a larger XBLOCK that doubles the per-CTA work and doubles the grid's waves_per_SM.

> **Applicable when**: Never effective for this kernel pattern. maxnreg capping is theoretically useful when high register count (≥32) is the sole cause of reduced Block Limit Registers, and the autotune would otherwise select the same XBLOCK tile size.

> **Avoid when**: Kernel already runs at high wave count (>400 waves/SM) with a small fixed XBLOCK. Adding maxnreg to autotune configs does not prevent the autotune from selecting a larger XBLOCK that multiplies total executed instructions. When the baseline kernel uses XBLOCK=128 with 16 regs/thread and 497 waves/SM, any autotune config that selects XBLOCK≥256 will double waves_per_SM and triple instruction count regardless of the maxnreg cap. Also avoid when the grid is fixed (not a lambda) — the grid must adapt to the chosen XBLOCK or the kernel will process each element multiple times.

---

## Cumulative Progress

```
Baseline:  647140 ns  (1.000×)

Final best: 2080000 ns  (0.3111×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,27 +5,29 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused_add_clone_exp_mul_permute_unsqueeze_view_16(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 134217728
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x4 = xindex
-    x5 = xindex // 64
     x0 = (xindex % 64)
     x1 = ((xindex // 64) % 64)
     x2 = ((xindex // 4096) % 8)
     x3 = xindex // 32768
-    tmp0 = tl.load(in_ptr0 + (x4), None)
-    tmp1 = tl.load(in_ptr1 + (x4), None)
-    tmp2 = tl.load(in_ptr2 + (x5), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x4), xmask)
+    tmp1 = tl.load(in_ptr1 + (x4), xmask)
+    x5_group0 = xoffset // 64
+    x5_group1 = xoffset // 64 + 1
+    tmp2_g0 = tl.load(in_ptr2 + x5_group0, mask=True)
+    tmp2_g1 = tl.load(in_ptr2 + x5_group1, mask=True)
+    tmp2 = tl.where(tl.arange(0, XBLOCK) < 64, tmp2_g0, tmp2_g1)
     tmp3 = tl.exp(tmp2)
     tmp4 = tmp1 * tmp3
     tmp5 = tmp0 + tmp4
-    tl.store(out_ptr0 + (x0 + 64*x2 + 512*x1 + 32768*x3), tmp5, None)
-
-
+    tl.store(out_ptr0 + (x0 + 64*x2 + 512*x1 + 32768*x3), tmp5, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
