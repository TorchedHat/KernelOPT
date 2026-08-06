# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_convolution_mul_native_group_norm_sigmoid_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 892930 ns — **Best latency**: 889790 ns (**1.0035×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`autotune_tile_sizes_to_reduce_register_pressure_and_improve_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_convolution_mul_native_group_norm_sigmoid_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_060/kernel_0_triton_red_fused_convolution_m/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 0.97
- **Registers/thread**: 128
- **Duration**: 892930 ns
- **SM Throughput**: 58.02 % | **Memory**: 23.03 % | **DRAM**: 23.03 %
- **Achieved Occupancy**: 21.37 %
- **IPC Active**: 2.40 inst/cycle | **No Eligible**: 40.03 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_r0block_tile_sizes | 0.701× | 0.703× | ✓ |  |
| B | autotune_tile_sizes_to_reduce_register_pressure_and_improve_occupancy | 1.114× | 1.004× | ✓ | **✓ best** |
| C | add_autotune_increase_xblock_reduce_register_pressure | 1.000× | 1.003× | ✓ |  |
| D | reduce_register_pressure_via_autotune_tile_sizes | 1.060× | 0.984× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_r... | Δ | autotune_tile_siz... | Δ | add_autotune_incr... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 892930 ns | 1270000 ns | +377070 ns (+42.2%) | 889790 ns | -3140 ns (-0.4%) | 890590 ns | -2340 ns (-0.3%) | 907330 ns | +14400 ns (+1.6%) |
| SM Throughput (%) | 58.02 % | 42.71 % | -15.31 pp | 58.15 % | +0.13 pp | 58.39 % | +0.37 pp | 57.23 % | -0.79 pp |
| Memory Throughput (%) | 23.03 % | 16.24 % | -6.79 pp | 23.10 % | +0.07 pp | 23.08 % | +0.05 pp | 22.66 % | -0.37 pp |
| DRAM Throughput (%) | 23.03 % | 16.24 % | -6.79 pp | 23.10 % | +0.07 pp | 23.08 % | +0.05 pp | 22.66 % | -0.37 pp |
| Achieved Occupancy (%) | 21.37 % | 11.70 % | -9.67 pp | 21.38 % | +0.01 pp | 21.38 % | +0.01 pp | 23.00 % | +1.63 pp |
| Registers/Thread | 128.00 register/thread | 255.00 register/thread | +127 | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 40.00 register/thread | -88 |
| Waves/SM | 0.97 | 1.94 | +0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.32 | -0.65 |
| No Eligible (%) | 40.03 % | 56.36 % | +16.33 pp | 40.05 % | +0.02 pp | 40.02 % | -0.01 pp | 41.34 % | +1.31 pp |
| IPC Active | 2.40 inst/cycle | 1.75 inst/cycle | -0.65 | 2.40 inst/cycle | +0.00 | 2.40 inst/cycle | +0.00 | 2.35 inst/cycle | -0.05 |


### Key Learnings

#### ❌ Plan A: `autotune_xblock_r0block_tile_sizes` (0.703×)

Adding @triton.autotune with larger XBLOCK and R0_BLOCK configs caused the compiler to allocate 255 registers per thread (up from 128), halving the block limit from 4 to 2 blocks/SM and dropping theoretical occupancy from 25% to 12.5%. This worsened scheduler eligibility (No Eligible rose from 40% to 56%) and increased kernel duration by ~42% despite waves/SM doubling from 0.97 to 1.94. The root cause is that autotuning configs with larger tile sizes (e.g. R0_BLOCK=4096, 8192) force the compiler to keep more live values in registers simultaneously, exhausting the register file and causing spills that dominate over any coalescing or wave-count benefit.

> **Applicable when**: Never apply this specific pattern when the kernel already uses 128+ registers/thread and the register block limit is already the binding occupancy constraint (Block Limit Registers < Block Limit Warps). Autotune with large tile configs is only safe when register pressure is low (e.g. <64 regs/thread) and occupancy is not register-limited.

> **Avoid when**: Kernel already has registers_per_thread >= 128 and Block Limit Registers is the binding occupancy constraint. Adding autotune configs with R0_BLOCK > current value will push register count to 255 (hardware max), halving occupancy and worsening scheduler stall rates. Also avoid when the kernel has a Welford reduction accumulating multiple live tensors across a large R0_BLOCK loop body.

---

## Iteration 2

**Best so far**: 1.0035× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_manual_silu_with_libdevice_fused_instruction | 1.114× | 1.002× | ✓ |  |
| B | fix_uncoalesced_stores_with_streaming_eviction_policy | 0.993× | 1.003× | ✓ |  |
| C | increase_num_warps_and_enable_software_pipelining_to_hide_l1tex_latency | 1.058× | 0.984× | ✓ |  |
| D | restrict_r0block_configs_to_restore_register_budget | 1.113× | 0.980× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_manual_si... | Δ | fix_uncoalesced_s... | Δ | increase_num_warp... | Δ | restrict_r0block_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 892930 ns | 891200 ns | -1730 ns (-0.2%) | 890050 ns | -2880 ns (-0.3%) | 907580 ns | +14650 ns (+1.6%) | 911010 ns | +18080 ns (+2.0%) |
| SM Throughput (%) | 58.02 % | 58.34 % | +0.32 pp | 58.24 % | +0.22 pp | 57.09 % | -0.93 pp | 57.19 % | -0.83 pp |
| Memory Throughput (%) | 23.03 % | 23.07 % | +0.04 pp | 23.10 % | +0.07 pp | 22.65 % | -0.38 pp | 22.57 % | -0.46 pp |
| DRAM Throughput (%) | 23.03 % | 23.07 % | +0.04 pp | 23.10 % | +0.07 pp | 22.65 % | -0.38 pp | 22.57 % | -0.46 pp |
| Achieved Occupancy (%) | 21.37 % | 21.38 % | +0.01 pp | 21.38 % | +0.01 pp | 23.00 % | +1.63 pp | 23.00 % | +1.63 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 40.00 register/thread | -88 | 40.00 register/thread | -88 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.32 | -0.65 | 0.32 | -0.65 |
| No Eligible (%) | 40.03 % | 40.06 % | +0.03 pp | 40.05 % | +0.02 pp | 41.26 % | +1.23 pp | 41.32 % | +1.29 pp |
| IPC Active | 2.40 inst/cycle | 2.40 inst/cycle | +0.00 | 2.40 inst/cycle | +0.00 | 2.35 inst/cycle | -0.05 | 2.35 inst/cycle | -0.05 |

---

## Iteration 3

**Best so far**: 1.0035× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_enable_autotune_xblock_coalescing | 1.000× | 1.002× | ✓ |  |
| B | fix_uncoalesced_stores_and_improve_l1_reuse_via_eviction_policy | 1.114× | 1.002× | ✓ |  |
| C | fix_strided_output_stores_with_contiguous_xblock_tiling | 1.060× | 0.982× | ✓ |  |
| D | enable_software_pipelining_via_num_stages2_with_tl_multiple_of_hints | 1.110× | 0.979× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | fix_uncoalesced_s... | Δ | fix_strided_outpu... | Δ | enable_software_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 892930 ns | 891010 ns | -1920 ns (-0.2%) | 891360 ns | -1570 ns (-0.2%) | 909310 ns | +16380 ns (+1.8%) | 912380 ns | +19450 ns (+2.2%) |
| SM Throughput (%) | 58.02 % | 58.27 % | +0.25 pp | 58.06 % | +0.04 pp | 57.21 % | -0.81 pp | 57.25 % | -0.77 pp |
| Memory Throughput (%) | 23.03 % | 23.07 % | +0.04 pp | 23.07 % | +0.04 pp | 22.61 % | -0.42 pp | 22.54 % | -0.49 pp |
| DRAM Throughput (%) | 23.03 % | 23.07 % | +0.04 pp | 23.07 % | +0.04 pp | 22.61 % | -0.42 pp | 22.54 % | -0.49 pp |
| Achieved Occupancy (%) | 21.37 % | 21.37 % | +0.00 pp | 21.38 % | +0.01 pp | 23.01 % | +1.64 pp | 23.01 % | +1.64 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 40.00 register/thread | -88 | 40.00 register/thread | -88 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.32 | -0.65 | 0.32 | -0.65 |
| No Eligible (%) | 40.03 % | 40.03 % | +0.00 pp | 40.03 % | +0.00 pp | 41.32 % | +1.29 pp | 41.25 % | +1.22 pp |
| IPC Active | 2.40 inst/cycle | 2.40 inst/cycle | +0.00 | 2.40 inst/cycle | +0.00 | 2.35 inst/cycle | -0.05 | 2.35 inst/cycle | -0.05 |

---

## Cumulative Progress

```
Baseline:  892930 ns  (1.000×)
Iter 1     889790 ns  (1.004×)  — autotune_tile_sizes_to_reduce_register_pressure_and_improve_occupancy

Final best: 889790 ns  (1.0035×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,19 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 2048}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 1024}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 256},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 256},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 128},  num_warps=8,  num_stages=1),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused_convolution_mul_native_group_norm_sigmoid_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 512
@@ -28,8 +41,8 @@

         rindex = r0_index
         r0_5 = r0_index
         r0_3 = r0_index // 123039
-        tmp0 = tl.load(in_ptr0 + (r0_5 + 492156*x4), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
-        tmp1 = tl.load(in_ptr1 + (r0_3 + 4*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp0 = tl.load(in_ptr0 + (r0_5 + 492156*x4), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp1 = tl.load(in_ptr1 + (r0_3 + 4*x0), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
         tmp2 = tmp0 + tmp1
         tmp3 = tl.sigmoid(tmp2)
         tmp4 = tmp3 * tmp2
@@ -44,10 +57,8 @@

     tmp6 = tmp7[:, None]
     tmp10 = tmp8[:, None]
     tmp11 = tmp9[:, None]
-    tl.store(out_ptr0 + (x4), tmp6, xmask)
-    tl.store(out_ptr1 + (x4), tmp10, xmask)
-
-
+    tl.store(out_ptr0 + (x4), tmp6, mask=xmask, eviction_policy='evict_first')
+    tl.store(out_ptr1 + (x4), tmp10, mask=xmask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -57,7 +68,7 @@

     xnumel = 512
     r0_numel = 492156
     grid = (triton.cdiv(xnumel, 1),)
-    triton_red_fused_convolution_mul_native_group_norm_sigmoid_0[grid](arg_0, arg_1, arg_2, arg_3, 512, 492156, XBLOCK=1, R0_BLOCK=2048)
+    triton_red_fused_convolution_mul_native_group_norm_sigmoid_0[grid](arg_0, arg_1, arg_2, arg_3, 512, 492156)
     return arg_3
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
