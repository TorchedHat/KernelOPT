# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 17250 ns — **Best latency**: 17020 ns (**1.0135×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`eliminate_modulo24_via_2d_grid_decomposition_static_channel_pid`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_019/kernel_2_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (23520,1,1), Waves/SM: 11.14
- **Registers/thread**: 17
- **Duration**: 17250 ns
- **SM Throughput**: 50.61 % | **Memory**: 22.46 % | **DRAM**: 14.25 %
- **Achieved Occupancy**: 56.38 %
- **IPC Active**: 2.32 inst/cycle | **No Eligible**: 41.11 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.728× | 0.705× | ✓ |  |
| B | autotune_xblock_to_hide_l1tex_latency | 0.741× | 0.720× | ✓ |  |
| C | vectorized_loads_larger_xblock_autotune | 0.739× | 0.735× | ✓ |  |
| D | increase_xblock_with_autotune_to_hide_l1tex_latency | 0.728× | 0.711× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_t... | Δ | vectorized_loads_... | Δ | increase_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17250 ns | 24480 ns | +7230 ns (+41.9%) | 23970 ns | +6720 ns (+39.0%) | 23460 ns | +6210 ns (+36.0%) | 24260 ns | +7010 ns (+40.6%) |
| SM Throughput (%) | 50.61 % | 78.64 % | +28.03 pp | 78.72 % | +28.11 pp | 78.29 % | +27.68 pp | 78.13 % | +27.52 pp |
| Memory Throughput (%) | 22.46 % | 22.90 % | +0.44 pp | 23.78 % | +1.32 pp | 24.05 % | +1.59 pp | 22.98 % | +0.52 pp |
| DRAM Throughput (%) | 14.25 % | 10.03 % | -4.22 pp | 10.25 % | -4.00 pp | 10.47 % | -3.78 pp | 10.12 % | -4.13 pp |
| Achieved Occupancy (%) | 56.38 % | 86.93 % | +30.55 pp | 86.18 % | +29.80 pp | 86.92 % | +30.54 pp | 86.63 % | +30.25 pp |
| Registers/Thread | 17.00 register/thread | 32.00 register/thread | +15 | 32.00 register/thread | +15 | 32.00 register/thread | +15 | 32.00 register/thread | +15 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 41.11 % | 14.08 % | -27.03 pp | 13.70 % | -27.41 pp | 13.77 % | -27.34 pp | 14.09 % | -27.02 pp |
| IPC Active | 2.32 inst/cycle | 3.43 inst/cycle | +1.11 | 3.41 inst/cycle | +1.09 | 3.40 inst/cycle | +1.08 | 3.44 inst/cycle | +1.12 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_with_autotune` (0.705×)

Increasing the XBLOCK tile size via autotune (from 128 to 512-2048) caused registers per thread to double from 17 to 32, which reduced the block limit from registers from 21 to 16 blocks/SM. Although achieved occupancy improved (56% → 87%) and eligible warps per scheduler increased (1.25 → 7.18), the total executed instruction count more than doubled (6.77M → 15.08M) and L1 hit rate dropped (79.9% → 63.1%), resulting in a 42% longer kernel duration (17.25us → 24.48us). The autotune decorator itself may have selected a suboptimal config, or the larger tile introduced additional computation/masking overhead that outweighed the latency-hiding benefit.

> **Applicable when**: Never: this specific change caused regression. In general, increasing XBLOCK is worth trying when register count is low (<20) and the tile increase does not push registers above ~24, keeping block-limit-registers above the warp-limit.

> **Avoid when**: Increasing XBLOCK causes register spill or doubles register count (e.g., 17→32), because the resulting reduction in block-limit-registers can negate occupancy gains and dramatically increase total instruction count. Also avoid when L1 hit rate is already high (>75%) and the kernel is not clearly memory-latency-bound — the larger tile may reduce cache efficiency. Do not apply when the instruction count increase from larger tiles outweighs latency-hiding benefits.

#### ❌ Plan B: `autotune_xblock_to_hide_l1tex_latency` (0.720×)

Increasing XBLOCK via autotune caused registers per thread to jump from 17 to 32, which halved the block limit from registers (21→16 blocks/SM) and more than doubled the total executed instructions (6.77M→14.63M). Although occupancy and eligible warps improved significantly, the 2.16x instruction count increase dominated, causing a 39% slowdown. For memory-bound pointwise kernels with very low register counts, adding autotune with large XBLOCK configs can trigger compiler register spilling or unrolling that inflates instruction count beyond what latency hiding can compensate.

> **Applicable when**: Kernel has high L1TEX stall fraction (>40% of stall cycles), low eligible warps per scheduler (<2), low achieved occupancy (<60%), and sufficient register headroom (registers/thread well below 32) to absorb larger tile sizes without spilling.

> **Avoid when**: Kernel already uses low registers per thread (≤17) where the compiler may aggressively unroll or spill when XBLOCK is increased 4-16x, causing instruction count to more than double. Also avoid when the kernel is already compute-bound or when the grid is large enough that occupancy is not the limiting factor. Do not add xmask guards when xnumel is exactly divisible by the original XBLOCK — the added predication overhead can compound with instruction inflation.

#### ❌ Plan D: `increase_xblock_with_autotune_to_hide_l1tex_latency` (0.711×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) caused the compiler to nearly double register usage per thread (17→32), which reduced Block Limit Registers from 21 to 16 and increased total executed instructions by 2.2x (6.77M→14.98M). Although occupancy improved (56%→87%) and eligible warps per scheduler rose (1.25→7.07), the massive instruction count increase dominated, extending kernel duration from 17.25 µs to 24.26 µs. The root cause is that larger XBLOCK tiles force the compiler to keep more live values in registers simultaneously, triggering instruction expansion even without register spilling to memory.

> **Applicable when**: Never apply this specific pattern (autotune large XBLOCK on pointwise kernels with low register count) when the compiler responds by nearly doubling register usage and instruction count.

> **Avoid when**: Kernel has low registers per thread (≤20) but autotune causes registers to jump to ≥28+; when executed instruction count increases by >50% after the change; when the kernel is a simple pointwise op with few arithmetic operations per load (low compute intensity per element). The instruction bloat from larger tiles outweighs the latency-hiding benefit.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | revert_xblock_and_fuse_bn_arithmetic_to_fma | 0.995× | 1.017× | ✓ |  |
| B | revert_autotune_to_restore_low_register_low_instruction_baseline | 0.987× | 0.995× | ✓ |  |
| C | remove_autotune_revert_xblock128_and_fuse_fp32_instructions | 1.008× | 1.019× | ✓ | **✓ best** |
| D | eliminate_non_fused_fp32_instructions_and_revert_xblock_regression | 1.007× | 1.019× | ✓ |  |
| E | add_triton_autotune | 0.985× | 1.008× | ✓ |  |


### Metric Comparison

| Metric | Baseline | revert_xblock_and... | Δ | revert_autotune_t... | Δ | remove_autotune_r... | Δ | eliminate_non_fus... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17250 ns | 16960 ns | -290 ns (-1.7%) | 17340 ns | +90 ns (+0.5%) | 16930 ns | -320 ns (-1.9%) | 16930 ns | -320 ns (-1.9%) | 17120 ns | -130 ns (-0.8%) |
| SM Throughput (%) | 50.61 % | 35.05 % | -15.56 pp | 50.63 % | +0.02 pp | 34.59 % | -16.02 pp | 34.05 % | -16.56 pp | 57.86 % | +7.25 pp |
| Memory Throughput (%) | 22.46 % | 22.20 % | -0.26 pp | 22.44 % | -0.02 pp | 22.40 % | -0.06 pp | 22.28 % | -0.18 pp | 22.28 % | -0.18 pp |
| DRAM Throughput (%) | 14.25 % | 14.50 % | +0.25 pp | 14.17 % | -0.08 pp | 14.51 % | +0.26 pp | 14.50 % | +0.25 pp | 14.36 % | +0.11 pp |
| Achieved Occupancy (%) | 56.38 % | 42.43 % | -13.95 pp | 56.39 % | +0.01 pp | 44.00 % | -12.38 pp | 40.86 % | -15.52 pp | 60.03 % | +3.65 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 17.00 register/thread | +0 | 16.00 register/thread | -1 | 16.00 register/thread | -1 | 18.00 register/thread | +1 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 41.11 % | 58.35 % | +17.24 pp | 41.51 % | +0.40 pp | 59.47 % | +18.36 pp | 66.63 % | +25.52 pp | 32.05 % | -9.06 pp |
| IPC Active | 2.32 inst/cycle | 1.65 inst/cycle | -0.67 | 2.31 inst/cycle | -0.01 | 1.61 inst/cycle | -0.71 | 1.32 inst/cycle | -1.00 | 2.71 inst/cycle | +0.39 |

---

## Iteration 3

**Best so far**: 1.0189× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_fixed_xblock_to_hide_l1tex_latency | 0.996× | 1.019× | ✓ |  |
| B | increase_num_warps_and_explicit_fma_to_hide_l1tex_latency | 0.999× | 1.011× | ✓ |  |
| C | increase_num_warps_to_hide_l1tex_latency_without_xblock_change | 0.998× | 1.004× | ✓ |  |
| D | fuse_bn_arithmetic_chain_to_fma_instructions | 0.989× | 1.009× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | increase_num_warp... | Δ | increase_num_warp... | Δ | fuse_bn_arithmeti... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17250 ns | 16930 ns | -320 ns (-1.9%) | 17060 ns | -190 ns (-1.1%) | 17180 ns | -70 ns (-0.4%) | 17090 ns | -160 ns (-0.9%) |
| SM Throughput (%) | 50.61 % | 42.03 % | -8.58 pp | 34.11 % | -16.50 pp | 35.22 % | -15.39 pp | 49.09 % | -1.52 pp |
| Memory Throughput (%) | 22.46 % | 22.31 % | -0.15 pp | 22.30 % | -0.16 pp | 22.35 % | -0.11 pp | 22.20 % | -0.26 pp |
| DRAM Throughput (%) | 14.25 % | 14.52 % | +0.27 pp | 14.41 % | +0.16 pp | 14.28 % | +0.03 pp | 14.39 % | +0.14 pp |
| Achieved Occupancy (%) | 56.38 % | 45.99 % | -10.39 pp | 41.79 % | -14.59 pp | 43.06 % | -13.32 pp | 55.31 % | -1.07 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 16.00 register/thread | -1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 41.11 % | 50.18 % | +9.07 pp | 66.04 % | +24.93 pp | 58.49 % | +17.38 pp | 43.63 % | +2.52 pp |
| IPC Active | 2.32 inst/cycle | 1.99 inst/cycle | -0.33 | 1.33 inst/cycle | -0.99 | 1.62 inst/cycle | -0.70 | 2.26 inst/cycle | -0.06 |

---

## Iteration 4

**Best so far**: 1.0189× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_reciprocal_sqrt_with_libdevice_rsqrt_to_fuse_fp32_instructions | 0.999× | 1.021× | ✓ |  |
| B | streaming_evict_first_on_main_load_to_reduce_l1tex_stall | 1.010× | 1.017× | ✓ | **✓ best** |
| C | evict_first_streaming_policy_for_non_reused_main_tensor | 0.987× | 1.002× | ✓ |  |
| D | eliminate_modulo24_channel_index_via_2d_pid_decomposition | 0.998× | 1.021× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_reciproca... | Δ | streaming_evict_f... | Δ | evict_first_strea... | Δ | eliminate_modulo2... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17250 ns | 16900 ns | -350 ns (-2.0%) | 16960 ns | -290 ns (-1.7%) | 17220 ns | -30 ns (-0.2%) | 16900 ns | -350 ns (-2.0%) |
| SM Throughput (%) | 50.61 % | 35.97 % | -14.64 pp | 33.60 % | -17.01 pp | 56.23 % | +5.62 pp | 45.02 % | -5.59 pp |
| Memory Throughput (%) | 22.46 % | 22.48 % | +0.02 pp | 21.98 % | -0.48 pp | 22.47 % | +0.01 pp | 22.36 % | -0.10 pp |
| DRAM Throughput (%) | 14.25 % | 14.54 % | +0.29 pp | 14.49 % | +0.24 pp | 14.27 % | +0.02 pp | 14.52 % | +0.27 pp |
| Achieved Occupancy (%) | 56.38 % | 43.40 % | -12.98 pp | 41.81 % | -14.57 pp | 56.48 % | +0.10 pp | 47.62 % | -8.76 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 16.00 register/thread | -1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 41.11 % | 57.50 % | +16.39 pp | 63.35 % | +22.24 pp | 35.19 % | -5.92 pp | 47.48 % | +6.37 pp |
| IPC Active | 2.32 inst/cycle | 1.69 inst/cycle | -0.63 | 1.48 inst/cycle | -0.84 | 2.60 inst/cycle | +0.28 | 2.11 inst/cycle | -0.21 |

---

## Iteration 5

**Best so far**: 1.0189× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_num_stages_software_pipelining_to_existing_autotune | 0.999× | 1.006× | ✓ |  |
| B | preload_bn_params_into_shared_memory_to_eliminate_l1tex_stalls | 1.000× | 1.006× | ✓ |  |
| C | eliminate_modulo24_via_2d_grid_decomposition_static_channel_pid | 1.010× | 1.014× | ✓ | **✓ best** |
| D | increase_block_warps_to_hide_l1tex_latency_without_xblock_change | 1.000× | 1.009× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_num_stages_so... | Δ | preload_bn_params... | Δ | eliminate_modulo2... | Δ | increase_block_wa... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17250 ns | 17150 ns | -100 ns (-0.6%) | 17150 ns | -100 ns (-0.6%) | 17020 ns | -230 ns (-1.3%) | 17090 ns | -160 ns (-0.9%) |
| SM Throughput (%) | 50.61 % | 35.97 % | -14.64 pp | 79.56 % | +28.95 pp | 34.11 % | -16.50 pp | 55.79 % | +5.18 pp |
| Memory Throughput (%) | 22.46 % | 22.46 % | +0.00 pp | 54.56 % | +32.10 pp | 22.37 % | -0.09 pp | 22.28 % | -0.18 pp |
| DRAM Throughput (%) | 14.25 % | 14.34 % | +0.09 pp | 14.30 % | +0.05 pp | 14.45 % | +0.20 pp | 14.38 % | +0.13 pp |
| Achieved Occupancy (%) | 56.38 % | 42.52 % | -13.86 pp | 53.83 % | -2.55 pp | 41.92 % | -14.46 pp | 56.27 % | -0.11 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 19.00 register/thread | +2 | 16.00 register/thread | -1 | 18.00 register/thread | +1 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 41.11 % | 57.59 % | +16.48 pp | 37.42 % | -3.69 pp | 63.04 % | +21.93 pp | 34.35 % | -6.76 pp |
| IPC Active | 2.32 inst/cycle | 1.68 inst/cycle | -0.64 | 2.50 inst/cycle | +0.18 | 1.48 inst/cycle | -0.84 | 2.60 inst/cycle | +0.28 |

---

## Cumulative Progress

```
Baseline:  17250 ns  (1.000×)
Iter 2     16930 ns  (1.019×)  — remove_autotune_revert_xblock128_and_fuse_fp32_instructions

Final best: 17020 ns  (1.0135×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,35 +5,26 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
-    xnumel = 3010560
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    # xnumel=3010560 is exactly divisible by XBLOCK=128 (3010560/128=23520) — no mask needed
     x2 = xindex
     x0 = (xindex % 24)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None)
+    tmp0 = tl.load(in_out_ptr0 + (x2), None, eviction_policy='evict_first')
     tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
     tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
     tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
     tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1], 1e-05, tl.float32)
-    tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
-    tmp16 = tl.full([1], 0, tl.int32)
-    tmp17 = tl.maximum(tmp16, tmp15)
-    tl.store(in_out_ptr0 + (x2), tmp17, None)
-
-
+    # FMA-friendly BN normalization
+    inv_std = tl.rsqrt(tmp3 + 1e-5)
+    # tl.math.fma(a,b,c) = a*b+c as single FFMA PTX instruction
+    norm   = tl.math.fma(-tmp1, inv_std, tmp0 * inv_std)  # (tmp0-tmp1)*inv_std
+    out    = tl.math.fma(norm, tmp12, tmp14)               # norm*tmp12+tmp14
+    result = tl.maximum(0.0, out)                          # ReLU
+    tl.store(in_out_ptr0 + (x2), result, None, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
