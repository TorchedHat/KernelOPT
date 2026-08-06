# Kernel Optimization Log

## Summary

- **Kernel**: `conv2d_implicit_gemm_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 12780000 ns — **Best latency**: 12780000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `conv2d_implicit_gemm_kernel`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_005/synthesis/kernel_10_fused_fused_convolution.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (96800,1,1), Waves/SM: 244.44
- **Registers/thread**: 139
- **Duration**: 12780000 ns
- **SM Throughput**: 69.30 % | **Memory**: 33.34 % | **DRAM**: 3.84 %
- **Achieved Occupancy**: 18.68 %
- **IPC Active**: 2.46 inst/cycle | **No Eligible**: 38.42 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores_conv2d_dot_minimal_change | — | fail | ✓ |  |
| B | fast_accumulation_fp16_tl_dot_conv2d | 0.837× | 1.000× | ✓ |  |
| C | fp8_precision_tl_dot_hopper | — | fail | ✓ |  |
| D | reduce_register_pressure_maxnreg_conv2d_implicit_gemm | 1.006× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fast_accumulation... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 12780000 ns | 12780000 ns | +0 ns (+0.0%) | 12780000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 69.30 % | 69.31 % | +0.01 pp | 69.32 % | +0.02 pp |
| Memory Throughput (%) | 33.34 % | 33.35 % | +0.01 pp | 33.35 % | +0.01 pp |
| DRAM Throughput (%) | 3.84 % | 3.84 % | +0.00 pp | 3.84 % | +0.00 pp |
| Achieved Occupancy (%) | 18.68 % | 18.67 % | -0.01 pp | 18.67 % | -0.01 pp |
| Registers/Thread | 139.00 register/thread | 139.00 register/thread | +0 | 139.00 register/thread | +0 |
| Waves/SM | 244.44 | 244.44 | +0.00 | 244.44 | +0.00 |
| No Eligible (%) | 38.42 % | 38.41 % | -0.01 pp | 38.40 % | -0.02 pp |
| IPC Active | 2.46 inst/cycle | 2.46 inst/cycle | +0.00 | 2.46 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_smaller_tiles_and_more_warps_conv2d | 1.018× | 1.000× | ✓ |  |
| C | hoist_loop_invariant_k_decode_and_expand_autotune_stages | 1.021× | 1.000× | ✓ |  |
| E | add_triton_autotune | 1.022× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | hoist_loop_invari... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12780000 ns | 12780000 ns | +0 ns (+0.0%) | 12780000 ns | +0 ns (+0.0%) | 12780000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 69.30 % | 69.30 % | +0.00 pp | 69.29 % | -0.01 pp | 69.29 % | -0.01 pp |
| Memory Throughput (%) | 33.34 % | 33.34 % | +0.00 pp | 33.33 % | -0.01 pp | 33.33 % | -0.01 pp |
| DRAM Throughput (%) | 3.84 % | 3.84 % | +0.00 pp | 3.84 % | +0.00 pp | 3.84 % | +0.00 pp |
| Achieved Occupancy (%) | 18.68 % | 18.67 % | -0.01 pp | 18.67 % | -0.01 pp | 18.67 % | -0.01 pp |
| Registers/Thread | 139.00 register/thread | 139.00 register/thread | +0 | 139.00 register/thread | +0 | 139.00 register/thread | +0 |
| Waves/SM | 244.44 | 244.44 | +0.00 | 244.44 | +0.00 | 244.44 | +0.00 |
| No Eligible (%) | 38.42 % | 38.42 % | +0.00 pp | 38.42 % | +0.00 pp | 38.41 % | -0.01 pp |
| IPC Active | 2.46 inst/cycle | 2.46 inst/cycle | +0.00 | 2.46 inst/cycle | +0.00 | 2.46 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tf32_tensor_cores_tl_dot_conv2d_and_linear | 1.010× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 12780000 ns | 12780000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 69.30 % | 69.29 % | -0.01 pp |
| Memory Throughput (%) | 33.34 % | 33.34 % | +0.00 pp |
| DRAM Throughput (%) | 3.84 % | 3.84 % | +0.00 pp |
| Achieved Occupancy (%) | 18.68 % | 18.67 % | -0.01 pp |
| Registers/Thread | 139.00 register/thread | 139.00 register/thread | +0 |
| Waves/SM | 244.44 | 244.44 | +0.00 |
| No Eligible (%) | 38.42 % | 38.42 % | +0.00 pp |
| IPC Active | 2.46 inst/cycle | 2.46 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | nhwc_output_layout_coalesced_store_conv2d | 1.007× | 1.000× | ✓ |  |
| C | hoist_k_decode_outside_loop_conv2d_implicit_gemm | 1.019× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | nhwc_output_layou... | Δ | hoist_k_decode_ou... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 12780000 ns | 12780000 ns | +0 ns (+0.0%) | 12770000 ns | -10000 ns (-0.1%) |
| SM Throughput (%) | 69.30 % | 69.32 % | +0.02 pp | 69.32 % | +0.02 pp |
| Memory Throughput (%) | 33.34 % | 33.35 % | +0.01 pp | 33.35 % | +0.01 pp |
| DRAM Throughput (%) | 3.84 % | 3.84 % | +0.00 pp | 3.84 % | +0.00 pp |
| Achieved Occupancy (%) | 18.68 % | 18.67 % | -0.01 pp | 18.67 % | -0.01 pp |
| Registers/Thread | 139.00 register/thread | 139.00 register/thread | +0 | 139.00 register/thread | +0 |
| Waves/SM | 244.44 | 244.44 | +0.00 | 244.44 | +0.00 |
| No Eligible (%) | 38.42 % | 38.40 % | -0.02 pp | 38.39 % | -0.03 pp |
| IPC Active | 2.46 inst/cycle | 2.46 inst/cycle | +0.00 | 2.46 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  12780000 ns  (1.000×)

Final best: 12780000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,7 @@

 import argparse
 
 # ─── Fused bias + ReLU pointwise kernel ───────────────────────────────────────
+
 @triton.autotune(
     configs=[
         triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
@@ -31,7 +32,6 @@

     tl.store(out_ptr + offs, out, mask=mask)
 
 
-# ─── ReLU pointwise kernel ────────────────────────────────────────────────────
 @triton.autotune(
     configs=[
         triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
@@ -42,6 +42,7 @@

     key=['numel'],
 )
 @triton.jit
+
 def relu_kernel(
     x_ptr, out_ptr,
     numel,
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
