# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 17020000 ns — **Best latency**: 16860000 ns (**1.0095×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`add_triton_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_086/kernel_1_triton_tem_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (131072,1,1), Waves/SM: 496.48
- **Registers/thread**: 255
- **Duration**: 17020000 ns
- **SM Throughput**: 37.55 % | **Memory**: 62.08 % | **DRAM**: 16.95 %
- **Achieved Occupancy**: 12.44 %
- **IPC Active**: 1.50 inst/cycle | **No Eligible**: 62.44 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 0.998× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 17020000 ns | 17050000 ns | +30000 ns (+0.2%) |
| SM Throughput (%) | 37.55 % | 37.49 % | -0.06 pp |
| Memory Throughput (%) | 62.08 % | 62.01 % | -0.07 pp |
| DRAM Throughput (%) | 16.95 % | 16.94 % | -0.01 pp |
| Achieved Occupancy (%) | 12.44 % | 12.44 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 496.48 | 496.48 | +0.00 |
| No Eligible (%) | 62.44 % | 62.35 % | -0.09 pp |
| IPC Active | 1.50 inst/cycle | 1.50 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | eviction_policies_streaming_A_evict_first_B_evict_last | 1.000× | 0.997× | ✓ |  |
| D | fix_b_matrix_load_correctness_and_coalescing | 1.001× | 0.997× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 17020000 ns | 17080000 ns | +60000 ns (+0.4%) | 17080000 ns | +60000 ns (+0.4%) |
| SM Throughput (%) | 37.55 % | 37.47 % | -0.08 pp | 37.64 % | +0.09 pp |
| Memory Throughput (%) | 62.08 % | 61.96 % | -0.12 pp | 62.42 % | +0.34 pp |
| DRAM Throughput (%) | 16.95 % | 16.91 % | -0.04 pp | 17.19 % | +0.24 pp |
| Achieved Occupancy (%) | 12.44 % | 12.44 % | +0.00 pp | 12.43 % | -0.01 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 496.48 | 496.48 | +0.00 | 496.48 | +0.00 |
| No Eligible (%) | 62.44 % | 62.40 % | -0.04 pp | 62.25 % | -0.19 pp |
| IPC Active | 1.50 inst/cycle | 1.50 inst/cycle | +0.00 | 1.51 inst/cycle | +0.01 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | tma_loads | 1.007× | 1.006× | ✓ |  |
| E | add_triton_autotune | 1.010× | 1.010× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tma_loads | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 17020000 ns | 16910000 ns | -110000 ns (-0.6%) | 16860000 ns | -160000 ns (-0.9%) |
| SM Throughput (%) | 37.55 % | 38.32 % | +0.77 pp | 37.85 % | +0.30 pp |
| Memory Throughput (%) | 62.08 % | 61.67 % | -0.41 pp | 60.68 % | -1.40 pp |
| DRAM Throughput (%) | 16.95 % | 16.05 % | -0.90 pp | 17.02 % | +0.07 pp |
| Achieved Occupancy (%) | 12.44 % | 12.43 % | -0.01 pp | 12.44 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 496.48 | 496.48 | +0.00 | 496.48 | +0.00 |
| No Eligible (%) | 62.44 % | 61.78 % | -0.66 pp | 62.00 % | -0.44 pp |
| IPC Active | 1.50 inst/cycle | 1.53 inst/cycle | +0.03 | 1.52 inst/cycle | +0.02 |

---

## Iteration 4

**Best so far**: 1.0095× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_load_address_precise_minimal_diff | 1.010× | 1.007× | ✓ |  |
| E | add_triton_autotune | 1.010× | 1.013× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 17020000 ns | 16900000 ns | -120000 ns (-0.7%) | 16800000 ns | -220000 ns (-1.3%) |
| SM Throughput (%) | 37.55 % | 37.97 % | +0.42 pp | 37.92 % | +0.37 pp |
| Memory Throughput (%) | 62.08 % | 60.88 % | -1.20 pp | 60.82 % | -1.26 pp |
| DRAM Throughput (%) | 16.95 % | 16.98 % | +0.03 pp | 17.08 % | +0.13 pp |
| Achieved Occupancy (%) | 12.44 % | 12.44 % | +0.00 pp | 12.44 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 496.48 | 496.48 | +0.00 | 496.48 | +0.00 |
| No Eligible (%) | 62.44 % | 62.10 % | -0.34 pp | 62.07 % | -0.37 pp |
| IPC Active | 1.50 inst/cycle | 1.52 inst/cycle | +0.02 | 1.52 inst/cycle | +0.02 |

---

## Iteration 5

**Best so far**: 1.0095× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | cap_registers_with_maxnreg_to_raise_occupancy | 0.914× | 0.308× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_registers_wit... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 17020000 ns | 55290000 ns | +38270000 ns (+224.9%) |
| SM Throughput (%) | 37.55 % | 22.28 % | -15.27 pp |
| Memory Throughput (%) | 62.08 % | 46.72 % | -15.36 pp |
| DRAM Throughput (%) | 16.95 % | 46.72 % | +29.77 pp |
| Achieved Occupancy (%) | 12.44 % | 24.96 % | +12.52 pp |
| Registers/Thread | 255.00 register/thread | 128.00 register/thread | -127 |
| Waves/SM | 496.48 | 248.24 | -248.24 |
| No Eligible (%) | 62.44 % | 83.98 % | +21.54 pp |
| IPC Active | 1.50 inst/cycle | 0.64 inst/cycle | -0.86 |


### Key Learnings

#### ❌ Plan C: `cap_registers_with_maxnreg_to_raise_occupancy` (0.308×)

Adding maxnreg=128 to Triton configs successfully doubled theoretical occupancy (12.5%→25%) and halved registers/thread (255→128), but caused a 3.25x slowdown. The no-eligible-warp fraction worsened from 62% to 84%, warp cycles per issued instruction jumped from 5.26 to 24.88, and DRAM throughput rose from 17% to 47% — indicating that the compiler aggressively spilled register-resident accumulator values to local memory (DRAM), generating ~6x more L2 traffic (524M→2990M sectors). The occupancy gain was completely overwhelmed by the spill traffic: when the accumulator tile is large (e.g. 128×128 FP32 = 16,384 values), capping registers forces massive spilling that is far more costly than the latency-hiding benefit of extra warps.

> **Applicable when**: Never applicable when the accumulator tile is large (BLOCK_M * BLOCK_N > ~4096 FP32 values). The spill traffic from capping registers on a large accumulator always exceeds the occupancy benefit.

> **Avoid when**: When the accumulator tile is large (e.g. BLOCK_M=128, BLOCK_N=128 → 16,384 FP32 values). Capping registers forces the compiler to spill accumulator values to local memory (DRAM), causing DRAM throughput to spike (17%→47%), warp stall cycles to increase 5x (5.26→24.88 cycles/instruction), and total executed instructions to grow 39% (5.0B→7.0B). The net result is a 3.25x slowdown despite doubling occupancy. Only consider maxnreg if the accumulator fits comfortably within the capped register budget.

---

## Cumulative Progress

```
Baseline:  17020000 ns  (1.000×)
Iter 3     16860000 ns  (1.010×)  — add_triton_autotune

Final best: 16860000 ns  (1.0095×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,16 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=3),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_convolution_1(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -65,12 +75,12 @@

         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 64*idx_m
-        a = tl.load(A + (xindex))
+        a = tl.load(A + (xindex), eviction_policy='evict_first')
 
         idx_m = b_k_idx_vals
         idx_n = offs_b_n[None, :]
         xindex = idx_n + 128*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 64*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        b = tl.load(B + ((tl.broadcast_to(idx_m + 64*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), eviction_policy='evict_last')
 
         
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
@@ -86,8 +96,6 @@

     # inductor generates a suffix
     xindex = idx_n + 128*idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
