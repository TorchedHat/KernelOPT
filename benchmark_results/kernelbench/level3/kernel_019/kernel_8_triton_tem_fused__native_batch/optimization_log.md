# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_8` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 68770 ns — **Best latency**: 68160 ns (**1.0089×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`cap_registers_with_maxnreg_to_boost_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_8`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_019/kernel_8_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (245,1,1), Waves/SM: 0.93
- **Registers/thread**: 255
- **Duration**: 68770 ns
- **SM Throughput**: 34.37 % | **Memory**: 51.92 % | **DRAM**: 5.27 %
- **Achieved Occupancy**: 11.77 %
- **IPC Active**: 1.52 inst/cycle | **No Eligible**: 62.21 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.004× | 1.010× | ✓ |  |
| B | tma_loads_with_autotune_to_reduce_register_pressure | 1.015× | 1.009× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_au... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 68770 ns | 68100 ns | -670 ns (-1.0%) | 68160 ns | -610 ns (-0.9%) |
| SM Throughput (%) | 34.37 % | 34.13 % | -0.24 pp | 34.38 % | +0.01 pp |
| Memory Throughput (%) | 51.92 % | 51.56 % | -0.36 pp | 51.41 % | -0.51 pp |
| DRAM Throughput (%) | 5.27 % | 5.33 % | +0.06 pp | 5.21 % | -0.06 pp |
| Achieved Occupancy (%) | 11.77 % | 11.78 % | +0.01 pp | 11.74 % | -0.03 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 0.93 | 0.93 | +0.00 | 0.93 | +0.00 |
| No Eligible (%) | 62.21 % | 62.14 % | -0.07 pp | 61.67 % | -0.54 pp |
| IPC Active | 1.52 inst/cycle | 1.52 inst/cycle | +0.00 | 1.53 inst/cycle | +0.01 |

---

## Iteration 2

**Best so far**: 1.0089× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cap_registers_with_maxnreg_to_boost_occupancy | 0.999× | 1.009× | ✓ |  |
| D | cap_registers_with_maxnreg_to_boost_occupancy | 0.921× | 0.157× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_registers_wit... | Δ | cap_registers_wit... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 68770 ns | 68160 ns | -610 ns (-0.9%) | 437820 ns | +369050 ns (+536.6%) |
| SM Throughput (%) | 34.37 % | 34.12 % | -0.25 pp | 9.82 % | -24.55 pp |
| Memory Throughput (%) | 51.92 % | 51.53 % | -0.39 pp | 20.01 % | -31.91 pp |
| DRAM Throughput (%) | 5.27 % | 5.29 % | +0.02 pp | 15.47 % | +10.20 pp |
| Achieved Occupancy (%) | 11.77 % | 11.77 % | +0.00 pp | 11.70 % | -0.07 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 128.00 register/thread | -127 |
| Waves/SM | 0.93 | 0.93 | +0.00 | 0.62 | -0.31 |
| No Eligible (%) | 62.21 % | 62.08 % | -0.13 pp | 91.85 % | +29.64 pp |
| IPC Active | 1.52 inst/cycle | 1.51 inst/cycle | -0.01 | 0.33 inst/cycle | -1.19 |


### Key Learnings

#### ❌ Plan A: `cap_registers_with_maxnreg_to_boost_occupancy` (0.157×)

Adding maxnreg=128 to Triton configs successfully reduced registers/thread from 255 to 128 and raised theoretical occupancy from 12.5% to 18.75%, but achieved occupancy barely changed (11.78% → 11.7%) because the register spilling introduced catastrophic L1TEX stalls. Warp cycles per issued instruction jumped from 4.96 to 22.94, and the CPIStall rule flagged that 80% of stall cycles were L1TEX scoreboard waits caused by spill traffic — the kernel spent 6.4× longer (68 µs → 438 µs) with 91.85% of scheduler cycles having no eligible warp (up from 62.14%). When a kernel's register pressure is driven by large accumulator tiles (e.g. BLOCK_M×BLOCK_N/threads = 128 fp32 values per thread), capping registers via maxnreg forces those accumulators to spill to local memory, creating dense strided L1TEX traffic that serializes execution and overwhelms any occupancy benefit.

> **Applicable when**: Never applicable when the kernel's register pressure is dominated by large accumulator tiles (BLOCK_M × BLOCK_N / num_threads > ~64 fp32 values per thread). In that regime, maxnreg forces accumulators to spill to local memory, creating strided L1TEX traffic that serializes execution.

> **Avoid when**: Do NOT apply maxnreg when: (1) registers/thread are high due to large accumulator arrays (BLOCK_M×BLOCK_N/threads ≥ 64 fp32 values), as spilling accumulators to local memory causes CPIStall L1TEX stalls that dominate execution time; (2) the kernel already has low achieved occupancy despite theoretical gains — the bottleneck is likely the accumulator size, not register file fragmentation; (3) Warp Cycles Per Issued Instruction is already low (< 10) before the change, indicating the kernel is not latency-bound in a way that occupancy can fix.

#### ❌ Plan D: `cap_registers_with_maxnreg_to_boost_occupancy` (0.157×)

Adding maxnreg=128 to Triton configs successfully reduced registers/thread from 255 to 128 and raised theoretical occupancy from 12.5% to 18.75%, but achieved occupancy barely changed (11.78% → 11.7%) because the register spilling introduced catastrophic L1TEX stalls. Warp cycles per issued instruction jumped from 4.96 to 22.94, and the CPIStall rule flagged that 80% of stall cycles were L1TEX scoreboard waits caused by spill traffic — the kernel spent 6.4× longer (68 µs → 438 µs) with 91.85% of scheduler cycles having no eligible warp (up from 62.14%). When a kernel's register pressure is driven by large accumulator tiles (e.g. BLOCK_M×BLOCK_N/threads = 128 fp32 values per thread), capping registers via maxnreg forces those accumulators to spill to local memory, creating dense strided L1TEX traffic that serializes execution and overwhelms any occupancy benefit.

> **Applicable when**: Never applicable when the kernel's register pressure is dominated by large accumulator tiles (BLOCK_M × BLOCK_N / num_threads > ~64 fp32 values per thread). In that regime, maxnreg forces accumulators to spill to local memory, creating strided L1TEX traffic that serializes execution.

> **Avoid when**: Do NOT apply maxnreg when: (1) registers/thread are high due to large accumulator arrays (BLOCK_M×BLOCK_N/threads ≥ 64 fp32 values), as spilling accumulators to local memory causes CPIStall L1TEX stalls that dominate execution time; (2) the kernel already has low achieved occupancy despite theoretical gains — the bottleneck is likely the accumulator size, not register file fragmentation; (3) Warp Cycles Per Issued Instruction is already low (< 10) before the change, indicating the kernel is not latency-bound in a way that occupancy can fix.

---

## Iteration 3

**Best so far**: 1.0089× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | remove_maxnreg_and_autotune_block_m_as_constexpr | 1.013× | 1.017× | ✓ |  |


### Metric Comparison

| Metric | Baseline | remove_maxnreg_an... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 68770 ns | 67650 ns | -1120 ns (-1.6%) |
| SM Throughput (%) | 34.37 % | 34.31 % | -0.06 pp |
| Memory Throughput (%) | 51.92 % | 51.32 % | -0.60 pp |
| DRAM Throughput (%) | 5.27 % | 5.22 % | -0.05 pp |
| Achieved Occupancy (%) | 11.77 % | 11.75 % | -0.02 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 0.93 | 0.93 | +0.00 |
| No Eligible (%) | 62.21 % | 61.62 % | -0.59 pp |
| IPC Active | 1.52 inst/cycle | 1.53 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  68770 ns  (1.000×)
Iter 1     68160 ns  (1.009×)  — tma_loads_with_autotune_to_reduce_register_pressure

Final best: 68160 ns  (1.0089×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_8(arg_A, arg_B, out_ptr0):
@@ -65,12 +66,12 @@

         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 128*idx_m
-        a = tl.load(A + (xindex))
+        a = tl.load(A + (xindex), eviction_policy='evict_first')
 
         idx_m = b_k_idx_vals
         idx_n = offs_b_n[None, :]
         xindex = idx_n + 128*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 128*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        b = tl.load(B + ((tl.broadcast_to(idx_m + 128*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), eviction_policy='evict_last')
 
         
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
@@ -85,9 +86,7 @@

 
     # inductor generates a suffix
     xindex = idx_n + 128*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
