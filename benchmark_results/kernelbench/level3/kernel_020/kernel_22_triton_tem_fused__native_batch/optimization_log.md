# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 29500 ns — **Best latency**: 29500 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_22_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (144,1,1), Waves/SM: 0.36
- **Registers/thread**: 168
- **Duration**: 29500 ns
- **SM Throughput**: 18.35 % | **Memory**: 41.28 % | **DRAM**: 0.64 %
- **Achieved Occupancy**: 7.07 %
- **IPC Active**: 1.16 inst/cycle | **No Eligible**: 70.77 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.999× | 0.996× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 29500 ns | 29630 ns | +130 ns (+0.4%) |
| SM Throughput (%) | 18.35 % | 18.44 % | +0.09 pp |
| Memory Throughput (%) | 41.28 % | 41.50 % | +0.22 pp |
| DRAM Throughput (%) | 0.64 % | 0.63 % | -0.01 pp |
| Achieved Occupancy (%) | 7.07 % | 7.07 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 0.36 | 0.36 | +0.00 |
| No Eligible (%) | 70.77 % | 70.62 % | -0.15 pp |
| IPC Active | 1.16 inst/cycle | 1.17 inst/cycle | +0.01 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | eviction_policies_streaming_a_preserve_b_in_l2 | 0.996× | 0.992× | ✓ |  |
| D | increase_num_warps_to_improve_scheduler_utilization | 0.697× | 0.678× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 29500 ns | 29730 ns | +230 ns (+0.8%) | 43490 ns | +13990 ns (+47.4%) |
| SM Throughput (%) | 18.35 % | 18.44 % | +0.09 pp | 12.63 % | -5.72 pp |
| Memory Throughput (%) | 41.28 % | 41.45 % | +0.17 pp | 48.39 % | +7.11 pp |
| DRAM Throughput (%) | 0.64 % | 0.63 % | -0.01 pp | 0.42 % | -0.22 pp |
| Achieved Occupancy (%) | 7.07 % | 7.07 % | +0.00 pp | 14.22 % | +7.15 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 128.00 register/thread | -40 |
| Waves/SM | 0.36 | 0.36 | +0.00 | 0.55 | +0.19 |
| No Eligible (%) | 70.77 % | 70.57 % | -0.20 pp | 77.67 % | +6.90 pp |
| IPC Active | 1.16 inst/cycle | 1.17 inst/cycle | +0.01 | 0.89 inst/cycle | -0.27 |


### Key Learnings

#### ❌ Plan D: `increase_num_warps_to_improve_scheduler_utilization` (0.678×)

Increasing num_warps from 4 to 8 (block_size 128→256) doubled achieved occupancy (7.07%→14.22%) and active warps per scheduler (1.13→2.27), but the kernel ran 47% slower (29.63µs→43.49µs). The root cause is that more warps simultaneously accessing shared memory amplified bank conflicts from 7.9-way to 11.8-way severity, and the fraction of no-eligible cycles actually increased (70.62%→77.67%) while warp cycles per issued instruction nearly tripled (3.86→10.18). When the dominant bottleneck is shared memory bank conflicts (CPIStall MIO stalls >44%), adding more warps increases contention on the shared memory banks rather than hiding latency, making the stall problem strictly worse.

> **Applicable when**: Never: this strategy caused regression in this context.

> **Avoid when**: When the kernel's primary bottleneck is shared memory bank conflicts (CPIStall MIO stalls >40%, bank conflict way >6x, SharedMemoryConflicts rule triggered). Increasing num_warps in this scenario amplifies bank conflict contention — more warps simultaneously issue conflicting shared memory requests, increasing average conflict severity and warp stall cycles. The no-eligible-warp percentage increases rather than decreases. Only apply num_warps increases when the bottleneck is compute latency or global memory latency, not shared memory bank conflicts.

---

## Cumulative Progress

```
Baseline:  29500 ns  (1.000×)

Final best: 29500 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,10 +5,11 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = False
-    USE_FAST_ACCUM : tl.constexpr = False
+    USE_FAST_ACCUM : tl.constexpr = True
     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 128
     BLOCK_N : tl.constexpr = 64
@@ -76,7 +77,7 @@

         b = tl.load(B + ((tl.broadcast_to(idx_m + 80*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
 
 
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
+        acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
 
     # rematerialize rm and rn to save registers
@@ -89,8 +90,6 @@

     # inductor generates a suffix
     xindex = idx_n + 480*idx_m
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
