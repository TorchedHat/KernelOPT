# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_relu_t_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 176450 ns — **Best latency**: 184740 ns (**0.9551×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_relu_t_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_003/kernel_1_triton_tem_fused_addmm_relu_t_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (128,1,1), Waves/SM: 0.32
- **Registers/thread**: 168
- **Duration**: 176450 ns
- **SM Throughput**: 27.02 % | **Memory**: 60.90 % | **DRAM**: 0.97 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 1.22 inst/cycle | **No Eligible**: 69.53 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.006× | 0.955× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 176450 ns | 184740 ns | +8290 ns (+4.7%) |
| SM Throughput (%) | 27.02 % | 25.72 % | -1.30 pp |
| Memory Throughput (%) | 60.90 % | 58.38 % | -2.52 pp |
| DRAM Throughput (%) | 0.97 % | 0.93 % | -0.04 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 0.32 | 0.32 | +0.00 |
| No Eligible (%) | 69.53 % | 72.19 % | +2.66 pp |
| IPC Active | 1.22 inst/cycle | 1.11 inst/cycle | -0.11 |

---

## Cumulative Progress

```
Baseline:  176450 ns  (1.000×)

Final best: 184740 ns  (0.9551×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,17 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({}, num_warps=4, num_stages=2),
+    triton.Config({}, num_warps=4, num_stages=3),
+    triton.Config({}, num_warps=4, num_stages=4),
+    triton.Config({}, num_warps=8, num_stages=2),
+    triton.Config({}, num_warps=8, num_stages=3),
+    triton.Config({}, num_warps=8, num_stages=4),
+    triton.Config({}, num_warps=4, num_stages=1),
+    triton.Config({}, num_warps=8, num_stages=1),
+], key=[])
 @triton.jit
 def triton_tem_fused_addmm_relu_t_1(arg_A, arg_B, in_ptr2, out_ptr1):
     EVEN_K : tl.constexpr = True
@@ -91,8 +102,6 @@

     tmp3 = tl.maximum(tmp2, tmp1)
     tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 1024*idx_m, [BLOCK_M, BLOCK_N])), tmp3, mask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
