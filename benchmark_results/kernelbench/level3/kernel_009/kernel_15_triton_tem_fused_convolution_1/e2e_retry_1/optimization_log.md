# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_15` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 23780 ns — **Best latency**: 23710 ns (**1.0030×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`fix_smem_bank_conflicts_and_autotune_tile_sizes`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_15`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_15_triton_tem_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (123,1,1), Waves/SM: 0.16
- **Registers/thread**: 80
- **Duration**: 23780 ns
- **SM Throughput**: 10.60 % | **Memory**: 74.15 % | **DRAM**: 5.20 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.50 inst/cycle | **No Eligible**: 87.53 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_smem_bank_conflicts_and_autotune_tile_sizes | 1.005× | 1.003× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_smem_bank_con... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 23780 ns | 23710 ns | -70 ns (-0.3%) |
| SM Throughput (%) | 10.60 % | 10.52 % | -0.08 pp |
| Memory Throughput (%) | 74.15 % | 73.55 % | -0.60 pp |
| DRAM Throughput (%) | 5.20 % | 5.22 % | +0.02 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp |
| Registers/Thread | 80.00 register/thread | 80.00 register/thread | +0 |
| Waves/SM | 0.16 | 0.16 | +0.00 |
| No Eligible (%) | 87.53 % | 87.48 % | -0.05 pp |
| IPC Active | 0.50 inst/cycle | 0.50 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  23780 ns  (1.000×)
Iter 1     23710 ns  (1.003×)  — fix_smem_bank_conflicts_and_autotune_tile_sizes

Final best: 23710 ns  (1.0030×)
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
 def triton_tem_fused_convolution_15(in_ptr0, arg_A, arg_B, out_ptr0):
@@ -56,15 +57,15 @@

         order=(1, 0),
     )
     # B transposed: shape=[N, K]=[32, 192], strides=[stride_bn, stride_bk]=[192, 1]
-    # order=(1, 0) means K is innermost (contiguous) in SMEM — N-contiguous layout
-    # This eliminates bank conflicts: N-row accesses hit consecutive banks
+    # order=(0, 1) means N is innermost (contiguous) in SMEM
+    # tl.trans(b_T) reads stride-1 in N direction, eliminating bank conflicts
     b_T_block_ptr = tl.make_block_ptr(
         base=B,
         shape=(N, K),
         strides=(stride_bn, stride_bk),
         offsets=(pid_n * BLOCK_N, 0),
         block_shape=(BLOCK_N, BLOCK_K),
-        order=(1, 0),
+        order=(0, 1),
     )
 
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
@@ -88,7 +89,6 @@

     tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
