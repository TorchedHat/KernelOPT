# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_clone_t_transpose_view_9` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4220000 ns — **Best latency**: 4220000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_clone_t_transpose_view_9`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_044/kernel_9_triton_tem_fused_addmm_clone_t/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3072,1,1), Waves/SM: 11.64
- **Registers/thread**: 255
- **Duration**: 4220000 ns
- **SM Throughput**: 40.58 % | **Memory**: 59.22 % | **DRAM**: 2.18 %
- **Achieved Occupancy**: 12.38 %
- **IPC Active**: 1.67 inst/cycle | **No Eligible**: 58.37 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_with_block_ptr_and_autotune | 0.171× | 0.165× | ✓ |  |
| C | eviction_policies_streaming_A_matrix | 0.998× | 1.000× | ✓ |  |
| D | add_autotune_reduce_register_pressure_via_tile_size | 1.022× | 0.998× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tma_loads_with_bl... | Δ | eviction_policies... | Δ | add_autotune_redu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4220000 ns | 25540000 ns | +21320000 ns (+505.2%) | 4220000 ns | +0 ns (+0.0%) | 4230000 ns | +10000 ns (+0.2%) |
| SM Throughput (%) | 40.58 % | 8.20 % | -32.38 pp | 40.61 % | +0.03 pp | 40.59 % | +0.01 pp |
| Memory Throughput (%) | 59.22 % | 16.38 % | -42.84 pp | 59.25 % | +0.03 pp | 59.22 % | +0.00 pp |
| DRAM Throughput (%) | 2.18 % | 11.43 % | +9.25 pp | 2.09 % | -0.09 pp | 2.18 % | +0.00 pp |
| Achieved Occupancy (%) | 12.38 % | 6.25 % | -6.13 pp | 12.34 % | -0.04 pp | 12.37 % | -0.01 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 11.64 | 23.27 | +11.63 | 11.64 | +0.00 | 11.64 | +0.00 |
| No Eligible (%) | 58.37 % | 91.68 % | +33.31 pp | 58.21 % | -0.16 pp | 58.35 % | -0.02 pp |
| IPC Active | 1.67 inst/cycle | 0.33 inst/cycle | -1.34 | 1.67 inst/cycle | +0.00 | 1.67 inst/cycle | +0.00 |


### Key Learnings

#### ❌ Plan B: `tma_loads_with_block_ptr_and_autotune` (0.165×)

Replacing scalar pointer arithmetic with tl.make_block_ptr + tl.advance did NOT reduce register pressure (still 255 registers/thread) but doubled dynamic shared memory per block from 65 KB to 131 KB. This caused Block Limit Shared Mem to drop from 3 to 1 block/SM, cutting theoretical occupancy from 12.5% to 6.25% and worsening scheduler starvation (No Eligible cycles rose from 58% to 92%, Warp Cycles Per Issued Instruction from 4.74 to 12.04). The kernel ran 6x slower (4.22 ms → 25.54 ms) because the shared memory increase was the binding occupancy constraint, not registers. When a kernel is already register-bound at 255 regs/thread, switching to block_ptr does not reduce register count in Triton — it may increase shared memory allocation, creating a new and worse occupancy bottleneck.

> **Applicable when**: Never apply this specific pattern when the kernel is already at 255 registers/thread and shared memory is already near the per-SM limit; block_ptr does not reduce register count in Triton and may increase shared memory allocation.

> **Avoid when**: Kernel is register-bound at max registers (255/thread) AND shared memory per block is already a binding occupancy constraint (Block Limit Shared Mem <= Block Limit Registers). Doubling BLOCK_K simultaneously with switching to block_ptr compounds the shared memory increase. Do not apply when the expected register reduction does not materialize — verify register count actually drops before deploying.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  4220000 ns  (1.000×)

Final best: 4220000 ns  (1.0000×)
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
 def triton_tem_fused_addmm_clone_t_transpose_view_9(arg_A, arg_B, out_ptr0):
@@ -65,12 +66,12 @@

         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 768*idx_m
-        a = tl.load(A + (xindex))
+        a = tl.load(A + (xindex), eviction_policy='evict_first')  # streaming: evict_first
 
         idx_m = b_k_idx_vals
         idx_n = offs_b_n[None, :]
         xindex = idx_n + 768*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), eviction_policy='evict_last')  # reused weight: evict_last
 
         
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
@@ -85,9 +86,7 @@

 
     # inductor generates a suffix
     xindex = idx_n + 768*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask=mask, eviction_policy='evict_first')  # output written once
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
