# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1710000 ns — **Best latency**: 1690000 ns (**1.0118×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_b_matrix_load_stride_and_reduce_register_pressure`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_006/kernel_4_triton_tem_fused_convolution_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3920,1,1), Waves/SM: 9.90
- **Registers/thread**: 168
- **Duration**: 1710000 ns
- **SM Throughput**: 40.45 % | **Memory**: 91.52 % | **DRAM**: 12.93 %
- **Achieved Occupancy**: 18.40 %
- **IPC Active**: 1.64 inst/cycle | **No Eligible**: 58.97 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 1.000× | ✓ |  |
| B | tma_loads_with_autotuned_tile_sizes | 0.999× | 1.000× | ✓ |  |
| C | fix_b_matrix_load_coalescing_and_reduce_register_pressure | 1.000× | 1.006× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_au... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1710000 ns | 1710000 ns | +0 ns (+0.0%) | 1710000 ns | +0 ns (+0.0%) | 1700000 ns | -10000 ns (-0.6%) |
| SM Throughput (%) | 40.45 % | 40.42 % | -0.03 pp | 40.44 % | -0.01 pp | 40.46 % | +0.01 pp |
| Memory Throughput (%) | 91.52 % | 91.44 % | -0.08 pp | 91.33 % | -0.19 pp | 91.55 % | +0.03 pp |
| DRAM Throughput (%) | 12.93 % | 12.87 % | -0.06 pp | 12.89 % | -0.04 pp | 12.95 % | +0.02 pp |
| Achieved Occupancy (%) | 18.40 % | 18.41 % | +0.01 pp | 18.41 % | +0.01 pp | 18.41 % | +0.01 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 9.90 | 9.90 | +0.00 | 9.90 | +0.00 | 9.90 | +0.00 |
| No Eligible (%) | 58.97 % | 58.96 % | -0.01 pp | 58.89 % | -0.08 pp | 58.96 % | -0.01 pp |
| IPC Active | 1.64 inst/cycle | 1.64 inst/cycle | +0.00 | 1.64 inst/cycle | +0.00 | 1.64 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_b_matrix_load_stride_and_reduce_register_pressure | 1.008× | 1.012× | ✓ | **✓ best** |
| C | fix_b_matrix_smem_bank_conflicts_via_transposed_load | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | fix_b_matrix_smem... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 1710000 ns | 1690000 ns | -20000 ns (-1.2%) | 1710000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 40.45 % | 40.74 % | +0.29 pp | 40.54 % | +0.09 pp |
| Memory Throughput (%) | 91.52 % | 92.16 % | +0.64 pp | 91.58 % | +0.06 pp |
| DRAM Throughput (%) | 12.93 % | 13.06 % | +0.13 pp | 12.94 % | +0.01 pp |
| Achieved Occupancy (%) | 18.40 % | 18.42 % | +0.02 pp | 18.41 % | +0.01 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 9.90 | 9.90 | +0.00 | 9.90 | +0.00 |
| No Eligible (%) | 58.97 % | 58.60 % | -0.37 pp | 58.89 % | -0.08 pp |
| IPC Active | 1.64 inst/cycle | 1.66 inst/cycle | +0.02 | 1.64 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0118× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_b_matrix_bank_conflicts_via_transposed_load_and_trans | 1.000× | 1.006× | ✓ |  |
| B | tma_loads | 1.000× | 1.006× | ✓ |  |
| C | fix_b_matrix_load_layout_to_eliminate_smem_bank_conflicts | 1.000× | 1.006× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_b_matri... | Δ | tma_loads | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1710000 ns | 1700000 ns | -10000 ns (-0.6%) | 1700000 ns | -10000 ns (-0.6%) | 1700000 ns | -10000 ns (-0.6%) |
| SM Throughput (%) | 40.45 % | 40.45 % | +0.00 pp | 40.43 % | -0.02 pp | 40.46 % | +0.01 pp |
| Memory Throughput (%) | 91.52 % | 91.52 % | +0.00 pp | 91.30 % | -0.22 pp | 91.38 % | -0.14 pp |
| DRAM Throughput (%) | 12.93 % | 12.94 % | +0.01 pp | 12.95 % | +0.02 pp | 12.94 % | +0.01 pp |
| Achieved Occupancy (%) | 18.40 % | 18.41 % | +0.01 pp | 18.41 % | +0.01 pp | 18.41 % | +0.01 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 9.90 | 9.90 | +0.00 | 9.90 | +0.00 | 9.90 | +0.00 |
| No Eligible (%) | 58.97 % | 58.98 % | +0.01 pp | 58.88 % | -0.09 pp | 58.88 % | -0.09 pp |
| IPC Active | 1.64 inst/cycle | 1.64 inst/cycle | +0.00 | 1.64 inst/cycle | +0.00 | 1.64 inst/cycle | +0.00 |


### Key Learnings

#### ⚠️ Plan B: `tma_loads` (0.000×)

Plan 'tma_loads' produced a kernel that crashed on execution after 4 attempt(s). Last error: exit 1
stderr:
loads.py", line 90, in kernel_function
    triton_tem_fused_convolution_4[grid](arg_0, arg_1, arg_2, arg_3)
  File "/home/ahpoddar/.local/lib/python3.11/site-packages/triton/runtime/jit.py", line 370, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *ar

> **Applicable when**: The kernel is a GEMM (M=501760, N=64, K=480) fused with bias add. The B matrix has stride_bn=480 (non-power-of-2, not a multiple of 32), so when Triton stages the (BLOCK_N=64, BLOCK_K=32) B tile through shared memory, the 480-element row stride creates systematic bank aliasing — 8-way conflicts on 69.79% of all shared load wavefronts. Prior attempts to fix this by changing BLOCK_K, BLOCK_M, or num_stages all introduced correctness bugs in the executor's rewrites. The diversity instruction mandates TMA loads. On Hopper (H100), replacing tl.make_block_ptr + tl.load with tl.make_tensor_descriptor + descriptor.load() uses the TMA hardware unit, which performs asynchronous bulk copies from global memory directly into shared memory via a dedicated DMA path — bypassing the warp-level load pipeline entirely. This eliminates the bank-conflict-prone scalar staging path and offloads address generation to the TMA unit, freeing SM cycles. The existing code already uses tl.make_block_ptr (a prerequisite pattern), making the migration to tl.make_tensor_descriptor straightforward.

> **Avoid when**: Dry-run failed: exit 1
stderr:
loads.py", line 90, in kernel_function
    triton_tem_fused_convolution_4[grid](arg_0, arg_1, arg_2, arg_3)
  File "/home/ahpoddar/.local/lib/python3.11/site-packages/triton/runtime/jit

---

## Iteration 4

**Best so far**: 1.0118× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | explicit_smem_padding_to_eliminate_b_matrix_bank_conflicts | 1.000× | 1.006× | ✓ |  |
| B | explicit_scalar_load_for_b_matrix_to_eliminate_smem_bank_conflicts | 1.002× | 1.000× | ✓ |  |
| C | fix_b_matrix_block_ptr_order_for_coalesced_smem_access | 0.999× | 1.006× | ✓ |  |


### Metric Comparison

| Metric | Baseline | explicit_smem_pad... | Δ | explicit_scalar_l... | Δ | fix_b_matrix_bloc... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1710000 ns | 1700000 ns | -10000 ns (-0.6%) | 1710000 ns | +0 ns (+0.0%) | 1700000 ns | -10000 ns (-0.6%) |
| SM Throughput (%) | 40.45 % | 40.53 % | +0.08 pp | 40.69 % | +0.24 pp | 40.52 % | +0.07 pp |
| Memory Throughput (%) | 91.52 % | 91.54 % | +0.02 pp | 91.71 % | +0.19 pp | 91.51 % | -0.01 pp |
| DRAM Throughput (%) | 12.93 % | 12.95 % | +0.02 pp | 12.91 % | -0.02 pp | 12.95 % | +0.02 pp |
| Achieved Occupancy (%) | 18.40 % | 18.41 % | +0.01 pp | 18.40 % | +0.00 pp | 18.41 % | +0.01 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 9.90 | 9.90 | +0.00 | 9.90 | +0.00 | 9.90 | +0.00 |
| No Eligible (%) | 58.97 % | 58.89 % | -0.08 pp | 58.73 % | -0.24 pp | 58.89 % | -0.08 pp |
| IPC Active | 1.64 inst/cycle | 1.64 inst/cycle | +0.00 | 1.65 inst/cycle | +0.01 | 1.64 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  1710000 ns  (1.000×)
Iter 2     1690000 ns  (1.012×)  — fix_b_matrix_load_stride_and_reduce_register_pressure

Final best: 1690000 ns  (1.0118×)
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
 def triton_tem_fused_convolution_4(in_ptr0, arg_A, arg_B, out_ptr0):
@@ -58,23 +59,21 @@

     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
+
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 480*idx_m
         a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 64*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 480*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        # Transposed B load: shape [BLOCK_N=64, BLOCK_K=32], coalesced in K (stride=1)
+        # b_T_ptrs[n, k] = B + n*480 + k = B[k, n] (same element as original b_ptrs[k, n])
+        b_T_ptrs = B + offs_b_n[:, None] * 480 + (offs_k[None, :] + k_idx * BLOCK_K) * 1
+        b_T = tl.load(b_T_ptrs)  # shape [BLOCK_N=64, BLOCK_K=32], coalesced in K
+        b = tl.trans(b_T)         # shape [BLOCK_K=32, BLOCK_N=64], correct for tl.dot
 
-        
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -88,8 +87,6 @@

     tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
