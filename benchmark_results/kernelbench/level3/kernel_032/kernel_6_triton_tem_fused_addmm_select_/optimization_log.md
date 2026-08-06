# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_select_t_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 9470 ns — **Best latency**: 9180 ns (**1.0316×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`persistent_kernel_one_cta_per_sm`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_select_t_6`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_032/kernel_6_triton_tem_fused_addmm_select_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (32,1,1), Waves/SM: 0.03
- **Registers/thread**: 38
- **Duration**: 9470 ns
- **SM Throughput**: 1.54 % | **Memory**: 14.93 % | **DRAM**: 1.16 %
- **Achieved Occupancy**: 6.23 %
- **IPC Active**: 0.32 inst/cycle | **No Eligible**: 91.85 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.980× | 0.967× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 9470 ns | 9790 ns | +320 ns (+3.4%) |
| SM Throughput (%) | 1.54 % | 1.59 % | +0.05 pp |
| Memory Throughput (%) | 14.93 % | 15.46 % | +0.53 pp |
| DRAM Throughput (%) | 1.16 % | 1.12 % | -0.04 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp |
| Registers/Thread | 38.00 register/thread | 38.00 register/thread | +0 |
| Waves/SM | 0.03 | 0.03 | +0.00 |
| No Eligible (%) | 91.85 % | 91.89 % | +0.04 pp |
| IPC Active | 0.32 inst/cycle | 0.31 inst/cycle | -0.01 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_shared_memory_bank_conflicts_b_matrix_padding | 0.988× | 0.993× | ✓ |  |
| B | persistent_kernel_one_cta_per_sm | 1.043× | 1.032× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_shared_memory... | Δ | persistent_kernel... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 9470 ns | 9540 ns | +70 ns (+0.7%) | 9180 ns | -290 ns (-3.1%) |
| SM Throughput (%) | 1.54 % | 1.40 % | -0.14 pp | 2.17 % | +0.63 pp |
| Memory Throughput (%) | 14.93 % | 15.24 % | +0.31 pp | 16.14 % | +1.21 pp |
| DRAM Throughput (%) | 1.16 % | 1.18 % | +0.02 pp | 1.27 % | +0.11 pp |
| Achieved Occupancy (%) | 6.23 % | 6.22 % | -0.01 pp | 6.25 % | +0.02 pp |
| Registers/Thread | 38.00 register/thread | 39.00 register/thread | +1 | 87.00 register/thread | +49 |
| Waves/SM | 0.03 | 0.03 | +0.00 | 0.06 | +0.03 |
| No Eligible (%) | 91.85 % | 92.89 % | +1.04 pp | 88.74 % | -3.11 pp |
| IPC Active | 0.32 inst/cycle | 0.28 inst/cycle | -0.04 | 0.44 inst/cycle | +0.12 |

---

## Iteration 3

**Best so far**: 1.0316× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reduce_block_sizes_to_increase_grid_coverage | 0.684× | 0.597× | ✓ |  |
| C | fix_b_block_ptr_smem_order_to_eliminate_bank_conflicts | 0.994× | 0.983× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_block_size... | Δ | fix_b_block_ptr_s... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 9470 ns | 15870 ns | +6400 ns (+67.6%) | 9630 ns | +160 ns (+1.7%) |
| SM Throughput (%) | 1.54 % | 1.58 % | +0.04 pp | 1.38 % | -0.16 pp |
| Memory Throughput (%) | 14.93 % | 17.65 % | +2.72 pp | 15.01 % | +0.08 pp |
| DRAM Throughput (%) | 1.16 % | 0.68 % | -0.48 pp | 1.16 % | +0.00 pp |
| Achieved Occupancy (%) | 6.23 % | 6.25 % | +0.02 pp | 6.22 % | -0.01 pp |
| Registers/Thread | 38.00 register/thread | 42.00 register/thread | +4 | 39.00 register/thread | +1 |
| Waves/SM | 0.03 | 0.03 | +0.00 | 0.03 | +0.00 |
| No Eligible (%) | 91.85 % | 92.39 % | +0.54 pp | 92.87 % | +1.02 pp |
| IPC Active | 0.32 inst/cycle | 0.30 inst/cycle | -0.02 | 0.29 inst/cycle | -0.03 |


### Key Learnings

#### ❌ Plan B: `reduce_block_sizes_to_increase_grid_coverage` (0.597×)

Reducing BLOCK_M and BLOCK_N in a persistent-kernel matmul (where the kernel loops over tiles internally using `for tile_id in range(sm_id, total_tiles, num_sms)`) does NOT increase the number of launched CTAs — the grid size remained at 32 blocks. Instead, the smaller tiles increased total_tiles from 32 to 126, but since the persistent kernel already handles tile distribution internally, the extra tiles only added more loop iterations per CTA, increasing instruction count from 158K to 200K and elapsed cycles from 13.7K to 23.8K. Achieved occupancy stayed flat at 6.25% while warp cycles per issued instruction worsened from 9.07 to 13.17, yielding a 0.60x regression. The grid tuple in the kernel_function call was NOT updated to match the new tile count (remained at 32), so the persistent loop ran more iterations per SM rather than distributing work across more SMs.

> **Applicable when**: Never — this specific change (reduce tile sizes without updating grid) is always a regression for persistent kernels.

> **Avoid when**: Any persistent-kernel matmul where the grid size is fixed and tile distribution is handled by an internal loop. Reducing tile sizes without proportionally increasing the grid size only adds more loop iterations per CTA, increasing instruction count and warp stall cycles without improving SM utilization. Achieved occupancy will not improve if the grid size does not change.

---

## Iteration 4

**Best so far**: 1.0316× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | fix_b_matrix_access_pattern_to_eliminate_bank_conflicts | 0.680× | 0.599× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_acce... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 9470 ns | 15810 ns | +6340 ns (+66.9%) |
| SM Throughput (%) | 1.54 % | 1.58 % | +0.04 pp |
| Memory Throughput (%) | 14.93 % | 17.70 % | +2.77 pp |
| DRAM Throughput (%) | 1.16 % | 0.69 % | -0.47 pp |
| Achieved Occupancy (%) | 6.23 % | 6.25 % | +0.02 pp |
| Registers/Thread | 38.00 register/thread | 42.00 register/thread | +4 |
| Waves/SM | 0.03 | 0.03 | +0.00 |
| No Eligible (%) | 91.85 % | 92.41 % | +0.56 pp |
| IPC Active | 0.32 inst/cycle | 0.30 inst/cycle | -0.02 |


### Key Learnings

#### ❌ Plan D: `fix_b_matrix_access_pattern_to_eliminate_bank_conflicts` (0.599×)

Attempting to fix 17-way shared memory bank conflicts by loading B as [BLOCK_N, BLOCK_K] with tl.trans() and simultaneously converting to a persistent kernel loop caused executed instructions to nearly double (107K → 200K) and elapsed cycles to increase by 64% (14.4K → 23.7K), yielding a 0.60x regression. The bank conflict rate (82% excessive wavefronts) was unchanged after the rewrite, meaning the tl.trans() approach did not actually eliminate the smem bank conflicts. The persistent loop structure added overhead without improving SM coverage since the grid remained at 32 blocks and waves_per_sm stayed at 0.03. Warp cycles per issued instruction improved marginally (14.12 → 13.14) but was overwhelmed by the instruction count increase.

> **Applicable when**: Never — this specific combination of tl.trans(b_T) for bank conflict elimination plus persistent loop wrapping causes regression when the grid size is not updated.

> **Avoid when**: Any kernel where: (1) the grid size remains fixed at 32 while adding a persistent tile loop — the loop adds overhead without improving SM coverage; (2) tl.trans() is used to attempt bank conflict elimination — the profiling shows bank conflicts (82% excessive wavefronts) were NOT reduced by this approach, meaning the smem staging still produces the same conflict pattern; (3) instruction count is already low (107K) — doubling it via persistent loop overhead will dominate any latency savings from reduced bank conflicts.

---

## Cumulative Progress

```
Baseline:  9470 ns  (1.000×)
Iter 2     9180 ns  (1.032×)  — persistent_kernel_one_cta_per_sm

Final best: 9180 ns  (1.0316×)
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
 def triton_tem_fused_addmm_select_t_6(in_ptr0, arg_A, arg_B, out_ptr0):
@@ -57,24 +58,33 @@

     offs_k = tl.arange(0, BLOCK_K)
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
+    # Create block pointers for conflict-free shared memory staging
+    # A is a slice of (10,2,128) tensor: stride_am=256, stride_ak=1
+    a_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(256, 1),           # stride_am=256, stride_ak=1
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0)                # K is innermost (contiguous) -> row-major tile
+    )
+    # B is (1000,128) row-major weight matrix: stride_bk=1, stride_bn=128
+    # Load [BLOCK_K, BLOCK_N] tile with N as fast-varying dim in smem to eliminate bank conflicts
+    b_block_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(1, 128),           # stride_bk=1, stride_bn=128
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(1, 0)                # N is fastest in smem: consecutive threads read consecutive N cols, zero bank conflicts
+    )
+
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 128*idx_m
-        a = tl.load(A + ((tl.broadcast_to(idx_n + 256*idx_m, [BLOCK_M, BLOCK_K])).broadcast_to(xindex.shape)))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 1000*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 128*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-        
+        a = tl.load(a_block_ptr)   # Triton stages through smem with order=(1,0)
+        b = tl.load(b_block_ptr)   # Triton stages through smem with order=(1,0), eliminates bank conflicts
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
+        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
+        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -88,8 +98,6 @@

     tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
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
