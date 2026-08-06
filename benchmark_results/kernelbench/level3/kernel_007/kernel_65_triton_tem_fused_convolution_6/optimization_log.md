# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_65` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 66660 ns — **Best latency**: 66500 ns (**1.0024×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`transpose_b_load_for_coalesced_shared_memory_access`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_65`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_65_triton_tem_fused_convolution_6/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (124,1,1), Waves/SM: 0.31
- **Registers/thread**: 150
- **Duration**: 66660 ns
- **SM Throughput**: 19.29 % | **Memory**: 74.94 % | **DRAM**: 1.44 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.86 inst/cycle | **No Eligible**: 78.58 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.998× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 66660 ns | 66690 ns | +30 ns (+0.0%) |
| SM Throughput (%) | 19.29 % | 19.30 % | +0.01 pp |
| Memory Throughput (%) | 74.94 % | 74.98 % | +0.04 pp |
| DRAM Throughput (%) | 1.44 % | 1.44 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 150.00 register/thread | 150.00 register/thread | +0 |
| Waves/SM | 0.31 | 0.31 | +0.00 |
| No Eligible (%) | 78.58 % | 78.54 % | -0.04 pp |
| IPC Active | 0.86 inst/cycle | 0.86 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_load_stride_correctness_bug | 0.999× | 0.998× | ✓ |  |
| B | fix_b_matrix_load_and_cap_registers_with_maxnreg | 0.999× | 1.000× | ✓ |  |
| C | fix_b_matrix_load_coalescing_to_eliminate_bank_conflicts | 1.000× | 1.004× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | fix_b_matrix_load... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 66660 ns | 66780 ns | +120 ns (+0.2%) | 66690 ns | +30 ns (+0.0%) | 66430 ns | -230 ns (-0.3%) |
| SM Throughput (%) | 19.29 % | 19.20 % | -0.09 pp | 19.35 % | +0.06 pp | 19.33 % | +0.04 pp |
| Memory Throughput (%) | 74.94 % | 74.64 % | -0.30 pp | 75.16 % | +0.22 pp | 75.12 % | +0.18 pp |
| DRAM Throughput (%) | 1.44 % | 1.44 % | +0.00 pp | 1.44 % | +0.00 pp | 1.44 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 150.00 register/thread | 150.00 register/thread | +0 | 150.00 register/thread | +0 | 150.00 register/thread | +0 |
| Waves/SM | 0.31 | 0.31 | +0.00 | 0.31 | +0.00 | 0.31 | +0.00 |
| No Eligible (%) | 78.58 % | 78.62 % | +0.04 pp | 78.54 % | -0.04 pp | 78.58 % | +0.00 pp |
| IPC Active | 0.86 inst/cycle | 0.85 inst/cycle | -0.01 | 0.86 inst/cycle | +0.00 | 0.86 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0035× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_load_transpose_to_eliminate_bank_conflicts | 1.000× | 1.001× | ✓ |  |
| B | fix_b_load_coalescing_via_transposed_tile_with_correct_addressing | 0.997× | 0.997× | ✓ |  |
| C | transpose_b_load_for_coalesced_shared_memory_access | 1.005× | 1.002× | ✓ | **✓ best** |
| D | add_triton_autotune | 1.001× | 0.884× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_load_transp... | Δ | fix_b_load_coales... | Δ | transpose_b_load_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 66660 ns | 66620 ns | -40 ns (-0.1%) | 66880 ns | +220 ns (+0.3%) | 66500 ns | -160 ns (-0.2%) | 75390 ns | +8730 ns (+13.1%) |
| SM Throughput (%) | 19.29 % | 19.35 % | +0.06 pp | 19.24 % | -0.05 pp | 19.25 % | -0.04 pp | 16.78 % | -2.51 pp |
| Memory Throughput (%) | 74.94 % | 75.17 % | +0.23 pp | 74.81 % | -0.13 pp | 74.83 % | -0.11 pp | 66.33 % | -8.61 pp |
| DRAM Throughput (%) | 1.44 % | 1.44 % | +0.00 pp | 1.43 % | -0.01 pp | 1.44 % | +0.00 pp | 1.27 % | -0.17 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 150.00 register/thread | 150.00 register/thread | +0 | 150.00 register/thread | +0 | 150.00 register/thread | +0 | 128.00 register/thread | -22 |
| Waves/SM | 0.31 | 0.31 | +0.00 | 0.31 | +0.00 | 0.31 | +0.00 | 0.23 | -0.08 |
| No Eligible (%) | 78.58 % | 78.54 % | -0.04 pp | 78.60 % | +0.02 pp | 78.54 % | -0.04 pp | 81.50 % | +2.92 pp |
| IPC Active | 0.86 inst/cycle | 0.86 inst/cycle | +0.00 | 0.85 inst/cycle | -0.01 | 0.86 inst/cycle | +0.00 | 0.74 inst/cycle | -0.12 |

---

## Iteration 4

**Best so far**: 1.0035× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_load_layout_eliminate_tl_trans_and_bank_conflicts | 0.995× | 0.995× | ✓ |  |
| B | cap_registers_with_maxnreg_to_boost_occupancy | 1.004× | 0.886× | ✓ |  |
| C | rewrite_b_load_with_block_ptr_and_cap_registers | 0.629× | 0.548× | ✓ |  |
| D | restore_b_load_correctness_and_reduce_register_pressure | 0.995× | 0.912× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_load_layout... | Δ | cap_registers_wit... | Δ | rewrite_b_load_wi... | Δ | restore_b_load_co... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 66660 ns | 66980 ns | +320 ns (+0.5%) | 75230 ns | +8570 ns (+12.9%) | 121600 ns | +54940 ns (+82.4%) | 73060 ns | +6400 ns (+9.6%) |
| SM Throughput (%) | 19.29 % | 19.31 % | +0.02 pp | 16.75 % | -2.54 pp | 18.00 % | -1.29 pp | 17.39 % | -1.90 pp |
| Memory Throughput (%) | 74.94 % | 74.83 % | -0.11 pp | 66.21 % | -8.73 pp | 58.96 % | -15.98 pp | 68.22 % | -6.72 pp |
| DRAM Throughput (%) | 1.44 % | 1.43 % | -0.01 pp | 1.27 % | -0.17 pp | 0.79 % | -0.65 pp | 1.31 % | -0.13 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 150.00 register/thread | 158.00 register/thread | +8 | 128.00 register/thread | -22 | 64.00 register/thread | -86 | 128.00 register/thread | -22 |
| Waves/SM | 0.31 | 0.31 | +0.00 | 0.23 | -0.08 | 0.16 | -0.15 | 0.23 | -0.08 |
| No Eligible (%) | 78.58 % | 78.42 % | -0.16 pp | 81.48 % | +2.90 pp | 83.41 % | +4.83 pp | 80.75 % | +2.17 pp |
| IPC Active | 0.86 inst/cycle | 0.86 inst/cycle | +0.00 | 0.74 inst/cycle | -0.12 | 0.66 inst/cycle | -0.20 | 0.77 inst/cycle | -0.09 |


### Key Learnings

#### ❌ Plan C: `rewrite_b_load_with_block_ptr_and_cap_registers` (0.548×)

Capping registers via maxnreg=64 (down from 150) while switching to tl.make_block_ptr caused a 1.82x slowdown (duration 66.5→121.6 µs). Although theoretical occupancy improved (18.75%→37.5%) and L1 hit rate jumped dramatically (2.66%→70.31%), achieved occupancy remained identical at 6.25% and executed instructions increased by 38% (10.2M→14.1M). The new MemoryCacheAccessPattern rule flagged severe uncoalesced local loads/stores (only 1 of 32 bytes per sector utilized), indicating the register cap forced spills to local memory, which introduced high-latency L1TEX stalls (40.4% of cycles) that more than offset the bank-conflict reduction. The anti-pattern is applying maxnreg aggressively on a kernel where the register file is genuinely needed — forced spills to local memory are far more expensive than the occupancy gain from freeing register slots.

> **Applicable when**: Never apply maxnreg to cap registers below what the compiler naturally allocates when the kernel has a large accumulator tile (e.g. 64x64 FP32 = 4096 elements). The register pressure is genuine and spills will be worse than low occupancy.

> **Avoid when**: Do not apply maxnreg register capping when: (1) the kernel has a large accumulator tile that genuinely requires many registers; (2) the MemoryCacheAccessPattern rule is not already present before the change (its appearance after indicates spills); (3) achieved occupancy does not improve despite theoretical occupancy improving — this signals the bottleneck is not occupancy-limited but instruction-count or latency-limited.

---

## Iteration 5

**Best so far**: 1.0035× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_load_as_NK_tile_with_tl_trans_for_coalesced_access | 0.997× | 1.000× | ✓ |  |
| B | eliminate_tl_trans_load_b_directly_as_k_n | 0.999× | 0.912× | ✓ |  |
| C | eliminate_tl_trans_by_loading_b_as_k_by_n_tile_directly | 0.628× | 0.513× | ✓ |  |
| D | stage_b_tile_through_padded_shared_memory_to_eliminate_bank_conflicts | 0.996× | 0.910× | ✓ |  |
| E | add_triton_autotune | 0.992× | 0.907× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_load_as_NK_... | Δ | eliminate_tl_tran... | Δ | eliminate_tl_tran... | Δ | stage_b_tile_thro... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 66660 ns | 66660 ns | +0 ns (+0.0%) | 73090 ns | +6430 ns (+9.6%) | 129950 ns | +63290 ns (+94.9%) | 73220 ns | +6560 ns (+9.8%) | 73500 ns | +6840 ns (+10.3%) |
| SM Throughput (%) | 19.29 % | 19.33 % | +0.04 pp | 17.48 % | -1.81 pp | 17.54 % | -1.75 pp | 17.39 % | -1.90 pp | 17.39 % | -1.90 pp |
| Memory Throughput (%) | 74.94 % | 74.89 % | -0.05 pp | 68.59 % | -6.35 pp | 55.85 % | -19.09 pp | 68.24 % | -6.70 pp | 68.23 % | -6.71 pp |
| DRAM Throughput (%) | 1.44 % | 1.44 % | +0.00 pp | 1.31 % | -0.13 pp | 0.74 % | -0.70 pp | 1.31 % | -0.13 pp | 1.30 % | -0.14 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 150.00 register/thread | 158.00 register/thread | +8 | 128.00 register/thread | -22 | 64.00 register/thread | -86 | 128.00 register/thread | -22 | 128.00 register/thread | -22 |
| Waves/SM | 0.31 | 0.31 | +0.00 | 0.23 | -0.08 | 0.16 | -0.15 | 0.23 | -0.08 | 0.23 | -0.08 |
| No Eligible (%) | 78.58 % | 78.37 % | -0.21 pp | 80.68 % | +2.10 pp | 84.58 % | +6.00 pp | 80.81 % | +2.23 pp | 80.87 % | +2.29 pp |
| IPC Active | 0.86 inst/cycle | 0.87 inst/cycle | +0.01 | 0.77 inst/cycle | -0.09 | 0.62 inst/cycle | -0.24 | 0.77 inst/cycle | -0.09 | 0.77 inst/cycle | -0.09 |


### Key Learnings

#### ❌ Plan C: `eliminate_tl_trans_by_loading_b_as_k_by_n_tile_directly` (0.513×)

Replacing tl.make_block_ptr + tl.trans(b) with scalar pointer arithmetic loading B as [BLOCK_K, BLOCK_N] directly, combined with maxnreg=64, caused a 1.95x slowdown (duration 66.98→129.95 µs). Although theoretical occupancy improved (18.75%→37.5%) and L1 hit rate jumped dramatically (2.66%→71.05%), achieved occupancy remained identical at 6.25% and executed instructions increased by 40% (10.2M→14.2M). The new MemoryCacheAccessPattern rule flagged severe uncoalesced local loads/stores (only 1 of 32 bytes per sector utilized), indicating the maxnreg=64 cap forced register spills to local memory, introducing high-latency L1TEX stalls (41.7% of cycles) that more than offset any bank-conflict reduction. The root cause of the regression is the same as iteration 4: aggressively capping registers via maxnreg on a kernel with a large accumulator tile causes spills that are far more expensive than the original shared memory bank conflicts.

> **Applicable when**: Never: this approach of eliminating tl.trans via direct [K,N] scalar pointer load combined with maxnreg capping consistently causes register spills that are worse than the original bank conflicts in this kernel class.

> **Avoid when**: Do not apply maxnreg register capping (e.g. maxnreg=64) when the kernel has a large accumulator tile (e.g. 64x64 FP32 = 4096 elements) that genuinely requires many registers. The MemoryCacheAccessPattern NCU rule appearing after the change (with only 1/32 bytes per sector utilized for local loads/stores) is a definitive signal that spills occurred and the approach caused regression. Also do not apply when achieved occupancy fails to improve despite theoretical occupancy improving — this signals the bottleneck is not occupancy-limited.

---

## Cumulative Progress

```
Baseline:  66660 ns  (1.000×)
Iter 2     66430 ns  (1.004×)  — fix_b_matrix_load_coalescing_to_eliminate_bank_conflicts

Final best: 66500 ns  (1.0024×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,19 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=5),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_convolution_65(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = False
@@ -60,24 +73,24 @@

     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
 
         a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
-        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
 
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 528*idx_m
         a = tl.load(A + (xindex), mask=a_mask, other=0.0)
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 256*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 528*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
-
+        # B layout: (N=256, K=528), stride_bn=528, stride_bk=1
+        # B[n, k] at offset: n * 528 + k * 1
+        # Load [BLOCK_K, BLOCK_N] tile directly: no transpose needed
+        rk_b = (k_idx * BLOCK_K + offs_k)[:, None]  # shape [BLOCK_K, 1], global k indices
+        rn_b = offs_b_n[None, :]                     # shape [1, BLOCK_N], global n indices
+        b_ptrs = B + rk_b * 1 + rn_b * 528          # = B_ptr + k + n*528, shape [BLOCK_K, BLOCK_N]
+        b_mask = rk_b < K                            # shape [BLOCK_K, 1], broadcasts to [BLOCK_K, BLOCK_N]
+        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # shape [BLOCK_K, BLOCK_N]
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -91,8 +104,6 @@

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
