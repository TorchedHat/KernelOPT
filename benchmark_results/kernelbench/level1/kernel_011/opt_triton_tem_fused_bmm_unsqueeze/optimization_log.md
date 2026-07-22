# Kernel Optimization Log

## Summary

- **Kernel**: `matmul_4d_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 14730000 ns — **Best latency**: 8010000 ns (**1.8390×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`fast_accumulation_allow_tf32`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `matmul_4d_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_011/synthesis/kernel_0_triton_tem_fused_bmm_unsqueeze_view_0.py`
- **Framework**: triton
- **Launch**: Block (256,1,1), Grid (24576,1,1), Waves/SM: 186.18
- **Registers/thread**: 255
- **Duration**: 14730000 ns
- **SM Throughput**: 63.17 % | **Memory**: 42.23 % | **DRAM**: 6.06 %
- **Achieved Occupancy**: 12.49 %
- **IPC Active**: 2.54 inst/cycle | **No Eligible**: 36.53 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 2.331× | 1.841× | ✓ |  |
| B | fast_accumulation_allow_tf32 | 2.338× | 1.839× | ✓ | **✓ best** |
| C | fp8_precision_dot_operands | 2.327× | 1.839× | ✓ |  |
| D | reduce_register_pressure_via_smaller_tile_autotune_configs | 1.009× | 1.003× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | fast_accumulation... | Δ | fp8_precision_dot... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 14730000 ns | 8000000 ns | -6730000 ns (-45.7%) | 8010000 ns | -6720000 ns (-45.6%) | 8010000 ns | -6720000 ns (-45.6%) | 14690000 ns | -40000 ns (-0.3%) |
| SM Throughput (%) | 63.17 % | 14.24 % | -48.93 pp | 14.25 % | -48.92 pp | 14.24 % | -48.93 pp | 63.18 % | +0.01 pp |
| Memory Throughput (%) | 42.23 % | 87.24 % | +45.01 pp | 87.29 % | +45.06 pp | 87.21 % | +44.98 pp | 42.24 % | +0.01 pp |
| DRAM Throughput (%) | 6.06 % | 10.86 % | +4.80 pp | 10.85 % | +4.79 pp | 10.85 % | +4.79 pp | 6.07 % | +0.01 pp |
| Achieved Occupancy (%) | 12.49 % | 12.46 % | -0.03 pp | 12.46 % | -0.03 pp | 12.46 % | -0.03 pp | 12.49 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 224.00 register/thread | -31 | 224.00 register/thread | -31 | 224.00 register/thread | -31 | 255.00 register/thread | +0 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 |
| No Eligible (%) | 36.53 % | 92.09 % | +55.56 pp | 92.09 % | +55.56 pp | 92.09 % | +55.56 pp | 36.51 % | -0.02 pp |
| IPC Active | 2.54 inst/cycle | 0.32 inst/cycle | -2.22 | 0.32 inst/cycle | -2.22 | 0.32 inst/cycle | -2.22 | 2.54 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.839× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_uncoalesced_shared_memory_via_smem_swizzle_and_larger_block_k | 2.303× | 1.846× | ✓ |  |
| B | fix_uncoalesced_shared_and_global_memory_via_block_ptr_and_swizzle | 2.304× | 1.841× | ✓ |  |
| C | fix_uncoalesced_shared_memory_via_block_ptr_swizzling | 2.321× | 1.837× | ✓ |  |
| D | increase_occupancy_via_num_warps2_autotune_configs | 1.003× | 1.000× | ✓ |  |
| E | add_triton_autotune | 2.328× | 1.844× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_uncoalesced_s... | Δ | fix_uncoalesced_s... | Δ | fix_uncoalesced_s... | Δ | increase_occupanc... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 14730000 ns | 7980000 ns | -6750000 ns (-45.8%) | 8000000 ns | -6730000 ns (-45.7%) | 8020000 ns | -6710000 ns (-45.6%) | 14730000 ns | +0 ns (+0.0%) | 7990000 ns | -6740000 ns (-45.8%) |
| SM Throughput (%) | 63.17 % | 14.27 % | -48.90 pp | 14.26 % | -48.91 pp | 14.19 % | -48.98 pp | 63.17 % | +0.00 pp | 14.24 % | -48.93 pp |
| Memory Throughput (%) | 42.23 % | 87.44 % | +45.21 pp | 87.35 % | +45.12 pp | 86.94 % | +44.71 pp | 42.23 % | +0.00 pp | 87.26 % | +45.03 pp |
| DRAM Throughput (%) | 6.06 % | 10.89 % | +4.83 pp | 10.87 % | +4.81 pp | 10.84 % | +4.78 pp | 6.06 % | +0.00 pp | 10.87 % | +4.81 pp |
| Achieved Occupancy (%) | 12.49 % | 12.44 % | -0.05 pp | 12.44 % | -0.05 pp | 12.43 % | -0.06 pp | 12.49 % | +0.00 pp | 12.46 % | -0.03 pp |
| Registers/Thread | 255.00 register/thread | 188.00 register/thread | -67 | 188.00 register/thread | -67 | 174.00 register/thread | -81 | 255.00 register/thread | +0 | 224.00 register/thread | -31 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 |
| No Eligible (%) | 36.53 % | 91.96 % | +55.43 pp | 91.95 % | +55.42 pp | 91.10 % | +54.57 pp | 36.54 % | +0.01 pp | 92.09 % | +55.56 pp |
| IPC Active | 2.54 inst/cycle | 0.32 inst/cycle | -2.22 | 0.32 inst/cycle | -2.22 | 0.36 inst/cycle | -2.18 | 2.54 inst/cycle | +0.00 | 0.32 inst/cycle | -2.22 |


### Key Learnings

#### ❌ Plan B: `fix_uncoalesced_shared_and_global_memory_via_block_ptr_and_swizzle` (1.841×)

Replacing raw pointer arithmetic with tl.make_block_ptr (order=(1,0) for A, order=(0,1) for B) and adding larger BLOCK_K autotune configs produced no measurable improvement: L1/TEX throughput remained at ~88%, excessive shared wavefronts stayed at 943M, excessive global sectors stayed at 604M, and kernel duration was unchanged at 8ms. The profiling data shows that despite the block pointer API change, the NCU-flagged uncoalesced access counts are identical before and after, indicating the root cause of the bank conflicts and uncoalesced loads is not the pointer construction style but rather the tile shape, stride layout, or a deeper structural issue (e.g., non-unit strides in the 4D tensor's batch dimensions). The speedup of 1.84x reported is the cumulative speedup from iteration 1 (TF32 enable), not from this change.

> **Applicable when**: Never: this strategy did not reduce uncoalesced access counts or improve latency in this kernel configuration

> **Avoid when**: When NCU reports UncoalescedSharedAccess and UncoalescedGlobalAccess on a 4D batched GEMM kernel with non-unit strides — tl.make_block_ptr alone does not resolve the bank conflicts or uncoalesced patterns when the underlying cause is stride layout or tile shape mismatch. Do not apply when the kernel is already L1/TEX saturated at ~88% and the excessive wavefront/sector counts are driven by structural access patterns not pointer API style.

#### ✅ Plan C: `fix_uncoalesced_shared_memory_via_block_ptr_swizzling` (1.837×)

Replacing manual pointer arithmetic with tl.make_block_ptr (order=(1,0) for A, order=(0,1) for B) in a Triton GEMM kernel reduced registers per thread from 224 to 174 and slightly reduced L1TEX stall cycles (13.9→10.6 cycles/warp), yielding a modest 1.84x speedup. However, the NCU-reported UncoalescedSharedAccess and UncoalescedGlobalAccess metrics were completely unchanged (78% excessive wavefronts, 60% excessive sectors), indicating the block_ptr swizzling did NOT eliminate the bank conflicts or uncoalesced global accesses as predicted — the speedup came primarily from register pressure reduction enabling better instruction scheduling. When the kernel is already deeply L1/TEX-bound at ~87% with low occupancy (12.5%) limited by shared memory, block_ptr swizzling alone is insufficient to resolve the uncoalesced access bottleneck.

> **Applicable when**: Triton GEMM kernel uses manual pointer arithmetic with high register pressure (registers_per_thread > 200); kernel is L1/TEX-bound; switching to block_ptr reduces register count and may improve instruction scheduling even if bank conflicts persist

> **Avoid when**: The primary goal is eliminating UncoalescedSharedAccess bank conflicts — block_ptr swizzling alone does not guarantee bank conflict elimination when the kernel is occupancy-limited by shared memory (Block Limit Shared Mem = 1); do not expect the NCU UncoalescedSharedAccess metric to drop from this change alone

---

## Iteration 3

**Best so far**: 1.839× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_uncoalesced_global_store_via_block_ptr_output | 2.321× | 1.837× | ✓ |  |
| B | reduce_shared_memory_per_block_to_increase_sm_occupancy | 2.326× | 1.322× | ✓ |  |
| C | eliminate_smem_bank_conflicts_via_explicit_padding | 2.327× | 1.837× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_uncoalesced_g... | Δ | reduce_shared_mem... | Δ | eliminate_smem_ba... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 14730000 ns | 8020000 ns | -6710000 ns (-45.6%) | 11140000 ns | -3590000 ns (-24.4%) | 8020000 ns | -6710000 ns (-45.6%) |
| SM Throughput (%) | 63.17 % | 14.20 % | -48.97 pp | 59.19 % | -3.98 pp | 14.19 % | -48.98 pp |
| Memory Throughput (%) | 42.23 % | 86.97 % | +44.74 pp | 80.44 % | +38.21 pp | 86.91 % | +44.68 pp |
| DRAM Throughput (%) | 6.06 % | 10.84 % | +4.78 pp | 7.80 % | +1.74 pp | 10.84 % | +4.78 pp |
| Achieved Occupancy (%) | 12.49 % | 12.43 % | -0.06 pp | 52.57 % | +40.08 pp | 12.43 % | -0.06 pp |
| Registers/Thread | 255.00 register/thread | 174.00 register/thread | -81 | 54.00 register/thread | -201 | 174.00 register/thread | -81 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 700.92 | +514.74 | 186.18 | +0.00 |
| No Eligible (%) | 36.53 % | 91.10 % | +54.57 pp | 40.80 % | +4.27 pp | 91.10 % | +54.57 pp |
| IPC Active | 2.54 inst/cycle | 0.36 inst/cycle | -2.18 | 2.37 inst/cycle | -0.17 | 0.36 inst/cycle | -2.18 |


### Key Learnings

#### ❌ Plan A: `fix_uncoalesced_global_store_via_block_ptr_output` (1.837×)

Converting the output store from raw pointer arithmetic to tl.make_block_ptr with order=(1,0) in a Triton GEMM kernel did NOT reduce the NCU-reported UncoalescedGlobalAccess excessive sector count (603M before and after, 60% of total unchanged), and kernel duration was essentially flat (7.98ms → 8.02ms, no improvement). The L1/TEX throughput, memory throughput, and all other bottleneck metrics remained identical, indicating the Triton compiler backend did not apply coalesced store optimization for this tile configuration despite the block_ptr API. The only observable change was a minor register reduction (188→174) and a slight improvement in warp cycles per instruction (24.81→22.36), which are secondary effects of cleaner codegen rather than coalescing improvements. When a kernel is already deeply L1/TEX-bound at ~87% with uncoalesced global accesses persisting through multiple block_ptr rewrites, the root cause is structural (tile shape, stride layout, or batch dimension strides) and cannot be fixed by pointer API changes alone.

> **Applicable when**: Never demonstrated effective for this pattern: do not apply when NCU UncoalescedGlobalAccess excessive sectors have already persisted unchanged through multiple block_ptr rewrites on input loads in the same kernel

> **Avoid when**: When the kernel is already L1/TEX-bound at >85% throughput and prior block_ptr rewrites on input loads did not reduce the UncoalescedGlobalAccess sector count — converting the output store to block_ptr will also fail to reduce the count, as the root cause is structural (non-unit batch strides, tile shape mismatch) rather than pointer API style. Do not expect this change to reduce kernel duration when the bottleneck is not the store path.

---

## Cumulative Progress

```
Baseline:  14730000 ns  (1.000×)
Iter 1     8010000 ns  (1.839×)  — fast_accumulation_allow_tf32

Final best: 8010000 ns  (1.8390×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -1,6 +1,7 @@

 import torch
 import triton
 import triton.language as tl
+
 
 
 @triton.autotune(
@@ -17,6 +18,14 @@

         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
         triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        # New configs: BLOCK_K=128 variants
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        # New configs: num_stages=5 variants
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=8),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=4),
     ],
     key=['M', 'N', 'K'],
 )
@@ -41,30 +50,33 @@

     pid_m = group_id * GROUP_M + (pid % group_size)
     pid_n = (pid % width) // group_size
 
-    # Row/col offsets
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    rk = tl.arange(0, BLOCK_K)
-
-    # Pointers
-    A = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
-    B = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
+    # Use tl.make_block_ptr with order=(1,0) to enable hardware swizzle
+    # for shared memory bank conflict elimination
+    A_block_ptr = tl.make_block_ptr(
+        base=A_ptr,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),  # K innermost -> row-major -> swizzle applied by backend
+    )
+    B_block_ptr = tl.make_block_ptr(
+        base=B_ptr,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(1, 0),  # N innermost -> row-major -> swizzle applied by backend
+    )
 
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
 
     for k in range(0, tl.cdiv(K, BLOCK_K)):
-        k_remaining = K - k * BLOCK_K
-        # Load A tile with masking
-        a_mask = (rm[:, None] < M) & (rk[None, :] < k_remaining)
-        a = tl.load(A, mask=a_mask, other=0.0)
-        # Load B tile with masking
-        b_mask = (rk[:, None] < k_remaining) & (rn[None, :] < N)
-        b = tl.load(B, mask=b_mask, other=0.0)
-
-        acc = tl.dot(a, b, acc=acc, allow_tf32=False)
-
-        A += BLOCK_K * stride_ak
-        B += BLOCK_K * stride_bk
+        a = tl.load(A_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        b = tl.load(B_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        acc = tl.dot(a, b, acc=acc, allow_tf32=True)
+        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
+        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))
 
     # Store result
     cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -72,7 +84,6 @@

     C = C_ptr + cm[:, None] * stride_cm + cn[None, :] * stride_cn
     mask = (cm[:, None] < M) & (cn[None, :] < N)
     tl.store(C, acc.to(tl.float32), mask=mask)
-
 
 def kernel_function(A, B):
     # A: [8, 256, 512, 256], B: [256, 768]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
