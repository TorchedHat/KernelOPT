# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_14` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 20770 ns — **Best latency**: 14300 ns (**1.4524×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`reduce_block_k_to_halve_shared_memory_bank_conflicts`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_14`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_14_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (123,1,1), Waves/SM: 0.19
- **Registers/thread**: 96
- **Duration**: 20770 ns
- **SM Throughput**: 10.04 % | **Memory**: 70.23 % | **DRAM**: 4.46 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.49 inst/cycle | **No Eligible**: 87.76 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_to_fix_bank_conflicts_and_occupancy | 1.266× | 0.997× | ✓ |  |
| C | add_eviction_policies_and_autotune_tile_sizes | 1.285× | 1.330× | ✓ | **✓ best** |
| D | add_autotune_to_reduce_bank_conflicts_and_improve_occupancy | 1.260× | 1.333× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | add_eviction_poli... | Δ | add_autotune_to_r... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 20770 ns | 20830 ns | +60 ns (+0.3%) | 15620 ns | -5150 ns (-24.8%) | 15580 ns | -5190 ns (-25.0%) |
| SM Throughput (%) | 10.04 % | 10.03 % | -0.01 pp | 12.73 % | +2.69 pp | 12.41 % | +2.37 pp |
| Memory Throughput (%) | 70.23 % | 70.14 % | -0.09 pp | 53.38 % | -16.85 pp | 52.09 % | -18.14 pp |
| DRAM Throughput (%) | 4.46 % | 4.45 % | -0.01 pp | 5.95 % | +1.49 pp | 5.97 % | +1.51 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp |
| Registers/Thread | 96.00 register/thread | 96.00 register/thread | +0 | 128.00 register/thread | +32 | 128.00 register/thread | +32 |
| Waves/SM | 0.19 | 0.19 | +0.00 | 0.12 | -0.07 | 0.12 | -0.07 |
| No Eligible (%) | 87.76 % | 87.79 % | +0.03 pp | 68.11 % | -19.65 pp | 68.04 % | -19.72 pp |
| IPC Active | 0.49 inst/cycle | 0.49 inst/cycle | +0.00 | 0.64 inst/cycle | +0.15 | 0.64 inst/cycle | +0.15 |

---

## Iteration 2

**Best so far**: 1.3297× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 3

**Best so far**: 1.3297× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_tile_strided_load_to_eliminate_local_memory_spills | 1.278× | 1.338× | ✓ |  |
| B | tma_loads_replace_scalar_pointer_arithmetic | 1.317× | 1.378× | ✓ | **✓ best** |
| C | add_evict_first_on_streaming_k_loop_tile_loads | 1.286× | 1.341× | ✓ |  |
| D | increase_block_n_to_eliminate_partial_tile_and_fix_b_indexing | 0.937× | 0.832× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_tile_stride... | Δ | tma_loads_replace... | Δ | add_evict_first_o... | Δ | increase_block_n_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 20770 ns | 15520 ns | -5250 ns (-25.3%) | 15070 ns | -5700 ns (-27.4%) | 15490 ns | -5280 ns (-25.4%) | 24960 ns | +4190 ns (+20.2%) |
| SM Throughput (%) | 10.04 % | 12.71 % | +2.67 pp | 12.92 % | +2.88 pp | 12.76 % | +2.72 pp | 14.66 % | +4.62 pp |
| Memory Throughput (%) | 70.23 % | 53.46 % | -16.77 pp | 55.24 % | -14.99 pp | 53.51 % | -16.72 pp | 33.48 % | -36.75 pp |
| DRAM Throughput (%) | 4.46 % | 6.00 % | +1.54 pp | 6.17 % | +1.71 pp | 6.00 % | +1.54 pp | 3.74 % | -0.72 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp |
| Registers/Thread | 96.00 register/thread | 128.00 register/thread | +32 | 128.00 register/thread | +32 | 128.00 register/thread | +32 | 168.00 register/thread | +72 |
| Waves/SM | 0.19 | 0.12 | -0.07 | 0.12 | -0.07 | 0.12 | -0.07 | 0.16 | -0.03 |
| No Eligible (%) | 87.76 % | 67.91 % | -19.85 pp | 67.12 % | -20.64 pp | 68.26 % | -19.50 pp | 62.34 % | -25.42 pp |
| IPC Active | 0.49 inst/cycle | 0.64 inst/cycle | +0.15 | 0.66 inst/cycle | +0.17 | 0.64 inst/cycle | +0.15 | 0.76 inst/cycle | +0.27 |


### Key Learnings

#### ✅ Plan A: `fix_b_tile_strided_load_to_eliminate_local_memory_spills` (1.338×)

When NCU reports MemoryCacheAccessPattern for both local loads and local stores with only 1/32 bytes utilized per sector, the kernel is spilling intermediate index tensors to local memory due to convoluted indexing (e.g., double tl.broadcast_to calls that force the compiler to materialize a full [BLOCK_K, BLOCK_N] index tensor). Replacing the double-broadcast pattern with direct pointer arithmetic (b_ptrs = B + k_offsets * stride_bk + n_offsets * stride_bn) eliminates the spill traffic and reduces executed instructions. In this case the speedup was modest (1.34x) because the local memory spill was not the dominant bottleneck — shared memory bank conflicts and grid underutilization remained — but L2 active cycles dropped from 9867 to 8394 and executed instructions fell by ~4000, confirming the spill elimination.

> **Applicable when**: NCU MemoryCacheAccessPattern rule triggered for both local loads AND local stores with ~1/32 bytes utilized per sector (hallmark of register spill with strided access). The kernel contains tl.broadcast_to chains or other convoluted index expressions inside a hot loop that force the compiler to materialize large index tensors. The B (or A) matrix has a known, fixed stride that can be expressed as direct pointer arithmetic.

> **Avoid when**: The local memory access pattern rule is absent or shows >4 bytes/sector utilization (spill is not the bottleneck). The indexing complexity is genuinely required (e.g., gather/scatter with dynamic indices). The kernel is already using direct pointer arithmetic for tile loads. The dominant bottleneck is shared memory bank conflicts or grid underutilization rather than local memory spill traffic — in that case, this change yields only marginal improvement.

#### ❌ Plan D: `increase_block_n_to_eliminate_partial_tile_and_fix_b_indexing` (0.833×)

Increasing BLOCK_N from 32 to 64 when the actual N dimension is 32 nearly doubled executed instructions (1.54M → 2.93M) and registers per thread (128 → 168), reducing theoretical occupancy from 25% to 18.75% and growing elapsed cycles from 22,543 to 37,293 — a 65% increase in kernel duration. Although bank conflict degree halved (13.2-way → 7.9-way) and Warp Cycles Per Issued Instruction improved slightly (3.04 → 2.65), the doubled accumulator tile size (BLOCK_M×BLOCK_N FMA work scales linearly with BLOCK_N) completely dominated any conflict reduction benefit. This confirms the pattern documented in exp_003: setting BLOCK_N > N causes a regression proportional to the BLOCK_N/N ratio, regardless of whether ALLOW_TF32 is also changed.

> **Applicable when**: Never: this strategy is a documented regression for this kernel shape (N=32, BLOCK_N=32).

> **Avoid when**: When BLOCK_N already equals or exceeds the actual N dimension. Doubling BLOCK_N beyond N doubles the accumulator size and FMA instruction count (scales as BLOCK_M × BLOCK_N × K_tiles), while only halving bank conflict degree. In a latency-bound kernel with 1 active warp/scheduler and tiny grid, the extra instruction count and register pressure (128→168 regs/thread) dominate over any conflict reduction. This regression pattern is robust: it occurred in both iteration 3 (exp_003, speedup=0.8321) and iteration 4 (this item, speedup=0.8331) with nearly identical metric signatures.

---

## Iteration 4

**Best so far**: 1.3782× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores_enable_allow_tf32_in_tl_dot | 1.311× | 1.381× | ✓ |  |
| C | block_ptr_with_column_major_b_order_to_break_bank_conflicts | 1.310× | 1.390× | ✓ |  |
| D | tf32_tensor_cores | 0.949× | 0.833× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores... | Δ | block_ptr_with_co... | Δ | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 20770 ns | 15040 ns | -5730 ns (-27.6%) | 14940 ns | -5830 ns (-28.1%) | 24930 ns | +4160 ns (+20.0%) |
| SM Throughput (%) | 10.04 % | 12.97 % | +2.93 pp | 13.15 % | +3.11 pp | 14.85 % | +4.81 pp |
| Memory Throughput (%) | 70.23 % | 55.43 % | -14.80 pp | 56.19 % | -14.04 pp | 33.93 % | -36.30 pp |
| DRAM Throughput (%) | 4.46 % | 6.18 % | +1.72 pp | 6.23 % | +1.77 pp | 3.75 % | -0.71 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp |
| Registers/Thread | 96.00 register/thread | 128.00 register/thread | +32 | 128.00 register/thread | +32 | 168.00 register/thread | +72 |
| Waves/SM | 0.19 | 0.12 | -0.07 | 0.12 | -0.07 | 0.16 | -0.03 |
| No Eligible (%) | 87.76 % | 67.14 % | -20.62 pp | 67.42 % | -20.34 pp | 62.38 % | -25.38 pp |
| IPC Active | 0.49 inst/cycle | 0.65 inst/cycle | +0.16 | 0.66 inst/cycle | +0.17 | 0.75 inst/cycle | +0.26 |


### Key Learnings

#### ✅ Plan A: `tf32_tensor_cores_enable_allow_tf32_in_tl_dot` (1.381×)

Flipping ALLOW_TF32 from False to True in a Triton tl.dot call produced a 1.381x wall-clock speedup, yet NCU profiling metrics are nearly identical before and after: Compute SM Throughput moved only from 12.92% to 12.97%, Elapsed Cycles from 22,543 to 22,490, and all memory/scheduler metrics are within noise. The NCU-visible kernel duration barely changed (15.07 µs → 15.04 µs), suggesting the wall-clock speedup is attributable to reduced kernel launch overhead, autotuning cache effects, or measurement variance rather than a genuine Tensor Core throughput improvement. When the kernel is severely grid-underutilized (0.12 waves/SM, 3.12% achieved occupancy) and dominated by shared memory bank conflicts (13.2-way, 79.6% conflict wavefront), enabling TF32 does not materially improve the dominant bottleneck and NCU metrics confirm no structural change.

> **Applicable when**: Triton GEMM kernel where Compute SM Throughput is the dominant bottleneck (>40% compute SOL gap vs memory SOL), the kernel is compute-bound, and the tl.dot accumulator dtype is float32. TF32 is most impactful when the MMA pipeline is the bottleneck and the kernel has sufficient occupancy to hide memory latency.

> **Avoid when**: The kernel is severely grid-underutilized (waves_per_sm < 0.2) and dominated by shared memory bank conflicts (>10-way) — in this regime, enabling TF32 does not address the dominant bottleneck and NCU metrics show no improvement. Also avoid when correctness tolerance is tight (atol < 0.01): TF32 truncates FP32 mantissa from 23 to 10 bits per operand, which can produce max_diff ~0.05 per element over large K reductions, failing strict atol=0.001 checks.

---

## Iteration 5

**Best so far**: 1.3782× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | xor_swizzle_b_tile_load_to_break_smem_bank_conflicts | 1.103× | 1.083× | ✓ |  |
| B | reduce_block_k_to_halve_shared_memory_bank_conflicts | 1.387× | 1.452× | ✓ | **✓ best** |
| C | reduce_block_n_to_match_actual_n_dimension_and_cut_register_pressure | 1.280× | 1.338× | ✓ |  |


### Metric Comparison

| Metric | Baseline | xor_swizzle_b_til... | Δ | reduce_block_k_to... | Δ | reduce_block_n_to... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 20770 ns | 19170 ns | -1600 ns (-7.7%) | 14300 ns | -6470 ns (-31.2%) | 15520 ns | -5250 ns (-25.3%) |
| SM Throughput (%) | 10.04 % | 11.96 % | +1.92 pp | 12.30 % | +2.26 pp | 12.71 % | +2.67 pp |
| Memory Throughput (%) | 70.23 % | 61.11 % | -9.12 pp | 51.85 % | -18.38 pp | 53.33 % | -16.90 pp |
| DRAM Throughput (%) | 4.46 % | 4.85 % | +0.39 pp | 6.52 % | +2.06 pp | 5.99 % | +1.53 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp |
| Registers/Thread | 96.00 register/thread | 96.00 register/thread | +0 | 128.00 register/thread | +32 | 128.00 register/thread | +32 |
| Waves/SM | 0.19 | 0.10 | -0.09 | 0.12 | -0.07 | 0.12 | -0.07 |
| No Eligible (%) | 87.76 % | 70.73 % | -17.03 pp | 68.78 % | -18.98 pp | 68.49 % | -19.27 pp |
| IPC Active | 0.49 inst/cycle | 0.59 inst/cycle | +0.10 | 0.62 inst/cycle | +0.13 | 0.64 inst/cycle | +0.15 |


### Key Learnings

#### ❌ Plan A: `xor_swizzle_b_tile_load_to_break_smem_bank_conflicts` (1.083×)

Applying an XOR swizzle (rk ^ (rk >> 2)) identically to both A's k-column and B's k-row indices when loading tiles via explicit scalar pointer arithmetic produced only a 1.08x wall-clock speedup, while actually increasing kernel duration from 14.94 µs to 19.17 µs (28% slower in NCU). The SharedMemoryConflicts metrics barely improved: bank conflict degree dropped only from 13.2-way to 11.9-way and conflict wavefront percentage from 79.6% to 77.35%. Critically, executed instructions increased from 1.54M to 1.79M (+16%), and the L1/TEX hit rate jumped from 0% to 80.2% — indicating Triton's compiler, when given explicit scalar pointer arithmetic instead of block_ptr, routes loads through L1 cache (registers/L1) rather than shared memory staging, so the XOR swizzle is applied to global memory addresses rather than SMEM addresses and does not break the bank conflict pattern. The wall-clock speedup (1.08x) is likely attributable to the register count reduction (128→96 regs/thread) enabling slightly higher theoretical occupancy (25%→28.12%), not to any bank conflict reduction.

> **Applicable when**: Never for the purpose of eliminating shared memory bank conflicts in Triton GEMM kernels. The XOR swizzle on global memory address indices does not control the physical SMEM layout.

> **Avoid when**: When the goal is to reduce SharedMemoryConflicts in a Triton GEMM kernel with non-power-of-2 strides. Replacing tl.make_block_ptr with explicit scalar pointer arithmetic and applying XOR swizzle to the k-dimension index: (1) routes loads through L1 cache instead of SMEM (L1 hit rate jumps from 0% to 80%), making the swizzle irrelevant to SMEM bank conflicts; (2) increases instruction count by ~16% due to explicit index computation overhead; (3) produces only marginal bank conflict reduction (13.2→11.9 way). The net NCU effect is a 28% duration increase despite a modest wall-clock speedup from register pressure reduction (128→96 regs/thread). Do not use this approach to fix SMEM bank conflicts in Triton — the compiler controls SMEM layout, not the programmer via index arithmetic.

#### ✅ Plan B: `reduce_block_k_to_halve_shared_memory_bank_conflicts` (1.452×)

Reducing BLOCK_K from 32 to 16 in a Triton GEMM kernel with a non-power-of-2 B-matrix stride (stride_bn=144) produced a 1.45x speedup (Duration: 14.94 µs → 14.30 µs, Elapsed Cycles: 22,310 → 21,375, Executed Instructions: 1.54M → 1.40M). The key causal mechanism is that halving BLOCK_K halves the number of threads simultaneously accessing the B shared memory tile per warp, which should reduce the bank conflict degree; however, NCU still reports 13.2-way conflicts at 79.56% conflict wavefront percentage — nearly identical to before. The actual gain appears to come from eliminating boundary_check overhead (EVEN_K changed from False to True since K=144 is divisible by 16) and the reduced instruction count (~9% fewer executed instructions), rather than a measurable reduction in bank conflict degree. The shared memory footprint halved (24.58 KB → 12.29 KB per block), which may improve cache residency.

> **Applicable when**: Triton GEMM kernel where: (1) SharedMemoryConflicts rule is triggered with >10-way bank conflict degree caused by non-power-of-2 B-matrix stride; (2) K dimension is divisible by a smaller BLOCK_K (e.g., K=144 divisible by 16), enabling EVEN_K=True and eliminating boundary_check overhead; (3) current BLOCK_K is large (≥32) and EVEN_K=False, meaning boundary checks are active on every K-loop iteration; (4) tl.dot minimum operand inner dimension constraint (≥16) is satisfied by the reduced BLOCK_K.

> **Avoid when**: K is not divisible by the reduced BLOCK_K (EVEN_K cannot be set to True, so boundary check overhead remains and the benefit is reduced). BLOCK_K is already at the minimum (16) for tl.dot. The bank conflict root cause is in the A-tile rather than B-tile (different stride pattern). The kernel is compute-bound rather than memory/shared-memory-bound. Note: this change does NOT reliably reduce the NCU-reported bank conflict degree — the 13.2-way conflict metric was unchanged — so do not expect this to resolve the SharedMemoryConflicts rule; the speedup comes primarily from eliminating boundary_check overhead and reducing instruction count.

---

## Cumulative Progress

```
Baseline:  20770 ns  (1.000×)
Iter 1     15620 ns  (1.330×)  — add_eviction_policies_and_autotune_tile_sizes
Iter 3     15070 ns  (1.378×)  — tma_loads_replace_scalar_pointer_arithmetic
Iter 5     14300 ns  (1.452×)  — reduce_block_k_to_halve_shared_memory_bank_conflicts

Final best: 14300 ns  (1.4524×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,14 +5,28 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=2),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_14(arg_A, arg_B, out_ptr0):
-    EVEN_K : tl.constexpr = False
+    EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 64
     BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 32
+    BLOCK_K : tl.constexpr = 16
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -54,30 +68,37 @@

         offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
     else:
         offs_b_n = rn % N
+
     offs_k = tl.arange(0, BLOCK_K)
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
+    # Build block pointers for A and B to replace scalar pointer arithmetic
+    # A layout: [M=7840, K=144], strides [stride_am=144, stride_ak=1]
+    a_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    # B layout: [K=144, N=32], strides [stride_bk=1, stride_bn=144]
+    # B[k, n] = B_ptr + k * stride_bk + n * stride_bn = B_ptr + k*1 + n*144
+    b_block_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(0, 1),
+    )
+
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-
-        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
-        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
-
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 144*idx_m
-        a = tl.load(A + (xindex), mask=a_mask, other=0.0)
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 32*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 144*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
-
-
+        a = tl.load(a_block_ptr)
+        b = tl.load(b_block_ptr)
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
+        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
+        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -89,8 +110,6 @@

     # inductor generates a suffix
     xindex = idx_n + 32*idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
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
