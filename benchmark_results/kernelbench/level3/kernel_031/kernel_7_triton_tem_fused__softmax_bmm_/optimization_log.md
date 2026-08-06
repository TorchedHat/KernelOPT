# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__softmax_bmm_exp_select_sub_transpose_view_7` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 852130 ns — **Best latency**: 1160000 ns (**0.7346×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__softmax_bmm_exp_select_sub_transpose_view_7`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_031/kernel_7_triton_tem_fused__softmax_bmm_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (128,1,1), Waves/SM: 0.19
- **Registers/thread**: 96
- **Duration**: 852130 ns
- **SM Throughput**: 47.24 % | **Memory**: 61.27 % | **DRAM**: 25.76 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 1.96 inst/cycle | **No Eligible**: 51.10 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_A_matrix | 0.508× | 1.000× | ✓ |  |
| D | fix_batch_grid_parallelism_and_autotune_tile_sizes | 1.023× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | fix_batch_grid_pa... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 852130 ns | 852060 ns | -70 ns (-0.0%) | 852320 ns | +190 ns (+0.0%) |
| SM Throughput (%) | 47.24 % | 47.23 % | -0.01 pp | 47.25 % | +0.01 pp |
| Memory Throughput (%) | 61.27 % | 61.25 % | -0.02 pp | 61.27 % | +0.00 pp |
| DRAM Throughput (%) | 25.76 % | 25.76 % | +0.00 pp | 25.75 % | -0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 96.00 register/thread | 96.00 register/thread | +0 | 96.00 register/thread | +0 |
| Waves/SM | 0.19 | 0.19 | +0.00 | 0.19 | +0.00 |
| No Eligible (%) | 51.10 % | 51.10 % | +0.00 pp | 51.12 % | +0.02 pp |
| IPC Active | 1.96 inst/cycle | 1.96 inst/cycle | +0.00 | 1.95 inst/cycle | -0.01 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_B_matrix_and_num_stages_pipeline | 1.034× | 0.735× | ✓ | **✓ best** |
| B | fix_batch_grid_dim1_only | 1.000× | 1.000× | ✓ |  |
| C | eviction_policies_streaming_A_reuse_B | 1.031× | 1.000× | ✓ |  |
| D | memory_coalescing_transpose_B_tile_load | 0.441× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | fix_batch_grid_di... | Δ | eviction_policies... | Δ | memory_coalescing... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 852130 ns | 1160000 ns | +307870 ns (+36.1%) | 852060 ns | -70 ns (-0.0%) | 852420 ns | +290 ns (+0.0%) | 852540 ns | +410 ns (+0.0%) |
| SM Throughput (%) | 47.24 % | 34.17 % | -13.07 pp | 46.97 % | -0.27 pp | 46.95 % | -0.29 pp | 47.24 % | +0.00 pp |
| Memory Throughput (%) | 61.27 % | 53.19 % | -8.08 pp | 60.91 % | -0.36 pp | 60.89 % | -0.38 pp | 61.26 % | -0.01 pp |
| DRAM Throughput (%) | 25.76 % | 18.94 % | -6.82 pp | 25.76 % | +0.00 pp | 25.75 % | -0.01 pp | 26.09 % | +0.33 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 96.00 register/thread | 114.00 register/thread | +18 | 96.00 register/thread | +0 | 96.00 register/thread | +0 | 96.00 register/thread | +0 |
| Waves/SM | 0.19 | 0.24 | +0.05 | 0.19 | +0.00 | 0.19 | +0.00 | 0.19 | +0.00 |
| No Eligible (%) | 51.10 % | 64.38 % | +13.28 pp | 51.09 % | -0.01 pp | 51.09 % | -0.01 pp | 51.09 % | -0.01 pp |
| IPC Active | 1.96 inst/cycle | 1.42 inst/cycle | -0.54 | 1.96 inst/cycle | +0.00 | 1.96 inst/cycle | +0.00 | 1.96 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_batch_grid_dim1_8_correct_index_arithmetic | 0.817× | 0.735× | ✓ |  |
| B | fix_batch_grid_dim0_and_dim1_correct_pid_mapping | 1.034× | 1.000× | ✓ |  |
| C | shared_memory_padding_B_tile_to_eliminate_bank_conflicts | 0.912× | 1.000× | ✓ |  |
| D | reduce_register_pressure_via_maxnreg_to_improve_occupancy | 0.744× | 0.344× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_batch_grid_di... | Δ | fix_batch_grid_di... | Δ | shared_memory_pad... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 852130 ns | 1160000 ns | +307870 ns (+36.1%) | 852420 ns | +290 ns (+0.0%) | 852450 ns | +320 ns (+0.0%) | 2480000 ns | +1627870 ns (+191.0%) |
| SM Throughput (%) | 47.24 % | 34.23 % | -13.01 pp | 47.24 % | +0.00 pp | 46.88 % | -0.36 pp | 22.44 % | -24.80 pp |
| Memory Throughput (%) | 61.27 % | 53.27 % | -8.00 pp | 61.27 % | +0.00 pp | 60.81 % | -0.46 pp | 44.99 % | -16.28 pp |
| DRAM Throughput (%) | 25.76 % | 19.02 % | -6.74 pp | 25.88 % | +0.12 pp | 25.75 % | -0.01 pp | 8.85 % | -16.91 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 96.00 register/thread | 114.00 register/thread | +18 | 96.00 register/thread | +0 | 96.00 register/thread | +0 | 64.00 register/thread | -32 |
| Waves/SM | 0.19 | 0.24 | +0.05 | 0.19 | +0.00 | 0.19 | +0.00 | 0.12 | -0.07 |
| No Eligible (%) | 51.10 % | 64.36 % | +13.26 pp | 51.09 % | -0.01 pp | 51.09 % | -0.01 pp | 74.42 % | +23.32 pp |
| IPC Active | 1.96 inst/cycle | 1.42 inst/cycle | -0.54 | 1.96 inst/cycle | +0.00 | 1.96 inst/cycle | +0.00 | 1.02 inst/cycle | -0.94 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_A_and_output_evict_first_reuse_B_in_L2 | 0.438× | 0.997× | ✓ |  |
| B | increase_block_n_to_eliminate_shared_memory_bank_conflicts | 0.483× | 0.484× | ✓ |  |
| C | fix_batch_grid_via_pid1_with_correct_stride_arithmetic | 1.014× | 0.999× | ✓ |  |
| D | fix_batch_grid_launch_dim1_equals_8_preserve_all_arithmetic | 0.749× | 0.345× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | increase_block_n_... | Δ | fix_batch_grid_vi... | Δ | fix_batch_grid_la... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 852130 ns | 854690 ns | +2560 ns (+0.3%) | 1760000 ns | +907870 ns (+106.5%) | 852640 ns | +510 ns (+0.1%) | 2470000 ns | +1617870 ns (+189.9%) |
| SM Throughput (%) | 47.24 % | 47.22 % | -0.02 pp | 43.00 % | -4.24 pp | 46.88 % | -0.36 pp | 22.73 % | -24.51 pp |
| Memory Throughput (%) | 61.27 % | 61.24 % | -0.03 pp | 29.37 % | -31.90 pp | 60.80 % | -0.47 pp | 45.56 % | -15.71 pp |
| DRAM Throughput (%) | 25.76 % | 25.68 % | -0.08 pp | 12.48 % | -13.28 pp | 25.74 % | -0.02 pp | 8.90 % | -16.86 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 96.00 register/thread | 96.00 register/thread | +0 | 168.00 register/thread | +72 | 96.00 register/thread | +0 | 64.00 register/thread | -32 |
| Waves/SM | 0.19 | 0.19 | +0.00 | 0.32 | +0.13 | 0.19 | +0.00 | 0.12 | -0.07 |
| No Eligible (%) | 51.10 % | 51.09 % | -0.01 pp | 46.75 % | -4.35 pp | 51.10 % | +0.00 pp | 74.47 % | +23.37 pp |
| IPC Active | 1.96 inst/cycle | 1.96 inst/cycle | +0.00 | 2.13 inst/cycle | +0.17 | 1.96 inst/cycle | +0.00 | 1.02 inst/cycle | -0.94 |


### Key Learnings

#### ❌ Plan B: `increase_block_n_to_eliminate_shared_memory_bank_conflicts` (0.484×)

Increasing BLOCK_N from 32 to 64 (to break shared memory bank conflicts) caused registers per thread to spike from 96 to 168, reducing Block Limit Registers from 5 to 3 and theoretical occupancy from 31.25% to 18.75%. Despite the bank conflict stall being partially resolved (No-Eligible dropped from 51.09% to 46.75%), the executed instruction count nearly doubled (318M → 599M) and elapsed cycles increased by ~2.07x (852 µs → 1.76 ms). The larger accumulator tile [BLOCK_M=128, BLOCK_N=64] requires 32 KB of FP32 registers, exhausting the register file and forcing the compiler to allocate 168 registers/thread — far exceeding the register-limited ceiling. When a kernel is already register-limited (Block Limit Registers ≤ 5), doubling the tile size in the N dimension worsens register pressure catastrophically and causes net regression even if it eliminates bank conflicts.

> **Applicable when**: Never apply BLOCK_N doubling when the kernel is already register-limited (Block Limit Registers ≤ 5, registers_per_thread ≥ 90) and BLOCK_M is large (≥ 128). The BLOCK_M × BLOCK_N accumulator lives in registers; doubling BLOCK_N doubles the accumulator register footprint, which at BLOCK_M=128 already consumes 32 KB of FP32 registers and pushes register count to 168+.

> **Avoid when**: Kernel has registers_per_thread ≥ 90 AND Block Limit Registers ≤ 5 AND BLOCK_M ≥ 128. In this regime, increasing BLOCK_N to fix bank conflicts causes register count to spike (96 → 168 observed), reducing Block Limit Registers from 5 to 3, cutting theoretical occupancy by ~40%, and nearly doubling executed instruction count. The bank conflict elimination benefit is completely overwhelmed by the register pressure regression.

---

## Cumulative Progress

```
Baseline:  852130 ns  (1.000×)

Final best: 1160000 ns  (0.7346×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,17 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=5),
+        triton.Config({}, num_warps=8, num_stages=5),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__softmax_bmm_exp_select_sub_transpose_view_7(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -60,7 +71,9 @@

 
     idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
     A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
-    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)
+    # Transposed B load: [BLOCK_N, BLOCK_K] with contiguous N-rows (stride_bn=1)
+    # to eliminate shared memory bank conflicts caused by stride_bk=256
+    B = B + (rbn[:, None] * stride_bn + rk[None, :] * stride_bk + idx_q*stride_bq)
 
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
     for k in range(K, 0, -BLOCK_K):
@@ -69,8 +82,10 @@

             b = tl.load(B)
         else:
             a = tl.load(A, mask=rk[None, :] < k, other=0.)
-            b = tl.load(B, mask=rk[:, None] < k, other=0.)
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
+            # rk is now the column (dim-1) index in the transposed [BLOCK_N, BLOCK_K] layout
+            b = tl.load(B, mask=rk[None, :] < k, other=0.)
+        # tl.trans(b) converts [BLOCK_N, BLOCK_K] back to [BLOCK_K, BLOCK_N] for correct matmul
+        acc += tl.dot(a, tl.trans(b), allow_tf32=ALLOW_TF32)
         A += BLOCK_K * stride_ak
         B += BLOCK_K * stride_bk
 
@@ -85,8 +100,6 @@

     # inductor generates a suffix
     xindex = idx_n + 32*idx_m + 524288*idx_q
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
