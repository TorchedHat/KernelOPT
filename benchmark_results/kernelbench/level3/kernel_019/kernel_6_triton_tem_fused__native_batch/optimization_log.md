# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 39330 ns — **Best latency**: 39100 ns (**1.0059×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`software_pipeline_num_stages_to_hide_latency_and_reduce_bank_conflicts`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_6`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_019/kernel_6_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (245,1,1), Waves/SM: 0.93
- **Registers/thread**: 255
- **Duration**: 39330 ns
- **SM Throughput**: 30.72 % | **Memory**: 49.10 % | **DRAM**: 5.00 %
- **Achieved Occupancy**: 11.68 %
- **IPC Active**: 1.39 inst/cycle | **No Eligible**: 65.22 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.996× | 0.990× | ✓ |  |
| C | eviction_policies_streaming_A_evict_first | 0.995× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | eviction_policies... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 39330 ns | 39710 ns | +380 ns (+1.0%) | 39260 ns | -70 ns (-0.2%) |
| SM Throughput (%) | 30.72 % | 30.65 % | -0.07 pp | 30.65 % | -0.07 pp |
| Memory Throughput (%) | 49.10 % | 48.98 % | -0.12 pp | 48.96 % | -0.14 pp |
| DRAM Throughput (%) | 5.00 % | 5.01 % | +0.01 pp | 5.01 % | +0.01 pp |
| Achieved Occupancy (%) | 11.68 % | 11.68 % | +0.00 pp | 11.67 % | -0.01 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 0.93 | 0.93 | +0.00 | 0.93 | +0.00 |
| No Eligible (%) | 65.22 % | 65.22 % | +0.00 pp | 65.39 % | +0.17 pp |
| IPC Active | 1.39 inst/cycle | 1.39 inst/cycle | +0.00 | 1.39 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_load_coalescing_to_eliminate_bank_conflicts | 1.000× | 1.002× | ✓ | **✓ best** |
| C | cap_registers_with_maxnreg_to_improve_occupancy | 0.886× | 0.232× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | cap_registers_wit... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 39330 ns | 39230 ns | -100 ns (-0.3%) | 169410 ns | +130080 ns (+330.7%) |
| SM Throughput (%) | 30.72 % | 30.68 % | -0.04 pp | 13.22 % | -17.50 pp |
| Memory Throughput (%) | 49.10 % | 49.00 % | -0.10 pp | 24.17 % | -24.93 pp |
| DRAM Throughput (%) | 5.00 % | 5.04 % | +0.04 pp | 21.83 % | +16.83 pp |
| Achieved Occupancy (%) | 11.68 % | 11.67 % | -0.01 pp | 11.49 % | -0.19 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 128.00 register/thread | -127 |
| Waves/SM | 0.93 | 0.93 | +0.00 | 0.46 | -0.47 |
| No Eligible (%) | 65.22 % | 65.35 % | +0.13 pp | 89.04 % | +23.82 pp |
| IPC Active | 1.39 inst/cycle | 1.39 inst/cycle | +0.00 | 0.44 inst/cycle | -0.95 |


### Key Learnings

#### ❌ Plan C: `cap_registers_with_maxnreg_to_improve_occupancy` (0.232×)

Capping registers via maxnreg (255→128) successfully doubled theoretical occupancy (12.5%→25%) and raised Block Limit Registers from 2 to 4, but achieved occupancy remained flat (11.67%→11.49%) while kernel duration increased 4.3x (39→169 µs). The register cap forced aggressive spilling to local memory, causing a new CPIStall bottleneck where warps spent 11.6 cycles stalled on L1TEX scoreboard dependencies (69% of all stall cycles), and No-Eligible warp cycles worsened from 65% to 89%. The anti-pattern is: when a large GEMM tile (128×128 FP32) already saturates registers, forcing spill via maxnreg trades register pressure for local-memory traffic that is far more expensive than the latency-hiding benefit of the marginal occupancy gain.

> **Applicable when**: Never: this strategy is a confirmed anti-pattern for large GEMM tiles. Do not apply maxnreg to kernels with 128x128+ FP32 accumulator tiles where register pressure is already at hardware maximum.

> **Avoid when**: Any kernel with a large accumulator tile (BLOCK_M×BLOCK_N ≥ 64×64 in FP32) where registers_per_thread is near the hardware maximum (≥200). Forcing maxnreg in this regime causes register spilling to local memory, introducing L1TEX scoreboard stalls that far outweigh any occupancy benefit. The achieved occupancy does not improve even when theoretical occupancy doubles, because the spill traffic creates new stalls that keep warps ineligible.

---

## Iteration 3

**Best so far**: 1.0025× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads | 1.019× | 1.016× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tma_loads | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 39330 ns | 38720 ns | -610 ns (-1.6%) |
| SM Throughput (%) | 30.72 % | 31.55 % | +0.83 pp |
| Memory Throughput (%) | 49.10 % | 49.86 % | +0.76 pp |
| DRAM Throughput (%) | 5.00 % | 4.55 % | -0.45 pp |
| Achieved Occupancy (%) | 11.68 % | 11.60 % | -0.08 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 0.93 | 0.93 | +0.00 |
| No Eligible (%) | 65.22 % | 64.53 % | -0.69 pp |
| IPC Active | 1.39 inst/cycle | 1.42 inst/cycle | +0.03 |

---

## Iteration 4

**Best so far**: 1.0158× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_shared_memory_order_to_eliminate_bank_conflicts | 1.016× | 1.017× | ✓ |  |
| B | software_pipeline_num_stages_to_hide_latency_and_reduce_bank_conflicts | 1.022× | 1.006× | ✓ | **✓ best** |
| C | fix_b_matrix_order_and_add_eviction_policies_for_streaming_A | 1.016× | 1.014× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_shar... | Δ | software_pipeline... | Δ | fix_b_matrix_orde... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 39330 ns | 38660 ns | -670 ns (-1.7%) | 39100 ns | -230 ns (-0.6%) | 38780 ns | -550 ns (-1.4%) |
| SM Throughput (%) | 30.72 % | 31.46 % | +0.74 pp | 30.52 % | -0.20 pp | 31.42 % | +0.70 pp |
| Memory Throughput (%) | 49.10 % | 49.71 % | +0.61 pp | 50.78 % | +1.68 pp | 49.64 % | +0.54 pp |
| DRAM Throughput (%) | 5.00 % | 4.56 % | -0.44 pp | 4.49 % | -0.51 pp | 4.55 % | -0.45 pp |
| Achieved Occupancy (%) | 11.68 % | 11.60 % | -0.08 pp | 11.62 % | -0.06 pp | 11.59 % | -0.09 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 0.93 | 0.93 | +0.00 | 0.93 | +0.00 | 0.93 | +0.00 |
| No Eligible (%) | 65.22 % | 64.48 % | -0.74 pp | 65.54 % | +0.32 pp | 64.53 % | -0.69 pp |
| IPC Active | 1.39 inst/cycle | 1.41 inst/cycle | +0.02 | 1.38 inst/cycle | -0.01 | 1.41 inst/cycle | +0.02 |

---

## Iteration 5

**Best so far**: 1.0158× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  39330 ns  (1.000×)
Iter 2     39230 ns  (1.002×)  — fix_b_matrix_load_coalescing_to_eliminate_bank_conflicts
Iter 3     38720 ns  (1.016×)  — tma_loads

Final best: 39100 ns  (1.0059×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=1),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=1),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=3),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_6(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -44,50 +56,44 @@

     tl.assume(pid_m >= 0)
     tl.assume(pid_n >= 0)
 
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
-        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
-    else:
-        offs_a_m = rm % M
-    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
-        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
-    else:
-        offs_b_n = rn % N
-    offs_k = tl.arange(0, BLOCK_K)
+    # A is [M=31360, K=64] row-major: strides=(stride_am=64, stride_ak=1)
+    a_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    # B stored as [N=128, K=64]: B[k,n]=B_ptr+n*64+k => accessed as [K=64,N=128]
+    # strides=(stride_bk=1, stride_bn=64), shape=(K=64, N=128)
+    b_block_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(0, 1),
+    )
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
+        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
+        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
+        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
 
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 64*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 128*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 64*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
-
-    # rematerialize rm and rn to save registers
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    idx_m = rm[:, None]
-    idx_n = rn[None, :]
-    mask = (idx_m < M) & (idx_n < N)
-
-    # inductor generates a suffix
-    xindex = idx_n + 128*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+    # Output store via block pointer (out_ptr0 is [M=31360,N=128] stride [128,1])
+    out_block_ptr = tl.make_block_ptr(
+        base=out_ptr0,
+        shape=(M, N),
+        strides=(128, 1),
+        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
+        block_shape=(BLOCK_M, BLOCK_N),
+        order=(1, 0),
+    )
+    tl.store(out_block_ptr, acc, boundary_check=(0, 1))
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
