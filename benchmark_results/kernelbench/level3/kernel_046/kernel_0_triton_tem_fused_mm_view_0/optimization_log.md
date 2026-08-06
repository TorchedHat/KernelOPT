# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_mm_view_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 406940 ns — **Best latency**: 406940 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_mm_view_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_046/kernel_0_triton_tem_fused_mm_view_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3200,1,1), Waves/SM: 4.85
- **Registers/thread**: 96
- **Duration**: 406940 ns
- **SM Throughput**: 76.49 % | **Memory**: 70.50 % | **DRAM**: 22.73 %
- **Achieved Occupancy**: 29.75 %
- **IPC Active**: 3.17 inst/cycle | **No Eligible**: 20.79 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | add_autotune_reduce_register_pressure_vectorized_loads | 0.779× | 0.783× | ✓ |  |
| D | autotune_tile_sizes_to_reduce_register_pressure_and_improve_occupancy | 0.771× | 0.774× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_autotune_redu... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 406940 ns | 519620 ns | +112680 ns (+27.7%) | 525700 ns | +118760 ns (+29.2%) |
| SM Throughput (%) | 76.49 % | 59.80 % | -16.69 pp | 59.26 % | -17.23 pp |
| Memory Throughput (%) | 70.50 % | 80.72 % | +10.22 pp | 79.79 % | +9.29 pp |
| DRAM Throughput (%) | 22.73 % | 17.92 % | -4.81 pp | 17.72 % | -5.01 pp |
| Achieved Occupancy (%) | 29.75 % | 12.41 % | -17.34 pp | 12.45 % | -17.30 pp |
| Registers/Thread | 96.00 register/thread | 128.00 register/thread | +32 | 128.00 register/thread | +32 |
| Waves/SM | 4.85 | 12.12 | +7.27 | 12.12 | +7.27 |
| No Eligible (%) | 20.79 % | 37.89 % | +17.10 pp | 37.89 % | +17.10 pp |
| IPC Active | 3.17 inst/cycle | 2.48 inst/cycle | -0.69 | 2.49 inst/cycle | -0.68 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_shared_memory_bank_conflicts_via_software_pipelining | 0.780× | 0.778× | ✓ |  |
| C | fix_shared_memory_bank_conflicts_via_num_stages_pipelining | 0.885× | 0.770× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_shared_memory... | Δ | fix_shared_memory... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 406940 ns | 522910 ns | +115970 ns (+28.5%) | 528480 ns | +121540 ns (+29.9%) |
| SM Throughput (%) | 76.49 % | 60.08 % | -16.41 pp | 59.59 % | -16.90 pp |
| Memory Throughput (%) | 70.50 % | 81.10 % | +10.60 pp | 80.22 % | +9.72 pp |
| DRAM Throughput (%) | 22.73 % | 17.81 % | -4.92 pp | 17.63 % | -5.10 pp |
| Achieved Occupancy (%) | 29.75 % | 12.41 % | -17.34 pp | 12.45 % | -17.30 pp |
| Registers/Thread | 96.00 register/thread | 128.00 register/thread | +32 | 128.00 register/thread | +32 |
| Waves/SM | 4.85 | 12.12 | +7.27 | 12.12 | +7.27 |
| No Eligible (%) | 20.79 % | 37.73 % | +16.94 pp | 37.70 % | +16.91 pp |
| IPC Active | 3.17 inst/cycle | 2.48 inst/cycle | -0.69 | 2.48 inst/cycle | -0.69 |


### Key Learnings

#### ❌ Plan A: `fix_shared_memory_bank_conflicts_via_software_pipelining` (0.778×)

Adding num_stages=2 and num_warps=4 to a Triton kernel launch call produced zero measurable change: bank conflict wavefronts remained at 33.24-33.25%, dynamic shared memory per block stayed at 81.92 KB, registers per thread stayed at 128, and duration increased slightly from 519.62 µs to 522.91 µs (0.78x speedup). The profiling data shows that all metrics — occupancy, warp cycles per instruction, memory throughput, and bank conflict rate — were statistically identical before and after. This indicates that when a Triton kernel is already compiled with a fixed tile configuration (BLOCK_M=128, BLOCK_N=32, BLOCK_K=64) and the shared memory layout is already determined at compile time, passing num_stages at the launch call site has no effect on the compiled kernel's shared memory layout or bank conflict behavior. The num_stages parameter only influences the Triton compiler's code generation when the kernel is being compiled (i.e., it must be a tl.constexpr or passed during JIT compilation), not at runtime launch.

> **Applicable when**: Never — this is a negative example. Do not attempt to fix shared memory bank conflicts by adding num_stages to the kernel launch call when the kernel is already compiled with fixed tl.constexpr tile parameters.

> **Avoid when**: When the Triton kernel is already compiled with fixed BLOCK_M, BLOCK_N, BLOCK_K as tl.constexpr values and the shared memory layout is determined at compile time. Adding num_stages at the launch call site cannot retroactively change the compiled kernel's smem layout, padding, or bank conflict behavior. To actually engage Triton's software pipelining and smem padding, num_stages must be a tl.constexpr parameter inside the kernel body or passed during the @triton.jit compilation phase.

#### ❌ Plan C: `fix_shared_memory_bank_conflicts_via_num_stages_pipelining` (0.770×)

Adding @triton.autotune with varying num_stages (2, 3, 4) and num_warps (4, 8) to address 4-way shared memory bank conflicts produced no measurable improvement: bank conflict percentage remained at ~33%, duration increased slightly from 519 µs to 528 µs, and all other metrics were statistically unchanged. The kernel was already at the shared memory limit (82 KB dynamic smem/block, Block Limit Shared Mem=2), meaning num_stages pipelining could not increase smem allocation further to add padding — the hardware constraint prevented the standard Triton bank-conflict fix from taking effect. When a kernel is already at the per-block shared memory ceiling, num_stages-based pipelining cannot insert the padding needed to break bank alignment, making this approach ineffective for bank conflict resolution in this regime.

> **Applicable when**: Never apply num_stages autotune as a bank conflict fix when the kernel is already at or near the per-block shared memory hardware limit (Block Limit Shared Mem ≤ 2, dynamic smem ≥ 80 KB/block). The pipelining padding cannot be inserted when smem is saturated.

> **Avoid when**: When dynamic shared memory per block is already ≥ 80 KB and Block Limit Shared Mem = 2, num_stages pipelining will not reduce bank conflicts — the smem ceiling prevents the compiler from adding padding. Also do not apply when the autotune key list is empty (key=[]) and tile dimensions are hardcoded constexprs, as the autotune cannot vary the parameters that actually affect bank alignment.

---

## Cumulative Progress

```
Baseline:  406940 ns  (1.000×)

Final best: 406940 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,14 +5,15 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_tem_fused_mm_view_0(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 64
-    BLOCK_N : tl.constexpr = 64
-    BLOCK_K : tl.constexpr = 32
+    BLOCK_M : tl.constexpr = 128
+    BLOCK_N : tl.constexpr = 32
+    BLOCK_K : tl.constexpr = 64
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -32,8 +33,8 @@

 
     # based on triton.ops.matmul
     pid = tl.program_id(0).to(INDEX_DTYPE)
-    grid_m = (M + BLOCK_M - 1) // BLOCK_M
-    grid_n = (N + BLOCK_N - 1) // BLOCK_N
+    grid_m = tl.cdiv(M, BLOCK_M)
+    grid_n = tl.cdiv(N, BLOCK_N)
 
     # re-order program ID for better L2 performance
     width = GROUP_M * grid_n
@@ -46,35 +47,21 @@

 
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
     rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
-        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
-    else:
-        offs_a_m = rm % M
-    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
-        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
-    else:
-        offs_b_n = rn % N
+
+    # Vectorization hints: aligned, contiguous blocks enable 128-bit loads
+    offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
+    offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
     offs_k = tl.arange(0, BLOCK_K)
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
+        a_k = tl.max_contiguous(tl.multiple_of(offs_k + k_idx * BLOCK_K, BLOCK_K), BLOCK_K)
+        b_k = tl.max_contiguous(tl.multiple_of(offs_k + k_idx * BLOCK_K, BLOCK_K), BLOCK_K)
 
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 512*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 48*idx_m
-        b = tl.load(B + (xindex))
-
+        a = tl.load(A + offs_a_m[:, None] * stride_am + a_k[None, :] * stride_ak)
+        b = tl.load(B + b_k[:, None] * stride_bk + offs_b_n[None, :] * stride_bn)
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -84,10 +71,7 @@

     mask = (idx_m < M) & (idx_n < N)
 
     # inductor generates a suffix
-    xindex = idx_n + 48*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+    tl.store(out_ptr0 + rm[:, None] * N + rn[None, :], acc, mask=mask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
