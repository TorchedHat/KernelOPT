# Kernel Optimization Log

## Summary

- **Kernel**: `conv1d_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 15630000 ns — **Best latency**: 15630000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `conv1d_kernel`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_076/synthesis/kernel_0_unknown.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (349516,1,1), Waves/SM: 882.62
- **Registers/thread**: 134
- **Duration**: 15630000 ns
- **SM Throughput**: 47.40 % | **Memory**: 48.04 % | **DRAM**: 29.80 %
- **Achieved Occupancy**: 18.57 %
- **IPC Active**: 1.83 inst/cycle | **No Eligible**: 54.26 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores_and_register_pressure_reduction | 0.999× | 0.998× | ✓ |  |
| B | reduce_register_pressure_hoist_loop_invariants | 0.996× | 1.147× | ✓ |  |
| C | eviction_policies_for_streaming_input_and_cached_weights | 1.000× | 1.003× | ✓ |  |
| D | expand_autotune_configs_with_higher_num_stages_and_register_pressure_relief | 0.999× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores... | Δ | reduce_register_p... | Δ | eviction_policies... | Δ | expand_autotune_c... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 15630000 ns | 15660000 ns | +30000 ns (+0.2%) | 13630000 ns | -2000000 ns (-12.8%) | 15590000 ns | -40000 ns (-0.3%) | 15630000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 47.40 % | 47.40 % | +0.00 pp | 56.74 % | +9.34 pp | 47.52 % | +0.12 pp | 47.40 % | +0.00 pp |
| Memory Throughput (%) | 48.04 % | 48.04 % | +0.00 pp | 56.00 % | +7.96 pp | 48.16 % | +0.12 pp | 48.04 % | +0.00 pp |
| DRAM Throughput (%) | 29.80 % | 29.76 % | -0.04 pp | 34.19 % | +4.39 pp | 29.88 % | +0.08 pp | 29.80 % | +0.00 pp |
| Achieved Occupancy (%) | 18.57 % | 18.57 % | +0.00 pp | 24.78 % | +6.21 pp | 18.57 % | +0.00 pp | 18.57 % | +0.00 pp |
| Registers/Thread | 134.00 register/thread | 134.00 register/thread | +0 | 110.00 register/thread | -24 | 134.00 register/thread | +0 | 134.00 register/thread | +0 |
| Waves/SM | 882.62 | 882.62 | +0.00 | 661.96 | -220.66 | 882.62 | +0.00 | 882.62 | +0.00 |
| No Eligible (%) | 54.26 % | 54.28 % | +0.02 pp | 47.99 % | -6.27 pp | 54.07 % | -0.19 pp | 54.28 % | +0.02 pp |
| IPC Active | 1.83 inst/cycle | 1.83 inst/cycle | +0.00 | 2.08 inst/cycle | +0.25 | 1.84 inst/cycle | +0.01 | 1.83 inst/cycle | +0.00 |


### Key Learnings

#### ✅ Plan B: `reduce_register_pressure_hoist_loop_invariants` (1.147×)

When a Triton kernel computes batch index, output position, and base pointer offsets (via integer division and modulo on wide vectors) inside a reduction loop, those intermediate tensors remain live across iterations and inflate register usage. Hoisting these computations outside the loop reduces registers/thread (134→110), which raises the block limit from registers (3→4 blocks/SM), improving theoretical occupancy (18.75%→25%) and eligible warps per scheduler (0.70→1.01), ultimately reducing kernel duration by ~13%. This pattern applies whenever NCU reports TheoreticalOccupancy limited by registers and the inner loop body contains index decode operations that are invariant to the loop induction variable.

> **Applicable when**: NCU reports TheoreticalOccupancy limited by registers (block_limit_registers < block_limit_warps), registers/thread > ~100, and the kernel inner loop body contains index decode operations (integer div/mod on SIMD-width vectors) that depend only on the outer program ID, not the loop induction variable. IssueSlotUtilization rule triggered with high no-eligible-warp percentage (>40%).

> **Avoid when**: Register pressure is not the occupancy bottleneck (e.g., block_limit_shared_mem is the binding constraint). Also avoid if the hoisted computation is genuinely loop-variant (depends on k_start or offs_k). If occupancy is already near hardware maximum, further register reduction yields diminishing returns.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores_for_tl_dot | 0.993× | 1.147× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 15630000 ns | 13630000 ns | -2000000 ns (-12.8%) |
| SM Throughput (%) | 47.40 % | 56.75 % | +9.35 pp |
| Memory Throughput (%) | 48.04 % | 56.01 % | +7.97 pp |
| DRAM Throughput (%) | 29.80 % | 34.19 % | +4.39 pp |
| Achieved Occupancy (%) | 18.57 % | 24.78 % | +6.21 pp |
| Registers/Thread | 134.00 register/thread | 110.00 register/thread | -24 |
| Waves/SM | 882.62 | 661.96 | -220.66 |
| No Eligible (%) | 54.26 % | 47.98 % | -6.28 pp |
| IPC Active | 1.83 inst/cycle | 2.08 inst/cycle | +0.25 |


### Key Learnings

#### ⚠️ Plan A: `tf32_tensor_cores_for_tl_dot` (1.147×)

Adding allow_tf32=True to tl.dot is expected to route matrix multiplications through Tensor Core TF32 MMA instructions (~989 TFLOPS on H200) instead of scalar FP32 FFMA (~67 TFLOPS). However, in this case all NCU metrics (compute SM throughput, duration, occupancy, scheduler statistics) remained essentially unchanged before and after the change, indicating the kernel was not compute-bound on the dot product and the TF32 path was either already selected by Triton or the bottleneck lies elsewhere (uncoalesced memory access, low occupancy). The reported 1.1467x speedup appears to be measurement noise or attributable to the prior iteration's changes, not this diff.

> **Applicable when**: Kernel is compute-bound (Compute SM Throughput >> Memory Throughput), tl.dot is the dominant instruction, K reduction dimension is large enough that TF32 error is acceptable (K < ~512 for atol=1e-3), and tile dimensions are multiples of 16 for Tensor Core compatibility.

> **Avoid when**: Kernel is memory-bound or latency-bound (as evidenced by low eligible warps per scheduler, high no-eligible-warp %, and uncoalesced access warnings dominating). When NCU metrics show no change in compute throughput after applying allow_tf32=True, the bottleneck is not the dot product throughput. Also avoid when K is large (>512) and numerical precision is critical.

---

## Cumulative Progress

```
Baseline:  15630000 ns  (1.000×)

Final best: 15630000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -9,6 +9,7 @@

 # Output length: (524280 - 4*(3-1) - 1) // 3 + 1 = (524280 - 9) // 3 + 1 = 174757 + 1 = 174758
 # Output: (64, 128, 174758)
 
+
 @triton.autotune(
     configs=[
         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
@@ -19,6 +20,12 @@

         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
         triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=4),
     ],
     key=['M', 'N', 'K'],
 )
@@ -46,10 +53,14 @@

     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
     offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
     
-    # Decode M index: batch and output position
+    # HOISTED outside K-loop: Decode M index: batch and output position
     batch = offs_m // L_out
     out_pos = offs_m % L_out
-    
+    # Precompute base pointer for X (includes batch offset and out_pos*stride)
+    x_base = X_ptr + batch * (Ci * L_in) + out_pos * stride
+    # Precompute valid_m mask (replaces valid_m & valid_batch inside loop)
+    valid_m = offs_m < M
+
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
     
     # Iterate over K = Ci * KERNEL_SIZE in blocks
@@ -60,19 +71,18 @@

         ci = offs_k // KERNEL_SIZE
         kpos = offs_k % KERNEL_SIZE
         
-        # Input position
+        # Input position (with padding subtracted for boundary check)
         in_pos = out_pos[:, None] * stride + kpos[None, :] * dilation - padding
         
-        # Validity mask
-        valid_m = offs_m[:, None] < M
+        # Validity mask (valid_batch is redundant since M = N * L_out bounds batch)
         valid_k = offs_k[None, :] < K
         valid_in = (in_pos >= 0) & (in_pos < L_in)
-        valid_batch = batch[:, None] < N
-        mask_x = valid_m & valid_k & valid_in & valid_batch
+        mask_x = valid_m[:, None] & valid_k & valid_in
         
         # Load X: shape [BLOCK_M, BLOCK_K]
         # X is (N, Ci, L_in), stride: Ci*L_in, L_in, 1
-        x_ptrs = X_ptr + batch[:, None] * (Ci * L_in) + ci[None, :] * L_in + in_pos
+        # x_base already includes batch * (Ci * L_in) + out_pos * stride
+        x_ptrs = x_base[:, None] + ci[None, :] * L_in + kpos[None, :] * dilation
         x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
         
         # Load W: shape [BLOCK_K, BLOCK_N]
@@ -88,7 +98,6 @@

     mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < Co)
     y_ptrs = Y_ptr + batch[:, None] * (Co * L_out) + offs_n[None, :] * L_out + out_pos[:, None]
     tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)
-
 
 def kernel_function(weight, x):
     """
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
