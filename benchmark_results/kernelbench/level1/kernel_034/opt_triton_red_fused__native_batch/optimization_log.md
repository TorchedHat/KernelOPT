# Kernel Optimization Log

## Summary

- **Kernel**: `instance_norm_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6240000 ns — **Best latency**: 8950000 ns (**0.6972×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `instance_norm_kernel`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_034/synthesis/kernel_0_triton_red_fused__native_batch_norm_legi.py`
- **Framework**: triton
- **Launch**: Block (256,1,1), Grid (7168,1,1), Waves/SM: 6.79
- **Registers/thread**: 32
- **Duration**: 6240000 ns
- **SM Throughput**: 49.22 % | **Memory**: 73.46 % | **DRAM**: 73.46 %
- **Achieved Occupancy**: 94.52 %
- **IPC Active**: 1.98 inst/cycle | **No Eligible**: 50.52 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_loads_stores | 0.924× | 0.992× | ✓ |  |
| B | vectorized_loads_with_multiple_of_hints | 0.763× | 0.828× | ✓ |  |
| C | single_pass_reduction_with_software_pipelining | 0.997× | 1.045× | ✓ |  |
| D | increase_block_size_and_pipeline_stages_to_hide_dram_latency | 1.007× | 0.697× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | vectorized_loads_... | Δ | single_pass_reduc... | Δ | increase_block_si... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6240000 ns | 6290000 ns | +50000 ns (+0.8%) | 7540000 ns | +1300000 ns (+20.8%) | 5970000 ns | -270000 ns (-4.3%) | 8950000 ns | +2710000 ns (+43.4%) |
| SM Throughput (%) | 49.22 % | 50.86 % | +1.64 pp | 27.59 % | -21.63 pp | 33.92 % | -15.30 pp | 59.70 % | +10.48 pp |
| Memory Throughput (%) | 73.46 % | 72.82 % | -0.64 pp | 81.14 % | +7.68 pp | 76.82 % | +3.36 pp | 73.11 % | -0.35 pp |
| DRAM Throughput (%) | 73.46 % | 72.82 % | -0.64 pp | 81.14 % | +7.68 pp | 76.82 % | +3.36 pp | 51.19 % | -22.27 pp |
| Achieved Occupancy (%) | 94.52 % | 94.49 % | -0.03 pp | 96.56 % | +2.04 pp | 96.24 % | +1.72 pp | 90.93 % | -3.59 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 30.00 register/thread | -2 |
| Waves/SM | 6.79 | 6.79 | +0.00 | 6.79 | +0.00 | 6.79 | +0.00 | 3.39 | -3.40 |
| No Eligible (%) | 50.52 % | 48.92 % | -1.60 pp | 74.43 % | +23.91 pp | 72.36 % | +21.84 pp | 41.10 % | -9.42 pp |
| IPC Active | 1.98 inst/cycle | 2.05 inst/cycle | +0.07 | 1.02 inst/cycle | -0.96 | 1.10 inst/cycle | -0.88 | 2.36 inst/cycle | +0.38 |


### Key Learnings

#### ❌ Plan B: `vectorized_loads_with_multiple_of_hints` (0.828×)

Adding tl.max_contiguous/tl.multiple_of hints and converting from one-pass Welford to a three-loop two-pass (mean, variance, normalize) design caused a 17% slowdown. Despite reducing total executed instructions by ~38%, the kernel duration increased from 6.24ms to 7.54ms. The root cause is that the two-pass approach adds an extra full data scan (3 passes over N elements instead of 2), which increased DRAM throughput from 73% to 81% and doubled warp stall cycles per issued instruction (30.5 → 60.4 cycles), indicating the extra memory pass amplified the L1TEX scoreboard stall bottleneck rather than alleviating it. The L2 hit rate also dropped from 33.8% to 25.7%, suggesting the extra pass evicted cached data. The vectorization hints alone did not compensate for the added memory traffic.

> **Applicable when**: Never — this specific combination of 3-loop two-pass design + vectorization hints caused regression in this memory-bound kernel.

> **Avoid when**: When the kernel is already memory-bound (DRAM throughput >70%) and the baseline uses a 2-loop design (Welford + normalize): adding a third data scan increases total DRAM traffic and worsens L1TEX stall cycles even if total instruction count decreases. The tl.max_contiguous/tl.multiple_of hints alone do not compensate for the extra memory pass. Avoid converting 2-loop Welford kernels to 3-loop two-pass designs when DRAM is the bottleneck.

#### ❌ Plan D: `increase_block_size_and_pipeline_stages_to_hide_dram_latency` (0.697×)

Halving the block size (256→128 threads) while keeping the same grid size cut waves/SM from 6.79 to 3.39, reducing achieved occupancy from 94.52% to 90.93% and nearly doubling kernel duration (6.24ms→8.95ms). Despite the autotuner selecting a smaller block (128 threads), total executed instructions nearly doubled (2.4B→4.1B), indicating the sum-of-squares Welford reformulation or loop structure generated significantly more instructions. Adding num_stages=3/4 to larger block configs did not help because the autotuner selected a small-block config that lacked pipelining, and the partial-wave tail effect (25% overhead flagged by LaunchConfiguration rule) compounded the regression. When a kernel already has high occupancy (>90%) and high waves/SM (>6), reducing block size to enable more pipeline stages is counterproductive — the occupancy loss and instruction count increase outweigh any latency-hiding benefit.

> **Applicable when**: Kernel is memory-latency bound (CPIStall rule, L1TEX stall >40% of cycles), low L1 hit rate (0%), moderate DRAM throughput (>60%), and autotuner selects small block sizes with low num_stages.

> **Avoid when**: Kernel already has high waves/SM (>6) and high achieved occupancy (>90%); reducing block size to accommodate more pipeline stages will cut waves/SM and increase instruction count, causing regression. Also avoid when the autotuner will select a smaller config than intended — verify the winning config post-autotune.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | explicit_fma_to_reduce_non_fused_fp32_instructions | 0.998× | 1.044× | ✓ |  |
| B | replace_welford_with_sum_of_squares_reduction | 0.932× | 1.035× | ✓ |  |
| C | fuse_mean_variance_passes_with_sum_of_squares_and_evict_first | 0.970× | 1.021× | ✓ |  |
| D | replace_welford_with_sum_of_squares_and_fix_autotune_configs | 1.006× | 0.746× | ✓ |  |
| E | add_triton_autotune | 0.998× | 1.044× | ✓ |  |


### Metric Comparison

| Metric | Baseline | explicit_fma_to_r... | Δ | replace_welford_w... | Δ | fuse_mean_varianc... | Δ | replace_welford_w... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6240000 ns | 5980000 ns | -260000 ns (-4.2%) | 6030000 ns | -210000 ns (-3.4%) | 6110000 ns | -130000 ns (-2.1%) | 8370000 ns | +2130000 ns (+34.1%) | 5980000 ns | -260000 ns (-4.2%) |
| SM Throughput (%) | 49.22 % | 33.81 % | -15.41 pp | 33.54 % | -15.68 pp | 33.06 % | -16.16 pp | 44.43 % | -4.79 pp | 33.71 % | -15.51 pp |
| Memory Throughput (%) | 73.46 % | 76.60 % | +3.14 pp | 75.98 % | +2.52 pp | 75.02 % | +1.56 pp | 56.19 % | -17.27 pp | 76.70 % | +3.24 pp |
| DRAM Throughput (%) | 73.46 % | 76.60 % | +3.14 pp | 75.98 % | +2.52 pp | 75.02 % | +1.56 pp | 54.73 % | -18.73 pp | 76.70 % | +3.24 pp |
| Achieved Occupancy (%) | 94.52 % | 96.10 % | +1.58 pp | 96.11 % | +1.59 pp | 95.90 % | +1.38 pp | 90.07 % | -4.45 pp | 96.15 % | +1.63 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 24.00 register/thread | -8 | 32.00 register/thread | +0 |
| Waves/SM | 6.79 | 6.79 | +0.00 | 6.79 | +0.00 | 6.79 | +0.00 | 3.39 | -3.40 | 6.79 | +0.00 |
| No Eligible (%) | 50.52 % | 73.01 % | +22.49 pp | 70.47 % | +19.95 pp | 71.27 % | +20.75 pp | 66.00 % | +15.48 pp | 72.33 % | +21.81 pp |
| IPC Active | 1.98 inst/cycle | 1.08 inst/cycle | -0.90 | 1.18 inst/cycle | -0.80 | 1.15 inst/cycle | -0.83 | 1.36 inst/cycle | -0.62 | 1.11 inst/cycle | -0.87 |


### Key Learnings

#### ✅ Plan A: `explicit_fma_to_reduce_non_fused_fp32_instructions` (1.044×)

When the FPInstructions NCU rule fires with a high ratio of non-fused to fused FP32 instructions (e.g., 10:1), two structural changes can reduce instruction count: (1) computing a masked tensor once and reusing it for multiple reductions (e.g., x_masked = tl.where(mask, x, 0.0) used for both sum and sum-of-squares) eliminates a redundant tl.where call that blocks FMA fusion; (2) pre-computing a scalar bias outside the loop and using libdevice.fma(x, scale, bias) inside the loop converts a FSUB+FMUL pair into a single FFMA per element. Together these changes reduced total executed instructions by ~2.3% (1280M → 1251M) and achieved a 4.35% wall-clock speedup despite the kernel remaining DRAM-bound (DRAM throughput ~76%), suggesting the instruction reduction slightly reduced the compute portion between memory stalls.

> **Applicable when**: FPInstructions NCU rule fires with non-fused to fused FP32 ratio >= 5:1; kernel has multiple tl.where calls on the same mask within one loop body; normalize pass uses (x - scalar) * scalar pattern; kernel is memory-bound but has non-trivial compute fraction (Compute SM Throughput > 20%).

> **Avoid when**: Triton compiler already auto-fuses the patterns (verify by checking fused vs non-fused instruction counts post-change); kernel is purely compute-bound where instruction count is not the bottleneck; libdevice.fma introduces precision differences that violate correctness tolerances (verify algebraic equivalence: fma(x, inv_std, -mean*inv_std) == (x - mean)*inv_std only when bias is exact).

#### ✅ Plan B: `replace_welford_with_sum_of_squares_reduction` (1.045×)

Welford's online algorithm for computing mean and variance performs per-chunk divisions and conditional merges inside the reduction loop, generating a large number of non-fused FP32 instructions (~925M before vs ~481M after). Replacing it with simple sum and sum-of-squares accumulation (mean = Σx/N, var = Σx²/N − mean²) eliminates these expensive per-iteration divisions and conditionals, cutting total executed instructions by ~47% and reducing kernel duration by ~4.5%. This strategy is applicable whenever a two-pass or online variance kernel is bottlenecked by high non-fused FP32 instruction count (FPInstructions NCU rule) and the input data is numerically well-conditioned (e.g., image tensors in [0,1] range where catastrophic cancellation is not a concern).

> **Applicable when**: Kernel uses Welford online algorithm for variance; FPInstructions NCU rule fires with high non-fused FP32 instruction count; input data is numerically well-conditioned (bounded range, no extreme outliers); CPIStall rule fires with significant L1TEX stall fraction.

> **Avoid when**: Input data has very large dynamic range or extreme outliers where E[x²]−E[x]² suffers catastrophic cancellation (e.g., large-magnitude activations far from zero); strict numerical equivalence to Welford is required; correctness tolerance is tighter than ~1e-3.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_tile_to_eliminate_second_dram_pass | 0.712× | 0.793× | ✓ |  |
| B | fma_fusion_via_masked_tensor_reuse_and_libdevice_fma | 0.933× | 1.035× | ✓ |  |
| C | fma_fusion_with_xmasked_reuse_and_pass1_evict_first | 0.933× | 1.035× | ✓ |  |
| D | evict_first_on_pass2_streaming_loads_and_stores | 0.993× | 0.742× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_til... | Δ | fma_fusion_via_ma... | Δ | fma_fusion_with_x... | Δ | evict_first_on_pa... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6240000 ns | 7870000 ns | +1630000 ns (+26.1%) | 6030000 ns | -210000 ns (-3.4%) | 6030000 ns | -210000 ns (-3.4%) | 8410000 ns | +2170000 ns (+34.8%) |
| SM Throughput (%) | 49.22 % | 26.45 % | -22.77 pp | 33.56 % | -15.66 pp | 33.53 % | -15.69 pp | 44.16 % | -5.06 pp |
| Memory Throughput (%) | 73.46 % | 77.62 % | +4.16 pp | 75.97 % | +2.51 pp | 75.97 % | +2.51 pp | 55.82 % | -17.64 pp |
| DRAM Throughput (%) | 73.46 % | 77.62 % | +4.16 pp | 75.97 % | +2.51 pp | 75.97 % | +2.51 pp | 54.46 % | -19.00 pp |
| Achieved Occupancy (%) | 94.52 % | 96.76 % | +2.24 pp | 96.16 % | +1.64 pp | 96.25 % | +1.73 pp | 90.10 % | -4.42 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 26.00 register/thread | -6 |
| Waves/SM | 6.79 | 6.79 | +0.00 | 6.79 | +0.00 | 6.79 | +0.00 | 3.39 | -3.40 |
| No Eligible (%) | 50.52 % | 77.86 % | +27.34 pp | 71.30 % | +20.78 pp | 71.40 % | +20.88 pp | 64.04 % | +13.52 pp |
| IPC Active | 1.98 inst/cycle | 0.88 inst/cycle | -1.10 | 1.15 inst/cycle | -0.83 | 1.15 inst/cycle | -0.83 | 1.44 inst/cycle | -0.54 |


### Key Learnings

#### ❌ Plan D: `evict_first_on_pass2_streaming_loads_and_stores` (0.742×)

Adding eviction_policy='evict_first' to Pass 2 loads and stores in a two-pass normalization kernel caused a 26% slowdown (5.98ms → 8.41ms). The L2 hit rate was essentially unchanged (34.09% → 33.83%), confirming that evict_first did not improve L2 utilization as predicted. The dominant regression driver was that the diff also changed the autotune config space, causing the autotuner to select a smaller block size (256 threads, 3.39 waves/SM vs. 6.79 waves/SM before), which nearly doubled total executed instructions (1.25B → 2.38B) and cut DRAM throughput from 76.6% to 54.46%. The evict_first hint itself appears ineffective for this kernel — the L2 hit rate did not improve — while the simultaneous autotune config reduction was the primary regression cause.

> **Applicable when**: Never — this strategy caused regression in this kernel and the L2 hit rate did not improve, indicating evict_first hints are ineffective when L2 hit rate is already low (~34%) due to working set exceeding L2 capacity.

> **Avoid when**: When the kernel's working set (N elements per row × concurrent blocks) exceeds L2 capacity, evict_first on streaming passes will not improve L2 hit rate because there is no reuse to protect — the L2 is already being thrashed by capacity misses, not eviction-policy misses. Also avoid when the autotune config change simultaneously reduces waves/SM, as the occupancy loss dominates any cache-hint benefit. Do not apply when DRAM throughput is already near peak (>70%) and L2 hit rate is low (~34%) — the bottleneck is bandwidth, not eviction policy.

---

## Cumulative Progress

```
Baseline:  6240000 ns  (1.000×)

Final best: 8950000 ns  (0.6972×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,14 +3,26 @@

 import triton.language as tl
 from triton.language.extra import libdevice
 
+
 @triton.autotune(
     configs=[
+        # num_stages=2 (original configs)
         triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
         triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
         triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=2),
         triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
         triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=2),
         triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
+        # num_stages=3 (new configs)
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=3),
+        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=3),
+        # num_stages=4 (new configs)
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=4),
     ],
     key=['N'],
 )
@@ -24,40 +36,35 @@

     # Each program handles one (batch, channel) pair
     row_idx = tl.program_id(0)
     row_start = row_idx * N
-    
-    # Pass 1: Welford online mean+variance
-    mean = tl.zeros((), dtype=tl.float32)
-    m2 = tl.zeros((), dtype=tl.float32)
-    count = tl.zeros((), dtype=tl.float32)
-    
+
+    # Pass 1: simple sum + sum-of-squares accumulation
+    x_sum = tl.zeros((), dtype=tl.float32)
+    x_sum2 = tl.zeros((), dtype=tl.float32)
+
     for off in tl.range(0, N, BLOCK_SIZE):
         cols = off + tl.arange(0, BLOCK_SIZE)
         mask = cols < N
         x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
-        
-        # Welford update
-        new_count = tl.sum(mask.to(tl.float32), axis=0)
-        new_mean = tl.sum(tl.where(mask, x, tl.zeros_like(x)), axis=0) / tl.maximum(new_count, 1.0)
-        new_m2 = tl.sum(tl.where(mask, (x - new_mean) * (x - new_mean), tl.zeros_like(x)), axis=0)
-        
-        # Combine with running stats
-        total_count = count + new_count
-        delta = new_mean - mean
-        mean = tl.where(total_count > 0, mean + delta * new_count / total_count, mean)
-        m2 = m2 + new_m2 + delta * delta * count * new_count / tl.maximum(total_count, 1.0)
-        count = total_count
-    
-    var = m2 / tl.maximum(count, 1.0)
+        x_masked = tl.where(mask, x, 0.0)
+        x_sum += tl.sum(x_masked, axis=0)
+        x_sum2 += tl.sum(x_masked * x_masked, axis=0)
+
+    # Compute mean and variance from sums
+    N_f = N.to(tl.float32)
+    mean = x_sum / N_f
+    var = x_sum2 / N_f - mean * mean
     inv_std = libdevice.rsqrt(var + eps)
-    
+
+    # Pre-compute scalar bias for FMA in Pass 2: bias = -mean * inv_std
+    bias = libdevice.fma(-mean, inv_std, 0.0)
+
     # Pass 2: Normalize
     for off in tl.range(0, N, BLOCK_SIZE):
         cols = off + tl.arange(0, BLOCK_SIZE)
         mask = cols < N
         x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
-        out = (x - mean) * inv_std
+        out = libdevice.fma(x, inv_std, bias)
         tl.store(out_ptr + row_start + cols, out.to(tl.float32), mask=mask)
-
 
 def kernel_function(*inputs):
     x, = inputs
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
