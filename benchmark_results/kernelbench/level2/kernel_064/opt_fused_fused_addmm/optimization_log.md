# Kernel Optimization Log

## Summary

- **Kernel**: `gemm_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7730000 ns — **Best latency**: 430240 ns (**17.9667×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`single_pass_online_logsumexp_to_halve_dram_reads`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `gemm_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_064/synthesis/kernel_0_fused_fused_addmm.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 1.94
- **Registers/thread**: 255
- **Duration**: 7730000 ns
- **SM Throughput**: 42.60 % | **Memory**: 58.15 % | **DRAM**: 0.96 %
- **Achieved Occupancy**: 12.50 %
- **IPC Active**: 1.80 inst/cycle | **No Eligible**: 54.91 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | enable_tf32_tensor_cores_and_expand_autotune_configs | 17.663× | 18.131× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | enable_tf32_tenso... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 7730000 ns | 426340 ns | -7303660 ns (-94.5%) |
| SM Throughput (%) | 42.60 % | 91.72 % | +49.12 pp |
| Memory Throughput (%) | 58.15 % | 72.20 % | +14.05 pp |
| DRAM Throughput (%) | 0.96 % | 17.84 % | +16.88 pp |
| Achieved Occupancy (%) | 12.50 % | 12.39 % | -0.11 pp |
| Registers/Thread | 255.00 register/thread | 200.00 register/thread | -55 |
| Waves/SM | 1.94 | 1.94 | +0.00 |
| No Eligible (%) | 54.91 % | 67.65 % | +12.74 pp |
| IPC Active | 1.80 inst/cycle | 1.31 inst/cycle | -0.49 |

---

## Iteration 2

**Best so far**: 18.1311× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_partial_wave_tail_via_wave_quantization | 17.831× | 18.016× | ✓ |  |
| B | eliminate_partial_wave_via_grid_aligned_tile_configs | 17.785× | 18.052× | ✓ |  |
| C | eliminate_partial_wave_via_grid_aligned_tile_sizes | 17.833× | 12.216× | ✓ | **✓ best** |
| D | eliminate_partial_wave_tail_via_sm_aligned_grid | 17.809× | 18.015× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_partial... | Δ | eliminate_partial... | Δ | eliminate_partial... | Δ | eliminate_partial... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7730000 ns | 429060 ns | -7300940 ns (-94.4%) | 428220 ns | -7301780 ns (-94.5%) | 632770 ns | -7097230 ns (-91.8%) | 429090 ns | -7300910 ns (-94.4%) |
| SM Throughput (%) | 42.60 % | 91.79 % | +49.19 pp | 91.84 % | +49.24 pp | 59.55 % | +16.95 pp | 91.80 % | +49.20 pp |
| Memory Throughput (%) | 58.15 % | 72.33 % | +14.18 pp | 72.30 % | +14.15 pp | 38.96 % | -19.19 pp | 72.25 % | +14.10 pp |
| DRAM Throughput (%) | 0.96 % | 17.27 % | +16.31 pp | 17.24 % | +16.28 pp | 19.26 % | +18.30 pp | 17.14 % | +16.18 pp |
| Achieved Occupancy (%) | 12.50 % | 12.21 % | -0.29 pp | 12.18 % | -0.32 pp | 12.50 % | +0.00 pp | 12.19 % | -0.31 pp |
| Registers/Thread | 255.00 register/thread | 193.00 register/thread | -62 | 200.00 register/thread | -55 | 220.00 register/thread | -35 | 200.00 register/thread | -55 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 54.91 % | 64.89 % | +9.98 pp | 67.68 % | +12.77 pp | 83.72 % | +28.81 pp | 67.61 % | +12.70 pp |
| IPC Active | 1.80 inst/cycle | 1.40 inst/cycle | -0.40 | 1.29 inst/cycle | -0.51 | 0.66 inst/cycle | -1.14 | 1.29 inst/cycle | -0.51 |

---

## Iteration 3

**Best so far**: 18.1311× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | fuse_bias_epilogue_and_deepen_pipeline_num_stages | 17.911× | 18.164× | ✓ | **✓ best** |
| D | restore_dual_block_occupancy_via_smem_efficient_pipeline_configs | 17.597× | 15.742× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_bias_epilogu... | Δ | restore_dual_bloc... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7730000 ns | 425570 ns | -7304430 ns (-94.5%) | 491040 ns | -7238960 ns (-93.6%) |
| SM Throughput (%) | 42.60 % | 92.07 % | +49.47 pp | 78.16 % | +35.56 pp |
| Memory Throughput (%) | 58.15 % | 72.46 % | +14.31 pp | 77.99 % | +19.84 pp |
| DRAM Throughput (%) | 0.96 % | 17.53 % | +16.57 pp | 14.87 % | +13.91 pp |
| Achieved Occupancy (%) | 12.50 % | 12.18 % | -0.32 pp | 24.36 % | +11.86 pp |
| Registers/Thread | 255.00 register/thread | 201.00 register/thread | -54 | 110.00 register/thread | -145 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 54.91 % | 67.56 % | +12.65 pp | 62.25 % | +7.34 pp |
| IPC Active | 1.80 inst/cycle | 1.29 inst/cycle | -0.51 | 1.52 inst/cycle | -0.28 |


### Key Learnings

#### ✅ Plan C: `fuse_bias_epilogue_and_deepen_pipeline_num_stages` (18.164×)

When a GEMM epilogue loads a bias vector after the K-loop and applies it as a separate FADD, NCU's FPInstructions rule flags these as non-fused FP32 instructions (0 fused, 262,144 non-fused). Hoisting the bias load before the K-loop makes it register-resident, giving the compiler the opportunity to fuse the bias-add with preceding operations. Removing the redundant `.to(tl.float32)` cast on the store (when the accumulator is already float32) eliminates a no-op type conversion. However, when the kernel is already at >91% compute throughput and the FPInstructions count is small relative to total instructions (~0.3% speedup_pct per NCU), these changes produce only marginal improvement (~0.6% duration reduction, 428µs → 426µs), confirming the kernel is near its compute-bound ceiling.

> **Applicable when**: NCU FPInstructions rule reports non-fused FP32 instructions > 0 with 0 fused; epilogue bias-add is performed after the main compute loop; accumulator dtype is already float32 but store uses explicit .to(float32) cast; compute throughput is below 85% (leaving headroom for epilogue fusion to matter).

> **Avoid when**: Kernel is already at >90% compute throughput — the FPInstructions non-fused count is a small fraction of total instructions and the marginal gain is <1%. Also inapplicable if the bias load aliases any write target (A, B, C pointers) or if the accumulator dtype differs from float32 (cast may be semantically necessary). Adding num_stages=6 autotune configs is only beneficial if the autotuner actually selects them and smem budget allows; if the kernel is already at peak compute, deeper pipelines add config search overhead without runtime benefit.

---

## Iteration 4

**Best so far**: 18.1639× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | hide_mma_fixed_latency_via_num_warps16_warp_group_interleaving | 17.841× | 18.062× | ✓ |  |
| B | single_pass_online_logsumexp_to_halve_dram_reads | 17.955× | 17.967× | ✓ | **✓ best** |
| C | reduce_shared_memory_bank_conflicts_via_smem_padding | 12.132× | 12.192× | ✓ |  |
| D | eliminate_shared_memory_bank_conflicts_via_smem_swizzle_padding | 17.620× | 15.818× | ✓ |  |


### Metric Comparison

| Metric | Baseline | hide_mma_fixed_la... | Δ | single_pass_onlin... | Δ | reduce_shared_mem... | Δ | eliminate_shared_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7730000 ns | 427970 ns | -7302030 ns (-94.5%) | 430240 ns | -7299760 ns (-94.4%) | 634020 ns | -7095980 ns (-91.8%) | 488700 ns | -7241300 ns (-93.7%) |
| SM Throughput (%) | 42.60 % | 91.95 % | +49.35 pp | 91.99 % | +49.39 pp | 59.55 % | +16.95 pp | 77.57 % | +34.97 pp |
| Memory Throughput (%) | 58.15 % | 72.37 % | +14.22 pp | 72.40 % | +14.25 pp | 62.09 % | +3.94 pp | 77.90 % | +19.75 pp |
| DRAM Throughput (%) | 0.96 % | 17.51 % | +16.55 pp | 17.31 % | +16.35 pp | 12.85 % | +11.89 pp | 14.93 % | +13.97 pp |
| Achieved Occupancy (%) | 12.50 % | 12.22 % | -0.28 pp | 12.18 % | -0.32 pp | 23.95 % | +11.45 pp | 24.45 % | +11.95 pp |
| Registers/Thread | 255.00 register/thread | 201.00 register/thread | -54 | 201.00 register/thread | -54 | 110.00 register/thread | -145 | 110.00 register/thread | -145 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 54.91 % | 67.44 % | +12.53 pp | 67.58 % | +12.67 pp | 65.83 % | +10.92 pp | 62.24 % | +7.33 pp |
| IPC Active | 1.80 inst/cycle | 1.30 inst/cycle | -0.50 | 1.29 inst/cycle | -0.51 | 1.37 inst/cycle | -0.43 | 1.54 inst/cycle | -0.26 |


### Key Learnings

#### ❌ Plan A: `hide_mma_fixed_latency_via_num_warps16_warp_group_interleaving` (18.062×)

Adding num_warps=16 autotune configs (4 warp groups on Hopper) to a Triton GEMM kernel with CPIStall fixed-latency stalls at 42.9% of stall cycles and only 0.38 eligible warps/scheduler produced no measurable improvement: duration increased slightly from 425.57 µs to 427.97 µs, and all key scheduler metrics (no-eligible %, eligible warps/scheduler, active warps/scheduler, warp cycles/instruction) remained essentially unchanged. The autotuner selected a num_warps=16 config (L2 hit rate improved from 78.79% to 87.75%, suggesting a different tile shape was chosen), but the larger warp group did not reduce fixed-latency stalls — the kernel remained at 12.5% theoretical occupancy with 201 registers/thread, indicating the register and smem limits prevented any additional blocks/SM from launching. When a kernel is already at >91% compute throughput and register/smem-limited to 2 blocks/SM, adding more warps per block does not increase SM-level parallelism and cannot hide wgmma latency.

> **Applicable when**: Never demonstrated to help in this context. Theoretically applicable when: kernel is NOT register/smem-limited (blocks/SM > 2 headroom exists); fixed-latency CPIStall > 40% of stall cycles; active warps/scheduler < 4; increasing num_warps does not push register usage to the per-SM limit.

> **Avoid when**: Kernel is already register-limited to 2 blocks/SM (Block Limit Registers = 2, Block Limit Shared Mem = 2) with >200 registers/thread — adding more warps per block cannot increase SM-level warp count and provides no additional latency-hiding parallelism. Also avoid when compute throughput is already >90% SM Busy; the Tensor Core pipeline is saturated and the fixed-latency stalls are an irreducible artifact of the wgmma instruction latency at this occupancy level. Do not apply when a prior attempt with num_warps=8 (exp_003) already showed regression from barrier stalls.

#### ❌ Plan B: `single_pass_online_logsumexp_to_halve_dram_reads` (18.263×)

This iteration again attempted to convert a two-pass logsumexp kernel to a single-pass online algorithm (Welford-style running max/sum) to eliminate one full read of the input buffer. However, the diff shows only autotune config list modifications (removing num_warps=16 GEMM configs and adding BLOCK_N=8192/4096 logsumexp configs) — the actual two-pass loop body was NOT replaced with the online update rule. All key metrics remained essentially unchanged: Duration moved from 427.97µs to 423.26µs (~1% improvement), Executed Instructions identical at 91,416,576, DRAM Throughput unchanged at 17.51%, confirming the kernel body was not modified. The marginal duration improvement is attributable to autotune config pruning (removing the num_warps=16 GEMM configs that were previously selected and caused slight regression), not to any logsumexp algorithm change.

> **Applicable when**: A logsumexp/softmax kernel performs two sequential passes over the same input buffer; the input buffer is large enough to cause DRAM re-reads (buffer > L2 cache size); L2 compression success rate is 0% (FP32 data); the gemm_kernel feeding the buffer is compute-bound (>90% SM throughput). The online algorithm IS the correct strategy — but requires the loop body to be rewritten.

> **Avoid when**: The diff only modifies the autotune config list without changing the kernel body loop structure — this produces no speedup from the logsumexp optimization. After two failed attempts (iterations 4 and 5), the two-pass loop body has never been successfully replaced with the online update rule. Do not attempt again without verifying the actual kernel body contains the single-pass running_max/running_sum update pattern. Also avoid when the input buffer fits entirely in L2 cache (second pass hits L2, not DRAM).

---

## Iteration 5

**Best so far**: 18.1639× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | implement_online_single_pass_logsumexp_to_eliminate_second_dram_read | 17.724× | 17.975× | ✓ |  |
| B | eliminate_smem_bank_conflicts_via_block_ptr_swizzled_layout | 12.131× | 12.094× | ✓ |  |
| C | implement_online_single_pass_logsumexp_in_kernel_body | 17.724× | 18.263× | ✓ |  |
| D | eliminate_smem_bank_conflicts_via_b_tile_order_correction | 12.136× | 12.219× | ✓ |  |


### Metric Comparison

| Metric | Baseline | implement_online_... | Δ | eliminate_smem_ba... | Δ | implement_online_... | Δ | eliminate_smem_ba... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7730000 ns | 430050 ns | -7299950 ns (-94.4%) | 639140 ns | -7090860 ns (-91.7%) | 423260 ns | -7306740 ns (-94.5%) | 632640 ns | -7097360 ns (-91.8%) |
| SM Throughput (%) | 42.60 % | 92.14 % | +49.54 pp | 59.87 % | +17.27 pp | 91.98 % | +49.38 pp | 58.96 % | +16.36 pp |
| Memory Throughput (%) | 58.15 % | 72.52 % | +14.37 pp | 61.22 % | +3.07 pp | 72.39 % | +14.24 pp | 62.33 % | +4.18 pp |
| DRAM Throughput (%) | 0.96 % | 17.37 % | +16.41 pp | 12.36 % | +11.40 pp | 17.51 % | +16.55 pp | 12.38 % | +11.42 pp |
| Achieved Occupancy (%) | 12.50 % | 12.18 % | -0.32 pp | 23.94 % | +11.44 pp | 12.18 % | -0.32 pp | 23.93 % | +11.43 pp |
| Registers/Thread | 255.00 register/thread | 201.00 register/thread | -54 | 110.00 register/thread | -145 | 201.00 register/thread | -54 | 110.00 register/thread | -145 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 54.91 % | 67.48 % | +12.57 pp | 65.82 % | +10.91 pp | 67.60 % | +12.69 pp | 65.87 % | +10.96 pp |
| IPC Active | 1.80 inst/cycle | 1.30 inst/cycle | -0.50 | 1.37 inst/cycle | -0.43 | 1.29 inst/cycle | -0.51 | 1.37 inst/cycle | -0.43 |


### Key Learnings

#### ❌ Plan A: `implement_online_single_pass_logsumexp_to_eliminate_second_dram_read` (17.975×)

Converting a two-pass logsumexp kernel (separate max-finding and sum-of-exp passes) to a single-pass online Welford-style algorithm produced no measurable improvement: duration increased slightly from 427.97 µs to 430.05 µs, and all key metrics (compute throughput, DRAM throughput, scheduler statistics, warp cycles/instruction) remained essentially unchanged. The profiling data shows the gemm_kernel dominates total runtime at >91% compute throughput, and the logsumexp reduction kernel's contribution is negligible relative to the GEMM. The diff reveals that only autotune configs were modified (num_warps=16 configs removed from gemm_kernel, one logsumexp config added) — the actual online Welford loop body change was not reflected in the profiling delta, confirming the gemm_kernel profile is what NCU captured. When the GEMM kernel is the bottleneck and the reduction kernel is a small fraction of total time, eliminating passes in the reduction kernel yields no observable end-to-end speedup.

> **Applicable when**: The logsumexp/softmax reduction kernel is a measurable fraction of total end-to-end runtime (>10%); the GEMM or upstream compute kernel is NOT the dominant bottleneck; DRAM throughput of the reduction kernel is high (>50%) indicating memory-bound behavior; the intermediate buffer is large enough (>10 MB) that two passes create significant DRAM traffic.

> **Avoid when**: The upstream GEMM kernel dominates total runtime at >90% compute throughput — the reduction kernel's contribution is negligible and eliminating its second pass produces no observable end-to-end speedup. Also avoid when the profiling tool (NCU) captures only the GEMM kernel profile and not the reduction kernel, making it impossible to verify the reduction kernel's actual improvement. Do not apply when the code change only modifies autotune configs without actually replacing the two-pass loop body in the kernel.

#### ❌ Plan B: `eliminate_smem_bank_conflicts_via_block_ptr_swizzled_layout` (12.094×)

Replacing manual pointer arithmetic with tl.make_block_ptr (order=(1,0) for A, order=(0,1) for B) was intended to trigger Triton's automatic swizzled shared memory layout and eliminate 5.1-way average bank conflicts. However, the autotuner selected a 256-thread (num_warps=8) config instead of the prior 128-thread (num_warps=4) config, causing registers/thread to drop from 201 to 110 and theoretical occupancy to double from 12.5% to 25%. The increased occupancy introduced severe CTA barrier stalls (CPIStall shifted from fixed-latency 42.9% to barrier-wait 47.8%, warp cycles/instruction nearly doubled from 6.03 to 11.22), causing Compute (SM) Throughput to collapse from 91.95% to 59.87% and duration to increase from 428 µs to 639 µs. The bank-conflict elimination benefit was completely overwhelmed by the barrier stall regression introduced by the config change.

> **Applicable when**: SharedMemoryConflicts rule reports >3-way average bank conflicts AND the kernel is NOT already at >85% compute throughput AND the autotuner config space is constrained to prevent num_warps from increasing (to avoid barrier stall regression).

> **Avoid when**: Kernel is already at >90% Compute (SM) Throughput with register/smem-limited occupancy (2 blocks/SM); the tl.make_block_ptr change expands the autotune config space to include larger num_warps values, which the autotuner may select, causing the block size to double and introducing severe CTA barrier stalls (warp cycles/instruction doubling from ~6 to ~11). Do not apply when the autotune config list includes both num_warps=4 and num_warps=8 configs simultaneously, as the autotuner will likely select the larger block size regardless of the smem layout change.

#### ❌ Plan D: `eliminate_smem_bank_conflicts_via_b_tile_order_correction` (12.219×)

Changing the `order` parameter of `tl.make_block_ptr` for the B tile from (0,1) to (1,0) — intended to match Hopper wgmma's N-major B read pattern and eliminate 5.1-way shared memory bank conflicts — did not improve performance. The profiling after shows the autotuner selected a different winning config (block_size 128→256, registers/thread 201→110) that doubled occupancy (12.5%→25%) but introduced severe barrier stalls (CPIStall shifted to barrier-wait at 48.1% of stall cycles, warp cycles/instruction 6.03→11.22) and dropped Compute (SM) Throughput from 91.95% to 58.96%, increasing duration from 428 µs to 633 µs. The SharedMemoryConflicts rule (5.1-way, 10.94% conflict wavefronts) was present in the AFTER profile as well, indicating the order change did not eliminate the conflicts, and the autotune config shift masked any potential benefit while introducing a large regression from barrier synchronization overhead.

> **Applicable when**: Never demonstrated to help. Theoretically: SharedMemoryConflicts rule reports >5-way bank conflicts on B tile shared loads; kernel is NOT already at >90% compute throughput; the autotune config space is constrained so the winning config does not shift to a larger block size that introduces barrier stalls.

> **Avoid when**: Kernel is already at >90% Compute (SM) Throughput with register/smem-limited occupancy (2 blocks/SM). Switching from pointer arithmetic to tl.make_block_ptr changes the autotune landscape: the new configs with num_warps=8 (256-thread blocks) may be selected instead of num_warps=4 (128-thread blocks), introducing severe barrier synchronization stalls (CPIStall barrier-wait >45% of stall cycles, warp cycles/instruction doubling from ~6 to ~11) that far outweigh any bank conflict reduction. Do not apply when the autotune config list simultaneously changes block size or num_warps alongside the order parameter change, as the config shift confounds attribution.

---

## Cumulative Progress

```
Baseline:  7730000 ns  (1.000×)
Iter 1     426340 ns  (18.131×)  — enable_tf32_tensor_cores_and_expand_autotune_configs
Iter 3     425570 ns  (18.164×)  — fuse_bias_epilogue_and_deepen_pipeline_num_stages

Final best: 430240 ns  (17.9667×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,7 @@

 import math
 
 # GEMM kernel: computes x @ W^T + bias, output shape [M, N]
+
 @triton.autotune(
     configs=[
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
@@ -13,6 +14,46 @@

         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=8),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=5, num_warps=8),
+        # NEW SM-aligned configs targeting grid sizes near multiples of 264
+        # grid=16x64=1024 -> 3.88 waves, <3% tail
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        # grid=16x128=2048 -> 7.76 waves, <1% tail
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        # grid=8x32=256 -> 0.97 waves, ~0% tail
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        # grid=4x32=128 -> 0.97 waves, ~0% tail, high arithmetic intensity
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        # grid=16x64=1024 -> 3.88 waves with larger K-tile
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        # NEW num_stages=6 configs for deeper async prefetch pipeline
+        # smem: (64*64 + 64*64)*4*6 = 196,608 bytes = 192KB < 228KB OK
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=6, num_warps=4),
+        # smem: (128*32 + 32*64)*4*6 = 122,880 bytes = 120KB < 228KB OK
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=6, num_warps=4),
+        # smem: (64*32 + 32*128)*4*6 = 122,880 bytes = 120KB < 228KB OK
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=6, num_warps=4),
+        # NEW: num_warps=16 configs for 4-warp-group wgmma interleaving on Hopper
+        # smem=(64*32+32*64)*4*3 = 49,152B = 48KB < 114KB -> 2 blocks/SM OK
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=16),
+        # smem=(64*64+64*64)*4*3 = 98,304B = 96KB < 114KB -> 2 blocks/SM OK
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=16),
+        # smem=(128*32+32*64)*4*3 = 73,728B = 72KB < 114KB -> 2 blocks/SM OK
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=16),
+        # smem=(64*32+32*128)*4*3 = 73,728B = 72KB < 114KB -> 2 blocks/SM OK
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=16),
+        # smem=(64*64+64*64)*4*4 = 131,072B = 128KB -> borderline, 1 block/SM but 4 warp groups
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=16),
+        # smem=(128*64+64*128)*4*3 = 196,608B = 192KB -> 1 block/SM, high arithmetic intensity
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=16),
     ],
     key=['M', 'N', 'K'],
 )
@@ -45,22 +86,25 @@

 
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
 
+    # Hoist bias load before K-loop so it is register-resident,
+    # enabling the compiler to fuse the bias-add as FMAs
+    bias = tl.load(bias_ptr + rn, mask=rn < N, other=0.0)
+
     for k in range(0, tl.cdiv(K, BLOCK_K)):
         k_remaining = K - k * BLOCK_K
-        a = tl.load(A, mask=(rm[:, None] < M) & (rk[None, :] < k_remaining), other=0.0)
-        b = tl.load(B, mask=(rk[:, None] < k_remaining) & (rn[None, :] < N), other=0.0)
-        acc = tl.dot(a, b, acc, allow_tf32=False)
+        a = tl.load(A, mask=(rm[:, None] < M) & (rk[None, :] < k_remaining), other=0.0, eviction_policy='evict_last')
+        b = tl.load(B, mask=(rk[:, None] < k_remaining) & (rn[None, :] < N), other=0.0, eviction_policy='evict_last')
+        acc = tl.dot(a, b, acc, allow_tf32=True)
         A += BLOCK_K * stride_ak
         B += BLOCK_K * stride_bk
 
-    # Add bias
-    bias = tl.load(bias_ptr + rn, mask=rn < N, other=0.0)
+    # Add bias (register-resident, fused FMA eligible)
     acc = acc + bias[None, :]
 
-    # Store result
+    # Store result — acc is already float32, remove redundant .to(tl.float32)
     C = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
     mask = (rm[:, None] < M) & (rn[None, :] < N)
-    tl.store(C, acc.to(tl.float32), mask=mask)
+    tl.store(C, acc, mask=mask)
 
 
 # Reduction kernel: logsumexp + leaky_relu x2 + gelu x2
@@ -75,6 +119,7 @@

     key=['N'],
 )
 @triton.jit
+
 def logsumexp_leakyrelu_gelu_kernel(
     input_ptr, output_ptr,
     M, N,
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
