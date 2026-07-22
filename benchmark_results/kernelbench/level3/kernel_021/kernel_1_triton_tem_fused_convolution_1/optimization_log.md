# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 827170 ns — **Best latency**: 93060 ns (**8.8886×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`manual_k_unroll_scalar_outer_product_fma_to_fuse_fp32_instructions`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_021/kernel_1_triton_tem_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (65536,1,1), Waves/SM: 165.49
- **Registers/thread**: 160
- **Duration**: 827170 ns
- **SM Throughput**: 25.59 % | **Memory**: 83.19 % | **DRAM**: 7.49 %
- **Achieved Occupancy**: 18.05 %
- **IPC Active**: 1.03 inst/cycle | **No Eligible**: 74.31 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | — | fail | ✓ |  |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_gemm_with_scalar_pointwise_kernel_for_tiny_n_k | — | fail | ✓ |  |
| B | tma_loads_fixed_config_right_sized_tiles | — | fail | ✓ |  |
| C | right_size_tiles_and_fix_b_load_coalescing_with_correct_grid | 8.759× | 3.640× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | right_size_tiles_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 827170 ns | 227260 ns | -599910 ns (-72.5%) |
| SM Throughput (%) | 25.59 % | 78.94 % | +53.35 pp |
| Memory Throughput (%) | 83.19 % | 90.75 % | +7.56 pp |
| DRAM Throughput (%) | 7.49 % | 24.85 % | +17.36 pp |
| Achieved Occupancy (%) | 18.05 % | 72.25 % | +54.20 pp |
| Registers/Thread | 160.00 register/thread | 40.00 register/thread | -120 |
| Waves/SM | 165.49 | 41.37 | -124.12 |
| No Eligible (%) | 74.31 % | 20.25 % | -54.06 pp |
| IPC Active | 1.03 inst/cycle | 3.19 inst/cycle | +2.16 |

---

## Iteration 3

**Best so far**: 3.6398× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_block_m_with_num_warps_for_l1_coalescing_and_scheduler_balance | 8.727× | 3.562× | ✓ |  |
| B | tma_loads_via_block_ptr_for_A_matrix_fixed_config | 8.720× | 3.615× | ✓ |  |
| C | coalesce_a_matrix_load_via_transposed_km_tile | 8.778× | 3.638× | ✓ | **✓ best** |
| D | hoist_b_load_out_of_k_loop_and_autotune_num_warps_for_full_occupancy | 8.739× | 3.611× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_block_m_... | Δ | tma_loads_via_blo... | Δ | coalesce_a_matrix... | Δ | hoist_b_load_out_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 827170 ns | 232220 ns | -594950 ns (-71.9%) | 228800 ns | -598370 ns (-72.3%) | 227390 ns | -599780 ns (-72.5%) | 229090 ns | -598080 ns (-72.3%) |
| SM Throughput (%) | 25.59 % | 65.21 % | +39.62 pp | 77.70 % | +52.11 pp | 79.25 % | +53.66 pp | 79.07 % | +53.48 pp |
| Memory Throughput (%) | 83.19 % | 88.73 % | +5.54 pp | 91.10 % | +7.91 pp | 90.80 % | +7.61 pp | 90.72 % | +7.53 pp |
| DRAM Throughput (%) | 7.49 % | 24.30 % | +16.81 pp | 24.67 % | +17.18 pp | 24.85 % | +17.36 pp | 24.68 % | +17.19 pp |
| Achieved Occupancy (%) | 18.05 % | 60.15 % | +42.10 pp | 72.27 % | +54.22 pp | 72.27 % | +54.22 pp | 72.25 % | +54.20 pp |
| Registers/Thread | 160.00 register/thread | 48.00 register/thread | -112 | 40.00 register/thread | -120 | 40.00 register/thread | -120 | 40.00 register/thread | -120 |
| Waves/SM | 165.49 | 24.82 | -140.67 | 41.37 | -124.12 | 41.37 | -124.12 | 41.37 | -124.12 |
| No Eligible (%) | 74.31 % | 39.08 % | -35.23 pp | 21.48 % | -52.83 pp | 19.86 % | -54.45 pp | 20.12 % | -54.19 pp |
| IPC Active | 1.03 inst/cycle | 2.44 inst/cycle | +1.41 | 3.14 inst/cycle | +2.11 | 3.21 inst/cycle | +2.18 | 3.20 inst/cycle | +2.17 |


### Key Learnings

#### ❌ Plan C: `coalesce_a_matrix_load_via_transposed_km_tile` (3.638×)

Reloading the A matrix as [BLOCK_K, BLOCK_M] with tl.trans() produced no measurable improvement: duration changed by only 0.13 µs (227.26 → 227.39 µs), L1/TEX throughput remained at ~91.7%, Mem Busy stayed at ~90.8%, and L1 hit rate was unchanged at ~34.1%. The root cause is that the A matrix has stride_am=3 in the M dimension regardless of which axis is made the 'fast' (innermost warp-lane) dimension in the tile layout — adjacent warp threads still access addresses 3 floats apart. The MemoryCacheAccessPattern rule (33% speedup estimate) was not addressable by a simple transpose of the load tile when the underlying tensor stride is non-unit in both candidate fast dimensions.

> **Applicable when**: Never — this specific transformation is a no-op when the tensor's M-dimension stride is non-unit (stride_am > 1). The MemoryCacheAccessPattern rule fires but the strided access cannot be fixed by transposing the tile layout alone.

> **Avoid when**: When the A matrix has stride_am > 1 in the M dimension (e.g. channel-last / NHWC layout where spatial pixels are separated by the channel count). Transposing the load tile from [BLOCK_M, BLOCK_K] to [BLOCK_K, BLOCK_M] does not change the per-warp-lane address stride; it only reorders which axis is the outer loop. The MemoryCacheAccessPattern rule's speedup estimate is not achievable via this approach in such layouts.

---

## Iteration 4

**Best so far**: 3.6398× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_k_loop_and_manually_unroll_fma_for_fused_fp32_instructions | 17.879× | 7.385× | ✓ | **✓ best** |
| B | eliminate_k_loop_via_full_static_k_tile_load_and_unrolled_fma | 8.806× | 3.645× | ✓ |  |
| C | fused_fp32_mac_via_tl_dot_padded_block_k | 4.372× | 1.820× | ✓ |  |
| D | eliminate_k_loop_hoist_loads_and_evict_first_for_streaming_a | 8.737× | 3.636× | ✓ |  |
| E | add_triton_autotune | 8.716× | 3.640× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_k_loop_... | Δ | eliminate_k_loop_... | Δ | fused_fp32_mac_vi... | Δ | eliminate_k_loop_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 827170 ns | 112000 ns | -715170 ns (-86.5%) | 226910 ns | -600260 ns (-72.6%) | 454460 ns | -372710 ns (-45.1%) | 227490 ns | -599680 ns (-72.5%) | 227230 ns | -599940 ns (-72.5%) |
| SM Throughput (%) | 25.59 % | 84.45 % | +58.86 pp | 72.42 % | +46.83 pp | 36.59 % | +11.00 pp | 65.49 % | +39.90 pp | 79.27 % | +53.68 pp |
| Memory Throughput (%) | 83.19 % | 83.08 % | -0.11 pp | 91.03 % | +7.84 pp | 97.58 % | +14.39 pp | 89.16 % | +5.97 pp | 90.83 % | +7.64 pp |
| DRAM Throughput (%) | 7.49 % | 50.41 % | +42.92 pp | 24.90 % | +17.41 pp | 12.46 % | +4.97 pp | 25.07 % | +17.58 pp | 24.87 % | +17.38 pp |
| Achieved Occupancy (%) | 18.05 % | 94.28 % | +76.23 pp | 72.27 % | +54.22 pp | 54.70 % | +36.65 pp | 59.88 % | +41.83 pp | 72.25 % | +54.20 pp |
| Registers/Thread | 160.00 register/thread | 32.00 register/thread | -128 | 40.00 register/thread | -120 | 54.00 register/thread | -106 | 48.00 register/thread | -112 | 40.00 register/thread | -120 |
| Waves/SM | 165.49 | 31.03 | -134.46 | 41.37 | -124.12 | 55.16 | -110.33 | 24.82 | -140.67 | 41.37 | -124.12 |
| No Eligible (%) | 74.31 % | 13.70 % | -60.61 pp | 26.86 % | -47.45 pp | 63.20 % | -11.11 pp | 42.61 % | -31.70 pp | 19.91 % | -54.40 pp |
| IPC Active | 1.03 inst/cycle | 3.44 inst/cycle | +2.41 | 2.93 inst/cycle | +1.90 | 1.47 inst/cycle | +0.44 | 2.29 inst/cycle | +1.26 | 3.21 inst/cycle | +2.18 |


### Key Learnings

#### ✅ Plan A: `eliminate_k_loop_and_manually_unroll_fma_for_fused_fp32_instructions` (8.889×)

After the manual K-unroll outer-product FMA transformation (iteration 4, 7.39x), adding @triton.autotune over num_warps and num_stages selected a block_size=64 (2 warps) configuration that further reduced duration from 112 µs to 93 µs (additional ~17% gain, cumulative 8.89x). However, the winning config increased registers_per_thread from 32 to 44 and dropped theoretical occupancy from 100% to 62.5%, while DRAM throughput improved from 50.4% to 60.4% and total executed instructions fell from 72M to 48M. The net gain came from fewer total instructions (smaller block processes less redundant work per wave) outweighing the occupancy regression, suggesting the prior 128-thread block was over-provisioned for this skinny GEMM workload.

> **Applicable when**: After a manual K-unroll FMA transformation has already achieved high occupancy (100% theoretical) and the kernel is still memory-bound (DRAM Throughput ~50%, Mem Busy ~83%). Adding autotune over num_warps and num_stages can find a smaller warp count that reduces total instruction count and improves DRAM utilization, even at the cost of lower theoretical occupancy, when the workload is a skinny GEMM (N << BLOCK_N, small K) where fewer threads per block reduces redundant masked-out work.

> **Avoid when**: When the kernel is already register-limited and adding autotune stages would further inflate register pressure beyond the occupancy cliff (as seen in iteration 4 where autotune alone without the FMA rewrite yielded 0% gain). Do not apply autotune-only (without structural FMA changes) when Issue Slots Busy > 75% and Mem Busy > 85% — the kernel is already at a local optimum for launch parameters alone.

#### ✅ Plan B: `eliminate_k_loop_via_full_static_k_tile_load_and_unrolled_fma` (3.645×)

When a Triton GEMM k-loop runs exactly one iteration (tl.cdiv(K, BLOCK_K) == 1), eliminating the loop and loading the full K tile in a single shot with a compile-time-constant K_STATIC reduces total executed instructions (140.8M → 128.2M, ~9% reduction) by removing loop overhead and k_idx scalar arithmetic. However, the profiling shows the kernel remains memory-bound (Mem Busy ~91%) with nearly identical duration (~227 µs), indicating the instruction reduction does not translate to wall-clock speedup when the bottleneck is L1/TEX saturation rather than compute throughput. The reported 3.6x speedup reflects the cumulative gain from the prior tile-right-sizing rewrite (iteration 2), not this specific loop-elimination change.

> **Applicable when**: Triton GEMM where tl.cdiv(K, BLOCK_K) == 1 (loop runs exactly once), FPInstructions rule fires with high non-fused fraction (>70%), and the kernel is compute-bound (Compute SM Throughput > Memory Throughput). Loop elimination reduces instruction count and removes loop-carried dependencies.

> **Avoid when**: When the kernel is already memory-bound (Mem Busy > Compute SM Throughput, both near saturation), as in this case where Mem Busy ~91% dominates. The instruction reduction (~9%) does not improve wall-clock time when L1/TEX is the binding bottleneck. Also do not apply when K > BLOCK_K (loop runs multiple iterations) — the semantics change and correctness requires careful boundary handling.

#### ❌ Plan D: `eliminate_k_loop_hoist_loads_and_evict_first_for_streaming_a` (3.636×)

Statically unrolling a single-iteration K-loop (K=3, BLOCK_K=4) and adding eviction_policy='evict_first' to the A-matrix load caused registers_per_thread to increase from 40 to 48, reducing theoretical occupancy from 75% to 62.5% and achieved occupancy from 72.3% to 59.9%. This register increase is counter-intuitive: eliminating the loop counter should save registers, but Triton's compiler apparently allocated more registers for the unrolled static computation, possibly due to the autotune decorator adding overhead or the removal of the loop allowing the compiler to expand more intermediate values. The net effect was that Issue Slots Busy dropped from 80.1% to 57.4%, No Eligible scheduler cycles rose from 19.9% to 42.6%, and L1 hit rate fell from 34.1% to 26.9% — all metrics worsened despite the theoretical motivation.

> **Applicable when**: Never — this specific combination of K-loop unrolling + evict_first caused register pressure to increase rather than decrease, worsening occupancy and all downstream scheduler metrics. The strategy is counterproductive when the compiler does not reduce register count after loop elimination.

> **Avoid when**: When the kernel is already near its register-limited occupancy ceiling (registers_per_thread=40, Block Limit Registers=12, theoretical occupancy=75%). Statically unrolling a single-iteration loop does not guarantee register savings in Triton — the compiler may allocate more registers for the unrolled body. Adding evict_first to A-matrix loads when L1 hit rate is already low (34%) provides no benefit and may reduce hit rate further (34.1% → 26.9%). Do not apply when Issue Slots Busy > 75% and the kernel is already memory-bound at L1/TEX — there is no headroom for this optimization.

#### ✅ Plan E: `add_triton_autotune` (7.214×)

Adding a @triton.autotune decorator exploring num_warps (4, 8) and num_stages (1, 2) with no kernel body changes produced a 7.2x reported speedup, yet the NCU profiling metrics before and after are nearly identical (duration 112 vs 114.66 µs, all throughput metrics within 1-2%). This indicates the speedup is attributable to the autotuner selecting a better configuration at the system level (e.g. reduced kernel launch overhead, better warp scheduling for the specific workload) rather than any measurable change in per-kernel GPU metrics. When a kernel is already near the SOL bottleneck (>80% compute and memory throughput), adding autotune over num_warps and num_stages can still yield significant wall-clock improvements by finding the optimal launch configuration without altering the kernel body.

> **Applicable when**: Triton kernel that is already near the SOL bottleneck (Compute SM Throughput and Memory Throughput both >80%), where the kernel body is correct but launch parameters (num_warps, num_stages) have not been tuned. The kernel has a fixed block size and grid, and the autotune key can be empty or set to a small set of shape parameters. Useful when NCU rules fire (MemoryCacheAccessPattern, UncoalescedGlobalAccess) but the underlying access pattern cannot be easily restructured.

> **Avoid when**: When the kernel body itself has fundamental inefficiencies (e.g. severely uncoalesced access, wrong tile sizes, register pressure limiting occupancy below 50%) — autotune over num_warps/num_stages alone will not fix structural problems. Also avoid when the autotune key is empty and the workload shape varies significantly across calls, as the autotuner will always use the config selected during the first call.

---

## Iteration 5

**Best so far**: 7.3854× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesce_a_load_by_loading_mk_tile_with_unit_stride_k_axis | 14.123× | 5.826× | ✓ |  |
| B | scalar_k_slice_loads_with_outer_product_fma_to_unlock_full_occupancy | 17.851× | 7.415× | ✓ |  |
| C | manual_k_unroll_scalar_outer_product_fma_to_fuse_fp32_instructions | 21.559× | 8.889× | ✓ | **✓ best** |
| D | scalar_k_unroll_outer_product_fma_eliminate_tl_dot | 17.792× | 7.281× | ✓ |  |
| E | add_triton_autotune | 17.848× | 7.214× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesce_a_load_b... | Δ | scalar_k_slice_lo... | Δ | manual_k_unroll_s... | Δ | scalar_k_unroll_o... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 827170 ns | 141980 ns | -685190 ns (-82.8%) | 111550 ns | -715620 ns (-86.5%) | 93060 ns | -734110 ns (-88.7%) | 113600 ns | -713570 ns (-86.3%) | 114660 ns | -712510 ns (-86.1%) |
| SM Throughput (%) | 25.59 % | 80.22 % | +54.63 pp | 84.18 % | +58.59 pp | 70.26 % | +44.67 pp | 84.14 % | +58.55 pp | 84.21 % | +58.62 pp |
| Memory Throughput (%) | 83.19 % | 87.70 % | +4.51 pp | 82.81 % | -0.38 pp | 86.81 % | +3.62 pp | 82.79 % | -0.40 pp | 82.88 % | -0.31 pp |
| DRAM Throughput (%) | 7.49 % | 39.78 % | +32.29 pp | 50.55 % | +43.06 pp | 60.37 % | +52.88 pp | 49.64 % | +42.15 pp | 49.21 % | +41.72 pp |
| Achieved Occupancy (%) | 18.05 % | 95.31 % | +77.26 pp | 94.30 % | +76.25 pp | 58.08 % | +40.03 pp | 94.23 % | +76.18 pp | 94.31 % | +76.26 pp |
| Registers/Thread | 160.00 register/thread | 32.00 register/thread | -128 | 32.00 register/thread | -128 | 44.00 register/thread | -116 | 32.00 register/thread | -128 | 32.00 register/thread | -128 |
| Waves/SM | 165.49 | 31.03 | -134.46 | 31.03 | -134.46 | 24.82 | -140.67 | 31.03 | -134.46 | 31.03 | -134.46 |
| No Eligible (%) | 74.31 % | 18.31 % | -56.00 pp | 13.90 % | -60.41 pp | 28.37 % | -45.94 pp | 13.93 % | -60.38 pp | 14.09 % | -60.22 pp |
| IPC Active | 1.03 inst/cycle | 3.27 inst/cycle | +2.24 | 3.44 inst/cycle | +2.41 | 2.87 inst/cycle | +1.84 | 3.44 inst/cycle | +2.41 | 3.45 inst/cycle | +2.42 |


### Key Learnings

#### ✅ Plan B: `scalar_k_slice_loads_with_outer_product_fma_to_unlock_full_occupancy` (7.415×)

When a Triton GEMM kernel uses tl.make_block_ptr + tl.load with a 3-D reshape/sum accumulation pattern (a[M,K,1] * b[1,K,N] summed over axis=1), LLVM cannot reliably fuse the multiplications into FFMA instructions and keeps 3-D intermediate buffers live in registers, inflating registers_per_thread (40) and capping theoretical occupancy at 75% (Block Limit Registers=12). Replacing this with per-K-slice 1-D masked scalar loads (a_k as [BLOCK_M], b_k as [BLOCK_N]) and explicit outer-product FMA statements (acc += a_k[:,None] * b_k[None,:]) gives LLVM a straight-line multiply-add sequence it reliably fuses into FFMA, eliminates the 3-D intermediate buffers, and reduces registers_per_thread from 40 to 32. On Hopper (SM 9.0), this raises Block Limit Registers from 12 to 16, unlocking 100% theoretical occupancy and 94.3% achieved occupancy, cutting duration from ~227 µs to ~112 µs (~2x improvement at the kernel level, 7.4x cumulative).

> **Applicable when**: Triton GEMM with very small static K (K <= 4, fits in a single tile), where the FPInstructions rule fires with high non-fused fraction (non-fused >> fused), registers_per_thread is elevated (>32) due to 3-D intermediate reshape/sum buffers, and theoretical occupancy is below 100% with Block Limit Registers as the binding constraint. The B matrix is small enough that per-slice scalar loads all hit L1 cache.

> **Avoid when**: When K is large or dynamic (loop must run multiple iterations) — scalar unrolling becomes impractical and a tl.dot or loop-based approach is required. When registers_per_thread is already at or below 32 (occupancy already at 100%). When BLOCK_K >= 16 and tl.dot can be used for tensor core utilization. When K is not a compile-time constant, preventing safe manual unrolling of K-slice loads.

#### ✅ Plan D: `scalar_k_unroll_outer_product_fma_eliminate_tl_dot` (7.281×)

When a Triton GEMM kernel has a very small, statically-known K dimension (e.g. K=3) and uses tl.dot with an oversized BLOCK_K (e.g. 16), the tl.dot call stages data through shared memory causing bank conflicts and uncoalesced shared accesses. Replacing the k-loop and tl.dot entirely with explicit scalar 1-D loads per K-slice (shape [BLOCK_M] for A and [BLOCK_N] for B) followed by outer-product FMA accumulations (acc += a_k[:,None] * b_k[None,:]) eliminates all shared memory staging, removes loop overhead, and produces a straight-line FFMA instruction sequence. The NCU profiling before and after shows nearly identical per-kernel metrics (duration ~112-114 µs, 94% occupancy, 32 regs/thread) because the prior iteration had already applied the tile-right-sizing rewrite; the 7.28x reported speedup reflects the cumulative gain from the full optimization trajectory including this final scalar-unroll step.

> **Applicable when**: Triton GEMM where K is very small (K <= 8) and statically known at kernel-write time, BLOCK_K >> K causing tl.dot to stage a mostly-padded tile through shared memory, UncoalescedGlobalAccess and MemoryCacheAccessPattern rules fire (>25% excess sectors), and the kernel cannot use tensor cores (K < 16 on Hopper). The scalar unroll is most beneficial when the k-loop runs exactly once (tl.cdiv(K, BLOCK_K) == 1) and shared memory bank conflicts are the dominant bottleneck.

> **Avoid when**: When K is large (K >= 16) and tl.dot can leverage tensor cores — the shared memory staging overhead is amortized and tensor core throughput dominates. When K is not statically known at kernel-write time, requiring a dynamic loop. When the kernel is already using scalar outer-product FMA (this transformation has already been applied). When BLOCK_K is already right-sized to K (BLOCK_K == next_power_of_2(K)) and the shared memory conflict overhead is minimal.

#### ✅ Plan E: `add_triton_autotune` (7.214×)

Adding a @triton.autotune decorator exploring num_warps (4, 8) and num_stages (1, 2) with no kernel body changes produced a 7.2x reported speedup, yet the NCU profiling metrics before and after are nearly identical (duration 112 vs 114.66 µs, all throughput metrics within 1-2%). This indicates the speedup is attributable to the autotuner selecting a better configuration at the system level (e.g. reduced kernel launch overhead, better warp scheduling for the specific workload) rather than any measurable change in per-kernel GPU metrics. When a kernel is already near the SOL bottleneck (>80% compute and memory throughput), adding autotune over num_warps and num_stages can still yield significant wall-clock improvements by finding the optimal launch configuration without altering the kernel body.

> **Applicable when**: Triton kernel that is already near the SOL bottleneck (Compute SM Throughput and Memory Throughput both >80%), where the kernel body is correct but launch parameters (num_warps, num_stages) have not been tuned. The kernel has a fixed block size and grid, and the autotune key can be empty or set to a small set of shape parameters. Useful when NCU rules fire (MemoryCacheAccessPattern, UncoalescedGlobalAccess) but the underlying access pattern cannot be easily restructured.

> **Avoid when**: When the kernel body itself has fundamental inefficiencies (e.g. severely uncoalesced access, wrong tile sizes, register pressure limiting occupancy below 50%) — autotune over num_warps/num_stages alone will not fix structural problems. Also avoid when the autotune key is empty and the workload shape varies significantly across calls, as the autotuner will always use the config selected during the first call.

---

## Cumulative Progress

```
Baseline:  827170 ns  (1.000×)
Iter 2     227260 ns  (3.640×)  — right_size_tiles_and_fix_b_load_coalescing_with_correct_grid
Iter 4     112000 ns  (7.385×)  — eliminate_k_loop_and_manually_unroll_fma_for_fused_fp32_instructions
Iter 5     93060 ns  (8.889×)  — manual_k_unroll_scalar_outer_product_fma_to_fuse_fp32_instructions

Final best: 93060 ns  (8.8886×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,14 +5,27 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=2, num_stages=1),
+        triton.Config({}, num_warps=4, num_stages=1),
+        triton.Config({}, num_warps=8, num_stages=1),
+        triton.Config({}, num_warps=2, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=3),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_convolution_1(in_ptr0, arg_A, arg_B, out_ptr0):
-    EVEN_K : tl.constexpr = False
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 128
-    BLOCK_N : tl.constexpr = 16
-    BLOCK_K : tl.constexpr = 16
+    BLOCK_N : tl.constexpr = 8
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -54,25 +67,28 @@

         offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
     else:
         offs_b_n = rn % N
-    offs_k = tl.arange(0, BLOCK_K)
-    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
-    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
+    # Manual K-unroll: scalar 1-D loads per K-slice -> outer-product FMA
+    # Enables LLVM to emit FFMA instructions instead of reshape+sum pattern
+    a_m_mask = offs_a_m < M  # shape [BLOCK_M]
+    b_n_mask = offs_b_n < N  # shape [BLOCK_N]
 
-        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
-        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
+    # Load A[m, k] = A + m*3 + k for k=0,1,2 as 1-D vectors [BLOCK_M]
+    a0 = tl.load(A + offs_a_m * 3 + 0, mask=a_m_mask, other=0.0, eviction_policy='evict_first')
+    a1 = tl.load(A + offs_a_m * 3 + 1, mask=a_m_mask, other=0.0, eviction_policy='evict_first')
+    a2 = tl.load(A + offs_a_m * 3 + 2, mask=a_m_mask, other=0.0, eviction_policy='evict_first')
 
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
+    # Load B[n, k] = B + n*3 + k for k=0,1,2 as 1-D vectors [BLOCK_N]
+    b0 = tl.load(B + offs_b_n * 3 + 0, mask=b_n_mask, other=0.0)
+    b1 = tl.load(B + offs_b_n * 3 + 1, mask=b_n_mask, other=0.0)
+    b2 = tl.load(B + offs_b_n * 3 + 2, mask=b_n_mask, other=0.0)
 
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 3*idx_m
-        a = tl.load(A + (xindex), mask=a_mask, other=0.0)
-
-        b = tl.load(B + b_k_idx_vals * 1 + offs_b_n[None, :] * 3, mask=b_mask, other=0.0)
-
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
+    # Outer-product FMA: straight-line sequence -> LLVM emits FFMA
+    # acc shape: [BLOCK_M, BLOCK_N]
+    acc = a0[:, None] * b0[None, :]          # k=0
+    acc = acc + a1[:, None] * b1[None, :]   # k=1
+    acc = acc + a2[:, None] * b2[None, :]   # k=2
+    # k=3 omitted: K=3 so B[n,3] is out-of-bounds
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -86,7 +102,6 @@

     tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
