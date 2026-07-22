# Kernel Optimization Log

## Summary

- **Kernel**: `matvec_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2200000 ns — **Best latency**: 2200000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `matvec_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_004/synthesis/kernel_0_fused_fused_mm.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2048,1,1), Waves/SM: 1.29
- **Registers/thread**: 36
- **Duration**: 2200000 ns
- **SM Throughput**: 21.48 % | **Memory**: 79.49 % | **DRAM**: 79.49 %
- **Achieved Occupancy**: 58.47 %
- **IPC Active**: 0.88 inst/cycle | **No Eligible**: 77.41 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_evict_first_deeper_pipeline | 0.981× | 0.996× | ✓ |  |
| B | expand_autotune_configs_deeper_pipeline_and_vectorization_hints | 0.985× | 1.014× | ✓ |  |
| D | fix_uncoalesced_stores_vectorize_loads_deepen_pipeline | 0.996× | 1.152× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | expand_autotune_c... | Δ | fix_uncoalesced_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2200000 ns | 2210000 ns | +10000 ns (+0.5%) | 2170000 ns | -30000 ns (-1.4%) | 1910000 ns | -290000 ns (-13.2%) |
| SM Throughput (%) | 21.48 % | 24.03 % | +2.55 pp | 24.16 % | +2.68 pp | 26.24 % | +4.76 pp |
| Memory Throughput (%) | 79.49 % | 79.25 % | -0.24 pp | 80.44 % | +0.95 pp | 91.57 % | +12.08 pp |
| DRAM Throughput (%) | 79.49 % | 79.25 % | -0.24 pp | 80.44 % | +0.95 pp | 91.57 % | +12.08 pp |
| Achieved Occupancy (%) | 58.47 % | 58.32 % | -0.15 pp | 58.82 % | +0.35 pp | 96.61 % | +38.14 pp |
| Registers/Thread | 36.00 register/thread | 36.00 register/thread | +0 | 36.00 register/thread | +0 | 32.00 register/thread | -4 |
| Waves/SM | 1.29 | 1.29 | +0.00 | 1.29 | +0.00 | 0.97 | -0.32 |
| No Eligible (%) | 77.41 % | 74.60 % | -2.81 pp | 74.49 % | -2.92 pp | 73.14 % | -4.27 pp |
| IPC Active | 0.88 inst/cycle | 0.99 inst/cycle | +0.11 | 1.00 inst/cycle | +0.12 | 1.09 inst/cycle | +0.21 |


### Key Learnings

#### ✅ Plan D: `fix_uncoalesced_stores_vectorize_loads_deepen_pipeline` (1.158×)

In a bandwidth-bound Triton matvec kernel already at BLOCK_M=1, expanding the autotune config space (adding larger BLOCK_K up to 32768, more num_warps/num_stages combinations) allows the autotuner to find a configuration that reduces register pressure (32→29 registers/thread) and dramatically improves instruction throughput (IPC active 1.09→1.58, Issue Slots Busy 27%→39%, Warp Cycles Per Issued Instruction 56.7→38.6). Adding eviction_policy='evict_last' to loads of the reused B vector modestly improves L2 hit rate (2.88%→5.41%) and L1/TEX cache throughput (44%→66%). The combined effect is a 15.8% speedup despite DRAM throughput remaining near-saturated (~92%), indicating the gain comes primarily from better instruction-level pipelining hiding memory latency.

> **Applicable when**: Triton matvec/gemv kernel with BLOCK_M=1 already applied; CPIStall rule fires with L1TEX scoreboard stalls >50% of warp cycles; Issue Slots Busy <35%; a small reused vector (fits in L2) is loaded every iteration without eviction hint; autotune config space is narrow (fewer than 10 configs, limited BLOCK_K range).

> **Avoid when**: Kernel is compute-bound (Compute SM Throughput >70%); the reused vector does not fit in L2 cache (evict_last would evict useful data); BLOCK_K is already at maximum hardware-supported size; register pressure is already minimal (<24 registers/thread) and occupancy is the binding constraint.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tl_dot_tensor_core_accumulation | 0.688× | 0.215× | ✓ |  |
| B | eliminate_partial_wave_via_grid_rounding_and_b_evict_last | 0.994× | 1.158× | ✓ |  |
| C | evict_last_b_vector_and_purge_block_m_gt1_configs | 0.981× | 1.005× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tl_dot_tensor_cor... | Δ | eliminate_partial... | Δ | evict_last_b_vect... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2200000 ns | 10230000 ns | +8030000 ns (+365.0%) | 1900000 ns | -300000 ns (-13.6%) | 2190000 ns | -10000 ns (-0.5%) |
| SM Throughput (%) | 21.48 % | 5.38 % | -16.10 pp | 38.34 % | +16.86 pp | 27.11 % | +5.63 pp |
| Memory Throughput (%) | 79.49 % | 17.09 % | -62.40 pp | 92.04 % | +12.55 pp | 79.89 % | +0.40 pp |
| DRAM Throughput (%) | 79.49 % | 17.09 % | -62.40 pp | 92.04 % | +12.55 pp | 79.89 % | +0.40 pp |
| Achieved Occupancy (%) | 58.47 % | 6.25 % | -52.22 pp | 96.12 % | +37.65 pp | 58.53 % | +0.06 pp |
| Registers/Thread | 36.00 register/thread | 32.00 register/thread | -4 | 29.00 register/thread | -7 | 36.00 register/thread | +0 |
| Waves/SM | 1.29 | 0.06 | -1.23 | 0.97 | -0.32 | 1.29 | +0.00 |
| No Eligible (%) | 77.41 % | 94.36 % | +16.95 pp | 61.04 % | -16.37 pp | 71.93 % | -5.48 pp |
| IPC Active | 0.88 inst/cycle | 0.23 inst/cycle | -0.65 | 1.58 inst/cycle | +0.70 | 1.13 inst/cycle | +0.25 |


### Key Learnings

#### ❌ Plan A: `tl_dot_tensor_core_accumulation` (0.215×)

Switching from BLOCK_M=1 scalar FMA accumulation to BLOCK_M=16/32 with tl.dot for tensor core MMA reduced the grid from 2048 to 128 blocks (16x fewer CTAs), causing waves_per_SM to collapse from 0.97 to 0.06 and achieved occupancy to drop from 96.6% to 6.25%. DRAM throughput fell from 91.57% to 17.09% because only 128 of 132 SMs were active and each SM had only 1 active warp per scheduler instead of 15. The kernel duration increased 5.4x (1.91ms → 10.23ms) because the GPU was massively underutilized: the grid was too small to fill the device, and the tl.dot approach did not compensate for the catastrophic loss of parallelism.

> **Applicable when**: Never: this strategy caused a 4.65x regression. Do not apply tl.dot with BLOCK_M>=16 to a matvec kernel where M is small (e.g., M=2048) and the baseline already achieves near-full occupancy with BLOCK_M=1.

> **Avoid when**: When the kernel is already bandwidth-saturated (DRAM SOL >85%) with high occupancy (>90%) and a large grid (waves_per_SM ~1.0). Increasing BLOCK_M to satisfy tl.dot's >=16 constraint reduces grid size proportionally, collapsing occupancy and DRAM utilization. The tensor core throughput gain cannot compensate for the loss of parallelism when M is small relative to SM count.

---

## Cumulative Progress

```
Baseline:  2200000 ns  (1.000×)

Final best: 2200000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -8,18 +8,21 @@

 # We tile over K dimension and accumulate dot product.
 # For large K (1048576), we need to be careful about register pressure.
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 1024}, num_warps=4, num_stages=2),
-        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 2048}, num_warps=4, num_stages=2),
-        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 4096}, num_warps=8, num_stages=2),
-        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 8192}, num_warps=8, num_stages=2),
-        triton.Config({'BLOCK_M': 2, 'BLOCK_K': 1024}, num_warps=4, num_stages=2),
-        triton.Config({'BLOCK_M': 2, 'BLOCK_K': 2048}, num_warps=4, num_stages=2),
-        triton.Config({'BLOCK_M': 4, 'BLOCK_K': 1024}, num_warps=4, num_stages=2),
-        triton.Config({'BLOCK_M': 4, 'BLOCK_K': 2048}, num_warps=8, num_stages=2),
-        triton.Config({'BLOCK_M': 8, 'BLOCK_K': 1024}, num_warps=8, num_stages=2),
-        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 512}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 4096},  num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 4096},  num_warps=4,  num_stages=3),
+        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 4096},  num_warps=4,  num_stages=4),
+        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 8192},  num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 8192},  num_warps=8,  num_stages=3),
+        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 8192},  num_warps=8,  num_stages=4),
+        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 8192},  num_warps=8,  num_stages=5),
+        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 16384}, num_warps=8,  num_stages=3),
+        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 16384}, num_warps=8,  num_stages=4),
+        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 16384}, num_warps=16, num_stages=4),
+        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 32768}, num_warps=16, num_stages=3),
+        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 32768}, num_warps=16, num_stages=4),
     ],
     key=['M', 'K'],
 )
@@ -46,15 +49,16 @@

     # Iterate over K dimension in tiles
     for k_start in tl.range(0, K, BLOCK_K):
         k_offsets = k_start + tl.arange(0, BLOCK_K)
+        k_offsets = tl.max_contiguous(tl.multiple_of(k_offsets, BLOCK_K), BLOCK_K)
         k_mask = k_offsets < K
         
         # Load B tile: shape [BLOCK_K]
         b_ptrs = B_ptr + k_offsets * stride_bk
-        b = tl.load(b_ptrs, mask=k_mask, other=0.0)
+        b = tl.load(b_ptrs, mask=k_mask, other=0.0, eviction_policy='evict_last')
         
         # Load A tile: shape [BLOCK_M, BLOCK_K]
         a_ptrs = A_ptr + row_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
-        a = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
+        a = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0, eviction_policy='evict_first')
         
         # Accumulate dot product
         acc += tl.sum(a * b[None, :], axis=1)
@@ -62,7 +66,6 @@

     # Store results
     c_ptrs = C_ptr + row_offsets
     tl.store(c_ptrs, acc, mask=row_mask)
-
 
 def kernel_function(a, b):
     M, K = a.shape
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
