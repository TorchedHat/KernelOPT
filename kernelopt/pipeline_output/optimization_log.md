# Kernel Optimization Pipeline — Full Log

## Environment
- **Device**: NVIDIA H200 (CC 9.0), 132 SMs, 143 GB HBM3e
- **CUDA**: 13.0, Driver 580.82.07
- **Torch**: 2.10.0+cu128, **Triton**: 3.6.0
- **GPU used**: GPU 2 (CUDA_VISIBLE_DEVICES=2)

## Baseline Kernel
- **Name**: `softmax_triton_forward_kernel` (naive 3-pass Triton softmax)
- **Shape**: (131072, 256) = batch=32 × seq=4096, hidden=256
- **Launch**: Block (128,1,1), Grid (131072,1,1) → 62.06 waves/SM
- **Duration**: 134.34 ns
- **SM Throughput**: 79.19% | **Memory**: 77.88% | **DRAM**: 37.94%
- **Occupancy**: 100% theoretical, 91.86% achieved
- **Registers/thread**: 22, Block Limit Registers: 21
- **IPC**: 3.22 | **No Eligible**: 19.16%
- **Benchmark**: 0.214 ms (fwd+bwd), 3764.6 GB/s

> **Note**: The baseline kernel on H200 with Triton 3.6 compiled much more efficiently than
> expected. The original profiling context was from a PyTorch ATen kernel with 44 regs/thread
> and 62.5% occupancy — the Triton compiler achieved 22 regs/thread and 100% theoretical
> occupancy automatically.

## Iteration 1 Results

```
Iteration 1/5  |  Framework: triton  |  Best: 1.06×  |  Memory items: 3
──────────────────────────────────────────────────────────────────────────
Direction                     | Duration(ns) | Speedup | Correctness | Kept?
──────────────────────────────┼──────────────┼─────────┼─────────────┼──────
eliminate_partial_wave        | 261.89       | 0.51×   | passed      | ✗ regression
fuse_passes_online_softmax    | 127.20       | 1.06×   | passed      | ✓ best
fuse_fp_instructions          | 134.72       | 1.00×   | passed      | ✗ neutral
increase_num_warps            | 246.21       | 0.55×   | passed      | ✗ regression
```

## Detailed Metric Comparison

| Metric | Baseline | Plan1 (grid) | Plan2 (fuse) | Plan3 (FP) | Plan4 (warps) |
|---|---|---|---|---|---|
| Duration (ns) | 134.34 | 261.89 | **127.20** | 134.72 | 246.21 |
| SM Throughput (%) | 79.19 | 36.27 | **82.99** | 78.22 | 77.20 |
| Memory Throughput (%) | 77.88 | 35.98 | 71.21 | 78.41 | 69.95 |
| DRAM Throughput (%) | 37.94 | 19.49 | **39.89** | 37.81 | 20.70 |
| Achieved Occupancy (%) | 91.86 | 30.99 | 91.08 | 91.75 | 91.65 |
| Registers/Thread | 22 | 30 | **21** | 22 | 20 |
| Block Limit Registers | 21 | 16 | 21 | 21 | 10 |
| Waves/SM | 62.06 | 0.31 | 62.06 | 62.06 | 124.12 |
| No Eligible (%) | 19.16 | 62.77 | **15.46** | 21.08 | 21.92 |
| IPC Active | 3.22 | 1.49 | **3.37** | 3.18 | 3.12 |

## Key Learnings

### Plan 1 REGRESSION (0.51x) — Grid reduction killed parallelism
The LaunchConfiguration rule's 50% speedup estimate was misleading for this kernel.
With 131072 rows → 62 waves/SM, the partial wave tail is negligible (<1.6%).
Reducing the grid to 660 blocks forced each program to loop over ~199 rows sequentially,
collapsing occupancy from 91.86% to 30.99%.

**Lesson**: Only apply grid-alignment when waves_per_sm < 2. For high-wave kernels,
one-program-per-row maximizes GPU parallelism.

### Plan 2 IMPROVEMENT (1.06x) — Online softmax fusion
Fusing three passes into two (online max+sum → normalize) eliminated one global memory
round-trip. IPC improved 3.22→3.37, No Eligible dropped 19.16%→15.46%, SM throughput
rose 79.19%→82.99%. Modest but real improvement.

### Plan 3 NEUTRAL (1.00x) — FP fusion had no effect
The exp→exp2 and div→mul-reciprocal changes had no measurable impact. The Triton compiler
likely already optimizes these patterns. The FPInstructions rule's 10.5% opportunity
was not realizable through source-level changes.

### Plan 4 REGRESSION (0.55x) — num_warps increase backfired
Increasing num_warps from 4 to 8 changed the compiled block size from 128 to 256 threads.
This doubled waves/SM from 62 to 124 and halved block_limit_registers from 21 to 10.
DRAM throughput dropped from 37.94% to 20.70%.

**Lesson**: Triton's num_warps controls block size, which has cascading effects on
register allocation. Don't increase num_warps when occupancy is already high.

## Phase 6 — Candidate Selection

Best kernel: **Plan 2 (fuse_passes_online_softmax)** at 127.20 ns (1.06× speedup)

Candidates for iteration 2: `[softmax_plan2_fuse_passes.py, softmax_baseline.py]` (B=2)

## Experience Memory

3 items stored in `optimization_memory.json`:
- `exp_001`: negative — grid reduction on high-wave kernel (0.51×)
- `exp_002`: positive — online softmax fusion (1.06×)
- `exp_003`: negative — blind num_warps increase (0.55×)

Plan 3 (FP fusion) skipped — neutral result (0.87 ≤ 1.00 ≤ 1.04).

---

## Iteration 2

### Baseline for Iteration 2
- **File**: `softmax_plan2_fuse_passes.py` (2-pass online softmax)
- **Duration**: 127.20 ns
- **SM**: 82.99% | **Memory**: 71.21% | **DRAM**: 39.89%
- **Occupancy**: 91.08% achieved, 100% theoretical
- **Registers/thread**: 21, IPC: 3.37, Issue Slots Busy: 84.36%
- **Remaining NCU OPT rules**: MemoryL2Compression (14.6%), FPInstructions (12.0%)

### Plans Generated (N=4, informed by experience memory)

```
Iteration 2/5  |  Framework: triton  |  Best: 1.275× (cumulative)  |  Memory items: 4
──────────────────────────────────────────────────────────────────────────────────────────
Direction                     | Duration(ns) | vs Iter1  | vs Orig  | Correctness | Kept?
──────────────────────────────┼──────────────┼───────────┼──────────┼─────────────┼──────
enable_software_pipelining    | 126.37       | 1.007×    | 1.063×   | passed      | ✗ marginal
fuse_fp_instructions_v2       | 126.72       | 1.004×    | 1.060×   | passed      | ✗ marginal
combined_pipelining_fp_fusion | 127.17       | 1.000×    | 1.056×   | passed      | ✗ neutral
single_pass_full_row          | 105.38       | 1.207×    | 1.275×   | passed      | ✓ BEST
```

### Detailed Metric Comparison (Iteration 2)

| Metric | Iter1 Best | A (pipeline) | B (FP fuse) | C (combined) | D (1-pass) |
|---|---|---|---|---|---|
| Duration (ns) | 127.20 | 126.37 | 126.72 | 127.17 | **105.38** |
| SM Throughput (%) | 82.99 | 83.12 | 79.66 | 79.50 | 68.48 |
| Memory Throughput (%) | 71.21 | 71.30 | 71.08 | 70.93 | **78.97** |
| DRAM Throughput (%) | 39.89 | 40.16 | 40.05 | 39.90 | **48.10** |
| Registers/Thread | 21 | 21 | 24 | 24 | **18** |
| No Eligible (%) | 15.46 | 16.06 | 18.69 | 18.90 | 30.18 |
| IPC Active | 3.37 | 3.39 | 3.22 | 3.24 | 2.80 |

### Key Learnings — Iteration 2

#### Plan D MAJOR IMPROVEMENT (1.275× cumulative) — Single-pass softmax
Since BLOCK_SIZE (256) >= n_cols (256), the entire row fits in registers after one `tl.load`.
This eliminates the second pass entirely — max, exp, sum, normalize all happen in-register.
Memory traffic halved (1 load + 1 store vs 2 loads + 1 store). DRAM throughput jumped
39.89% → 48.10%, registers dropped 21 → 18.

The tradeoff: SM throughput dropped 82.99% → 68.48% and IPC 3.37 → 2.80 because fewer
instructions are needed (less compute work). But the wall-clock time improved 20.7%
because less memory traffic dominates.

#### Plans A, B, C — Marginal/neutral (<1% improvement)
Software pipelining and FP fusion had negligible effect on a kernel that already runs
in ~127 ns with single-iteration loops. The Triton compiler already handles these
optimizations internally for such simple patterns.

### Candidate Selection

Best kernel: **Plan D (single_pass_full_row)** at 105.38 ns (**1.275× over original**)

### Experience Memory

4 items stored:
- `exp_001`: negative — grid reduction (0.51×)
- `exp_002`: positive — online softmax fusion (1.06×)
- `exp_003`: negative — blind num_warps increase (0.55×)
- `exp_004`: positive — single-pass when row fits in block (1.275×)

### Cumulative Progress (after iteration 2)

```
Original baseline:  134.34 ns  (1.00×)
After iteration 1:  127.20 ns  (1.06×)  — 2-pass online softmax
After iteration 2:  105.38 ns  (1.28×)  — single-pass full row
```

---

## Iteration 3

### Baseline for Iteration 3
- **File**: `softmax_iter2_planD_single_pass.py` (single-pass softmax)
- **Duration**: 105.38 ns — **now memory-bound** (SOLBottleneck: Memory > Compute)
- **SM**: 68.48% | **Memory**: 78.97% | **DRAM**: 48.10% | **L1/TEX**: 80.64%
- **Occupancy**: 90.10% achieved, 100% theoretical
- **Registers/thread**: 18, Block: (128,1,1), num_warps=4, num_stages=1
- **CPIStall**: 32% MIO scoreboard + 32% L1TEX scoreboard (memory latency)
- **No Eligible**: 30.18% — scheduler starved waiting on memory

### Plans Generated (N=4)

```
Iteration 3/5  |  Framework: triton  |  Best: 1.603× (cumulative)  |  Memory items: 6
──────────────────────────────────────────────────────────────────────────────────────────
Direction                     | Duration(ns) | vs Iter2  | vs Orig  | Correctness | Kept?
──────────────────────────────┼──────────────┼───────────┼──────────┼─────────────┼──────
aggressive_pipelining (stg=4) | 105.47       | 0.999×    | 1.274×   | passed      | ✗ neutral
fp_fusion_single_pass         | 106.46       | 0.990×    | 1.262×   | passed      | ✗ neutral
process_2_rows_per_program    | 103.74       | 1.016×    | 1.295×   | passed      | ✗ marginal
combined_warps2_exp2_recip    |  83.81       | 1.257×    | 1.603×   | passed      | ✓ BEST
```

### Detailed Metric Comparison (Iteration 3)

| Metric | Iter2 Best | A (stg=4) | B (FP) | C (2row) | D (combo) |
|---|---|---|---|---|---|
| Duration (ns) | 105.38 | 105.47 | 106.46 | 103.74 | **83.81** |
| SM Throughput (%) | 68.48 | 68.20 | 64.83 | 64.41 | 44.22 |
| Memory Throughput (%) | 78.97 | 78.63 | 79.24 | 78.83 | 60.42 |
| DRAM Throughput (%) | 48.10 | 48.07 | 47.62 | 48.86 | **60.42** |
| L1/TEX Throughput (%) | 80.64 | 80.24 | 81.01 | 80.32 | 59.06 |
| Achieved Occupancy (%) | 90.10 | 90.06 | 90.04 | **94.09** | 54.71 |
| Registers/Thread | 18 | 18 | 17 | 22 | 21 |
| Waves/SM | 62.06 | 62.06 | 62.06 | 31.03 | 31.03 |
| No Eligible (%) | 30.18 | 30.42 | 33.82 | 35.47 | 54.07 |
| IPC Active | 2.80 | 2.78 | 2.65 | 2.57 | 1.84 |
| CPI (cycles/instr) | 20.50 | 20.44 | 21.66 | 23.24 | **19.08** |
| Block Size | (128,1,1) | (128,1,1) | (128,1,1) | (128,1,1) | **(64,1,1)** |

### Key Learnings — Iteration 3

#### Plan D MAJOR IMPROVEMENT (1.603× cumulative) — num_warps=2 changes everything
This is the most counterintuitive result of the pipeline. Reducing num_warps from 4 to 2:
- Changed block size from 128 to 64 threads
- **Occupancy dropped** 90% → 55%
- **No Eligible rose** 30% → 54%
- **But wall-clock time improved 20.5%** (105→84 ns)

Why? **DRAM throughput jumped 48% → 60%**. Smaller blocks changed the memory access pattern:
- More blocks per SM → better interleaving of memory requests
- CPI improved 20.50 → 19.08 (less latency per instruction despite more stalls)
- The kernel is doing less total work per unit time, but finishing the work faster

This demonstrates that **occupancy is not always correlated with performance**. For
memory-bound kernels with simple compute, smaller blocks with better memory access
patterns can win despite lower occupancy.

Combined with exp2 fusion and reciprocal multiplication (num_stages=2), the total
effect is 1.60× over the original kernel.

#### Plans A, B — Neutral (pipelining and FP fusion alone don't help)
num_stages=4 and exp2/reciprocal on the single-pass kernel had no effect — confirming
that the bottleneck was the memory access pattern, not instruction scheduling.

#### Plan C — Marginal (1.6%) — 2 rows per program
Processing 2 rows per program gave a small improvement (103.74 ns) by improving
occupancy to 94%. But this was dwarfed by Plan D's approach.

### Candidate Selection

Best kernel: **Plan D (combined_warps2_exp2_recip)** at 83.81 ns (**1.603× over original**)

### Experience Memory

6 items stored:
- `exp_001`: negative — grid reduction (0.51×)
- `exp_002`: positive — online softmax fusion (1.06×)
- `exp_003`: negative — num_warps increase (0.55×)
- `exp_004`: positive — single-pass full row (1.275×)
- `exp_005`: positive — **num_warps=2 + exp2 + reciprocal (1.603×)** ← new best
- `exp_006`: positive — 2 rows per program (1.016× marginal)

### Cumulative Progress

```
Original baseline:  134.34 ns  (1.00×)
After iteration 1:  127.20 ns  (1.06×)  — 2-pass online softmax
After iteration 2:  105.38 ns  (1.28×)  — single-pass full row
After iteration 3:   83.81 ns  (1.60×)  — num_warps=2 + exp2 + reciprocal
```
