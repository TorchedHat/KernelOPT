# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_clone_permute_unsqueeze_view_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 163460 ns — **Best latency**: 163460 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_clone_permute_unsqueeze_view_4`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_048/kernel_4_triton_poi_fused_clone_permute/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (262144,1,1), Waves/SM: 124.12
- **Registers/thread**: 16
- **Duration**: 163460 ns
- **SM Throughput**: 30.95 % | **Memory**: 47.20 % | **DRAM**: 47.20 %
- **Achieved Occupancy**: 51.75 %
- **IPC Active**: 1.23 inst/cycle | **No Eligible**: 69.55 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_and_xblock_increase | 0.998× | 0.995× | ✓ |  |
| B | vectorized_loads_with_contiguity_hints_and_larger_xblock | 0.991× | 0.684× | ✓ |  |
| C | autotune_xblock_increase_tile_size | 0.733× | 0.744× | ✓ |  |
| D | increase_xblock_with_autotune_to_hide_l1tex_latency | 0.997× | 0.681× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | vectorized_loads_... | Δ | autotune_xblock_i... | Δ | increase_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 163460 ns | 164350 ns | +890 ns (+0.5%) | 239040 ns | +75580 ns (+46.2%) | 219650 ns | +56190 ns (+34.4%) | 240160 ns | +76700 ns (+46.9%) |
| SM Throughput (%) | 30.95 % | 38.09 % | +7.14 pp | 49.81 % | +18.86 pp | 27.05 % | -3.90 pp | 50.13 % | +19.18 pp |
| Memory Throughput (%) | 47.20 % | 45.24 % | -1.96 pp | 32.16 % | -15.04 pp | 35.02 % | -12.18 pp | 32.00 % | -15.20 pp |
| DRAM Throughput (%) | 47.20 % | 45.24 % | -1.96 pp | 32.16 % | -15.04 pp | 35.02 % | -12.18 pp | 32.00 % | -15.20 pp |
| Achieved Occupancy (%) | 51.75 % | 51.79 % | +0.04 pp | 52.37 % | +0.62 pp | 44.99 % | -6.76 pp | 51.42 % | -0.33 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 124.12 | 124.12 | +0.00 | 248.24 | +124.12 | 124.12 | +0.00 | 248.24 | +124.12 |
| No Eligible (%) | 69.55 % | 60.74 % | -8.81 pp | 53.53 % | -16.02 pp | 73.38 % | +3.83 pp | 53.59 % | -15.96 pp |
| IPC Active | 1.23 inst/cycle | 1.56 inst/cycle | +0.33 | 1.84 inst/cycle | +0.61 | 1.06 inst/cycle | -0.17 | 1.83 inst/cycle | +0.60 |


### Key Learnings

#### ❌ Plan C: `autotune_xblock_increase_tile_size` (0.744×)

Adding a @triton.autotune decorator with larger XBLOCK configs (512–2048) to a kernel that was already latency-bound (CPIStall ~80% of stall cycles, L1 hit rate 0%) caused a 26% regression. The profiling shows that duration increased from 163 µs to 220 µs, DRAM throughput dropped from 47% to 35%, achieved occupancy fell from 51.7% to 45%, and eligible warps per scheduler dropped from 0.39 to 0.31 — all metrics moved in the wrong direction. The autotune overhead and the introduction of a non-trivial xmask (changing from tl.full([XBLOCK], True) to xindex < xnumel) added extra predication instructions (+3.7M executed instructions), increasing per-warp work without improving memory access quality. When the root bottleneck is L1TEX stall latency and L2 hit rate is already ~67%, simply enlarging the tile size does not improve cache reuse and instead reduces occupancy, making the latency problem worse.

> **Applicable when**: Never: this specific direction caused regression. In general, increasing XBLOCK via autotune may help when L1 hit rate is 0% AND the access pattern is truly strided in a way that larger tiles fix coalescing. However, if the kernel is already latency-bound with CPIStall dominating and L2 hit rate is moderate (~67%), tile size increases alone will not help.

> **Avoid when**: Kernel is latency-bound (CPIStall rule triggered, >70% of stall cycles from L1TEX waits) AND L2 hit rate is already moderate (>50%) AND xnumel is exactly divisible by the original XBLOCK (making masking unnecessary). Adding autotune with larger XBLOCK in this scenario reduces occupancy, increases instruction count from unnecessary masking, and lowers both DRAM throughput and eligible warps per scheduler.

#### ❌ Plan D: `increase_xblock_with_autotune_to_hide_l1tex_latency` (0.681×)

Increasing block size (XBLOCK) from 128 to 256 via autotune improved scheduler eligibility (eligible warps per scheduler: 0.39→1.28) and reduced warp stall cycles (26.6→17.6 cycles/instruction), but the kernel still ran 47% slower (163→240 µs). The autotune selected XBLOCK=256 rather than the intended 512 or 1024, doubling total thread count and waves-per-SM (124→248) without proportionally improving memory bandwidth utilization (DRAM throughput dropped from 47.2% to 32.0%), indicating the larger grid introduced overhead that outweighed latency-hiding gains. When the kernel is already memory-bandwidth-limited with high waves-per-SM and the grid is fixed rather than scaled with XBLOCK, doubling block size doubles total work dispatched and increases kernel duration.

> **Applicable when**: Never: this specific implementation has a critical bug where the grid is hardcoded to xnumel//128 regardless of the autotuned XBLOCK, causing 2x thread count when XBLOCK=256 is selected.

> **Avoid when**: Do not use autotune with a fixed grid that does not adapt to the selected XBLOCK. The grid must be a lambda: `grid = lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)`. Without this, larger XBLOCK values double (or more) the total threads launched, increasing duration proportionally. Also avoid adding unnecessary xmask computation when xnumel is divisible by all candidate XBLOCK values — the predicated stores add overhead.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | store_eviction_policy_evict_first_for_streaming_outputs | 0.998× | 0.997× | ✓ |  |
| B | eviction_policies_stores_evict_first_load_evict_last | 0.752× | 0.758× | ✓ |  |
| C | eviction_policies_streaming_load_evict_first_store_evict_last | 0.997× | 0.657× | ✓ |  |
| D | eviction_policies_streaming_load_evict_first | 0.996× | 0.675× | ✓ |  |


### Metric Comparison

| Metric | Baseline | store_eviction_po... | Δ | eviction_policies... | Δ | eviction_policies... | Δ | eviction_policies... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 163460 ns | 163970 ns | +510 ns (+0.3%) | 215520 ns | +52060 ns (+31.8%) | 248860 ns | +85400 ns (+52.2%) | 242140 ns | +78680 ns (+48.1%) |
| SM Throughput (%) | 30.95 % | 34.32 % | +3.37 pp | 32.25 % | +1.30 pp | 54.55 % | +23.60 pp | 51.08 % | +20.13 pp |
| Memory Throughput (%) | 47.20 % | 46.90 % | -0.30 pp | 35.53 % | -11.67 pp | 29.91 % | -17.29 pp | 31.71 % | -15.49 pp |
| DRAM Throughput (%) | 47.20 % | 46.90 % | -0.30 pp | 35.53 % | -11.67 pp | 29.91 % | -17.29 pp | 31.71 % | -15.49 pp |
| Achieved Occupancy (%) | 51.75 % | 51.75 % | +0.00 pp | 43.13 % | -8.62 pp | 69.77 % | +18.02 pp | 67.65 % | +15.90 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 124.12 | 124.12 | +0.00 | 124.12 | +0.00 | 248.24 | +124.12 | 248.24 | +124.12 |
| No Eligible (%) | 69.55 % | 65.21 % | -4.34 pp | 67.10 % | -2.45 pp | 44.88 % | -24.67 pp | 48.77 % | -20.78 pp |
| IPC Active | 1.23 inst/cycle | 1.39 inst/cycle | +0.16 | 1.31 inst/cycle | +0.08 | 2.20 inst/cycle | +0.97 | 2.04 inst/cycle | +0.81 |


### Key Learnings

#### ❌ Plan B: `eviction_policies_stores_evict_first_load_evict_last` (0.758×)

Adding eviction_policy='evict_last' to loads and eviction_policy='evict_first' to write-once stores on a streaming permute kernel caused a 24% regression (164 µs → 216 µs). Despite the theoretical motivation (L2 hit rate ~67% on strided loads, write-once outputs polluting L2), the L2 hit rate was essentially unchanged (66.77% → 67.02%), while DRAM throughput dropped from 45.2% to 35.5%, achieved occupancy fell from 51.8% to 43.1%, eligible warps per scheduler dropped from 0.54 to 0.39, and executed instruction count increased by ~7.4% (49.3M → 52.9M). The eviction policy hints appear to have introduced additional instruction overhead and disrupted the GPU's native L2 management without improving cache residency, indicating that the hardware's default eviction policy was already near-optimal for this access pattern. The concurrent introduction of an @triton.autotune decorator (with a fixed grid bug from prior iterations) and an unnecessary xmask predication further inflated instruction count and reduced occupancy.

> **Applicable when**: Never for this kernel: eviction policy hints did not improve L2 hit rate and caused occupancy regression. In general, evict_first on stores and evict_last on loads is theoretically sound for streaming kernels with write-once outputs and strided inputs, but only when the hardware's default policy is measurably suboptimal (e.g., L2 hit rate drops after adding stores, or DRAM throughput is clearly store-limited).

> **Avoid when**: Do not apply eviction_policy hints when: (1) the L2 hit rate is already stable (~67%) and does not change after adding hints, indicating the hardware default is already near-optimal; (2) the change is bundled with other modifications (autotune, masking) that independently cause regression, making it impossible to isolate the eviction policy effect; (3) the kernel is already latency-bound (CPIStall dominant) rather than bandwidth-bound, since eviction hints address cache residency not stall latency. Also avoid when xnumel is exactly divisible by XBLOCK, making xmask = xindex < xnumel unnecessary predication overhead.

#### ❌ Plan D: `eviction_policies_streaming_load_evict_first` (0.675×)

Adding eviction_policy='evict_first' to tl.load and tl.store on a pure streaming permute kernel was intended to reduce L2 pollution, but the kernel regressed by 33% (164→242 µs). The root cause was not the eviction policy itself but a compounding implementation flaw: the autotune decorator was added with a fixed grid hardcoded to xnumel//128, so when autotune selected XBLOCK=256 the total thread count doubled (33M→67M) and waves-per-SM doubled (124→248), processing each element twice. Despite the eviction policy change, L2 hit rate barely moved (66.77%→67.32%), DRAM throughput dropped from 45.24% to 31.71%, and executed instructions nearly doubled (49M→95M), confirming the regression was driven by the doubled grid rather than the eviction policy. The evict_first policy alone cannot be evaluated from this experiment because the grid bug dominates.

> **Applicable when**: Never in this form. The evict_first policy on streaming one-shot data is theoretically sound but cannot be evaluated here due to the grid bug. If the grid bug is fixed (using lambda meta), evict_first on streaming permutes with 0% L1 hit rate and moderate L2 hit rate (~67%) may be worth retrying.

> **Avoid when**: Do not combine @triton.autotune with a hardcoded grid (not a lambda). This is the same anti-pattern as prior iterations: the grid must be `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)` to adapt to the autotuned XBLOCK. Do not add autotune when the only intended change is an eviction policy hint — autotune introduces XBLOCK variability that requires a dynamic grid. Also, evict_first on stores may be counterproductive if downstream kernels read those outputs from L2 cache.

---

## Cumulative Progress

```
Baseline:  163460 ns  (1.000×)

Final best: 163460 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_poi_fused_clone_permute_unsqueeze_view_4(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
@@ -16,11 +17,9 @@

     x2 = ((xindex // 1024) % 8)
     x3 = xindex // 8192
     x4 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0 + 16*x2 + 128*x1 + 8192*x3), None)
-    tl.store(out_ptr0 + (x4), tmp0, None)
-    tl.store(out_ptr1 + (x4), tmp0, None)
-
-
+    tmp0 = tl.load(in_ptr0 + (x0 + 16*x2 + 128*x1 + 8192*x3), None, eviction_policy='evict_first')
+    tl.store(out_ptr0 + (x4), tmp0, None, eviction_policy='evict_last')
+    tl.store(out_ptr1 + (x4), tmp0, None, eviction_policy='evict_last')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
