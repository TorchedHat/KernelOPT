# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 286140 ns — **Best latency**: 139650 ns (**2.0490×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`enable_software_pipelining_via_num_stages_2_for_coalesced_kernel`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_002/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 286140 ns
- **SM Throughput**: 19.35 % | **Memory**: 78.77 % | **DRAM**: 72.11 %
- **Achieved Occupancy**: 94.04 %
- **IPC Active**: 0.79 inst/cycle | **No Eligible**: 80.41 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_stores_and_software_pipelining_for_strided_transpose | 0.999× | 1.002× | ✓ |  |
| C | software_pipelining_via_num_stages_autotune | 0.998× | 1.004× | ✓ |  |
| D | autotune_yblock_and_num_stages_for_l2_reuse | 1.488× | 1.002× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_stores... | Δ | software_pipelini... | Δ | autotune_yblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 286140 ns | 285600 ns | -540 ns (-0.2%) | 285090 ns | -1050 ns (-0.4%) | 285600 ns | -540 ns (-0.2%) |
| SM Throughput (%) | 19.35 % | 19.84 % | +0.49 pp | 19.65 % | +0.30 pp | 19.60 % | +0.25 pp |
| Memory Throughput (%) | 78.77 % | 78.90 % | +0.13 pp | 79.07 % | +0.30 pp | 78.93 % | +0.16 pp |
| DRAM Throughput (%) | 72.11 % | 72.23 % | +0.12 pp | 72.37 % | +0.26 pp | 72.24 % | +0.13 pp |
| Achieved Occupancy (%) | 94.04 % | 94.08 % | +0.04 pp | 94.04 % | +0.00 pp | 94.04 % | +0.00 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.41 % | 80.06 % | -0.35 pp | 80.30 % | -0.11 pp | 79.91 % | -0.50 pp |
| IPC Active | 0.79 inst/cycle | 0.79 inst/cycle | +0.00 | 0.79 inst/cycle | +0.00 | 0.80 inst/cycle | +0.01 |

---

## Iteration 3

**Best so far**: 1.0019× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | vectorized_loads_via_scalar_contiguous_hints_on_load_index | 0.702× | 0.696× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 286140 ns | 411230 ns | +125090 ns (+43.7%) |
| SM Throughput (%) | 19.35 % | 13.43 % | -5.92 pp |
| Memory Throughput (%) | 78.77 % | 59.94 % | -18.83 pp |
| DRAM Throughput (%) | 72.11 % | 57.71 % | -14.40 pp |
| Achieved Occupancy (%) | 94.04 % | 94.91 % | +0.87 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 80.41 % | 86.34 % | +5.93 pp |
| IPC Active | 0.79 inst/cycle | 0.54 inst/cycle | -0.25 |


### Key Learnings

#### ❌ Plan B: `vectorized_loads_via_scalar_contiguous_hints_on_load_index` (0.696×)

Applying tl.max_contiguous and tl.multiple_of hints to a 1D index that is then broadcast into a strided 2D load address does not change the physical memory access pattern — consecutive warp lanes still access addresses 128 floats apart. In this case the hints caused a regression: duration increased from 285 µs to 411 µs, Warp Cycles Per Issued Instruction worsened from 74.5 to 112.5, and eligible warps per scheduler dropped from 0.28 to 0.20, indicating increased latency stalls. The simultaneous change from num_stages=2/3/4 to including num_stages=1 in the autotune search likely selected a lower-pipelining config, amplifying the L1TEX stall penalty and causing the regression.

> **Applicable when**: Never beneficial for this pattern — do not apply.

> **Avoid when**: When the load stride between consecutive warp lanes is large (e.g., 128 floats), tl.max_contiguous/tl.multiple_of hints on the index do not change the physical access pattern and cannot coalesce the loads. Adding num_stages=1 to the autotune config set can select an under-pipelined configuration that worsens L1TEX stall cycles. Avoid this combination when the kernel is already L2-bound with high warp stall cycles.

---

## Iteration 4

**Best so far**: 1.0019× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_tiled_transpose_with_verified_indices_and_safe_autotune | 1.943× | 2.021× | ✓ | **✓ best** |
| E | add_triton_autotune | 0.718× | 0.701× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_til... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 286140 ns | 141570 ns | -144570 ns (-50.5%) | 408030 ns | +121890 ns (+42.6%) |
| SM Throughput (%) | 19.35 % | 49.21 % | +29.86 pp | 13.52 % | -5.83 pp |
| Memory Throughput (%) | 78.77 % | 42.07 % | -36.70 pp | 60.50 % | -18.27 pp |
| DRAM Throughput (%) | 72.11 % | 34.78 % | -37.33 pp | 57.98 % | -14.13 pp |
| Achieved Occupancy (%) | 94.04 % | 91.44 % | -2.60 pp | 94.95 % | +0.91 pp |
| Registers/Thread | 25.00 register/thread | 26.00 register/thread | +1 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.41 % | 55.78 % | -24.63 pp | 86.60 % | +6.19 pp |
| IPC Active | 0.79 inst/cycle | 1.78 inst/cycle | +0.99 | 0.54 inst/cycle | -0.25 |


### Key Learnings

#### ❌ Plan E: `add_triton_autotune` (0.701×)

When a kernel's dominant bottleneck is uncoalesced global loads (0% L1 hit rate, ~87% L2 throughput, ~112 warp cycles per issued instruction due to L1TEX stalls), varying XBLOCK via autotune configs (128→256→512→1024) with num_stages=1 produces no meaningful improvement. The profiling metrics are nearly identical before and after (Duration: 411 µs vs 408 µs, all other metrics within noise), because the access stride between consecutive warp lanes is unchanged regardless of tile width. Autotune parameter sweeps cannot fix structural memory access pattern problems; coalescing must be addressed by changing the index formula or using shared memory staging.

> **Applicable when**: Never beneficial when the root cause is uncoalesced global loads with large inter-lane stride.

> **Avoid when**: When the kernel has 0% L1 hit rate, ~87% L2 throughput, and ~112 warp cycles per issued instruction caused by strided global loads (e.g., consecutive warp lanes access addresses N*sizeof(float) apart where N>>1), sweeping XBLOCK via autotune will not improve performance. The access stride is determined by the index formula, not the tile size. Autotune is only useful when the bottleneck is tunable (e.g., occupancy, register pressure, software pipelining depth) rather than structural memory access patterns.

---

## Iteration 5

**Best so far**: 2.0212× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | software_pipelining_num_stages_and_eviction_policy_for_latency_hiding | 1.883× | 1.934× | ✓ |  |
| B | enable_software_pipelining_via_num_stages_2_for_coalesced_kernel | 1.943× | 2.049× | ✓ | **✓ best** |
| C | coalesced_2d_tile_remapping_for_nhwc_to_nchw_transpose | 0.727× | 0.670× | ✓ |  |


### Metric Comparison

| Metric | Baseline | software_pipelini... | Δ | enable_software_p... | Δ | coalesced_2d_tile... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 286140 ns | 147970 ns | -138170 ns (-48.3%) | 139650 ns | -146490 ns (-51.2%) | 427100 ns | +140960 ns (+49.3%) |
| SM Throughput (%) | 19.35 % | 47.46 % | +28.11 pp | 49.31 % | +29.96 pp | 53.14 % | +33.79 pp |
| Memory Throughput (%) | 78.77 % | 40.46 % | -38.31 pp | 42.25 % | -36.52 pp | 70.39 % | -8.38 pp |
| DRAM Throughput (%) | 72.11 % | 32.63 % | -39.48 pp | 35.22 % | -36.89 pp | 12.14 % | -59.97 pp |
| Achieved Occupancy (%) | 94.04 % | 91.84 % | -2.20 pp | 91.76 % | -2.28 pp | 59.13 % | -34.91 pp |
| Registers/Thread | 25.00 register/thread | 26.00 register/thread | +1 | 26.00 register/thread | +1 | 45.00 register/thread | +20 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 95.81 | +35.93 |
| No Eligible (%) | 80.41 % | 54.37 % | -26.04 pp | 56.20 % | -24.21 pp | 48.78 % | -31.63 pp |
| IPC Active | 0.79 inst/cycle | 1.82 inst/cycle | +1.03 | 1.78 inst/cycle | +0.99 | 2.04 inst/cycle | +1.25 |


### Key Learnings

#### ✅ Plan C: `coalesced_2d_tile_remapping_for_nhwc_to_nchw_transpose` (2.021×)

For a layout-permutation kernel (e.g., NHWC→NCHW transpose) where the load has a large stride along the warp's innermost thread dimension (causing 0% L1 hit rate, high L2 throughput, and high Warp Cycles Per Issued Instruction from CPIStall), restructure the 2D tile so that the innermost thread dimension aligns with the stride-1 axis of the input for loads and the stride-1 axis of the output for stores. In this case, making YBLOCK iterate over the channel (C) dimension (stride=1 in input) and XBLOCK iterate over the spatial (HW) dimension (stride=1 in output) made both loads and stores coalesced without shared memory or tl.trans(). This reduced Warp Cycles Per Issued Instruction from 112.5 to 32.5, boosted Executed IPC Active from 0.54 to 1.78, and cut duration from 411 µs to 142 µs (2.02× speedup). All autotune configs must use num_stages=1 to avoid crashes with strided access patterns.

> **Applicable when**: Kernel performs a layout permutation (e.g., NHWC→NCHW) where the load index has a large stride (>>1 element) along the warp's innermost thread dimension, causing 0% L1 hit rate, L2 throughput >80%, and Warp Cycles Per Issued Instruction >70 due to CPIStall on L1TEX. The input has a stride-1 dimension (e.g., channel C) and the output has a different stride-1 dimension (e.g., spatial HW). Both can be made coalesced by choosing the 2D tile axes appropriately.

> **Avoid when**: When the load and store stride-1 dimensions are the same axis (no transpose needed), or when the tensor dimensions do not cleanly decompose into two independent tile axes. Also avoid num_stages>1 in autotune configs when the access pattern involves strided global loads, as this causes autotune crashes in Triton.

---

## Cumulative Progress

```
Baseline:  286140 ns  (1.000×)
Iter 2     285600 ns  (1.002×)  — autotune_yblock_and_num_stages_for_l2_reuse
Iter 4     141570 ns  (2.021×)  — shared_memory_tiled_transpose_with_verified_indices_and_safe_autotune
Iter 5     139650 ns  (2.049×)  — enable_software_pipelining_via_num_stages_2_for_coalesced_kernel

Final best: 139650 ns  (2.0490×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,69 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
-def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
-    ynumel = 4096
-    xnumel = 31500
-    yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
-    xmask = xindex < xnumel
-    x2 = xindex
-    y0 = (yindex % 128)
-    y1 = yindex // 128
-    y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask)
+def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
+    # TENSOR LAYOUT:
+    # Input shape (32,128,126,250), strides (4032000,1,32000,128) = NHWC-like
+    #   elem[n,c,h,w] = n*4032000 + c*1 + hw*128  where hw = h*250+w
+    # Output shape (32,128,126,250), strides (4032000,31500,250,1) = NCHW
+    #   elem[n,c,h,w] = n*4032000 + c*31500 + hw*1
+    # NC = 32*128 = 4096, HW = 126*250 = 31500
+    #
+    # COALESCING STRATEGY:
+    # Use XBLOCK as TILE_HW (hw outer dim) and YBLOCK as TILE_C (nc inner dim).
+    # 2D tile shape: [XBLOCK, YBLOCK] = [TILE_HW, TILE_NC]
+    # Load: inner dim (YBLOCK) varies in nc -> c varies (stride=1 in input) -> coalesced
+    # Store: outer dim (XBLOCK) varies in hw (stride=1 in output) -> coalesced
+    C = 128
+    HW = 31500
+    NC = 4096
 
+    pid_hw = tl.program_id(0)   # drives hw dimension
+    pid_nc = tl.program_id(1)   # drives nc dimension
 
+    # hw_off: outer dimension, shape [XBLOCK]
+    hw_off = pid_hw * XBLOCK + tl.arange(0, XBLOCK)
+    hw_mask = hw_off < HW
+
+    # nc_off: inner dimension, shape [YBLOCK]
+    nc_off = pid_nc * YBLOCK + tl.arange(0, YBLOCK)
+    nc_mask = nc_off < NC
+
+    # Decompose nc into n and c
+    n_idx = nc_off // C   # shape [YBLOCK]
+    c_idx = nc_off % C    # shape [YBLOCK]
+
+    # 2D tile: [XBLOCK, YBLOCK] = [TILE_HW, TILE_NC]
+    # Load index: n*4032000 + c*1 + hw*128
+    # Inner dim (YBLOCK) varies in c (stride=1 in input) -> coalesced load
+    load_idx = n_idx[None, :] * 4032000 + c_idx[None, :] * 1 + hw_off[:, None] * 128
+    mask2d = hw_mask[:, None] & nc_mask[None, :]
+    vals = tl.load(in_ptr0 + load_idx, mask=mask2d)
+
+    # Store index: n*4032000 + c*31500 + hw*1
+    # Inner dim (YBLOCK) varies in c (stride=31500 in output)
+    # Outer dim (XBLOCK) varies in hw (stride=1 in output) -> coalesced store
+    store_idx = n_idx[None, :] * 4032000 + c_idx[None, :] * 31500 + hw_off[:, None] * 1
+    tl.store(out_ptr0 + store_idx, vals, mask=mask2d)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +75,7 @@

     ynumel = 4096
     xnumel = 31500
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
