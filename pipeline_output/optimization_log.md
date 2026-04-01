# Optimization Log

- **Kernel**: `test/flash_attention_triton.py`
- **Framework**: unknown
- **Baseline latency**: 303870 ns
- **Best latency**: 298240 ns
- **Overall speedup**: 1.02x
- **Iterations**: 4
- **Config**: T=4, N=4, K=1, TopK=4

## Iteration 1

| Direction | Speedup | Correctness | Kept | ncu-rep |
|-----------|---------|-------------|------|---------|
| fuse_floating_point_operations | 1.00x | passed |  | [link](pipeline_output/flash_attention_triton_iter1_planD_fuse_floating_point_operations.ncu-rep) |

## Iteration 2

| Direction | Speedup | Correctness | Kept | ncu-rep |
|-----------|---------|-------------|------|---------|
| increase_eligible_warps_per_scheduler | 1.01x | passed | Y | [link](pipeline_output/flash_attention_triton_iter2_planA_increase_eligible_warps_per_scheduler.ncu-rep) |
| increase_theoretical_occupancy | 0.95x | passed |  | [link](pipeline_output/flash_attention_triton_iter2_planB_increase_theoretical_occupancy.ncu-rep) |
| eliminate_partial_wave | 1.01x | passed |  | [link](pipeline_output/flash_attention_triton_iter2_planC_eliminate_partial_wave.ncu-rep) |
| fuse_fp32_operations | 1.00x | passed |  | [link](pipeline_output/flash_attention_triton_iter2_planD_fuse_fp32_operations.ncu-rep) |

## Iteration 3

| Direction | Speedup | Correctness | Kept | ncu-rep |
|-----------|---------|-------------|------|---------|
| increase_eligible_warps_per_scheduler | 0.46x | passed |  | [link](pipeline_output/flash_attention_triton_iter3_planA_increase_eligible_warps_per_scheduler.ncu-rep) |
| increase_theoretical_occupancy | 0.47x | passed |  | [link](pipeline_output/flash_attention_triton_iter3_planB_increase_theoretical_occupancy.ncu-rep) |
| fuse_floating_point_operations | 1.00x | passed | Y | [link](pipeline_output/flash_attention_triton_iter3_planD_fuse_floating_point_operations.ncu-rep) |

## Iteration 4

| Direction | Speedup | Correctness | Kept | ncu-rep |
|-----------|---------|-------------|------|---------|
| reduce_registers_per_thread | 1.00x | passed | Y | [link](pipeline_output/flash_attention_triton_iter4_planA_reduce_registers_per_thread.ncu-rep) |
| enable_l2_compression_optimization | 0.87x | passed |  | [link](pipeline_output/flash_attention_triton_iter4_planD_enable_l2_compression_optimization.ncu-rep) |

