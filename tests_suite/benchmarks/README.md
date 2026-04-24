# Kernel Benchmarks

This directory contains integration of standard GPU kernel benchmarks for validating and measuring the kernel optimization tool's performance.

## Benchmarks Included

### 1. KernelBench (ScalingIntelligence/Stanford)
**Location:** `kernelbench/`
**Dataset:** https://huggingface.co/datasets/ScalingIntelligence/KernelBench

**Description:**
- 270 GPU kernel problems across 4 difficulty levels
- Covers 100 different neural network operation types
- Progressive difficulty: Level 1 (100) → Level 2 (100) → Level 3 (50) → Level 4 (20)

**Categories:**
- Matrix operations (9 types)
- Tensor operations (9 types)
- Activation functions (14 types)
- Normalization layers (8 types)
- Pooling operations (6 types)
- Reduction operations (7 types)
- Convolution operations (37 types)
- Cumulative operations (5 types)
- Loss functions (6 types)
- Attention & special ops (3 types)

**Use Case:**
Tests the tool's ability to optimize diverse neural network operations across progressive difficulty levels.

### 2. SOL-ExecBench (NVIDIA)
**Location:** `sol_execbench/`
**Dataset:** https://huggingface.co/datasets/nvidia/SOL-ExecBench

**Description:**
- 235 real-world CUDA kernel specifications from HuggingFace production models
- Evaluates performance against theoretical hardware "Speed of Light" limits
- Includes text, vision, and speech model workloads

**Composition:**
- 26 FlashInfer-Bench kernels
- 94 Level 1 (L1) kernels
- 82 Level 2 (L2) kernels
- 33 Quantized kernels (FP32, BF16, FP16, FP8, NVFP4)

**Categories:**
- Fused operations (Add + RMSNorm)
- GEMM operations (various N×K configurations)
- Grouped Query Attention (GQA)
- Matrix multiplications
- Attention mechanisms

**Use Case:**
Validates optimization quality against theoretical hardware limits and real production workloads.

## Why These Benchmarks Matter

### For the Kernel Profiling Tool:

1. **Validation:** Prove optimization strategies work on industry-standard benchmarks
2. **Coverage:** Test across diverse kernel types (GEMM, attention, fused ops, quantized)
3. **Performance Baseline:** Compare against theoretical limits (SOL metrics)
4. **Regression Testing:** Ensure optimizations don't break on edge cases
5. **Real-World Relevance:** Test on actual production model kernels

### Integration Value:

- **KernelBench** validates breadth (100 operation types, progressive difficulty)
- **SOL-ExecBench** validates depth (real production kernels, theoretical limits)
- Combined: comprehensive evaluation of optimization pipeline

## Directory Structure

```
benchmarks/
├── README.md                    # This file
├── kernelbench/
│   ├── README.md               # KernelBench documentation
│   ├── loader.py               # Dataset loader
│   ├── kernels/                # Downloaded kernel implementations
│   ├── test_kernelbench.py    # Integration tests
│   └── run_benchmark.py       # Benchmark runner
└── sol_execbench/
    ├── README.md               # SOL-ExecBench documentation
    ├── loader.py               # Dataset loader
    ├── kernels/                # Downloaded kernel specifications
    ├── test_sol_execbench.py  # Integration tests
    └── run_benchmark.py       # Benchmark runner
```

## Quick Start

### Install Dependencies
```bash
pip install datasets huggingface-hub
```

### Download KernelBench
```bash
cd benchmarks/kernelbench
python loader.py --download
```

### Download SOL-ExecBench
```bash
cd benchmarks/sol_execbench
python loader.py --download
```

### Run Benchmarks
```bash
# Run single kernel from KernelBench
python benchmarks/kernelbench/run_benchmark.py --kernel matmul_square --gpu 0

# Run full KernelBench Level 1
python benchmarks/kernelbench/run_benchmark.py --level 1 --gpu 0

# Run SOL-ExecBench
python benchmarks/sol_execbench/run_benchmark.py --gpu 0
```

## Performance Expectations

### Runtime Estimates (T=3, N=4, single GPU)
- **Single kernel:** ~40-50 minutes
- **KernelBench Level 1 (100 kernels):** ~75 hours
- **Full KernelBench (270 kernels):** ~200 hours
- **SOL-ExecBench (235 kernels):** ~175 hours

### With 8× H200 GPUs (parallel)
- **KernelBench Level 1:** ~9.4 hours
- **Full KernelBench:** ~25 hours
- **SOL-ExecBench:** ~22 hours

## Citation

### KernelBench
```bibtex
@misc{ouyang2024kernelbench,
  title={KernelBench: Can LLMs Write GPU Kernels?},
  author={Anne Ouyang and Simon Guo and Azalia Mirhoseini},
  year={2024},
  url={https://scalingintelligence.stanford.edu/blogs/kernelbench/}
}
```

### SOL-ExecBench
```bibtex
@article{sol-execbench2026,
  title={SOL-ExecBench: Speed-of-Light Kernel Execution Benchmark},
  author={NVIDIA Research},
  year={2026},
  url={https://research.nvidia.com/benchmarks/sol-execbench}
}
```
