# Kernel Benchmarks

This directory contains integration of standard GPU kernel benchmarks for validating and measuring the kernel optimization tool's performance.

## Benchmarks Included

### 1. KernelBench (ScalingIntelligence/Stanford)
**Location:** `kernelbench/`
**Dataset:** https://huggingface.co/datasets/ScalingIntelligence/KernelBench
**Paper:** KernelBench: Can LLMs Write GPU Kernels? (2024)

| Property | Value |
|----------|-------|
| Problems | 270 (L1:100, L2:100, L3:50, L4:20) |
| Source | Curated + models |
| Metric | fast_p |
| Precision | FP32/FP16/BF16 |
| Status | **Set up and verified** |

Covers 100 different neural network operation types with progressive difficulty.

### 2. SOL-ExecBench (NVIDIA)
**Location:** `sol_execbench/`
**Dataset:** https://huggingface.co/datasets/nvidia/SOL-ExecBench
**Paper:** [arXiv:2603.19173](https://arxiv.org/abs/2603.19173)
**Leaderboard:** https://research.nvidia.com/benchmarks/sol-execbench

| Property | Value |
|----------|-------|
| Problems | 235 (FlashInfer:26, L1:94, L2:82, Quant:33) |
| Source | Model subgraphs from 124 production models |
| Metric | SOL score (vs hardware speed-of-light) |
| Precision | BF16/FP8/NVFP4 |
| Status | **Set up and verified** |

Real-world CUDA kernel specifications evaluated against theoretical hardware limits using SOLAR-derived SOL bounds.

### 3. BackendBench (Meta/PyTorch)
**Location:** `backendbench/`
**Repo:** https://github.com/meta-pytorch/BackendBench
**Cloned to:** `/mnt/podman_storage/ahpoddar/home/kernelopt/backendbench`

| Property | Value |
|----------|-------|
| Problems | 271 PyTorch OpInfo operators |
| Source | PyTorch OpInfo + FACTO test suites |
| Metric | Speedup + Correctness |
| Precision | FP32/FP16/BF16 |
| Status | **Set up (runtime-discovery benchmark)** |

Evaluates LLM-generated Triton kernels for PyTorch backend operators using PyTorch's own OpInfo and FACTO correctness test suites, plus TorchBench performance traces.

### 4. ComputeEval (NVIDIA)
**Location:** `computeeval/`
**Dataset:** https://huggingface.co/datasets/nvidia/compute-eval
**Repo:** https://github.com/NVIDIA/compute-eval
**Blog:** https://developer.nvidia.com/blog/benchmarking-llms-on-ai-generated-cuda-code-with-computeeval-2025-2/

| Property | Value |
|----------|-------|
| Problems | 566 (2026.1 release) |
| Source | Handcrafted by NVIDIA engineers |
| Metric | pass@1 + performance benchmarks |
| Precision | FP32/FP16 |
| Status | **Set up (566 problems extracted)** |

Groups: cuda-kernels (114), cudnn (113), mathlibs (101), cublas (81), cccl (80), cutile (48), cuda-runtime (29).

### 5. CUDABench
**Location:** `cudabench/`
**Repo:** https://github.com/CUDA-Bench/CUDABench
**Paper:** [arXiv:2603.02236](https://arxiv.org/abs/2603.02236)

| Property | Value |
|----------|-------|
| Problems | 500 tasks (1,500 prompts across 3 difficulty levels) |
| Source | Open-source CUDA codebases |
| Metric | Roofline-based Performance-Score |
| Precision | FP32/FP16/BF16 |
| Status | **Set up (500 tasks extracted)** |

Covers AI, scientific computing, data analytics, signal processing, graphics, and simulation domains.

### 6. TritonBench (Meta/PyTorch)
**Location:** (external) `/mnt/podman_storage/ahpoddar/home/kernelopt/tritonbench`
**Repo:** https://github.com/pytorch/tritonbench

| Property | Value |
|----------|-------|
| Problems | 53 operators (184 benchmark configs per paper) |
| Source | GitHub repos |
| Metric | Correctness + Performance |
| Precision | FP32/FP16/BF16 |
| Status | **Repo cloned (not integrated into test suite)** |

### 7. Additional Resources
- **Helion Tutorials:** `helion_tutorials/` (10 examples)
- **Triton Tutorials:** `triton_tutorials/` (10 examples)
- **KernelBook:** `kernelbook/` (PyTorch + Triton reference kernels)

## Benchmark Comparison (from SOL-ExecBench paper)

| Benchmark | Problems | Source | Metric | Precision | HW Target | Train |
|-----------|----------|--------|--------|-----------|-----------|-------|
| KernelBench | 270 | Curated + models | fast_p | FP32/FP16/BF16 | Any | No |
| FlashInfer-Bench | 26 | Inference traces | fast_p | FP16/BF16/FP8 | BW+ | No |
| BackendBench | 271 | PyTorch OpInfo | Speedup | FP32/FP16/BF16 | Any | Partial |
| TritonBench | 184 | GitHub repos | Correctness+ | FP32/FP16/BF16 | Any | No |
| ComputeEval | 566 | NVIDIA engineers | pass@1 | FP32/FP16 | Any | No |
| CUDABench | 500 | Open-source | Roofline | FP32/FP16/BF16 | Any | No |
| SOL-ExecBench | 235 | Model subgraphs | SOL score | BF16/FP8/NVFP4 | BW+ | Yes |

## Directory Structure

```
benchmarks/
├── README.md
├── download_datasets.py
├── kernelbench/            # 270 problems (L1-L4)
│   ├── level_1/            # 100 single operations
│   ├── level_2/            # 100 operator fusions
│   ├── level_3/            # 50 model architectures
│   └── level_4/            # 20 aspirational tasks
├── sol_execbench/          # 235 problems
│   ├── FlashInfer-Bench/train/  # 26 FlashInfer kernels
│   ├── L1/train/           # 94 single-kernel ops
│   ├── L2/train/           # 82 fused/backward ops
│   └── Quant/train/        # 33 quantized kernels
├── backendbench/           # 271 ops (runtime-discovery)
│   └── config.json         # Benchmark config + repo reference
├── computeeval/            # 566 problems
│   ├── cuda-kernels/       # 114 problems
│   ├── cudnn/              # 113 problems
│   ├── mathlibs/           # 101 problems
│   ├── cublas/             # 81 problems
│   ├── cccl/               # 80 problems
│   ├── cutile/             # 48 problems
│   └── cuda-runtime/       # 29 problems
├── cudabench/              # 500 tasks
│   └── tasks/              # 500 individual task JSONs
├── helion_tutorials/       # 10 Helion examples
├── triton_tutorials/       # 10 Triton examples
└── kernelbook/             # Reference kernel implementations
    ├── pytorch/
    └── triton/
```

## Quick Start

```bash
pip install datasets huggingface-hub

# Download/refresh all datasets
python download_datasets.py
```

## Performance Expectations

### Runtime Estimates (T=3, N=4, single GPU)
- **Single kernel:** ~40-50 minutes
- **KernelBench Level 1 (100 kernels):** ~75 hours
- **Full KernelBench (270 kernels):** ~200 hours
- **SOL-ExecBench (235 kernels):** ~175 hours

### With 8x H200 GPUs (parallel)
- **KernelBench Level 1:** ~9.4 hours
- **Full KernelBench:** ~25 hours
- **SOL-ExecBench:** ~22 hours

## Citations

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
@misc{lin2026solexecbench,
  title={SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits},
  author={Edward Lin and Sahil Modi and others},
  year={2026},
  eprint={2603.19173},
  archivePrefix={arXiv}
}
```

### BackendBench
```bibtex
@misc{saroufim2025backendbench,
  title={BackendBench: Benchmarking LLM-Generated Triton Kernels for PyTorch Backend Operators},
  author={Mark Saroufim and others},
  year={2025},
  url={https://github.com/meta-pytorch/BackendBench}
}
```

### ComputeEval
```bibtex
@misc{nvidia2026computeeval,
  title={ComputeEval: Benchmarking LLMs on AI-Generated CUDA Code},
  author={NVIDIA},
  year={2026},
  url={https://github.com/NVIDIA/compute-eval}
}
```

### CUDABench
```bibtex
@article{zhu2026cudabench,
  title={CUDABench: Benchmarking LLMs for Text-to-CUDA Generation},
  author={Jiace Zhu and Wentao Chen and others},
  journal={arXiv preprint arXiv:2603.02236},
  year={2026}
}
```
