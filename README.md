<h1 align="center">KernelOPT</h1>
<p align="center">Automatic GPU kernel optimization for PyTorch models</p>

<p align="center">
  <a href="https://github.com/TorchedHat/KernelOPT/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
</p>

---

## Demo

> End-to-end optimization run on a KernelBench kernel (1.75x playback speed)

https://github.com/user-attachments/assets/6710ff29-b44e-4a66-b4b8-1d04da69b847

---

## About

KernelOPT optimizes GPU kernels through profiling-guided LLM code transformations. Give it a PyTorch model, a standalone Triton kernel, a CUDA file, or a Helion kernel — it profiles with NVIDIA Nsight Compute, figures out what's slow, rewrites the code, checks correctness, and repeats until it can't find further improvements.

Five specialized agents (profiler, planner, executor, correctness verifier, summarizer) run in a closed feedback loop. Every optimization is verified for numerical correctness and end-to-end model performance before being accepted — no false speedups.

### Inputs

| Input | How it works |
|-------|-------------|
| **PyTorch model** (`.py` with `nn.Module`) | Runs `torch.compile` → extracts Triton kernels from TorchInductor → optimizes each kernel → re-stitches into the model |
| **Triton kernel** (`.py` with `@triton.jit`) | Profiles and optimizes directly |
| **CUDA kernel** (`.cu`) | Compiles with nvcc, profiles with NCU, optimizes |
| **Helion kernel** (`.hl` / `@helion.kernel`) | Profiles and optimizes directly |

### What it does

- Profiles GPU kernels using NCU — memory throughput, compute utilization, occupancy, bottleneck rules
- Generates optimization plans grounded in profiling data, not blind code generation
- Applies transformations: autotuning, tiling, precision casting, memory coalescing, algorithmic restructuring
- Verifies every change: element-wise correctness, float64 reference disambiguation, model-level weight-extracted verification, wall-clock performance gating
- Learns from experience: four-layer memory system carries optimization knowledge across runs
- Searches broadly: beam search over the transformation space avoids premature convergence

### Hardware

Any NVIDIA GPU with compute capability ≥ 6.0. Tested on H100 and H200.

---

## Getting Started

### Installation

```bash
git clone https://github.com/TorchedHat/KernelOPT.git
cd KernelOPT/kernelopt
pip install -e ".[triton]"
```

**Requirements:** Python 3.10+, NVIDIA GPU, `ncu` (Nsight Compute) on PATH, Google Cloud project with Vertex AI API enabled.

### Setup

```bash
# Interactive setup — configures Vertex AI project, region, and authentication
kernelopt setup

# Verify your environment
kernelopt doctor
```

Or set environment variables manually:

```bash
export ANTHROPIC_VERTEX_PROJECT_ID=your-project-id
export VERTEX_REGION=us-east5
```

### Running an optimization

```bash
# PyTorch model — extracts and optimizes all Triton kernels
CUDA_VISIBLE_DEVICES=0 kernelopt run model.py --multi-kernel

# Standalone Triton kernel
CUDA_VISIBLE_DEVICES=0 kernelopt run softmax_triton.py -T 3 -N 4

# CUDA kernel
CUDA_VISIBLE_DEVICES=0 kernelopt run matmul.cu -T 3 -N 4

# With full configuration
CUDA_VISIBLE_DEVICES=0 kernelopt run model.py \
    -T 5 -N 4 -K 4 \
    --search-policy beam --beam-width 4 \
    --e2e-retries 3 \
    --e2e-perf-gate --e2e-perf-margin 1.03 \
    --multi-kernel
```

The output directory contains the optimized model (`final_model.py`), profiling artifacts, optimization logs, and a detailed markdown report.

---

## Usage

### Commands

```bash
kernelopt run <model.py>          # Full optimization pipeline
kernelopt profile <kernel.py>     # Single NCU profiling run
kernelopt show <ncu.csv>          # Pretty-print profiling results
kernelopt plan <ncu.csv> <kernel> # Generate plans without executing
kernelopt setup                   # Interactive environment setup
kernelopt doctor                  # Verify environment health
```

### Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `-T` | 5 | Optimization iterations |
| `-N` | 4 | Plans per iteration |
| `-K` | 1 | Executor attempts per plan (with in-conversation retry) |
| `--beam-width` | 1 | Concurrent optimization chains |
| `--search-policy` | `greedy` | Search strategy (`beam` or `greedy`) |
| `--framework` | `auto` | Input framework: `auto`, `triton`, `helion`, `cuda`, or `pytorch` |
| `--multi-kernel` / `--single-kernel` | on | Multi-kernel: trace PyTorch model → extract all Inductor kernels → optimize each → re-stitch. Use `--single-kernel` for standalone Triton/Helion/CUDA files |
| `--e2e-perf-gate` / `--no-e2e-perf-gate` | on | Reject optimizations that slow down the full model |
| `--e2e-perf-margin` | 1.03 | Performance noise tolerance (3%) |
| `--e2e-retries` | 2 | Max retries on e2e correctness or performance failure |
| `--e2e-correctness-timeout` | 180 | Seconds for e2e correctness subprocess |
| `-o` | `pipeline_output/` | Output directory |
| `--no-run-subdir` | off | Write directly to `-o` path (no `run_<timestamp>/` subfolder) |
| `--checkpoint` | — | Resume from a previous run directory |
| `--hud` | off | Full-screen terminal dashboard with live system metrics |
| `--model` | `claude-sonnet-4-6` | Default LLM model for all agents |
| `--planner-model` | — | Override model for planner agent |
| `--executor-model` | — | Override model for executor agent |
| `--profiler-model` | — | Override model for profiler config agent |
| `--summarizer-model` | — | Override model for summarizer agent |
| `--kernel-name` | auto | Regex filter for NCU `--kernel-name` |
| `--kernel-args` | — | Extra args passed to kernel program (e.g. `--profile-only`) |
| `--replay-mode` | `kernel` | NCU replay: `kernel` (fast) or `application` (lower memory) |
| `--entry-point` | auto | Callable to compile: `ClassName`, `ClassName.method`, or `function_name` |
| `--input-shapes` | — | Tensor shapes for synthesis, e.g. `"(8,512) (8,512,512)"` |
| `--verbose` | off | Verbose output |
| `--debug` | off | Debug prints (profiler config, correctness subprocess) |

### Resuming a run

```bash
# Start a run
kernelopt run model.py -T 5 -N 4 -o output/

# Resume if interrupted (continues from last checkpoint)
kernelopt run model.py --checkpoint output/run_20260625_120000/

# Extend with more iterations
kernelopt run model.py -T 10 --checkpoint output/run_20260625_120000/
```

### GPU selection

```bash
# Use a specific GPU
CUDA_VISIBLE_DEVICES=0 kernelopt run model.py

# Check available GPUs
nvidia-smi
```

---

## How It Works

### Pipeline

```
PyTorch model
  → torch.compile (Inductor) → Triton kernel extraction
  → NCU baseline profiling
  → Strategy analysis (memory-bound / compute-bound / underutilized)
  → Iterative optimization:
      Planner → Executor → Profiler → Summarizer → Beam selection
  → E2E verification at re-stitch time
  → Optimized model
```

### Agents

| Agent | Role |
|-------|------|
| **Profiler** | Configures and runs NCU, parses metrics into structured context |
| **Planner** | Generates diverse optimization plans from NCU bottleneck analysis |
| **Executor** | Implements plans with bounded retries and 6-stage static validation |
| **Correctness** | Verifies numerical equivalence across multiple random seeds |
| **Summarizer** | Extracts generalizable optimization insights for cross-run learning |

### Verification

Every accepted optimization passes four independent gates:

1. **Executor validation** — signature preservation, mask consistency, API validity, codegen checks
2. **Element-wise correctness** — strict allclose with float64 reference disambiguation
3. **Model-level verification** — real model weights via `torch.compile` capture backend
4. **Performance gate** — wall-clock timing of re-stitched model vs baseline

### Memory system

KernelOPT maintains four layers of optimization knowledge:

1. **Session memory** — bounded queue of recent experiences with asymmetric curation
2. **Cross-run strategy records** — success/failure rates per kernel type and strategy
3. **Technique library** — 14 hardware-aware GPU optimization patterns with applicability conditions
4. **Semantic memory bank** — embedding-based retrieval with automatic constraint extraction

---

## Project Structure

```
kernelopt/
├── kernelopt/
│   ├── cli.py                    # CLI entry point
│   ├── langgraph_pipeline.py     # Optimization pipeline (LangGraph)
│   ├── multi_kernel.py           # Multi-kernel orchestration + e2e gates
│   ├── state.py                  # Config and state
│   ├── agents/                   # LLM agents (planner, executor, profiler, summarizer)
│   ├── synthesis/                # torch.compile → Triton extraction
│   ├── fusion/                   # Cross-kernel pattern matching
│   └── functions/                # Benchmarking, correctness, profiling utilities
├── scripts/                      # Evaluation and benchmark scripts
└── tests_suite/
    ├── unit/                     # Unit tests
    ├── benchmarks/               # KernelBench, Triton tutorials, Helion tutorials
    └── fixtures/                 # Test data
```

---

## Contributing

We welcome contributions. Please open an issue to discuss before submitting a PR.

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests_suite/unit/
```

## License

KernelOPT is released under the [Apache 2.0 License](LICENSE).
