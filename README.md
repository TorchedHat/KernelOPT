<h1 align="center">
  <img src="assets/kernelopt-mark.png" width="40" height="40" align="absmiddle" alt="">&nbsp; KernelOPT
</h1>
<p align="center">Dispatch-Aware Agentic Search for GPU Kernel Optimization</p>

<p align="center">
  <a href="https://github.com/TorchedHat/KernelOPT/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://triton-lang.org/"><img src="https://img.shields.io/badge/Triton-3.0%2B-7B68EE.svg" alt="Triton"></a>
  <a href="https://developer.nvidia.com/cuda-gpus"><img src="https://img.shields.io/badge/GPU-NVIDIA%20CC%20%E2%89%A5%206.0-76B900.svg?logo=nvidia&logoColor=white" alt="GPU"></a>
  <a href="https://arxiv.org/abs/2609.30059"><img src="https://img.shields.io/badge/arXiv-2609.30059-b31b1b.svg" alt="arXiv"></a>

</p>

---

## Demo

> End-to-end optimization run on KernelBench L1 014 kernel (playback speedup)



https://github.com/user-attachments/assets/fe1b6a07-c6c4-4ddc-aac7-c077f676c4b3



---

## About

KernelOPT is a multi-agent LLM system that optimizes GPU kernels through profiling-guided code transformations. Give it a PyTorch model, a standalone Triton kernel, a CUDA file, or a Helion kernel. It profiles with NVIDIA Nsight Compute, identifies bottlenecks, generates targeted optimization plans, applies transformations, verifies correctness, and iterates until convergence.

Five specialized agents (profiler, planner, executor, correctness verifier, summarizer) operate in a closed feedback loop with beam search over the optimization space. Every accepted change passes four independent verification gates. The result is either a faster kernel or a guaranteed-safe fallback to the `torch.compile` baseline.

### Key properties

- **Profiling-grounded** --- optimization plans are derived from NCU metrics (memory throughput, compute utilization, occupancy), not blind code generation
- **Dispatch-aware** --- detects when Inductor dispatches to cuBLAS/cuDNN and preserves those calls instead of replacing them with slower Triton
- **Zero regressions** --- four-gate verification (static analysis, element-wise correctness, float64 reference, E2E performance gate) ensures no shipped result is slower than the baseline
- **Multi-framework** --- accepts PyTorch models, Triton kernels, and Helion kernels as input
- **Pluggable LLM backend** --- works with Vertex AI (default), Anthropic API, OpenAI, or any OpenAI-compatible local endpoint

### Inputs

| Input | How it works |
|-------|-------------|
| **PyTorch model** (`.py` with `nn.Module`) | Traces through `torch.compile` / TorchInductor, extracts generated Triton kernels, optimizes each, re-stitches into the model with E2E verification |
| **Triton kernel** (`.py` with `@triton.jit`) | Profiles with NCU and optimizes the Triton source directly |
| **Helion kernel** (`.py` with `@helion.kernel`) | Profiles and optimizes directly |

---

## Results

Evaluated on [KernelBench](https://huggingface.co/datasets/ScalingIntelligence/KernelBench) (250 kernels across 3 difficulty levels). All results on NVIDIA H200, FP32, versus `torch.compile` baseline, without compiler context (`--use-compiler-context` off).

| Level | Kernels | Optimized | Matched | Fallback | Regressions | Geomean (all 100) | Geomean (optimized only) | Max |
|-------|---------|-----------|---------|----------|-------------|-------------------|--------------------------|-----|
| **L1** (single ops) | 100 | 51 | 34 | 15 | **0** | 1.40x | **1.94x** | 88.63x |
| **L2** (fused ops) | 100 | 31 | 35 | 34 | **0** | 1.15x | **1.61x** | 33.11x |
| **L3** (architectures) | 50 | 12 | 3 | 35 | **0** | 1.07x | **1.47x** | 4.12x |

- **0 regressions** across all 250 kernels --- every result either improves or safely falls back to `torch.compile`
- "Geomean (all)" includes matched (1.0x) and fallback (1.0x) kernels; "Geomean (optimized only)" counts only kernels where KernelOPT found an improvement
- L2/L3 fallbacks are primarily GEMM/Conv-dominant kernels where cuBLAS/cuDNN are faster than any Triton alternative
- Also evaluated on Triton Tutorials (10 kernels, best 1.70x) and Helion Tutorials (10 kernels, best 1.12x)

---

## Getting Started

### Installation

```bash
git clone https://github.com/TorchedHat/KernelOPT.git
cd KernelOPT/kernelopt
pip install -e ".[triton]"
```

Optional extras:

```bash
pip install -e ".[semantic]"        # Embedding-based memory retrieval
pip install -e ".[local]"           # OpenAI-compatible API backend
pip install -e ".[all-providers]"   # All LLM backends
```

**Requirements:** Python 3.10+, NVIDIA GPU (compute capability >= 6.0), `ncu` (Nsight Compute CLI) on PATH.

### LLM Backend Setup

KernelOPT supports multiple LLM providers. Configure one:

```bash
# Option 1: Vertex AI (default)
kernelopt setup  # Interactive setup
# Or manually:
export ANTHROPIC_VERTEX_PROJECT_ID=your-project-id
export VERTEX_REGION=us-east5

# Option 2: Anthropic API
export ANTHROPIC_API_KEY=your-key
kernelopt run model.py --backend anthropic

# Option 3: OpenAI-compatible endpoint
kernelopt run model.py --backend openai --api-base http://localhost:8000/v1 --api-key your-key
```

Verify your environment:

```bash
kernelopt doctor
```

### Running an optimization

```bash
# PyTorch model --- extracts and optimizes all Triton kernels
CUDA_VISIBLE_DEVICES=0 kernelopt run model.py

# Standalone Triton kernel
CUDA_VISIBLE_DEVICES=0 kernelopt run softmax_triton.py --framework triton -T 3 -N 4

# Full configuration with beam search
CUDA_VISIBLE_DEVICES=0 kernelopt run model.py \
    -T 5 -N 4 -K 4 \
    --search-policy beam --beam-width 4 \
    --e2e-retries 3 \
    --e2e-perf-gate --e2e-perf-margin 1.03
```

The output directory contains the optimized model (`final_model.py`), per-kernel profiling artifacts (`.ncu-rep`, `.csv`), optimization logs, and a detailed markdown report.

---

## Usage

### Commands

```bash
kernelopt run <kernel.py>           # Full optimization pipeline
kernelopt profile <kernel.py>       # Single NCU profiling run
kernelopt show <ncu.csv>            # Pretty-print profiling results
kernelopt plan <ncu.csv> <kernel>   # Generate plans without executing
kernelopt setup                     # Interactive environment setup
kernelopt doctor                    # Verify environment health
```

### Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `-T` | 5 | Optimization iterations |
| `-N` | 4 | Plans per iteration |
| `-K` | 1 | Executor attempts per plan |
| `--beam-width` | 1 | Concurrent optimization chains |
| `--search-policy` | `greedy` | `greedy`, `beam`, `top_k`, or `ucb` (UCB1 with c=1.4) |
| `--framework` | `auto` | `auto`, `triton`, `helion`, or `pytorch` |
| `--backend` | `vertex` | LLM backend: `vertex`, `anthropic`, `openai` |
| `--multi-kernel` / `--single-kernel` | on | Multi-kernel mode for PyTorch models |
| `--e2e-perf-gate` | on | Reject optimizations that slow down the full model |
| `--e2e-perf-margin` | 1.03 | Performance noise tolerance (3%) |
| `--e2e-retries` | 2 | Retries on E2E correctness or performance failure |
| `--use-compiler-context` | off | Inject Inductor IR/fusion metadata into LLM prompts |
| `-o` | `pipeline_output/` | Output directory |
| `--checkpoint` | --- | Resume from a previous run |
| `--hud` | off | Full-screen terminal dashboard |
| `--model` | `claude-sonnet-4-6` | Default LLM model |
| `--planner-model` | --- | Override model for planner agent |
| `--executor-model` | --- | Override model for executor agent |

### Resuming a run

```bash
kernelopt run model.py -T 5 -N 4 -o output/
kernelopt run model.py --checkpoint output/run_20260625_120000/
```

---

## How It Works

### Pipeline

```
Input (PyTorch / Triton / Helion)
  --> torch.compile (Inductor) --> Triton kernel extraction [PyTorch only]
  --> NCU baseline profiling
  --> Strategy analysis (memory-bound / compute-bound / launch-bound)
  --> Iterative optimization loop:
        Planner --> Executor --> Profiler --> Summarizer
        --> Beam selection (greedy / top-k / UCB1)
  --> E2E verification + performance gate
  --> Optimized kernel or safe fallback
```

### Agents

| Agent | Role |
|-------|------|
| **Profiler** | Configures NCU, parses 40+ hardware metrics into structured bottleneck analysis |
| **Planner** | Generates diverse optimization plans grounded in NCU data and cross-run memory |
| **Executor** | Implements plans with 6-stage static validation and bounded retries |
| **Correctness** | Verifies numerical equivalence across multiple random seeds with float64 disambiguation |
| **Summarizer** | Extracts optimization insights with deduplication for cross-run learning |

### Verification gates

Every accepted optimization passes four independent checks:

1. **Static validation** --- signature preservation, mask consistency, API validity, codegen checks
2. **Element-wise correctness** --- `allclose(rtol=1e-3, atol=1e-3)` with float64 reference when needed
3. **Model-level verification** --- real model weights via `torch.compile` capture backend
4. **Performance gate** --- E2E wall-clock timing; reject if optimized > baseline * 1.03

### Dispatch-aware synthesis

When tracing a PyTorch model, KernelOPT detects Inductor's dispatch decisions:

- **cuBLAS/cuDNN calls** (`extern_kernels.mm`, `extern_kernels.convolution`) are preserved, not replaced with Triton
- Only the surrounding operations (epilogues, fusions, reductions) are synthesized as Triton
- If the LLM ignores this guidance and writes a full Triton GEMM, the E2E performance gate catches the regression

### Memory system

Four layers of optimization knowledge:

1. **Session memory** --- bounded queue of recent experiences with asymmetric success/failure curation
2. **Cross-run strategy records** --- success/failure rates per kernel type and strategy
3. **Technique library** --- 14 hardware-aware GPU optimization patterns with applicability conditions
4. **Semantic memory bank** --- embedding-based retrieval with automatic constraint extraction

---

## `kernelopt.compiler`

Standalone subpackage for extracting structured metadata from GPU compilation pipelines. Usable independently of the optimization loop.

### Extraction

Hooks into PyTorch's compilation pipeline to capture:

- **IR nodes** --- buffer shapes, dtypes, read/write dependencies, reduction metadata
- **Fusion groups** --- pre/post-fusion node mapping, which ops fused and which didn't (WhyNoFuse reasons)
- **Decomposition paths** --- how high-level ops decompose into primitives
- **Codegen strategies** --- tile sizes, reduction types chosen by Inductor
- **Triton compilation** --- JIT metadata, cache behavior

```python
from kernelopt.compiler.extraction.subprocess_script import build_extraction_script

# Generates a self-contained script that traces compilation and dumps structured context
script = build_extraction_script(model_path="model.py")
```

### Custom passes

- **`hook_installer`** --- install/uninstall Inductor hooks (pre/post-grad, fusion, WhyNoFuse, Triton listener)
- **`ProfileGuidedChoices`** --- subclass InductorChoices to inject profiling-derived tile sizes and reduction strategies
- **`fusion_patterns`** --- detect common fusion patterns (norm chains, GEMM epilogues) from extracted IR
- **`template_registration`** --- register optimized Triton kernels as ExternalTritonTemplateKernel for Inductor reuse

### Status

The compiler subpackage is functional and usable standalone for compiler analysis. Its integration into the optimization loop (`--use-compiler-context`) is experimental. Preliminary ablation (V6, single-run, 30 kernels) showed marginal benefit on reduction/norm kernels (e.g. RMSNorm: 1.73x with context vs 1.25x without) but no benefit on GEMM-dominant compound kernels. The benchmark results in this README were obtained without compiler context. Further validation with multiple runs is needed before this can be recommended as a default.

### Cross-framework support

Experimental extraction for JAX/XLA (stages API, HLO fusion dump) and StableHLO analysis via torch-mlir.

---

## Project Structure

```
kernelopt/
  kernelopt/
    cli.py                          # CLI entry point (Typer)
    langgraph_pipeline.py           # LangGraph optimization pipeline
    multi_kernel.py                 # Multi-kernel orchestration + E2E gates
    state.py                        # Pipeline config and state
    llm.py                          # Pluggable LLM backend (Vertex/Anthropic/OpenAI)
    search_tree.py                  # Beam search with greedy/top_k/UCB selection
    agents/                         # LLM agents (planner, executor, profiler, summarizer)
    synthesis/                      # torch.compile --> Triton extraction + wrapping
      compile_extract.py            # Inductor code extraction
      graph_tracer.py               # Computation graph + extern kernel detection
      validate_eager.py             # Correctness validation
      wrap_kernel.py                # Standalone kernel wrapping
    compiler/                       # Standalone compiler analysis (kernelopt.compiler)
      context.py                    # CompilerContext dataclass
      extraction/                   # Dynamo, AOT, Inductor, Triton, XLA hooks
      passes/                       # Custom Inductor passes
      rendering/                    # Prompt section generation
    fusion/                         # Cross-kernel fusion pattern matching
    functions/                      # Benchmarking, profiling, correctness utilities
    data/                           # Optimization technique library
  tests_suite/
    unit/                           # Unit tests
    benchmarks/                     # KernelBench, SOL-ExecBench, Triton/Helion tutorials
    fixtures/                       # Test data and sample kernels
```

---

## Contributing

We welcome contributions. Please open an issue to discuss before submitting a PR.

```bash
pip install -e ".[dev]"
pytest tests_suite/unit/
```

## Acknowledgments

> [!NOTE]
> KernelOPT was developed during an internship with the PyTorch team at Red Hat.

## Citation

If you use KernelOPT in your research, please cite our paper:

```bibtex
@misc{poddar2026kerneloptdispatchawareagenticsearch,
      title={KernelOPT: Dispatch-Aware Agentic Search for GPU Kernel Optimization}, 
      author={Aheli Poddar and Sanskar Prasad and Arindam Samanta and Subha Chakraborty and Vishal Goyal and Rohit Singh Rathaur},
      year={2026},
      eprint={2609.30059},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={[https://arxiv.org/abs/2609.30059](https://arxiv.org/abs/2609.30059)}, 
}
```

## License

KernelOPT is released under the [Apache 2.0 License](LICENSE).
