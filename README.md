# kernelopt

Multi-agent GPU kernel optimizer powered by LangGraph and NVIDIA Nsight Compute (NCU). Automatically profiles, plans, rewrites, and re-profiles GPU kernels in a closed loop to find faster implementations.

## How It Works

```
                    +---> Planner (x N) ---> Executor (x N) ---+
                    |                                          |
Kernel --> Profiler-+                                          +--> Profiler --> Summarizer --> Select Best
                    |                                          |
                    +<---- loop (T iterations) <---------------+
```

Each iteration:

1. **Profiler** runs NCU on the kernel, parses the CSV into a structured `profiling_context` (metrics, rules, launch config)
2. **Planner** reads profiling data + optimization memory, generates N diverse optimization plans ranked by NCU rule speedup potential
3. **Executor** applies each plan to produce optimized kernel code, verified by a correctness gate
4. **Profiler** re-profiles valid candidates with NCU
5. **Summarizer** compares before/after profiling, extracts reusable experience items into memory
6. **Select Best** picks the fastest candidate for the next iteration

The pipeline stops after T iterations, 2 consecutive no-improvement iterations, or when all NCU optimization opportunities are exhausted.

## Supported Frameworks

- **Triton** -- tile sizes, num_warps, num_stages, `tl.math.*` substitutions
- **CUDA** -- shared memory tiling, vectorized loads, `__launch_bounds__`, instruction fusion
- **Helion** -- tile hierarchy, memory access ordering

Framework is auto-detected from the NCU kernel name.

## Installation

```bash
git clone https://gitlab.com/Sanskaaar1/kernel-profiling-tool.git
cd kernel-profiling-tool/kernelopt

# Core install
pip install -e .

# With TorchServe backend (batch inference on H200)
pip install -e ".[torchserve]"

# With local vLLM backend
pip install -e ".[local]"
```

**Requirements:**
- Python >= 3.10
- NVIDIA GPU with CUDA
- [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) (`ncu` on PATH)
- An LLM backend (TorchServe, vLLM, or any OpenAI-compatible API)

## Quick Start

### Run the full pipeline

```bash
# Using TorchServe (auto-starts servers, auto-shuts down)
kernelopt run my_kernel.py --torchserve --gpu 2 -T 4 -N 4

# Using a local vLLM server (must be running separately)
kernelopt run my_kernel.py --local -T 4 -N 4

# Using any OpenAI-compatible endpoint
kernelopt run my_kernel.py --base-url http://localhost:8000/v1 --model Qwen/Qwen3-30B-A3B
```

### Direct inference (no server)

```bash
# Uses transformers directly on GPU -- slower but no server setup needed
CUDA_VISIBLE_DEVICES=2 python test/run_pipeline.py -T 4 -N 4 --kernel test/softmaxtri.py
```

### Profile only

```bash
kernelopt profile my_kernel.py --kernel-name 'softmax_kernel' --kernel-args '--profile-only'
```

### Inspect a profile

```bash
kernelopt show output.csv
```

### Generate plans without executing

```bash
kernelopt plan output.csv my_kernel.py -N 4
```

### List kernels in a CSV

```bash
kernelopt kernels output.csv
```

## CLI Reference

### `kernelopt run`

| Flag | Default | Description |
|------|---------|-------------|
| `kernel_path` | *required* | Kernel file (.cu, .py, .hl) |
| `-T, --iterations` | 5 | Max optimization iterations |
| `-N, --plans` | 4 | Plans generated per iteration |
| `-K, --attempts` | 1 | Executor attempts per plan |
| `--top-k` | 4 | Max experience items per iteration |
| `-o, --output-dir` | pipeline_output | Output directory |
| `--framework` | auto | cuda\|triton\|helion\|auto |
| `--csv` | | Existing baseline NCU CSV (skip profiling) |
| `--memory` | | Load existing optimization_memory.json |
| `--kernel-name` | | Regex for NCU `--kernel-name` |
| `--kernel-args` | | Extra args for the kernel program |
| `--torchserve` | false | Use TorchServe + LangGraph backend |
| `--local` | false | Use dual vLLM servers |
| `--gpu` | 2 | GPU device for TorchServe models |
| `--planner-model` | | Override model for planner agent |
| `--executor-model` | | Override model for executor agent |
| `--summarizer-model` | | Override model for summarizer agent |

## Architecture

```
kernelopt/
  cli.py                  # Typer CLI entry point
  langgraph_pipeline.py   # LangGraph StateGraph (7 nodes + conditional loop)
  graph.py                # Legacy linear pipeline (still works)
  state.py                # PipelineState, PipelineConfig, GraphState
  llm.py                  # LLM factory (ChatOpenAI wrapper)
  memory.py               # Bounded experience queue (FIFO, ExpN=8)
  schemas.py              # Pydantic models for all data contracts
  agents/
    profiler.py           # NCU runner + CSV parser + profiling_context builder
    planner.py            # LLM agent: profiling data -> optimization plan
    executor.py           # LLM agent: plan + kernel -> optimized kernel
    summarizer.py         # LLM agent: before/after -> experience item
  functions/
    csv_parser.py         # NCU sections-based CSV parser
    correctness.py        # Subprocess correctness checker (.py and .cu)
    selector.py           # Best candidate selector
  serving/
    manager.py            # TorchServeManager context manager (start/stop)
    gateway.py            # FastAPI proxy: OpenAI format <-> TorchServe
    qwen3_handler.py      # TorchServe handler for Qwen3-30B-A3B (BF16)
    qwen3_coder_handler.py # TorchServe handler for Qwen3-Coder-Next (FP8)
    config.properties     # TorchServe server config
```

### LangGraph Pipeline

```
profile_baseline -> generate_plans -> execute_plans -> profile_candidates
                         ^                                     |
                   increment_iter                       summarize_results
                         ^                                     |
                   {should_continue} <---------------- select_best
                      | stop
                     END
```

- **generate_plans** and **summarize_results** use `ThreadPoolExecutor` for concurrent LLM calls (enables TorchServe batch inference)
- **execute_plans** and **profile_candidates** run sequentially (GPU-bound)

### Model Configuration

| Agent | Default Model | Purpose |
|-------|--------------|---------|
| Planner | Qwen/Qwen3-30B-A3B | Analyze profiling data, generate optimization hypotheses |
| Summarizer | Qwen/Qwen3-30B-A3B | Extract reusable experience from results |
| Executor | Qwen/Qwen3-Coder-Next-FP8 | Rewrite kernel code per plan |

### TorchServe Backend

When `--torchserve` is used, the CLI automatically:
1. Packages model archives (.mar)
2. Starts TorchServe on the specified GPU
3. Registers both models with batch inference config
4. Starts a FastAPI gateway on port 8000
5. Runs the LangGraph pipeline
6. Shuts everything down on exit

Both models fit on a single H200 (150GB): ~60GB BF16 + ~30GB FP8 = ~90GB with 60GB headroom for KV cache.

## Output

After a run, `pipeline_output/` contains:

| File | Description |
|------|-------------|
| `best_kernel.py` | Fastest kernel found |
| `optimization_log.md` | Markdown table of all iterations, speedups, and candidates |
| `optimization_memory.json` | Experience items for future runs (pass via `--memory`) |
| `*_iter{N}_plan{X}_*.py` | Each candidate kernel |
| `*_iter{N}_plan{X}_*.csv` | NCU CSV for each profiled candidate |
| `*_iter{N}_plan{X}_*.ncu-rep` | NCU report for each profiled candidate |
| `baseline.csv` / `baseline.ncu-rep` | Baseline profiling data |

## Memory System

The optimization memory is a bounded FIFO queue (capacity 8) of experience items. Each item captures:
- What profiling signal triggered the optimization
- What code change was made (as pseudocode)
- Whether it helped or hurt (positive/negative rewrite)
- When to apply and when not to apply

Memory persists across runs via `--memory optimization_memory.json`, enabling the planner to learn from past attempts.

## License

MIT
