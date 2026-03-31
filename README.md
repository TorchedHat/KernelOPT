# KernelOpt

Multi-agent GPU kernel optimizer powered by **Claude on Vertex AI**, **LangGraph**, and **NVIDIA Nsight Compute**. Automatically profiles, plans, rewrites, and re-profiles GPU kernels in a closed loop to find faster implementations.

## How It Works

```
                    +---> Planner (Claude) x N ---> Executor (Claude) x N ---+
                    |                                                        |
Kernel --> Profiler-+                                                        +--> Profiler --> Summarizer (Claude) --> Select Best
                    |                                                        |
                    +<---- loop (T iterations) <-----------------------------+
```

Each iteration:

1. **Profiler** runs NCU on the kernel, parses the CSV into a structured `profiling_context` (metrics, rules, launch config)
2. **Planner** (Claude) reads profiling data + optimization memory, generates N diverse optimization plans ranked by NCU rule speedup potential
3. **Executor** (Claude) applies each plan to produce optimized kernel code
4. **Correctness** (Claude) generates CPU reference implementation and validates GPU kernel outputs
5. **Profiler** re-profiles valid candidates with NCU
6. **Summarizer** (Claude) compares before/after profiling, extracts reusable experience items into memory
7. **Select Best** picks the fastest candidate for the next iteration

The pipeline stops after T iterations, 2 consecutive no-improvement iterations, or when all NCU optimization opportunities are exhausted.

## Supported Frameworks

- **Triton** — tile sizes, num_warps, num_stages, `tl.math.*` substitutions
- **CUDA** — shared memory tiling, vectorized loads, `__launch_bounds__`, instruction fusion
- **Helion** — tile hierarchy, memory access ordering

Framework is auto-detected from the NCU kernel name.

## Installation

```bash
cd /workspaces/pytorch-devcontainers/TOOL/kernelopt
pip install -e .
```

**Requirements:**
- Python >= 3.10
- NVIDIA GPU with CUDA
- [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) (`ncu` on PATH)
- Vertex AI authentication (gcloud)
- Environment variable: `ANTHROPIC_VERTEX_PROJECT_ID`

## Quick Start

```bash
# Quick example run
./run_example.sh

# Or manually:
kernelopt run test/softmaxtri.py \
  --csv pipeline_output/baseline.csv \
  -T 4 -N 4 \
  -o results \
  --kernel-name softmax_triton_forward_kernel \
  --kernel-args '--profile-only'
```

See [QUICKSTART.md](QUICKSTART.md) for complete guide.

## Basic Commands

```bash
# Profile a kernel
kernelopt profile test/softmaxtri.py \
  --kernel-name 'softmax_kernel' \
  --kernel-args '--profile-only' \
  -o baseline

# View profiling results
kernelopt show baseline.csv

# Generate optimization plans (uses Claude)
kernelopt plan baseline.csv test/softmaxtri.py -N 4

# Run full optimization pipeline
kernelopt run test/softmaxtri.py \
  --csv baseline.csv \
  -T 4 -N 4 \
  -o results

# List kernels in a CSV
kernelopt kernels baseline.csv
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
| `--model` | claude-sonnet-4@20250514 | Claude model to use |
| `--project-id` | $ANTHROPIC_VERTEX_PROJECT_ID | Vertex AI project ID |
| `--region` | us-east5 | Vertex AI region |
| `--planner-model` | | Override model for planner agent |
| `--executor-model` | | Override model for executor agent |
| `--summarizer-model` | | Override model for summarizer agent |

## Architecture

```
kernelopt/
  cli.py                  # Typer CLI entry point
  langgraph_pipeline.py   # LangGraph StateGraph (7 nodes + conditional loop)
  graph.py                # Legacy linear pipeline
  state.py                # PipelineState, PipelineConfig, GraphState
  llm.py                  # LLM factory (Claude Vertex AI wrapper)
  memory.py               # Bounded experience queue (FIFO, ExpN=8)
  schemas.py              # Pydantic models for all data contracts
  agents/
    profiler.py           # NCU runner + CSV parser + profiling_context builder
    planner.py            # Claude agent: profiling data -> optimization plan
    executor.py           # Claude agent: plan + kernel -> optimized kernel
    correctness.py        # Claude agent: generate CPU reference + validate
    summarizer.py         # Claude agent: before/after -> experience item
  functions/
    csv_parser.py         # NCU sections-based CSV parser
    correctness.py        # Correctness wrapper (agentic + subprocess modes)
    selector.py           # Best candidate selector
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

- **generate_plans** and **summarize_results** use `ThreadPoolExecutor` for concurrent Claude API calls
- **execute_plans** and **profile_candidates** run sequentially (GPU-bound)

### Model Configuration

| Agent | Default Model | Purpose |
|-------|--------------|---------|
| Profiler | (rule-based) | Parse NCU CSV into structured profiling context |
| Planner | claude-sonnet-4@20250514 | Analyze profiling data, generate optimization hypotheses |
| Executor | claude-sonnet-4@20250514 | Rewrite kernel code per plan |
| Correctness | claude-sonnet-4@20250514 | Generate CPU reference and validate GPU outputs |
| Summarizer | claude-sonnet-4@20250514 | Extract reusable experience from results |

All AI agents use **Claude Sonnet 4** on Vertex AI by default.

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

## Testing

```bash
# Test Claude API connection
python3 test_claude.py

# Test individual agents
python3 test_agents_individual.py

# Run example pipeline
./run_example.sh
```

## Features

- 🤖 **AI-Powered**: Claude analyzes NCU profiles and generates optimizations
- 🔄 **Iterative**: Multi-iteration optimization with learning memory
- 🎯 **Targeted**: NCU rules guide specific bottleneck fixes
- ✅ **Safe**: Agentic correctness checking (Claude generates CPU reference tests)
- 📊 **Transparent**: Full NCU reports + optimization logs
- 🧠 **Learning**: Builds reusable optimization memory across runs

## License

MIT
