# kernelopt — Feature Reference

Multi-agent GPU kernel optimizer. Automatically profiles, plans, executes, verifies, and iterates — building an experience memory that improves plan quality over time.

---

## Framework Support

- **Triton** — Python kernels decorated with `@triton.jit`
- **CUDA** — Native `.cu` files compiled via `nvcc`
- **Helion** — Python kernels decorated with `@helion.kernel` / `@hl.kernel`
- Auto-detection from file extension and source content (AST for Python, regex for CUDA)

---

## 6-Phase Optimization Pipeline (LangGraph)

Each iteration runs six phases in a LangGraph pipeline:

| Phase | Agent | Description |
|-------|-------|-------------|
| 1. Baseline | Profiler (stateless) | Runs NCU on the original kernel once; parses profiling context |
| 2. Planning | N concurrent Planner agents | Generates N diverse `OptimizationPlan` objects (direction, evidence, steps, confidence) |
| 3. Execution | N concurrent Executor agents | Applies each plan; correctness-checks each candidate (agentic or static); up to K attempts per plan |
| 4. Profiling | Profiler (stateless) | Runs NCU on each passing candidate; skipped for correctness failures |
| 5. Summarization | Summarizer agent | Extracts a generalizable experience item; updates optimization memory |
| 6. Selection | Pipeline (pure function) | `select_best()` picks the lowest-latency passing candidate; updates `best_kernel` and `optimization_log.md` |

### Stopping Conditions

- Max T iterations reached
- 2 consecutive iterations with no improvement
- All plans report low confidence

---

## Agents

### Planner Agent
- N instances run **concurrently** via `ThreadPoolExecutor`
- Each instance receives a **diversity hint** pointing at a different NCU optimization rule
- Cross-iteration direction history prevents repeating tried strategies
- Outputs a Pydantic-validated `OptimizationPlan` with `direction`, `evidence`, `steps`, `confidence`

### Executor Agent
- N instances run **concurrently** (one per plan)
- Up to **K attempts** per plan; each attempt validates kernel markers before accepting
- For CUDA: compile errors from `nvcc` are fed back to the executor as context for the next attempt
- Returns `(optimized_source, execution_result)` with `ready_for_profiling` flag

### Profiler Agent
- Stateless — no LLM involved
- Runs `ncu --set full` → `.ncu-rep`, then exports CSV via `ncu --import`
- Parses sections, rules, and launch stats into a structured `profiling_context` dict
- Enriches context with **GPU hardware specs** (SM count, HBM bandwidth, compute capability)

### Summarizer Agent
- Single LLM call with structured output
- Diffs the slow/fast kernel pair
- Produces an `experience_item` with generalizable insight and a `memory_update` action (`append` / `replace` / `skip`)
- Correctness-failure items bypass speedup thresholds so the planner avoids repeating broken implementations

---

## Correctness Checking

| Mode | How it works |
|------|-------------|
| **Agentic (default)** | Claude generates a CPU reference implementation; numerical outputs are compared |
| **Non-agentic** (`--no-agentic-correctness`) | Simpler static check (exit code + failure markers) |
| **CUDA builtin self-test** | Detects `int main(` — the only reliable indicator that a `.cu` file is a standalone binary; runs it directly |
| **CUDA agentic harness** | Claude writes a complete self-contained `.cu` test harness with `main()`, compiles it, and checks output markers |

Correctness failures skip NCU profiling but still generate negative experience items so the planner learns from them.

---

## Native CUDA `.cu` File Support

- Upfront compilation with `nvcc` before baseline profiling
- Clean, human-readable error messages (temp-file paths stripped from nvcc stderr)
- Executor retry loop: compile errors fed back to the LLM for up to K fix attempts
- Binary reused for both correctness checking and NCU profiling (no redundant compilation)
- Builtin self-test auto-detection heuristic (scans for `int main(` — the only reliable indicator of a standalone binary)
- Agentic fallback: LLM generates a complete `.cu` test harness when no self-test is detected

---

## Optimization Memory

A bounded FIFO queue of `experience_items` that accumulates across iterations.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ExpN` | 8 | Queue capacity |
| `TopK` | 4 | Max items added per iteration |
| `t_pos` | 1.04 | Min speedup to store a positive item |
| `t_neg` | 1.15 | Min slowdown factor to store a negative item |

Each item records: `direction`, `speedup`, `rewrite_type`, `profiling_signal`, `strategy_description`, `slow_pseudocode`, `fast_pseudocode`, `applicable_when`, `do_not_apply_when`.

**Memory update actions**: `append` (new direction), `replace` (same direction, higher speedup), `skip` (insignificant or duplicate). The planner reads this memory each iteration to avoid dead ends and reinforce successful strategies.

---

## Pipeline Checkpointing & Resume

Checkpoint saved after **every major stage**:

```
after_baseline → after_plans → after_execute → after_profile → after_summarize → after_select → iteration_N
```

**What's saved**: full config (T, N, K, models, NCU flags, kernel args), current iteration, best speedup, optimization memory, profiling context, all candidate results, last completed stage.

### Resume

```bash
# Resume from where the run stopped
kernelopt run softmax.py --memory pipeline_output/run_20260413_120000_000000/

# Resume and extend iterations
kernelopt run softmax.py -T 10 --memory pipeline_output/run_20260413_120000_000000/

# Resume with increased N/K
kernelopt run softmax.py -T 5 -N 8 -K 2 --memory pipeline_output/run_20260413_120000_000000/
```

- Reloads full config from `pipeline_checkpoint.json`; only `-T`/`-N`/`-K` and `--hud-tty` can be overridden
- Continues from the last completed stage (e.g., skips plan generation if plans are already saved)
- Use cases: job timeouts, cluster time limits, iterative tuning, checkpoint inspection

---

## GPU Monitoring

### Pre-run System Status

Displayed before optimization starts:
- System RAM and GPU VRAM (clearly labeled to avoid confusion)
- GPU utilization and temperature
- Active GPU processes with per-process memory usage
- Warnings if GPU memory is low (<40% free) or GPU is busy (>20% utilization)

### Per-iteration Monitoring

Silent when idle; prints GPU activity summary at the start of each iteration if the GPU is busy.

### Container Support

- GPU memory per process and total utilization are accurate inside containers
- Processes outside the container namespace labeled as `"host"` (PID namespace isolation)

---

## Live HUD (`--hud-tty`)

A compact status panel rendered on a **second terminal** while the main terminal keeps the full scrolling log.

- Second terminal switches to **alternate screen** during the run (restores on exit)
- Shows: current phase, iteration, best speedup, recent events, system/GPU snapshot
- GPU VRAM / utilization / temperature refresh ~1 Hz from `nvidia-smi`
- Log coalesced at ~2 Hz to avoid strobing; phase changes force immediate refresh
- NCU output stays on the main terminal — no UI mixing

```bash
# Run `tty` in terminal B, then:
kernelopt run softmax.py --hud-tty /dev/pts/6
# or: export KERNELOPT_HUD_TTY=/dev/pts/6
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `kernelopt run <kernel>` | Full optimization pipeline |
| `kernelopt profile <kernel>` | Single NCU profiling run only |
| `kernelopt show <ncu.csv>` | Pretty-print a profile CSV (no LLM) |
| `kernelopt plan <ncu.csv> <kernel>` | Generate plans only, no execution |
| `kernelopt kernels <ncu.csv>` | List all kernels found in an NCU CSV |
| `kernelopt setup` | Interactive wizard — configure Vertex AI credentials |
| `kernelopt doctor` | Read-only environment health check |

### `kernelopt run` Key Options

| Flag | Default | Description |
|------|---------|-------------|
| `-T` | 5 | Max optimization iterations |
| `-N` | 4 | Plans per iteration (concurrent planner agents) |
| `-K` | 1 | Executor attempts per plan |
| `--top-k` | 4 | Max experience items added per iteration |
| `-o` | `pipeline_output/` | Output base directory |
| `--run-subdir` / `--no-run-subdir` | on | Create timestamped `run_<timestamp>/` subdirectory |
| `--csv` | — | Skip baseline profiling; use an existing NCU CSV |
| `--memory` | — | Resume from a previous run directory |
| `--hud-tty` | — | Device path for the live HUD terminal |
| `--kernel-name` | auto | NCU `--kernel-name` regex |
| `--launch-skip` | 0 | NCU `--launch-skip` |
| `--launch-count` | — | NCU `--launch-count` |
| `--replay-mode` | `kernel` | `kernel` or `application` (use `application` for kernels that OOM with kernel replay) |
| `--kernel-args` | — | Extra args forwarded to the kernel program |
| `--model` | `claude-sonnet-4@20250514` | Default model for all agents |
| `--planner-model` | — | Model override for the planner only |
| `--executor-model` | — | Model override for the executor only |
| `--summarizer-model` | — | Model override for the summarizer only |
| `--project-id` | `$ANTHROPIC_VERTEX_PROJECT_ID` | Google Cloud project |
| `--region` | `us-east5` | Vertex AI region |
| `--agentic-correctness` / `--no-agentic-correctness` | on | Use Claude for correctness checks |

---

## `kernelopt setup` — Interactive Wizard

One-time setup after install:
1. Prompts for Google Cloud project ID and Vertex AI region (shows current values as defaults)
2. Runs OAuth via `google-auth-oauthlib` (opens browser, caches ADC credentials)
3. Writes `~/.config/kernelopt/config.toml`
4. Runs environment checks with Rich spinners: GPU, NCU, Vertex AI connectivity
5. Prints a summary table and a ready message

---

## `kernelopt doctor` — Environment Health Check

Read-only. Reads config + env vars, runs all checks, prints a Rich status table:

| Check | What it verifies |
|-------|-----------------|
| Config file | `~/.config/kernelopt/config.toml` present |
| `project_id` | Set via config or env var |
| `region` | Set via config or env var |
| Auth credentials | `application_default_credentials.json` found |
| GPU | CUDA-capable GPU detected (compute capability) |
| NCU binary | `ncu` found on `$PATH` |
| Vertex AI | API reachable in the configured region |

Exit code 0 if all checks pass, exit code 1 if any fail — scriptable for CI.

---

## Configuration Precedence

For `project_id` and `region`:

```
CLI flag  >  env var  >  ~/.config/kernelopt/config.toml  >  error
```

---

## Output Artifacts

Every `kernelopt run` creates a timestamped directory `pipeline_output/run_YYYYMMDD_HHMMSS_microseconds/`:

| File | Description |
|------|-------------|
| `best_kernel.{py\|cu\|hl}` | Best optimized kernel found so far |
| `optimization_memory.json` | Accumulated experience items (FIFO queue) |
| `pipeline_checkpoint.json` | Full pipeline state for resume |
| `optimization_log.md` | Human-readable per-iteration log with metric tables |
| `baseline.csv` / `baseline.ncu-rep` | Baseline NCU profiling artifacts |
| `baseline_profiling_context.json` | Parsed baseline metrics |
| `{kernel}_iter{N}_plan{L}_{direction}.{ext}` | Each candidate's kernel source |
| `{stem}.csv` / `{stem}.ncu-rep` | Candidate NCU profiling data (passing candidates only) |
| `{stem}_profiling_context.json` | Parsed candidate metrics |

Use `--no-run-subdir` to write flat into `-o` (predictable paths for CI/scripting).

---

## LLM Integration

- **Backend**: Claude on Vertex AI via `langchain-anthropic` (`ChatAnthropic` / `AnthropicVertex`)
- **Thinking model support**: `<think>...</think>` blocks stripped from output automatically
- **Per-agent model overrides**: planner, executor, and summarizer can each use a different model
- **Structured output**: planner and summarizer use Pydantic-validated structured output schemas

---

## NCU Integration

- Runs `ncu --set full` for comprehensive sections-based profiling
- Exports `.ncu-rep` binary reports (openable in Nsight Compute UI)
- Exports CSV via `ncu --import` for machine-readable parsing
- Parses three sections: **Sections** (throughput, occupancy), **Rules** (NCU guided recommendations), **Launch** (grid/block/register/shared memory config)
- Kernel name filtering via `--kernel-name` regex
- Launch skip/count control via `--launch-skip` / `--launch-count`
- Kernel vs. application replay mode selection

---

## Architecture Highlights

- **LangGraph**: all agents are LangGraph sub-graphs (`agent → tools → agent → ... → submit`)
- **Concurrent execution**: planner and executor agents run in parallel via `ThreadPoolExecutor`
- **Stateless profiler**: no LLM, deterministic, reusable
- **Dependency injection**: agents accept LLM config as parameters (not hardcoded)
- **Clean layer boundaries**: CSV parser, correctness, GPU monitoring, and selector are pure functions independent of agent internals
- **490 tests** across unit, component, integration, e2e, and quality layers
