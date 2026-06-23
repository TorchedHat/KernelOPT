# kernelopt Architecture

## System Overview

kernelopt is a multi-agent GPU kernel optimization pipeline that takes any GPU kernel
(PyTorch, Triton, CUDA, Helion) and produces an optimized version using LLM-guided
hardware-aware optimization with NCU profiling.

**Core differentiators:**
- Compiler-first: uses `torch.compile` (Inductor) to synthesize Triton from PyTorch
- Dual metric: NCU for hardware diagnosis, wall-clock do_bench for speedup measurement
- Multi-kernel awareness: profiles ALL GPU kernels, optimizes the right bottleneck
- Semantic memory bank: learns from past runs with embedding-based retrieval
- Framework-agnostic: auto-detects PyTorch, Triton, CUDA, Helion

---

## Pipeline Flow

```mermaid
flowchart TD
    subgraph input [Input Detection]
        A[User Kernel] --> B{Framework?}
        B -->|PyTorch| C[maybe_synthesize]
        B -->|Triton/CUDA/Helion| D[Direct Path]
        C -->|torch.compile + Inductor| E[Synthesized Triton + Original PyTorch Saved]
        C -->|Synthesis fails| D
        E --> F[Kernel File]
        D --> F
    end

    subgraph baseline [Baseline Profiling]
        F --> G[profile_baseline]
        G --> G1[NCU --set full]
        G --> G2[do_bench wall-clock]
        G1 --> G3[baseline_ncu.csv + profiling_context]
        G2 --> G4[baseline_wall_ms]
        G3 --> H[Multi-Kernel Discovery]
        H --> H1[Classify: user / library / runtime]
        H1 --> H2[Select target kernel for optimization]
    end

    subgraph strategy [Strategy Analysis]
        H2 --> I[analyze_strategy]
        I --> I1[Compute NCU Tier]
        I1 -->|SOL > 80%| I2[Tier 0: tune_parameters]
        I1 -->|Mem SOL > Compute| I3[Tier 1: memory_bound]
        I1 -->|Compute > Mem| I4[Tier 2: compute_bound]
        I1 -->|Both < 30%| I5[Tier 3: underutilized]
        I --> I6[Query Memory Bank]
        I6 --> I7[Past Failures + Successes + Constraints + Templates]
        I7 --> I8[LLM Strategy Decision]
        I8 --> I9[strategy_type + plan_guidance]
    end

    subgraph iterloop [Optimization Loop x T iterations]
        I9 --> J[generate_plans]
        J -->|N parallel plans| J1[Planner Agent x N]
        J1 --> J2[NCU metrics + kernel source + strategy + memory]
        J2 --> J3[N optimization plans]

        J3 --> K[execute_plans]
        K -->|K attempts per plan| K1[Executor Agent]
        K1 --> K2[Modified kernel source]

        K2 --> L[correctness_check]
        L -->|Has _original_pytorch_path?| L1[PyTorch Reference Check]
        L -->|No PyTorch source| L2[LLM Reference Check]
        L1 --> L3{atol=1e-2?}
        L2 --> L3
        L3 -->|Pass| M[profile_candidates]
        L3 -->|Fail| N[correctness_failure_items]

        M --> M1[do_bench wall-clock]
        M --> M2[NCU profiling]
        M1 --> M3[wall_speedup = baseline_wall_ms / candidate_wall_ms]
        M2 --> M4[Differential Profiling: NCU metric deltas]

        M3 --> O[summarize_results]
        M4 --> O
        O --> O1[Update Memory Bank]
        O1 --> O2[Semantic Embeddings]
        O1 --> O3[Code Diff Templates]
        O1 --> O4[Metric Deltas]
        O1 --> O5[Constraint Extraction]

        O --> P[select_best]
        P -->|Primary: wall_ms| P1{Improved?}
        P1 -->|Yes| P2[Update best_kernel + best_wall_ms]
        P1 -->|No| P3[no_improvement_streak++]

        P2 --> Q[increment_iteration]
        P3 --> Q
        Q --> R{should_continue?}
        R -->|iter < T AND streak < 2| J
        R -->|iter >= T OR streak >= 2| S[END]
    end

    S --> T[Output]
    T --> T1[best_kernel.py]
    T --> T2[optimization_log.md]
    T --> T3[pipeline_checkpoint.json]
    T --> T4[Memory Bank Updated]
```

---

## Agent Architecture

```mermaid
flowchart LR
    subgraph agents [Multi-Agent System]
        PA[Profiler Agent] -->|NCU config| NCU[NCU Runner]
        SA[Strategy Analyst] -->|strategy_type| PL
        PL[Planner Agent x N] -->|optimization plan| EX
        EX[Executor Agent x K] -->|modified kernel| CR
        CR[Correctness Check] -->|pass/fail| PR
        PR[Profiler + Benchmarker] -->|metrics| SU
        SU[Summarizer Agent] -->|memory entries| MB
    end

    subgraph memory [Memory Bank]
        MB[Memory Bank]
        MB --> SE[Semantic Embeddings]
        MB --> CE[Constraint Extraction]
        MB --> CT[Code Templates]
        MB --> DP[Differential Profiling]
        MB --> CW[Confidence Weighting]
        SE -->|all-MiniLM-L6-v2 / F2LLM-v2| EMB[Cosine Similarity Search]
    end

    subgraph measurement [Dual Metric System]
        NCU2[NCU Profiling] -->|per-kernel hardware diagnosis| PL
        DB[do_bench Wall-Clock] -->|end-to-end speedup| PS[select_best]
    end
```

---

## Memory Bank Architecture

```mermaid
flowchart TD
    subgraph membank [Persistent Memory Bank]
        direction TB
        ME[MemoryEntry]
        ME --> KS[kernel_structure: type, has_library, multi_kernel]
        ME --> HW[hardware: gpu, cc, shared_memory, sm_count]
        ME --> ST[strategy: type, direction]
        ME --> OC[outcome: success / failure]
        ME --> SP[speedup: float]
        ME --> CD[code_diff: unified diff template]
        ME --> MD[metric_deltas: NCU before/after]
        ME --> FR[failure_reason + constraint]
    end

    subgraph retrieval [Retrieval Methods]
        R1[Structural Query] -->|kernel_type + hardware + outcome| MB2[Keyword Matching]
        R2[Semantic Query] -->|kernel_source + NCU_summary| EMB2[Embedding Cosine Similarity]
        R3[Template Query] -->|similar successful optimizations| TMPL[Code Diff Templates]
        R4[Constraint Query] -->|auto-extracted failure rules| CONS[Structured Rules]
    end

    subgraph scoring [Confidence Scoring]
        SC1[Structural Score: type=10, library=3, multi=2, gpu=5]
        SC2[Speedup Weight: score * speedup_magnitude]
        SC3[Recency Decay: score * max 0.5, 1 - age_days/365]
    end

    subgraph models [Embedding Models Priority]
        EM1[1. codefuse-ai/F2LLM-v2-330M]
        EM2[2. ibm-granite/granite-embedding-311m-multilingual-r2]
        EM3[3. jinaai/jina-embeddings-v3]
        EM4[4. nomic-ai/nomic-embed-text-v1.5]
        EM5[5. sentence-transformers/all-mpnet-base-v2]
        EM6[6. sentence-transformers/all-MiniLM-L6-v2]
    end
```

---

## Correctness Verification

```mermaid
flowchart TD
    C[Optimized Kernel] --> D{Has _original_pytorch_path?}
    D -->|Yes: Synthesized from PyTorch| E[Compiler-First Correctness]
    D -->|No: Direct Triton/CUDA input| F{Framework?}

    E --> E1[Run original PyTorch Model]
    E --> E2[Run optimized Triton kernel_function]
    E1 --> E3[torch.allclose atol=1e-2, rtol=1e-2]
    E2 --> E3
    E3 --> G{Pass?}

    F -->|Python + agentic| H[LLM Correctness Agent]
    F -->|CUDA with main| I[Compile + Run Binary]
    F -->|CUDA without main| J[LLM CUDA Harness]

    H --> H1[LLM generates reference function]
    H1 --> H2[GPU vs CPU comparison, 3 seeds]
    H2 --> G

    I --> I1[nvcc -O2, run, check exit code]
    I1 --> G

    G -->|Pass| K[Proceed to Profiling]
    G -->|Fail| L[correctness_failure_items]
```

---

## NCU-Guided Optimization Tiers

| Tier | Condition | Strategy | Recommended Optimizations |
|------|-----------|----------|--------------------------|
| 0 | SOL% > 80% | `tune_parameters` | XBLOCK/num_warps/num_stages tuning, eviction hints |
| 1 | Memory SOL% > Compute SOL% | `optimize_kernel` (memory) | Coalescing, vectorized loads, L2 compression, tile locality |
| 2 | Compute SOL% > Memory SOL% | `optimize_kernel` (compute) | Register pressure, divergence, precision reduction |
| 3 | Both SOL% < 30% | `optimize_kernel` (rewrite) | Buffer elimination, pointwise fusion, algorithmic changes |

---

## Benchmark Runner Architecture

```mermaid
flowchart TD
    subgraph runner [eval_benchmark.py]
        R[eval_benchmark.py] -->|per GPU| G1[GPU 2: kb_l1]
        R --> G2[GPU 3: kb_l3 + kb_l4]
        R --> G3[GPU 4: sol_l1]
        R --> G4[GPU 5: kb_l2]
        R --> G5[GPU 6: sol_l2]
        R --> G6[GPU 7: sol_train + FlashInfer + Quant]
    end

    subgraph perproblem [Per Problem Flow]
        PP1[Measure Baselines] --> PP2[eager_ns + compile_ns]
        PP2 --> PP3[Convert SOL JSON to .py if needed]
        PP3 --> PP4[kernelopt run -T 4 -N 3 -K 2]
        PP4 --> PP5[Parse pipeline_checkpoint.json]
        PP5 --> PP6[Write result_id.json + results.json]
    end

    subgraph resilience [Resilience]
        RS1[Resume: skip completed problems via results.json]
        RS2[Cascade Recovery: os.execv on CUDA cascade error]
        RS3[SOL Axis Resolution: const + var + expr evaluation]
        RS4[EXCLUSIVE_PROCESS GPU mode for isolation]
    end
```

---

## File Structure

```
kernelopt/
├── cli.py                          # Typer CLI entry point
├── langgraph_pipeline.py           # LangGraph optimization pipeline
│   ├── maybe_synthesize            # PyTorch → Triton via torch.compile
│   ├── profile_baseline            # NCU + do_bench baseline
│   ├── analyze_strategy            # NCU-tier + memory-guided strategy
│   ├── generate_plans              # N parallel planner instances
│   ├── execute_plans               # K executor attempts per plan
│   ├── profile_candidates          # NCU + do_bench per candidate
│   ├── summarize_results           # Memory bank update
│   ├── select_best                 # Wall-clock primary metric
│   └── should_continue             # Loop control
├── state.py                        # PipelineState, GraphState, PlanCandidate
├── memory_bank.py                  # Semantic memory with embeddings
├── reference_kernels.py            # Triton tutorial reference database
├── agents/
│   ├── profiler/                   # NCU config + profiling
│   │   ├── __init__.py             # run_profiler_agent
│   │   └── ncu_runner.py           # NCU subprocess management
│   ├── planner/                    # Optimization planning
│   │   ├── __init__.py             # N-parallel planner orchestration
│   │   ├── prompts.py             # System prompt + diversity hints
│   │   └── tools.py               # query_profiling_context, search_memory
│   ├── executor/                   # Code modification
│   │   ├── __init__.py            # Executor agent
│   │   └── prompts.py            # System prompt + optimization guidance
│   ├── summarizer.py             # Experience summarization
│   ├── correctness.py            # LLM-based correctness verification
│   └── schemas.py                # Pydantic schemas for structured output
├── functions/
│   ├── benchmark.py              # Wall-clock do_bench (CUDA Events)
│   ├── correctness.py            # Correctness gate (PyTorch ref + LLM ref)
│   ├── compiler.py               # nvcc compilation
│   ├── csv_parser.py             # NCU CSV parsing + bottleneck discovery
│   └── gpu_monitoring.py         # GPU selection + activity monitoring
├── synthesis/
│   ├── compile_extract.py        # torch.compile → Triton extraction
│   └── wrap_kernel.py            # Wrap Inductor output for kernelopt
├── llm.py                        # LLM client (Anthropic / Vertex AI)
├── checkpoint.py                 # Pipeline checkpoint save/load
└── memory.py                     # Short-term optimization memory

eval_benchmark.py                 # Batch benchmark runner (531 problems)
launch_benchmark.sh               # tmux orchestrator for multi-GPU runs
collect_benchmark.py              # Result aggregation + report generation
```

---

## Key Design Decisions

1. **Compiler-first, LLM-second**: torch.compile generates correct Triton deterministically.
   The LLM only optimizes — never synthesizes from scratch.

2. **Dual metric system**: NCU diagnoses per-kernel hardware bottlenecks (what to fix).
   do_bench measures end-to-end wall-clock (whether the fix helped).

3. **Soft optimization boundary**: The LLM is guided to focus on @triton.jit kernels
   but retains context about library calls for informed decisions.

4. **Compiler-first correctness**: For synthesized files, the original PyTorch program
   is the ground truth — no LLM-generated reference needed.

5. **Semantic memory with constraint learning**: Embeddings enable `matmul+relu ≈ matmul+gelu`
   matching. Auto-extracted constraints prevent repeating known failures.

6. **Differential profiling**: Verifies the intended NCU metric actually changed after
   optimization, providing causal evidence not just correlations.
