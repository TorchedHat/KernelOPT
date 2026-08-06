# Contributing to KernelOPT

Thank you for your interest in contributing to KernelOPT. This document covers the development workflow, coding standards, and how to submit changes.

## Getting Started

### Prerequisites

- Python 3.10+
- NVIDIA GPU with compute capability >= 6.0
- NVIDIA Nsight Compute CLI (`ncu`) on PATH
- An LLM backend configured (see README)

### Development Setup

```bash
git clone https://github.com/TorchedHat/KernelOPT.git
cd KernelOPT/kernelopt
pip install -e ".[dev,triton,semantic]"
```

This installs KernelOPT in editable mode with development dependencies (pytest, ruff, mypy).

### Verify Installation

```bash
kernelopt doctor    # Check environment
pytest tests_suite/unit/ -x    # Run unit tests
```

## Development Workflow

1. **Open an issue first** --- describe the bug, feature, or improvement before writing code
2. **Create a branch** from `main` for your work
3. **Make focused commits** --- each commit should be a single logical change
4. **Run tests** before submitting
5. **Open a pull request** against `main` with a clear description

## Project Layout

```
kernelopt/
  kernelopt/
    cli.py                    # CLI entry point
    langgraph_pipeline.py     # Core optimization pipeline
    multi_kernel.py           # Multi-kernel orchestration
    state.py                  # Config and state dataclasses
    llm.py                    # LLM backend registry
    agents/                   # Planner, executor, profiler, summarizer
    synthesis/                # torch.compile extraction and kernel wrapping
    compiler/                 # Standalone compiler analysis subpackage
    fusion/                   # Cross-kernel pattern matching
    functions/                # Benchmarking, profiling, correctness
    data/                     # Optimization technique library (JSON)
  tests_suite/
    unit/                     # Unit tests (fast, no GPU required)
    integration/              # Integration tests (requires GPU)
    e2e/                      # End-to-end tests (requires GPU + LLM)
    fixtures/                 # Test data and sample kernels
    benchmarks/               # KernelBench, SOL-ExecBench
```

## Coding Standards

### Style

- Follow existing code patterns in the file you're modifying
- No comments explaining what the code does --- use clear names instead
- Comments only for non-obvious *why* (hidden constraints, workarounds, surprising behavior)
- No multi-line docstrings or comment blocks --- one short line max

### Formatting

```bash
ruff check kernelopt/    # Lint
ruff format kernelopt/   # Format
```

### Type Hints

Use type hints for function signatures. We use `from __future__ import annotations` in all modules.

### Imports

- Standard library first, then third-party, then local
- Prefer explicit imports over `from module import *`
- Runtime-optional imports (torch, triton) should use try/except with graceful fallback

## Testing

### Running Tests

```bash
# Unit tests (no GPU needed)
pytest tests_suite/unit/ -x

# Specific test file
pytest tests_suite/unit/test_compiler_context.py -v

# With coverage
pytest tests_suite/unit/ --cov=kernelopt
```

### Writing Tests

- Unit tests go in `tests_suite/unit/`
- Test files are named `test_<module>.py`
- Use fixtures from `tests_suite/fixtures/` for sample kernels and data
- Mock LLM calls --- don't hit real APIs in unit tests
- GPU-dependent tests should be marked and skippable

## Areas for Contribution

### Good First Issues

- Adding support for new NCU metrics in the profiler parser
- Improving error messages in `kernelopt doctor`
- Adding new optimization techniques to `data/optimization_techniques.json`

### Medium Complexity

- New LLM backend integrations (add a builder in `llm.py`)
- Improving the correctness verifier with additional checks
- Better handling of edge cases in `wrap_kernel.py` (multi-output kernels, dynamic shapes)

### Advanced

- New search strategies in `search_tree.py`
- Cross-kernel fusion optimization in `fusion/`
- Compiler pass development in `compiler/passes/`
- Support for new input frameworks beyond PyTorch/Triton/CUDA/Helion

## Commit Messages

- Use imperative mood: "Fix correctness check" not "Fixed correctness check"
- First line under 72 characters
- Reference issue numbers where relevant: "Fix validation crash (Fixes #42)"
- No trivial commits ("fix fix", "wip", "temp")

## Pull Request Guidelines

- PR title should be concise and descriptive (under 70 characters)
- Include a summary of what changed and why
- Link the issue being addressed
- Ensure all tests pass
- Keep PRs focused --- one logical change per PR

## Security

If you discover a security vulnerability, please report it privately by opening a security advisory on GitHub rather than a public issue.

## License

By contributing to KernelOPT, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
