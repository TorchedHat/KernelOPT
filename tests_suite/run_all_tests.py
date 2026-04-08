"""
Comprehensive test runner for the kernel optimization tool.

This script provides a convenient way to run different test suites:
- Unit tests: Fast tests for individual functions and components
- Integration tests: Tests for component interactions
- Optimization tests: Full optimization pipeline tests (slow)
- Agent tests: Tests for individual agents (requires GPU)

Usage:
    python test/run_all_tests.py                    # Run all tests
    python test/run_all_tests.py --unit             # Run only unit tests
    python test/run_all_tests.py --integration      # Run integration tests
    python test/run_all_tests.py --optimization     # Run optimization pipeline tests
    python test/run_all_tests.py --agents           # Run agent tests
    python test/run_all_tests.py --fast             # Run only fast tests
    python test/run_all_tests.py --gpu 3            # Specify GPU device
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_sanity_check() -> int:
    """Run sanity check before tests."""
    print(f"\n{'='*70}")
    print("🔍 Running Sanity Check...")
    print(f"{'='*70}\n")

    sanity_check_path = Path(__file__).parent / "core" / "sanity_check.py"
    result = subprocess.call([sys.executable, str(sanity_check_path)])

    if result != 0:
        print(f"\n{'='*70}")
        print("❌ Sanity check failed!")
        print("Please fix the issues before running tests.")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'='*70}")
        print("✅ Sanity check passed!")
        print(f"{'='*70}\n")

    return result


def run_pytest(
    markers: list[str] | None = None,
    paths: list[str] | None = None,
    extra_args: list[str] | None = None,
    verbose: bool = True,
) -> int:
    """Run pytest with specified markers and paths."""
    cmd = ["pytest"]

    if verbose:
        cmd.append("-v")

    # Add markers
    if markers:
        for marker in markers:
            cmd.extend(["-m", marker])

    # Add paths
    if paths:
        cmd.extend(paths)
    else:
        # Default to test directory
        cmd.append(str(Path(__file__).parent))

    # Add extra args
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Run kernel optimization tool tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Test suite selection
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run only unit tests (functions, state, schemas)",
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run only integration tests",
    )
    parser.add_argument(
        "--optimization",
        action="store_true",
        help="Run only optimization pipeline tests (slow)",
    )
    parser.add_argument(
        "--agents",
        action="store_true",
        help="Run only agent tests (requires GPU)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run only fast tests (exclude slow and e2e)",
    )

    # Test configuration
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device ID for tests (default: interactive selection or auto-select)",
    )
    parser.add_argument(
        "--skip-gpu-check",
        action="store_true",
        help="Skip GPU resource checking (use with caution)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "-k",
        type=str,
        help="Run tests matching the given pattern (pytest -k)",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage reporting",
    )
    parser.add_argument(
        "--html",
        type=str,
        help="Generate HTML coverage report at specified path",
    )
    parser.add_argument(
        "--skip-sanity-check",
        action="store_true",
        help="Skip sanity check before running tests",
    )

    args = parser.parse_args()

    # Run sanity check first (unless skipped)
    if not args.skip_sanity_check:
        sanity_result = run_sanity_check()
        if sanity_result != 0:
            print("⚠️  Sanity check failed, but continuing with tests...")
            print("    (Use --skip-sanity-check to skip this check)\n")

    # Build pytest arguments
    markers = []
    paths = []
    extra_args = []

    # Set GPU environment variable for tests
    import os
    if args.gpu is not None:
        os.environ["TEST_GPU_ID"] = str(args.gpu)

    # Skip GPU checks if requested
    if args.skip_gpu_check:
        os.environ["SKIP_GPU_CHECK"] = "1"

    # Determine which tests to run
    if args.unit:
        paths.extend([
            "test/core/test_llm.py",
            "test/core/test_memory.py",
            "test/core/test_state_schemas.py",
            "test/functions/test_csv_parser.py",
            "test/functions/test_correctness.py",
        ])
    elif args.integration:
        paths.append("test/integration/test_integration.py")
    elif args.optimization:
        markers.append("optimization")
    elif args.agents:
        paths.append("test/agents/")
    elif args.fast:
        markers.append("not slow and not optimization")
    else:
        # Run all tests by default
        pass

    # Add pattern matching
    if args.k:
        extra_args.extend(["-k", args.k])

    # Add GPU option
    if args.gpu is not None:
        extra_args.extend(["--gpu", str(args.gpu)])

    if args.skip_gpu_check:
        extra_args.append("--skip-gpu-check")

    # Add coverage
    if args.coverage:
        extra_args.extend([
            "--cov=kernelopt",
            "--cov-report=term-missing",
        ])
        if args.html:
            extra_args.append(f"--cov-report=html:{args.html}")

    # Run tests
    exit_code = run_pytest(
        markers=markers if markers else None,
        paths=paths if paths else None,
        extra_args=extra_args,
        verbose=args.verbose,
    )

    # Print summary
    print(f"\n{'='*70}")
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed with exit code {exit_code}")
    print(f"{'='*70}\n")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
