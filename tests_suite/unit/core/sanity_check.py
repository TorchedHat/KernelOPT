#!/usr/bin/env python3
"""
Sanity check script for the kernel optimization test suite.

This script validates:
- Test suite structure and completeness
- H200 GPU configuration
- Required dependencies
- Test file integrity
- Configuration files

Run this before executing the full test suite to ensure everything is set up correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Colors for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header(text: str):
    """Print colored header."""
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}{text}{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{GREEN}✓{RESET} {text}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{YELLOW}⚠{RESET} {text}")


def print_error(text: str):
    """Print error message."""
    print(f"{RED}✗{RESET} {text}")


def check_file_exists(file_path: Path, description: str) -> bool:
    """Check if a file exists."""
    if file_path.exists():
        print_success(f"{description}: {file_path.name}")
        return True
    else:
        print_error(f"{description} not found: {file_path}")
        return False


def check_directory_structure() -> bool:
    """Check test directory structure."""
    print_header("Directory Structure")

    test_dir = Path(__file__).parent.parent  # Go up from core/ to test/
    all_good = True

    # Required directories
    required_dirs = [
        ("core", "Core tests directory"),
        ("agents", "Agent tests directory"),
        ("functions", "Function tests directory"),
        ("integration", "Integration tests directory"),
        ("optimization", "End-to-end tests directory"),
        ("utils", "Test utilities directory"),
        ("fixtures", "Test fixtures directory"),
        ("fixtures/kernels", "Test kernels directory"),
        ("fixtures/data", "Test data directory"),
    ]

    for dir_name, description in required_dirs:
        dir_path = test_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print_success(f"{description}: {dir_name}/")
        else:
            print_error(f"{description} not found: {dir_name}/")
            all_good = False

    return all_good


def check_test_files() -> bool:
    """Check that all test files exist."""
    print_header("Test Files")

    test_dir = Path(__file__).parent.parent  # Go up from core/ to test/
    all_good = True

    # Root test files
    root_files = [
        ("conftest.py", "Pytest configuration"),
        ("pytest.ini", "Pytest settings"),
        ("run_all_tests.py", "Main test runner"),
    ]

    for file_name, description in root_files:
        all_good &= check_file_exists(test_dir / file_name, description)

    # Core test files (in core/ subdirectory)
    core_test_files = [
        ("test_llm.py", "LLM module tests"),
        ("test_memory.py", "Memory module tests"),
        ("test_state_schemas.py", "State/schema tests"),
        ("sanity_check.py", "Sanity check script"),
    ]

    for file_name, description in core_test_files:
        all_good &= check_file_exists(test_dir / "core" / file_name, description)

    # Agent tests
    agent_tests = [
        "test_profiler.py",
        "test_planner.py",
        "test_executor.py",
        "test_summarizer.py",
    ]

    for file_name in agent_tests:
        all_good &= check_file_exists(
            test_dir / "agents" / file_name,
            f"Agent test: {file_name}"
        )

    # Function tests
    function_tests = [
        "test_csv_parser.py",
        "test_correctness.py",
    ]

    for file_name in function_tests:
        all_good &= check_file_exists(
            test_dir / "functions" / file_name,
            f"Function test: {file_name}"
        )

    # Utils
    # Integration and optimization tests
    integration_tests = [
        ("integration/test_integration.py", "Integration tests"),
    ]

    optimization_tests = [
        ("optimization/test_optimization.py", "End-to-end tests"),
    ]

    for file_name, description in integration_tests + optimization_tests:
        all_good &= check_file_exists(test_dir / file_name, description)

    # Utility files
    util_files = [
        "gpu_utils.py",
        "test_gpu_utils.py",
    ]

    for file_name in util_files:
        all_good &= check_file_exists(
            test_dir / "utils" / file_name,
            f"Utility: {file_name}"
        )

    return all_good


def check_documentation() -> bool:
    """Check documentation files."""
    print_header("Documentation")

    test_dir = Path(__file__).parent.parent  # Go up from core/ to test/
    all_good = True

    docs = [
        ("README.md", "Main test README"),
        ("fixtures/README.md", "Fixtures documentation"),
        (".env.example", "Environment configuration template"),
    ]

    for file_name, description in docs:
        all_good &= check_file_exists(test_dir / file_name, description)

    return all_good


def check_dependencies() -> bool:
    """Check required dependencies."""
    print_header("Dependencies")

    all_good = True

    # Core dependencies
    dependencies = [
        ("pytest", "Test framework"),
        ("torch", "PyTorch (for CUDA)"),
    ]

    for module_name, description in dependencies:
        try:
            __import__(module_name)
            print_success(f"{description}: {module_name}")
        except ImportError:
            print_error(f"{description} not installed: {module_name}")
            all_good = False

    # Optional but recommended
    optional_deps = [
        ("pytest_cov", "Coverage reporting"),
        ("pytest_xdist", "Parallel execution"),
    ]

    for module_name, description in optional_deps:
        try:
            __import__(module_name)
            print_success(f"{description}: {module_name}")
        except ImportError:
            print_warning(f"{description} not installed (optional): {module_name}")

    return all_good


def check_gpu_availability() -> bool:
    """Check GPU availability."""
    print_header("GPU Configuration (H200 Optimized)")

    try:
        import torch
    except ImportError:
        print_error("PyTorch not installed - cannot check GPU")
        return False

    if not torch.cuda.is_available():
        print_error("CUDA is not available")
        return False

    gpu_count = torch.cuda.device_count()
    print_success(f"CUDA available: {gpu_count} GPU(s) detected")

    # Try to import GPU utils
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))  # Go up to test/ directory
        from utils.gpu_utils import get_gpu_info, print_gpu_summary, is_h200_gpu

        print("\nGPU Details:")
        for gpu_id in range(gpu_count):
            try:
                info = get_gpu_info(gpu_id)
                is_h200 = is_h200_gpu(gpu_id)
                h200_marker = " [H200 - OPTIMIZED]" if is_h200 else ""

                print(f"\n  GPU {gpu_id}: {info['name']}{h200_marker}")
                print(f"    Memory: {info['memory_total_mb']:.0f} MB total")
                print(f"    Free: {info['memory_free_mb']:.0f} MB ({(info['memory_free_mb']/info['memory_total_mb'])*100:.1f}%)")

                if is_h200:
                    print_success(f"    H200 detected - using optimized settings")
            except Exception as e:
                print_error(f"    Error getting info for GPU {gpu_id}: {e}")

        return True

    except Exception as e:
        print_error(f"Cannot import GPU utils: {e}")
        return False


def check_configuration_files() -> bool:
    """Check configuration integrity."""
    print_header("Configuration Files")

    test_dir = Path(__file__).parent.parent  # Go up from core/ to test/
    all_good = True

    # Check conftest.py has GPU selection
    conftest_path = test_dir / "conftest.py"
    if conftest_path.exists():
        content = conftest_path.read_text()
        if "H200" in content and "gpu_id" in content:
            print_success("conftest.py configured for H200 GPU selection")
        else:
            print_warning("conftest.py may not have H200 configuration")

    # Check pytest.ini has markers
    pytest_ini_path = test_dir / "pytest.ini"
    if pytest_ini_path.exists():
        content = pytest_ini_path.read_text()
        required_markers = ["gpu", "slow", "agent", "optimization", "profiling", "h200"]
        missing_markers = [m for m in required_markers if m not in content]

        if not missing_markers:
            print_success(f"pytest.ini has all required markers: {', '.join(required_markers)}")
        else:
            print_warning(f"pytest.ini missing markers: {', '.join(missing_markers)}")

    return all_good


def check_test_integrity() -> bool:
    """Check test files for common issues."""
    print_header("Test Integrity Checks")

    test_dir = Path(__file__).parent.parent  # Go up from core/ to test/
    all_good = True

    # Find all test files
    test_files = list(test_dir.rglob("test_*.py"))

    print(f"Found {len(test_files)} test files")

    issues = []

    for test_file in test_files:
        try:
            content = test_file.read_text()

            # Check for common issues
            if "test_" not in content and "Test" not in content:
                issues.append(f"{test_file.name}: No test functions or classes")

            if "@pytest.mark.gpu" in content and "skip_if_no_cuda" not in content:
                # This is OK, just a note
                pass

        except Exception as e:
            issues.append(f"{test_file.name}: Error reading file - {e}")

    if issues:
        print_warning(f"Found {len(issues)} potential issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
        all_good = False
    else:
        print_success("All test files look good")

    return all_good


def count_tests() -> dict:
    """Count tests in each category."""
    print_header("Test Statistics")

    test_dir = Path(__file__).parent

    stats = {
        "unit": 0,
        "agent": 0,
        "integration": 0,
        "optimization": 0,
        "utils": 0,
        "total": 0,
    }

    # Count test functions in each category
    for category in ["unit", "agent", "integration", "optimization", "utils"]:
        if category == "unit":
            patterns = ["test_*.py"]
            dirs = [test_dir, test_dir / "functions"]
        elif category == "agent":
            patterns = ["test_*.py"]
            dirs = [test_dir / "agents"]
        elif category == "integration":
            patterns = ["test_*.py"]
            dirs = [test_dir / "integration"]
        elif category == "optimization":
            patterns = ["test_*.py"]
            dirs = [test_dir / "optimization"]
        elif category == "utils":
            patterns = ["test_*.py"]
            dirs = [test_dir / "utils"]
        else:
            continue

        count = 0
        for dir_path in dirs:
            if not dir_path.exists():
                continue
            for pattern in patterns:
                for file_path in dir_path.glob(pattern):
                    try:
                        content = file_path.read_text()
                        # Count test functions and methods
                        count += content.count("def test_")
                    except Exception:
                        pass

        stats[category] = count
        stats["total"] += count

    for category, count in stats.items():
        if category != "total":
            print(f"  {category.capitalize()}: {count} tests")

    print(f"\n  {BOLD}Total: {stats['total']} tests{RESET}")

    return stats


def main():
    """Run all sanity checks."""
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}Kernel Optimization Test Suite - Sanity Check{RESET}")
    print(f"{BOLD}H200 GPU Optimized - Claude Sonnet 4 on Vertex AI{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")

    results = {}

    # Run all checks
    results["directory_structure"] = check_directory_structure()
    results["test_files"] = check_test_files()
    results["documentation"] = check_documentation()
    results["dependencies"] = check_dependencies()
    results["gpu"] = check_gpu_availability()
    results["configuration"] = check_configuration_files()
    results["integrity"] = check_test_integrity()

    # Count tests
    stats = count_tests()

    # Summary
    print_header("Summary")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for check, result in results.items():
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {check.replace('_', ' ').title()}: {status}")

    print(f"\n{BOLD}Overall: {passed}/{total} checks passed{RESET}")
    print(f"{BOLD}Total Tests: {stats['total']}{RESET}")

    if all(results.values()):
        print(f"\n{GREEN}{BOLD}✓ Test suite is ready!{RESET}")
        print(f"\nRun tests with:")
        print(f"  {BLUE}pytest test/{RESET}                  # Interactive GPU selection")
        print(f"  {BLUE}pytest test/ --gpu 2{RESET}          # Specify GPU")
        print(f"  {BLUE}python test/run_all_tests.py{RESET}  # Full test runner")
        return 0
    else:
        print(f"\n{RED}{BOLD}✗ Some checks failed. Please fix the issues above.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
