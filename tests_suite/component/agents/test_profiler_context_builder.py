"""
Tests for profiler context_builder module.

Tests build_profiling_context and get_latency_ns functions.
"""
from __future__ import annotations

import pytest
from kernelopt.agents.profiler.context_builder import build_profiling_context, get_latency_ns


SAMPLE_SECTIONS = {
    "GPU Speed Of Light Throughput": {
        "Compute (SM) Throughput": {"value": 18.4, "unit": "%"},
        "Memory Throughput": {"value": 72.1, "unit": "%"},
        "Duration": {"value": 25.6, "unit": "us"},
    },
    "Launch Statistics": {
        "Registers Per Thread": {"value": 48, "unit": ""},
        "# SMs": {"value": 132, "unit": ""},
        "Dynamic Shared Memory Per Block": {"value": 0, "unit": "byte"},
        "Static Shared Memory Per Block": {"value": 0, "unit": "byte"},
        "Threads": {"value": 1048576, "unit": ""},
        "Waves Per SM": {"value": 0.62, "unit": ""},
    },
    "Occupancy": {
        "Theoretical Occupancy": {"value": 62.5, "unit": "%"},
        "Achieved Occupancy": {"value": 50.5, "unit": "%"},
    },
}

SAMPLE_RULES = [
    {
        "name": "TheoreticalOccupancy",
        "type": "OPT",
        "description": "Register usage limits occupancy to 62.5%.",
        "speedup_type": "global",
        "speedup_pct": 24.0,
    },
]

SAMPLE_LAUNCH = {
    "kernel_name": "triton__0d1d2d3_softmax_kernel<int=1024>",
    "block_size": 1024,
    "grid_size": 4096,
    "cc": "9.0",
    "device": "NVIDIA H100 SXM5",
}


def test_build_profiling_context_top_level_fields():
    ctx = build_profiling_context(SAMPLE_SECTIONS, SAMPLE_RULES, SAMPLE_LAUNCH, "fake_kernel.py")
    assert ctx["kernel_name"] == "softmax_kernel"
    assert ctx["kernel_name_full"] == "triton__0d1d2d3_softmax_kernel<int=1024>"
    assert ctx["kernel_id"] == 0
    assert ctx["compute_capability"] == "9.0"
    assert ctx["device"] == "NVIDIA H100 SXM5"


def test_build_profiling_context_launch_fields():
    ctx = build_profiling_context(SAMPLE_SECTIONS, SAMPLE_RULES, SAMPLE_LAUNCH, "fake_kernel.py")
    launch = ctx["launch"]
    assert launch["block_size"] == 1024
    assert launch["grid_size"] == 4096
    assert launch["sm_count"] == 132
    assert launch["registers_per_thread"] == 48
    assert launch["waves_per_sm"] == 0.62


def test_build_profiling_context_sections_preserved():
    ctx = build_profiling_context(SAMPLE_SECTIONS, SAMPLE_RULES, SAMPLE_LAUNCH, "fake_kernel.py")
    assert "GPU Speed Of Light Throughput" in ctx["sections"]
    assert ctx["sections"]["GPU Speed Of Light Throughput"]["Duration"]["value"] == 25.6


def test_build_profiling_context_rules_preserved():
    ctx = build_profiling_context(SAMPLE_SECTIONS, SAMPLE_RULES, SAMPLE_LAUNCH, "fake_kernel.py")
    assert len(ctx["rules"]) == 1
    assert ctx["rules"][0]["name"] == "TheoreticalOccupancy"


def test_build_profiling_context_gpu_specs_present():
    ctx = build_profiling_context(SAMPLE_SECTIONS, SAMPLE_RULES, SAMPLE_LAUNCH, "fake_kernel.py")
    assert "gpu_specs" in ctx
    assert "gpu_model" in ctx["gpu_specs"]


def test_get_latency_ns_microseconds():
    ctx = {
        "sections": {
            "GPU Speed Of Light Throughput": {
                "Duration": {"value": 25.6, "unit": "us"}
            }
        }
    }
    assert get_latency_ns(ctx) == pytest.approx(25600.0)


def test_get_latency_ns_nanoseconds():
    ctx = {
        "sections": {
            "GPU Speed Of Light Throughput": {
                "Duration": {"value": 1500.0, "unit": "ns"}
            }
        }
    }
    assert get_latency_ns(ctx) == pytest.approx(1500.0)


def test_get_latency_ns_missing_returns_inf():
    ctx = {"sections": {}}
    assert get_latency_ns(ctx) == float("inf")
