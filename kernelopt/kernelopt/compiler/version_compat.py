"""PyTorch version detection and capability flags.

Checks which compiler hooks are available in the current PyTorch
installation. Must be called INSIDE the extraction subprocess
(same Python environment as torch).
"""

from __future__ import annotations


def detect_capabilities() -> dict[str, bool | str]:
    """Detect which compiler hooks are available.

    Returns dict with boolean flags for each capability.
    """
    caps: dict[str, bool | str] = {
        "pytorch_version": "unknown",
        "triton_version": "unknown",
        "has_pre_grad_custom_pass": False,
        "has_post_grad_custom_pre_pass": False,
        "has_post_grad_custom_post_pass": False,
        "has_pre_fusion_pass": False,
        "has_post_fusion_pass": False,
        "has_joint_custom_passes": False,
        "has_algorithm_selector_cache": False,
        "has_graph_name_to_buffer": False,
        "has_why_no_fuse": False,
        "has_triton_compilation_listener": False,
        "has_triton_autotune_listener": False,
        "has_triton_stages_inspection_hook": False,
    }

    try:
        import torch
        caps["pytorch_version"] = torch.__version__
    except ImportError:
        return caps

    try:
        import torch._inductor.config as cfg
        caps["has_pre_grad_custom_pass"] = hasattr(cfg, "pre_grad_custom_pass")
        caps["has_post_grad_custom_pre_pass"] = hasattr(cfg, "post_grad_custom_pre_pass")
        caps["has_post_grad_custom_post_pass"] = hasattr(cfg, "post_grad_custom_post_pass")
        caps["has_pre_fusion_pass"] = hasattr(cfg, "_pre_fusion_custom_pass")
        caps["has_post_fusion_pass"] = hasattr(cfg, "_post_fusion_custom_pass")
        caps["has_joint_custom_passes"] = (
            hasattr(cfg, "joint_custom_pre_pass") and
            hasattr(cfg, "joint_custom_post_pass")
        )
    except ImportError:
        pass

    try:
        from torch._inductor.select_algorithm import get_algorithm_selector_cache
        caps["has_algorithm_selector_cache"] = True
    except ImportError:
        pass

    try:
        from torch._inductor.scheduler import WhyNoFuse
        caps["has_why_no_fuse"] = hasattr(WhyNoFuse, "__call__")
    except ImportError:
        pass

    try:
        import triton
        caps["triton_version"] = triton.__version__
        caps["has_triton_compilation_listener"] = hasattr(
            triton.knobs.compilation, "listener"
        )
        caps["has_triton_autotune_listener"] = hasattr(
            triton.knobs.autotuning, "listener"
        )
        caps["has_triton_stages_inspection_hook"] = hasattr(
            triton.knobs.runtime, "add_stages_inspection_hook"
        )
    except (ImportError, AttributeError):
        pass

    return caps
