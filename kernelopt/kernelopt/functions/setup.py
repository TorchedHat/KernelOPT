"""Pure environment check functions — each returns (ok: bool, message: str)."""
from __future__ import annotations

import subprocess
import sys


def check_python_version() -> tuple[bool, str]:
    v = sys.version_info
    ok = (v[0], v[1]) >= (3, 10)
    ver = f"{v[0]}.{v[1]}.{v[2]}"
    return ok, f"Python {ver}" + ("" if ok else " (need >= 3.10)")


def check_gpu() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap,name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        line = result.stdout.strip()
        if not line:
            return False, "no GPU detected by nvidia-smi"
        parts = line.split(",", 1)
        cap = float(parts[0].strip())
        name = parts[1].strip() if len(parts) > 1 else "Unknown GPU"
        if cap >= 6.0:
            return True, f"{name} (compute {cap})"
        return False, f"{name} has compute {cap} (need >= 6.0)"
    except FileNotFoundError:
        return False, "nvidia-smi not found — install NVIDIA drivers"
    except subprocess.TimeoutExpired:
        return False, "nvidia-smi timed out"
    except subprocess.CalledProcessError:
        return False, "nvidia-smi failed — check driver installation"
    except (ValueError, IndexError):
        return False, "could not parse GPU info from nvidia-smi"


def check_ncu() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["ncu", "--version"], capture_output=True, text=True, check=True, timeout=5,
        )
        for line in result.stdout.split("\n"):
            if "Version" in line:
                return True, f"NCU {line.split('Version')[-1].strip()}"
        return True, "NCU installed"
    except FileNotFoundError:
        return False, "not found — install from https://developer.nvidia.com/nsight-compute"
    except subprocess.CalledProcessError:
        return False, "ncu failed to run"
    except subprocess.TimeoutExpired:
        return False, "ncu timed out"


def check_auth() -> tuple[bool, str]:
    try:
        import google.auth
        _creds, project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        return True, f"application default credentials (project: {project or 'unknown'})"
    except Exception:
        return False, "no credentials — run: gcloud auth application-default login"


def check_local_connection(base_url: str) -> tuple[bool, str]:
    """Check if a local OpenAI-compatible server is reachable."""
    import urllib.request
    import urllib.error

    url = base_url.rstrip("/") + "/models"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5):
            return True, f"server reachable at {base_url}"
    except urllib.error.URLError as exc:
        return False, f"not reachable at {base_url}: {exc.reason}"
    except Exception as exc:
        return False, f"not reachable at {base_url}: {exc}"


def check_vertex_connection(project_id: str, region: str) -> tuple[bool, str]:
    try:
        from anthropic import AnthropicVertex
        AnthropicVertex(project_id=project_id, region=region)
        return True, f"connected ({region})"
    except ImportError:
        return False, "anthropic[vertex] not installed"
    except Exception as exc:
        msg = str(exc)
        if "403" in msg or "permission" in msg.lower():
            return False, "permission denied (need Vertex AI User role)"
        if "404" in msg:
            return False, f"project '{project_id}' or region '{region}' not found"
        return False, f"connection failed: {msg[:80]}"
