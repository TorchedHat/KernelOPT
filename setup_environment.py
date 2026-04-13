#!/usr/bin/env python3
"""
Interactive environment setup script for kernelopt.

Sets up:
- Vertex AI credentials and project configuration
- GPU environment variables
- System dependencies checks
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple


# ============================================================================
# TROUBLESHOOTING GUIDES
# ============================================================================

TROUBLESHOOTING = {
    "gcloud_not_found": """
╔════════════════════════════════════════════════════════════════════╗
║                  gcloud CLI Not Found                              ║
╚════════════════════════════════════════════════════════════════════╝

OPTION 1: Automatic Installation
  curl -sSL https://sdk.cloud.google.com | bash
  exec -l $SHELL

OPTION 2: Manual Installation
  Visit: https://cloud.google.com/sdk/docs/install

After installing, verify: gcloud --version
""",

    "auth_failed": """
╔════════════════════════════════════════════════════════════════════╗
║              Authentication Failed                                 ║
╚════════════════════════════════════════════════════════════════════╝

Try these steps:
  1. Run: gcloud auth application-default login
  2. Verify: gcloud auth application-default print-access-token
  3. Check account: gcloud auth list

Alternative: Use a service account JSON key file
  Set: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
""",

    "vertex_api_disabled": """
╔════════════════════════════════════════════════════════════════════╗
║           Vertex AI API Not Enabled                                ║
╚════════════════════════════════════════════════════════════════════╝

Enable the API:
  Web: https://console.cloud.google.com/apis/library/aiplatform.googleapis.com
  CLI: gcloud services enable aiplatform.googleapis.com --project=YOUR_PROJECT

Verify:
  gcloud services list --enabled --project=YOUR_PROJECT | grep aiplatform
""",

    "ncu_not_found": """
╔════════════════════════════════════════════════════════════════════╗
║         NVIDIA Nsight Compute (ncu) Not Found                      ║
╚════════════════════════════════════════════════════════════════════╝

Install from: https://developer.nvidia.com/nsight-compute
Or install via CUDA Toolkit: https://developer.nvidia.com/cuda-downloads

Add to PATH: export PATH="/usr/local/cuda/bin:$PATH"

Verify: ncu --version
""",

    "gpu_not_found": """
╔════════════════════════════════════════════════════════════════════╗
║              NVIDIA GPU Not Detected                               ║
╚════════════════════════════════════════════════════════════════════╝

Check GPU: nvidia-smi

If nvidia-smi fails:
  - Install NVIDIA driver: sudo ubuntu-drivers autoinstall
  - For Docker: Use --gpus all flag

Requirements:
  • CUDA Compute Capability >= 6.0 (Pascal or newer)
  • NVIDIA Driver >= 450.80.02
""",

    "python_version": """
╔════════════════════════════════════════════════════════════════════╗
║              Python Version Too Old                                ║
╚════════════════════════════════════════════════════════════════════╝

kernelopt requires Python 3.10+

Upgrade with pyenv:
  curl https://pyenv.run | bash
  pyenv install 3.11
  pyenv global 3.11

Or use system package manager:
  Ubuntu: sudo apt-get install python3.11
""",

    "vertex_connection_failed": """
╔════════════════════════════════════════════════════════════════════╗
║         Cannot Connect to Vertex AI                                ║
╚════════════════════════════════════════════════════════════════════╝

Checklist:
  ☐ Vertex AI API enabled
  ☐ Proper IAM permissions (Vertex AI User role)
  ☐ Correct project ID
  ☐ Valid region (us-east5, us-central1, etc.)

Check permissions:
  gcloud projects get-iam-policy YOUR_PROJECT --filter=YOUR_EMAIL
""",

    "kernelopt_not_installed": """
╔════════════════════════════════════════════════════════════════════╗
║           kernelopt Package Not Installed                          ║
╚════════════════════════════════════════════════════════════════════╝

Install the kernelopt package:

  cd /path/to/kernel-profiling-tool/kernelopt
  pip install -e ".[triton]"      # For Triton support
  pip install -e ".[dev]"         # For development/testing

Verify installation:
  pip show kernelopt
  which kernelopt
  kernelopt --help

If using a virtual environment:
  source .venv/bin/activate
  cd kernelopt
  pip install -e ".[triton]"
""",
}


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def print_step(step: int, text: str):
    """Print a step indicator."""
    print(f"\n[{step}] {text}")
    print("-" * 70)


def print_troubleshooting(error_key: str):
    """Print detailed troubleshooting instructions."""
    if error_key in TROUBLESHOOTING:
        print(TROUBLESHOOTING[error_key])
    else:
        print(f"\n⚠️  No specific troubleshooting guide for: {error_key}")


def check_command_exists(command: str) -> bool:
    """Check if a command exists in PATH."""
    try:
        subprocess.run(
            ["which", command],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=5
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def enhanced_check(
    check_name: str,
    check_func: callable,
    error_key: str,
    required: bool = True
) -> bool:
    """
    Run a check with retry/skip/help options on failure.

    Args:
        check_name: Display name for the check
        check_func: Function that returns (success: bool, message: str)
        error_key: Key for troubleshooting guide
        required: Whether this check must pass

    Returns:
        bool: True if check passed, False if skipped
    """
    while True:
        success, message = check_func()

        if success:
            print(f"✓ {message}")
            return True

        # Check failed
        print(f"✗ {message}")

        if not required:
            print(f"  ⚠️  This check is optional")

        print("\n  Options:")
        print("    [V] View troubleshooting guide")
        print("    [R] Retry")
        if not required:
            print("    [S] Skip")
        print("    [Q] Quit")

        choice = input("\n  Choice: ").strip().upper()

        if choice == 'V':
            print_troubleshooting(error_key)
            input("\n  Press Enter to continue...")
        elif choice == 'R':
            print("\n  Retrying...")
            continue
        elif choice == 'S' and not required:
            print("  ⚠️  Skipping {}".format(check_name))
            return False
        elif choice == 'Q':
            print("\n  Setup cancelled.")
            sys.exit(0)
        else:
            print("  ⚠️  Invalid choice.")


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version >= 3.10"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version >= (3, 10):
        return True, f"Python {version_str}"
    return False, f"Python {version_str} (need >= 3.10)"


def check_vertex_ai_api(project_id: str) -> Tuple[bool, str]:
    """Verify Vertex AI API is enabled"""
    try:
        result = subprocess.run(
            [
                "gcloud", "services", "list", "--enabled",
                f"--project={project_id}",
                "--filter=aiplatform.googleapis.com",
                "--format=value(name)"
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15
        )

        if "aiplatform.googleapis.com" in result.stdout:
            return True, "Vertex AI API enabled"

        return False, f"Vertex AI API NOT enabled in project '{project_id}'"

    except subprocess.TimeoutExpired:
        return False, "Timeout checking API (network issue?)"
    except subprocess.CalledProcessError as e:
        return False, f"Failed to check API: {e.stderr.strip() if e.stderr else 'unknown error'}"
    except FileNotFoundError:
        return False, "gcloud not found"


def check_gpu_compute_capability() -> Tuple[bool, str]:
    """Verify GPU compute capability >= 6.0"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )

        line = result.stdout.strip()
        if not line:
            return False, "No GPU detected"

        parts = line.split(',', 1)
        compute_cap = float(parts[0].strip())
        gpu_name = parts[1].strip() if len(parts) > 1 else "Unknown GPU"

        if compute_cap >= 6.0:
            return True, "{} (compute {})".format(gpu_name, compute_cap)

        return False, "{} has compute {} (need >= 6.0)".format(gpu_name, compute_cap)

    except subprocess.TimeoutExpired:
        return False, "nvidia-smi timeout"
    except subprocess.CalledProcessError:
        return False, "nvidia-smi failed (driver issue?)"
    except (ValueError, IndexError):
        return False, "Could not parse GPU info"
    except FileNotFoundError:
        return False, "nvidia-smi not found"


def check_ncu_detailed() -> Tuple[bool, str]:
    """Check NCU with version info"""
    try:
        result = subprocess.run(
            ["ncu", "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )

        for line in result.stdout.split('\n'):
            if 'Version' in line:
                version = line.split('Version')[-1].strip()
                return True, "NCU {}".format(version)

        return True, "NCU installed"

    except subprocess.TimeoutExpired:
        return False, "ncu timeout"
    except subprocess.CalledProcessError:
        return False, "ncu failed"
    except FileNotFoundError:
        return False, "ncu not found"


def test_vertex_connection(project_id: str, region: str) -> Tuple[bool, str]:
    """Test actual Vertex AI connection"""
    try:
        from anthropic import AnthropicVertex

        # This validates auth, API enabled, and permissions
        _ = AnthropicVertex(
            project_id=project_id,
            region=region
        )

        return True, "Connected to Vertex AI ({})".format(region)

    except ImportError:
        return False, "anthropic package not installed"
    except Exception as e:
        error_msg = str(e)

        if "403" in error_msg or "permission" in error_msg.lower():
            return False, "Permission denied (need Vertex AI User role)"
        elif "404" in error_msg:
            return False, "Project '{}' or region '{}' not found".format(project_id, region)
        elif "api not enabled" in error_msg.lower():
            return False, "Vertex AI API not enabled"
        elif "auth" in error_msg.lower() or "credential" in error_msg.lower():
            return False, "Authentication failed"
        else:
            return False, "Connection failed: {}".format(error_msg[:80])


def check_kernelopt_installed() -> Tuple[bool, str]:
    """Check if kernelopt package and CLI are installed"""
    # Check if package is installed
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "kernelopt"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )

        # Extract version
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                version = line.split(':', 1)[1].strip()

                # Also check if CLI is on PATH
                if check_command_exists("kernelopt"):
                    return True, "kernelopt {} (CLI ready)".format(version)
                else:
                    return True, "kernelopt {} (installed, but CLI not on PATH)".format(version)

        return True, "kernelopt installed"

    except subprocess.CalledProcessError:
        return False, "kernelopt package not installed"
    except subprocess.TimeoutExpired:
        return False, "pip timeout"
    except (FileNotFoundError, IndexError):
        return False, "Cannot check package status"


def print_system_summary():
    """Print system information summary"""
    print("\n╔════════════════════════════════════════════════════════════════════╗")
    print("║                    System Information                              ║")
    print("╚════════════════════════════════════════════════════════════════════╝\n")

    # Python
    success, msg = check_python_version()
    status = "✓" if success else "✗"
    print("  Python:          {} {}".format(msg, status))

    # GPU
    success, msg = check_gpu_compute_capability()
    status = "✓" if success else "✗"
    print("  GPU:             {} {}".format(msg, status))

    # NVIDIA Driver
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=5
        )
        print("  NVIDIA Driver:   {} ✓".format(result.stdout.strip()))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        print("  NVIDIA Driver:   Not detected ✗")

    # NCU
    success, msg = check_ncu_detailed()
    status = "✓" if success else "✗"
    print("  Nsight Compute:  {} {}".format(msg, status))

    # kernelopt package
    success, msg = check_kernelopt_installed()
    status = "✓" if success else "✗"
    print("  kernelopt:       {} {}".format(msg, status))

    # gcloud
    if check_command_exists("gcloud"):
        try:
            result = subprocess.run(
                ["gcloud", "--version"],
                capture_output=True, text=True, check=True, timeout=5
            )
            version = result.stdout.split('\n')[0].replace('Google Cloud SDK ', '')
            print("  gcloud CLI:      {} ✓".format(version))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, IndexError):
            print("  gcloud CLI:      Installed ✓")
    else:
        print("  gcloud CLI:      Not found ✗")

    print()


def prompt_with_default(prompt: str, default: Optional[str] = None) -> str:
    """Prompt user with optional default value."""
    if default:
        user_input = input("{} [{}]: ".format(prompt, default)).strip()
        return user_input if user_input else default
    else:
        while True:
            user_input = input("{}: ".format(prompt)).strip()
            if user_input:
                return user_input
            print("  ⚠️  This field is required. Please enter a value.")


def check_gcloud_installed() -> bool:
    """Check if gcloud CLI is installed."""
    return check_command_exists("gcloud")


def install_gcloud():
    """Guide user to install gcloud CLI."""
    print("\n⚠️  gcloud CLI not found. Installing...")

    print("\nChoose installation method:")
    print("  1. Automatic install (recommended)")
    print("  2. Manual install (I'll do it myself)")
    print("  3. Skip (I'll use service account key file)")

    choice = input("\nChoice [1]: ").strip() or "1"

    if choice == "1":
        print("\nInstalling gcloud CLI...")
        try:
            subprocess.run(
                "curl -sSL https://sdk.cloud.google.com | bash",
                shell=True,
                check=True
            )
            print("\n✓ gcloud installed. Please restart your shell and run this script again.")
            print("  Run: exec -l $SHELL")
            sys.exit(0)
        except subprocess.CalledProcessError:
            print("\n❌ Installation failed. Please install manually:")
            print("  https://cloud.google.com/sdk/docs/install")
            sys.exit(1)

    elif choice == "2":
        print("\nPlease install gcloud CLI manually:")
        print("  https://cloud.google.com/sdk/docs/install")
        print("\nThen run this script again.")
        sys.exit(0)

    # Choice 3: skip, will use service account key


def setup_gcloud_auth():
    """Setup gcloud authentication."""
    print("\nAuthenticating with Google Cloud...")
    print("This will open a browser window for authentication.")

    proceed = input("\nProceed with authentication? [Y/n]: ").strip().lower()
    if proceed and proceed != 'y':
        print("Skipping authentication. You'll need to set it up manually.")
        return False

    try:
        subprocess.run(
            ["gcloud", "auth", "application-default", "login", "--no-launch-browser"],
            check=True
        )
        print("\n✓ Authentication successful!")
        return True
    except subprocess.CalledProcessError:
        print("\n⚠️  Authentication failed. You can:")
        print("  1. Run: gcloud auth application-default login")
        print("  2. Use a service account key file instead")
        return False


def verify_auth() -> bool:
    """Verify that authentication is working."""
    try:
        subprocess.run(
            ["gcloud", "auth", "application-default", "print-access-token"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def create_env_file(config: dict):
    """Create .env file with configuration."""
    env_path = Path(__file__).parent / ".env"

    lines = [
        "# Vertex AI Configuration",
        f"export ANTHROPIC_VERTEX_PROJECT_ID={config['project_id']}",
        f"export VERTEX_REGION={config['region']}",
        "",
        "# Performance Configuration",
        "export OPENBLAS_NUM_THREADS=1",
        "",
    ]

    if config.get('sa_key_path'):
        lines.extend([
            "# Service Account Key (if using)",
            f"export GOOGLE_APPLICATION_CREDENTIALS={config['sa_key_path']}",
            "",
        ])

    lines.extend([
        "# Add gcloud to PATH (if needed)",
        'export PATH="$HOME/google-cloud-sdk/bin:$PATH"',
        "",
    ])

    env_path.write_text("\n".join(lines))
    print("\n✓ Configuration saved to: {}".format(env_path))
    print("\n  To use in future sessions, run:")
    print("    source {}".format(env_path))

    # Also update shell RC file
    shell_rc = Path.home() / ".zshrc" if Path.home() / ".zshrc" else Path.home() / ".bashrc"

    update_rc = input("\nAdd to {} for automatic loading? [Y/n]: ".format(shell_rc.name)).strip().lower()
    if not update_rc or update_rc == 'y':
        with open(shell_rc, 'a') as f:
            f.write("\n# kernelopt environment (added by setup_environment.py)\n")
            f.write("[ -f {} ] && source {}\n".format(env_path, env_path))
        print("✓ Added to {}".format(shell_rc))


def main():
    print_header("kernelopt Environment Setup")

    print("This script will help you set up the environment for kernelopt.")
    print("You'll need:")
    print("  • Google Cloud project with Vertex AI API enabled")
    print("  • gcloud CLI installed (or we can install it)")
    print("  • CUDA-capable GPU (for running the tool)")

    proceed = input("\nContinue? [Y/n]: ").strip().lower()
    if proceed and proceed != 'y':
        print("Setup cancelled.")
        sys.exit(0)

    # Show system summary
    print_system_summary()

    config = {}

    # Step 0: Check Python version
    print_step(0, "Checking Python Version")
    enhanced_check(
        "Python Version",
        check_python_version,
        "python_version",
        required=True
    )

    # Step 1: Check gcloud installation
    print_step(1, "Checking gcloud CLI")

    if check_gcloud_installed():
        print("✓ gcloud CLI is installed")
    else:
        install_gcloud()

    # Step 2: Authentication method
    print_step(2, "Authentication Setup")

    print("\nChoose authentication method:")
    print("  1. Application Default Credentials (recommended)")
    print("  2. Service Account Key File")

    auth_choice = input("\nChoice [1]: ").strip() or "1"

    if auth_choice == "1":
        if not verify_auth():
            setup_gcloud_auth()
        else:
            print("✓ Already authenticated")
    else:
        sa_key_path = prompt_with_default(
            "Enter path to service account key file",
            "/root/sa-key.json"
        )
        config['sa_key_path'] = sa_key_path

        if not Path(sa_key_path).exists():
            print(f"\n⚠️  Warning: File not found: {sa_key_path}")
            print("  Make sure to copy the key file before running kernelopt")

    # Step 3: Project configuration
    print_step(3, "Vertex AI Configuration")

    print("\nWhere to find your credentials:")
    print("  • Project ID: https://console.cloud.google.com/")
    print("    (Top navbar → Select project → Copy project ID)")
    print("  • Common regions: us-east5, us-central1, europe-west4")
    print("    See: https://cloud.google.com/vertex-ai/docs/general/locations\n")

    config['project_id'] = prompt_with_default(
        "Enter your Google Cloud project ID",
        None
    )

    config['region'] = prompt_with_default(
        "Enter Vertex AI region (e.g., us-east5, us-central1)",
        None
    )

    # Step 3a: Check Vertex AI API is enabled
    print("\nVerifying Vertex AI API is enabled...")
    enhanced_check(
        "Vertex AI API",
        lambda: check_vertex_ai_api(config['project_id']),
        "vertex_api_disabled",
        required=False  # Can skip if they want to enable later
    )

    # Step 4: Verify Python dependencies
    print_step(4, "Checking Python Dependencies")

    try:
        __import__('anthropic')
        print("✓ anthropic package installed")
    except ImportError:
        print("⚠️  anthropic package not found")
        install = input("  Install now? [Y/n]: ").strip().lower()
        if not install or install == 'y':
            subprocess.run([sys.executable, "-m", "pip", "install", "anthropic[vertex]"], check=True)

    try:
        __import__('google.cloud.aiplatform')
        print("✓ google-cloud-aiplatform installed")
    except ImportError:
        print("⚠️  google-cloud-aiplatform not found")
        install = input("  Install now? [Y/n]: ").strip().lower()
        if not install or install == 'y':
            subprocess.run([sys.executable, "-m", "pip", "install", "google-cloud-aiplatform"], check=True)

    # Step 5: Check CUDA
    print_step(5, "Checking GPU/CUDA")

    # Check GPU with compute capability
    enhanced_check(
        "GPU Compute Capability",
        check_gpu_compute_capability,
        "gpu_not_found",
        required=False  # Can skip for CPU-only testing
    )

    # Check NCU installation
    enhanced_check(
        "NVIDIA Nsight Compute",
        check_ncu_detailed,
        "ncu_not_found",
        required=False  # Can skip if only testing setup
    )

    # Step 6: Create configuration files
    print_step(6, "Saving Configuration")

    create_env_file(config)

    # Step 7: Verify setup
    print_step(7, "Verifying Setup")

    # Set environment variables for verification
    os.environ['ANTHROPIC_VERTEX_PROJECT_ID'] = config['project_id']
    os.environ['VERTEX_REGION'] = config['region']
    if config.get('sa_key_path'):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config['sa_key_path']

    # Test basic auth first
    print("Checking Google Cloud authentication...")
    try:
        from google.auth import default
        credentials, project = default()
        print("✓ Authentication working")
        print("  Project: {}".format(project))
    except Exception as e:
        print("⚠️  Authentication verification failed: {}".format(e))
        print("  You may need to run: gcloud auth application-default login")

    # Test actual Vertex AI connection
    print("\nTesting Vertex AI connection...")
    enhanced_check(
        "Vertex AI Connection",
        lambda: test_vertex_connection(config['project_id'], config['region']),
        "vertex_connection_failed",
        required=False  # Can skip and verify later when running
    )

    # Check kernelopt installation
    print("\nChecking kernelopt installation...")
    kernelopt_ok = enhanced_check(
        "kernelopt Package",
        check_kernelopt_installed,
        "kernelopt_not_installed",
        required=False  # Can skip if they want to install later
    )

    # Final summary
    print_header("Setup Complete!")

    print("Configuration summary:")
    print("  Project ID: {}".format(config['project_id']))
    print("  Region: {}".format(config['region']))
    print("  Auth method: {}".format('Service Account' if config.get('sa_key_path') else 'Application Default'))

    print("\nNext steps:")

    if not kernelopt_ok:
        print("  ⚠️  WARNING: kernelopt package not installed!")
        print("\n  1. Install kernelopt package:")
        print("       cd {}".format(Path(__file__).parent / "kernelopt"))
        print("       pip install -e \".[triton]\"")
        print("\n  2. Load environment variables:")
        print("       source {}".format(Path(__file__).parent / '.env'))
        print("\n  3. Test the tool:")
        print("       kernelopt run tests_suite/fixtures/kernels/matmul_triton.py -T 1 -N 1")
    else:
        print("  1. Load environment variables:")
        print("       source {}".format(Path(__file__).parent / '.env'))
        print("\n  2. Test the tool:")
        print("       kernelopt run tests_suite/fixtures/kernels/matmul_triton.py -T 1 -N 1")

    print("\n✓ All done! Happy optimizing! 🚀")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
