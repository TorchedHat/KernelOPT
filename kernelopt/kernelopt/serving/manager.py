"""TorchServeManager: lifecycle manager for TorchServe + FastAPI gateway.

Used as a context manager from the CLI — starts everything on __enter__,
shuts everything down on __exit__ (including on Ctrl+C or exceptions).
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

HANDLER_DIR = Path(__file__).parent
PLANNER_HANDLER = HANDLER_DIR / "qwen3_handler.py"
CODER_HANDLER = HANDLER_DIR / "qwen3_coder_handler.py"

DEFAULT_TS_INFERENCE_PORT = 8085
DEFAULT_TS_MANAGEMENT_PORT = 8086
DEFAULT_TS_METRICS_PORT = 8087
DEFAULT_GATEWAY_PORT = 8000


class TorchServeManager:
    """Context manager that starts TorchServe + gateway, stops on exit."""

    def __init__(
        self,
        gpu_id: int = 2,
        ts_inference_port: int = DEFAULT_TS_INFERENCE_PORT,
        ts_management_port: int = DEFAULT_TS_MANAGEMENT_PORT,
        ts_metrics_port: int = DEFAULT_TS_METRICS_PORT,
        gateway_port: int = DEFAULT_GATEWAY_PORT,
        planner_model: str = "Qwen/Qwen3-30B-A3B",
        executor_model: str = "Qwen/Qwen3-Coder-Next-FP8",
        planner_batch_size: int = 8,
        planner_batch_delay: int = 500,
        executor_batch_size: int = 4,
        executor_batch_delay: int = 200,
    ):
        self.gpu_id = gpu_id
        self.ts_inference_port = ts_inference_port
        self.ts_management_port = ts_management_port
        self.ts_metrics_port = ts_metrics_port
        self.gateway_port = gateway_port
        self.planner_model = planner_model
        self.executor_model = executor_model
        self.planner_batch_size = planner_batch_size
        self.planner_batch_delay = planner_batch_delay
        self.executor_batch_size = executor_batch_size
        self.executor_batch_delay = executor_batch_delay

        self._model_store: str = ""
        self._config_path: str = ""
        self._ts_process: subprocess.Popen | None = None
        self._gw_process: subprocess.Popen | None = None
        self._tmpdir: str = ""

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def start(self):
        """Package MARs, start TorchServe, register models, start gateway."""
        self._tmpdir = tempfile.mkdtemp(prefix="kernelopt_ts_")
        self._model_store = os.path.join(self._tmpdir, "model_store")
        os.makedirs(self._model_store, exist_ok=True)

        logger.info("Packaging model archives...")
        self._package_mar("qwen3-30b-a3b", str(PLANNER_HANDLER))
        self._package_mar("qwen3-coder-next-fp8", str(CODER_HANDLER))

        self._write_config()

        logger.info("Starting TorchServe on GPU %d...", self.gpu_id)
        self._start_torchserve()
        self._wait_for_torchserve()

        logger.info("Registering models...")
        self._register_models()

        logger.info("Starting gateway on port %d...", self.gateway_port)
        self._start_gateway()
        self._wait_for_gateway()

        logger.info("All services ready.")

    def stop(self):
        """Stop gateway and TorchServe."""
        if self._gw_process:
            logger.info("Stopping gateway...")
            self._gw_process.terminate()
            try:
                self._gw_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._gw_process.kill()
            self._gw_process = None

        if self._ts_process:
            logger.info("Stopping TorchServe...")
            subprocess.run(["torchserve", "--stop"], capture_output=True, timeout=10)
            try:
                self._ts_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._ts_process.kill()
            self._ts_process = None

        if self._tmpdir and os.path.exists(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = ""

    def wait_ready(self, timeout: int = 300):
        """Wait until both TorchServe models and gateway are healthy."""
        # Already waited in start(), but allow explicit re-check
        deadline = time.time() + timeout
        urls = [
            f"http://localhost:{self.ts_inference_port}/ping",
            f"http://localhost:{self.gateway_port}/health",
        ]
        for url in urls:
            while time.time() < deadline:
                try:
                    resp = httpx.get(url, timeout=5)
                    if resp.status_code == 200:
                        break
                except (httpx.ConnectError, httpx.ReadTimeout):
                    pass
                time.sleep(2)
            else:
                raise TimeoutError(f"Service at {url} not ready after {timeout}s")

    # -- Internal helpers --

    def _package_mar(self, model_name: str, handler_path: str):
        """Create a .mar archive using torch-model-archiver."""
        cmd = [
            "torch-model-archiver",
            "--model-name", model_name,
            "--version", "1.0",
            "--handler", handler_path,
            "--export-path", self._model_store,
            "--force",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                f"torch-model-archiver failed for {model_name}: {result.stderr[:500]}"
            )

    def _write_config(self):
        """Write TorchServe config.properties."""
        self._config_path = os.path.join(self._tmpdir, "config.properties")
        config = (
            f"inference_address=http://0.0.0.0:{self.ts_inference_port}\n"
            f"management_address=http://0.0.0.0:{self.ts_management_port}\n"
            f"metrics_address=http://0.0.0.0:{self.ts_metrics_port}\n"
            f"model_store={self._model_store}\n"
            "number_of_netty_threads=4\n"
            "job_queue_size=100\n"
        )
        Path(self._config_path).write_text(config)

    def _start_torchserve(self):
        """Launch TorchServe as a subprocess."""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        cmd = [
            "torchserve", "--start",
            "--model-store", self._model_store,
            "--ts-config", self._config_path,
            "--ncs",
            "--foreground",
        ]
        self._ts_process = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def _wait_for_torchserve(self, timeout: int = 300):
        """Poll TorchServe health endpoint."""
        url = f"http://localhost:{self.ts_inference_port}/ping"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = httpx.get(url, timeout=5)
                if resp.status_code == 200:
                    logger.info("TorchServe is ready")
                    return
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass

            # Check if process died
            if self._ts_process and self._ts_process.poll() is not None:
                stderr = self._ts_process.stderr.read().decode() if self._ts_process.stderr else ""
                raise RuntimeError(
                    f"TorchServe exited with code {self._ts_process.returncode}: "
                    f"{stderr[:1000]}"
                )
            time.sleep(3)

        raise TimeoutError(f"TorchServe not ready after {timeout}s")

    def _register_models(self):
        """Register models via TorchServe management API."""
        mgmt = f"http://localhost:{self.ts_management_port}"

        # Register planner model (device_ids=0 because CUDA_VISIBLE_DEVICES
        # maps the real GPU to device 0 inside TorchServe)
        self._register_model(
            mgmt, "qwen3-30b-a3b",
            batch_size=self.planner_batch_size,
            max_batch_delay=self.planner_batch_delay,
        )
        self._register_model(
            mgmt, "qwen3-coder-next-fp8",
            batch_size=self.executor_batch_size,
            max_batch_delay=self.executor_batch_delay,
        )

    def _register_model(
        self, mgmt_url: str, model_name: str,
        batch_size: int = 1, max_batch_delay: int = 100,
    ):
        """Register a single model."""
        url = (
            f"{mgmt_url}/models"
            f"?url={model_name}.mar"
            f"&initial_workers=1"
            f"&batch_size={batch_size}"
            f"&max_batch_delay={max_batch_delay}"
            f"&response_timeout=300"
        )
        resp = httpx.post(url, timeout=30)
        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"Failed to register {model_name}: {resp.status_code} {resp.text[:300]}"
            )
        logger.info("Registered model %s (batch=%d, delay=%dms)", model_name, batch_size, max_batch_delay)

        # Wait for workers to be ready
        self._wait_for_model(mgmt_url, model_name)

    def _wait_for_model(self, mgmt_url: str, model_name: str, timeout: int = 300):
        """Wait until model workers are ready."""
        url = f"{mgmt_url}/models/{model_name}"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = httpx.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    workers = data.get("workers", [])
                    if any(w.get("status") == "READY" for w in workers):
                        return
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass
            time.sleep(5)
        raise TimeoutError(f"Model {model_name} workers not ready after {timeout}s")

    def _start_gateway(self):
        """Start the FastAPI gateway as a subprocess."""
        # Configure routes before starting
        from kernelopt.serving.gateway import configure_routes
        configure_routes(
            ts_inference_url=f"http://localhost:{self.ts_inference_port}",
            planner_model=self.planner_model,
            executor_model=self.executor_model,
        )

        cmd = [
            sys.executable, "-m", "uvicorn",
            "kernelopt.serving.gateway:app",
            "--host", "0.0.0.0",
            "--port", str(self.gateway_port),
            "--log-level", "warning",
        ]
        env = os.environ.copy()
        # Pass route config via env vars so the subprocess can configure itself
        env["KERNELOPT_TS_URL"] = f"http://localhost:{self.ts_inference_port}"
        env["KERNELOPT_PLANNER_MODEL"] = self.planner_model
        env["KERNELOPT_EXECUTOR_MODEL"] = self.executor_model

        self._gw_process = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def _wait_for_gateway(self, timeout: int = 30):
        """Wait for gateway to be ready."""
        url = f"http://localhost:{self.gateway_port}/health"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = httpx.get(url, timeout=5)
                if resp.status_code == 200:
                    logger.info("Gateway is ready on port %d", self.gateway_port)
                    return
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass
            if self._gw_process and self._gw_process.poll() is not None:
                stderr = self._gw_process.stderr.read().decode() if self._gw_process.stderr else ""
                raise RuntimeError(f"Gateway exited: {stderr[:500]}")
            time.sleep(1)
        raise TimeoutError(f"Gateway not ready after {timeout}s")
