"""
llama.cpp server manager — starts and stops llama-server as a subprocess.

Pulse owns the server process: starts it on boot, monitors health,
and shuts it down gracefully (waiting for in-flight inference to finish).
"""

import asyncio
import logging
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class LlamaServer:
    """Manages a llama-server subprocess."""

    def __init__(self, config: dict):
        server_cfg = config.get("server", {})
        model_cfg = config.get("model", {})

        # Paths
        self.server_path = Path(server_cfg.get("llama_cpp_dir", "C:\\llama-cpp")) / "llama-server.exe"
        model_name = model_cfg.get("model_file", "")
        models_dir = Path(server_cfg.get("models_dir", server_cfg.get("llama_cpp_dir", "C:\\llama-cpp")))
        self.model_path = models_dir / model_name if model_name else None

        # Server settings
        self.host = server_cfg.get("host", "127.0.0.1")
        self.port = server_cfg.get("port", 8012)
        self.gpu_layers = server_cfg.get("gpu_layers", -1)
        self.context_size = model_cfg.get("max_context", 16384)
        self.parallel = server_cfg.get("parallel", 1)
        self.flash_attn = server_cfg.get("flash_attention", True)

        # Reasoning model support
        self.reasoning = model_cfg.get("reasoning", False)

        # Vision support (optional mmproj file)
        mmproj_file = model_cfg.get("mmproj_file", "")
        self.mmproj_path = models_dir / mmproj_file if mmproj_file else None

        # Process handle
        self._process: Optional[subprocess.Popen] = None
        self._output_lines: deque[str] = deque(maxlen=80)
        self._output_thread: Optional[threading.Thread] = None
        self._healthy = False

    @property
    def endpoint(self) -> str:
        """OpenAI-compatible API endpoint."""
        return f"http://{self.host}:{self.port}/v1"

    @property
    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}/health"

    def _build_command(self) -> list[str]:
        """Build the llama-server command line."""
        if not self.server_path.exists():
            raise FileNotFoundError(f"llama-server not found: {self.server_path}")
        if not self.model_path or not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        cmd = [
            str(self.server_path),
            "--model", str(self.model_path),
            "--host", self.host,
            "--port", str(self.port),
            "--ctx-size", str(self.context_size),
            "--parallel", str(self.parallel),
            "-ngl", str(self.gpu_layers),
            "--flash-attn", "on" if self.flash_attn else "off",
            "--reasoning", "on" if self.reasoning else "off",
        ]

        if self.mmproj_path:
            if not self.mmproj_path.exists():
                logger.warning(f"mmproj file not found: {self.mmproj_path} — starting without vision")
            else:
                cmd.extend(["--mmproj", str(self.mmproj_path)])

        return cmd

    async def start(self, timeout: int = 120) -> bool:
        """Start the llama-server and wait until it's healthy.

        Args:
            timeout: Max seconds to wait for the server to become healthy.

        Returns:
            True if server started and is healthy, False otherwise.
        """
        if self._process and self._process.poll() is None:
            logger.info("llama-server is already running.")
            return True

        cmd = self._build_command()
        logger.info(f"Starting llama-server...")
        logger.info(f"  Model: {self.model_path.name}")
        logger.info(f"  Port: {self.port} | GPU layers: {self.gpu_layers} | Context: {self.context_size}")
        logger.info(f"  Reasoning: {'on' if self.reasoning else 'off'}")
        if self.mmproj_path:
            logger.info(f"  Vision: {self.mmproj_path.name}")
        logger.debug(f"  Command: {' '.join(cmd)}")

        try:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
        except OSError as e:
            logger.error(f"Failed to start llama-server: {e}")
            return False
        self._start_output_drain()

        # Wait for the server to become healthy
        logger.info(f"Waiting for llama-server to load model (up to {timeout}s)...")
        try:
            healthy = await self._wait_for_health(timeout)
        except Exception as e:
            logger.exception(f"Error while waiting for llama-server health: {e}")
            await self.stop()
            return False

        if healthy:
            self._healthy = True
            logger.info("llama-server is ready!")
            return True
        else:
            logger.error("llama-server failed to become healthy in time.")
            await self.stop()
            return False

    async def _wait_for_health(self, timeout: int) -> bool:
        """Poll the health endpoint until the server is ready."""
        start = time.time()
        async with httpx.AsyncClient() as client:
            while time.time() - start < timeout:
                # Check if process died
                if self._process and self._process.poll() is not None:
                    logger.error(f"llama-server exited with code {self._process.returncode}")
                    self._log_recent_output()
                    return False

                try:
                    resp = await client.get(self.health_url, timeout=2)
                    if resp.status_code == 200:
                        return True
                except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError):
                    pass

                await asyncio.sleep(1)

        logger.error("llama-server health check timed out.")
        self._log_recent_output()
        return False

    def _start_output_drain(self) -> None:
        """Continuously drain llama-server stdout so the child cannot block."""
        if not self._process or not self._process.stdout:
            return

        def drain() -> None:
            try:
                for line in self._process.stdout:
                    cleaned = line.rstrip()
                    if cleaned:
                        self._output_lines.append(cleaned)
                        logger.debug(f"llama-server: {cleaned}")
            except Exception as e:
                logger.debug(f"llama-server output reader stopped: {e}")

        self._output_thread = threading.Thread(
            target=drain,
            name="llama-server-output",
            daemon=True,
        )
        self._output_thread.start()

    def _log_recent_output(self) -> None:
        if not self._output_lines:
            logger.error("  server: no output captured")
            return
        for line in list(self._output_lines)[-20:]:
            logger.error(f"  server: {line}")

    async def stop(self):
        """Stop the llama-server gracefully.

        Waits for any in-flight inference to complete before terminating.
        """
        if not self._process or self._process.poll() is not None:
            self._process = None
            self._healthy = False
            return

        logger.info("Stopping llama-server (waiting for in-flight requests)...")

        # Check if the server is busy — poll /health which returns 503 during inference
        try:
            async with httpx.AsyncClient() as client:
                for _ in range(30):  # wait up to 30s for in-flight work
                    try:
                        resp = await client.get(self.health_url, timeout=2)
                        data = resp.json() if resp.status_code == 200 else {}
                        # llama-server /health returns {"status": "ok"} when idle
                        # and {"status": "loading model"} or 503 when busy
                        if resp.status_code == 200:
                            break
                    except (httpx.ConnectError, httpx.ReadTimeout):
                        break  # server already gone
                    await asyncio.sleep(1)
        except Exception:
            pass  # best effort — proceed with shutdown

        # Terminate the process
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
            logger.info("llama-server stopped.")
        except subprocess.TimeoutExpired:
            logger.warning("llama-server didn't stop gracefully — killing it.")
            self._process.kill()
            self._process.wait(timeout=5)

        self._process = None
        self._healthy = False

    def is_running(self) -> bool:
        """Check if the server process is alive."""
        return self._process is not None and self._process.poll() is None

    async def check_health(self) -> bool:
        """Quick health check — is the server responding?"""
        if not self.is_running():
            self._healthy = False
            return False
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(self.health_url, timeout=3)
                self._healthy = resp.status_code == 200
                return self._healthy
        except (httpx.ConnectError, httpx.ReadTimeout):
            self._healthy = False
            return False
