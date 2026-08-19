"""
Docker-based execution sandbox for the operations that need real isolation:
run_command, run_tests, apply_patch, and git clone. Everything else
(list_files/read_file/search_code/retrieve_context) stays a direct,
path-validated, read-only host filesystem read -- no isolation benefit to
shelling those into a container.

Design: one persistent, --network none container per repo_root ("session
sandbox"), exec'd into repeatedly and stopped when the session ends -- an
LLM-issued run_command can't reach the network regardless of what
tools.py's DANGEROUS_COMMAND_PATTERNS denylist did or didn't catch. git
clone is the one op that legitimately needs network and is human- (not
LLM-) initiated, so it gets its own ephemeral, network-enabled container
(run_ephemeral()) instead of loosening the session sandbox's policy.
Resource limits + a non-root user (see Dockerfile) on every container.

This is a denylist-free boundary: the guarantee isn't "we blocked the
dangerous command", it's "even one we failed to block only affects this
container". Opt-in via $AGENT_SANDBOX_ENABLED=1; falls back to direct host
execution when unset or Docker isn't reachable (see tools/tools.py and
api/server.py's _clone_repo).
"""

from __future__ import annotations

import atexit
import hashlib
import os
import subprocess
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

DEFAULT_IMAGE = "agent-sandbox:latest"
DOCKERFILE_DIR = str(Path(__file__).parent)
CONTAINER_WORKDIR = "/workspace"

_docker_available: Optional[bool] = None  # cached after first check
_image_built = False


def is_sandbox_enabled() -> bool:
    return os.environ.get("AGENT_SANDBOX_ENABLED", "0").strip().lower() in {"1", "true", "yes"}


def is_docker_available(force_recheck: bool = False) -> bool:
    global _docker_available
    if _docker_available is not None and not force_recheck:
        return _docker_available
    try:
        proc = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        _docker_available = proc.returncode == 0
    except Exception:
        _docker_available = False
    return _docker_available


def ensure_image_built(image: str = DEFAULT_IMAGE, dockerfile_dir: str = DOCKERFILE_DIR, timeout: int = 300) -> None:
    """Build the sandbox image if it doesn't already exist. Idempotent."""
    global _image_built
    if _image_built:
        return

    check = subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=10)
    if check.returncode == 0:
        _image_built = True
        return

    proc = subprocess.run(
        ["docker", "build", "-t", image, dockerfile_dir],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to build sandbox image: {proc.stderr[-2000:]}")
    _image_built = True


class DockerSandbox:
    """A persistent, network-disabled container for one repo_root."""

    def __init__(
        self,
        repo_root: str,
        image: str = DEFAULT_IMAGE,
        network: str = "none",
        memory: str = "512m",
        cpus: str = "1.0",
        pids_limit: int = 256,
        name: Optional[str] = None,
    ):
        self.repo_root = str(Path(repo_root).resolve())
        self.image = image
        self.network = network
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        # sha1, not Python's built-in hash(): hash() is salted with a random
        # seed per process (PYTHONHASHSEED), so the "same" repo_root would
        # get a different container name every time the server restarts --
        # a container orphaned by a crash (skips the atexit cleanup) could
        # never be found and removed by a later process, since it wouldn't
        # know what name to look for. sha1 is stable across restarts.
        self.container_name = name or f"agent-sandbox-{hashlib.sha1(self.repo_root.encode()).hexdigest()[:16]}"
        self._started = False
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        if not self._started:
            return False
        proc = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", self.container_name],
            capture_output=True, text=True, timeout=10,
        )
        return proc.returncode == 0 and proc.stdout.strip() == "true"

    def start(self, timeout: int = 60) -> None:
        with self._lock:
            if self.is_running:
                return

            ensure_image_built(self.image)
            Path(self.repo_root).mkdir(parents=True, exist_ok=True)

            # Clear out any stale container of the same name (e.g. left over
            # from a crash) before starting a fresh one.
            subprocess.run(["docker", "rm", "-f", self.container_name], capture_output=True, timeout=10)

            cmd = [
                "docker", "run", "-d",
                "--name", self.container_name,
                "--network", self.network,
                "--memory", self.memory,
                "--cpus", self.cpus,
                "--pids-limit", str(self.pids_limit),
                "-v", f"{self.repo_root}:{CONTAINER_WORKDIR}",
                "-w", CONTAINER_WORKDIR,
                self.image,
                "sleep", "infinity",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if proc.returncode != 0:
                raise RuntimeError(f"docker run failed: {proc.stderr.strip()}")
            self._started = True

    def exec(self, command: str, timeout: int = 60, workdir: Optional[str] = None) -> Tuple[int, str, str]:
        """Run `command` (via `sh -c`) inside the container. Returns (exit_code, stdout, stderr)."""
        if not self.is_running:
            self.start()

        cmd = ["docker", "exec"]
        if workdir:
            cmd += ["-w", workdir]
        cmd += [self.container_name, "sh", "-c", command]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout} seconds."

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            subprocess.run(["docker", "rm", "-f", self.container_name], capture_output=True, timeout=15)
            self._started = False


def run_ephemeral(
    cwd: str,
    command: str,
    image: str = DEFAULT_IMAGE,
    network: str = "bridge",
    timeout: int = 120,
    memory: str = "512m",
    cpus: str = "1.0",
) -> Tuple[int, str, str]:
    """
    Run one command in a throwaway (--rm) container bind-mounting `cwd` at
    /workspace. For one-off operations that need network (git clone), where
    a persistent, network-disabled session sandbox would be the wrong tool.
    """
    ensure_image_built(image)
    Path(cwd).mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        "--network", network,
        "--memory", memory,
        "--cpus", cpus,
        "--pids-limit", "128",
        "-v", f"{Path(cwd).resolve()}:{CONTAINER_WORKDIR}",
        "-w", CONTAINER_WORKDIR,
        image, "sh", "-c", command,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout} seconds."


# ── process-wide registry (mirrors agent_mcp/client.py's shared-client cache) ──

_sandboxes: Dict[str, DockerSandbox] = {}
_registry_lock = threading.Lock()


def get_shared_sandbox(repo_root: str, **kwargs) -> DockerSandbox:
    """One persistent DockerSandbox per repo_root, reused across calls."""
    key = str(Path(repo_root).resolve())

    with _registry_lock:
        existing = _sandboxes.get(key)
        if existing is not None:
            return existing

    candidate = DockerSandbox(repo_root=repo_root, **kwargs)
    candidate.start()

    with _registry_lock:
        winner = _sandboxes.setdefault(key, candidate)
        if winner is not candidate:
            candidate.stop()
        return winner


def close_shared_sandbox(repo_root: str) -> None:
    key = str(Path(repo_root).resolve())
    with _registry_lock:
        sandbox = _sandboxes.pop(key, None)
    if sandbox is not None:
        sandbox.stop()


def close_all_shared_sandboxes() -> None:
    with _registry_lock:
        sandboxes = list(_sandboxes.values())
        _sandboxes.clear()
    for s in sandboxes:
        try:
            s.stop()
        except Exception:
            pass


atexit.register(close_all_shared_sandboxes)
