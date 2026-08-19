"""
Live-Docker tests for sandbox/docker_sandbox.py.

Split out of test_all.py on purpose, same reasoning as test_db_store.py:
test_all.py's contract is "no external services required". These tests
start real containers and need Docker actually reachable:

    open -a Docker   # or: colima start
    docker build -t agent-sandbox:latest sandbox/
    AGENT_SANDBOX_ENABLED=1 python -m pytest testcase/test_sandbox.py -v

Skipped automatically (not failed) when Docker isn't reachable, so this file
is still safe to include in a default `pytest testcase/` sweep.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sandbox.docker_sandbox import is_docker_available  # noqa: E402

DOCKER_UP = is_docker_available(force_recheck=True)


@unittest.skipUnless(DOCKER_UP, "Docker daemon not reachable -- see module docstring to run this file")
class TestDockerSandboxLive(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from sandbox.docker_sandbox import ensure_image_built
        try:
            ensure_image_built(timeout=300)
        except Exception as e:
            raise unittest.SkipTest(f"could not build sandbox image: {e}")

    def setUp(self):
        self.repo = Path(tempfile.mkdtemp(prefix="sandbox_test_"))
        (self.repo / "hello.py").write_text("print('hi')\n")

    def tearDown(self):
        from sandbox.docker_sandbox import close_shared_sandbox
        close_shared_sandbox(str(self.repo))
        shutil.rmtree(self.repo, ignore_errors=True)

    def test_exec_runs_inside_container_not_host(self):
        from sandbox.docker_sandbox import get_shared_sandbox
        sb = get_shared_sandbox(str(self.repo))
        code, out, err = sb.exec("pwd")
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "/workspace")  # host cwd would never be this

    def test_bind_mount_is_two_way(self):
        from sandbox.docker_sandbox import get_shared_sandbox
        sb = get_shared_sandbox(str(self.repo))

        # host -> container
        code, out, _ = sb.exec("cat hello.py")
        self.assertEqual(code, 0)
        self.assertIn("hi", out)

        # container -> host
        sb.exec("echo written > from_container.txt")
        self.assertEqual((self.repo / "from_container.txt").read_text().strip(), "written")

    def test_network_is_disabled_by_default(self):
        from sandbox.docker_sandbox import get_shared_sandbox
        sb = get_shared_sandbox(str(self.repo))
        (self.repo / "nettest.py").write_text(
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('8.8.8.8', 53), timeout=3)\n"
            "    print('REACHABLE')\n"
            "except Exception:\n"
            "    print('BLOCKED')\n"
        )
        code, out, _ = sb.exec("python3 nettest.py", timeout=10)
        self.assertEqual(code, 0)
        self.assertIn("BLOCKED", out)

    def test_pytest_is_available_in_image(self):
        from sandbox.docker_sandbox import get_shared_sandbox
        sb = get_shared_sandbox(str(self.repo))
        code, out, _ = sb.exec("python3 -m pytest --version")
        self.assertEqual(code, 0)
        self.assertIn("pytest", out.lower())

    def test_shared_sandbox_is_reused(self):
        from sandbox.docker_sandbox import get_shared_sandbox
        a = get_shared_sandbox(str(self.repo))
        b = get_shared_sandbox(str(self.repo))
        self.assertIs(a, b)

    def test_close_shared_sandbox_stops_container(self):
        from sandbox.docker_sandbox import get_shared_sandbox, close_shared_sandbox
        sb = get_shared_sandbox(str(self.repo))
        self.assertTrue(sb.is_running)
        close_shared_sandbox(str(self.repo))
        self.assertFalse(sb.is_running)

    def test_run_command_tool_routes_through_sandbox(self):
        import os
        from core.state import AgentState
        from tools.tools import run_command

        os.environ["AGENT_SANDBOX_ENABLED"] = "1"
        try:
            state = AgentState(input_query="x", repo_root=str(self.repo))
            r = run_command(state, command="pwd")
            self.assertTrue(r.success)
            self.assertIn("/workspace", r.output)
        finally:
            os.environ.pop("AGENT_SANDBOX_ENABLED", None)
            from sandbox.docker_sandbox import close_shared_sandbox
            close_shared_sandbox(str(self.repo))

    def test_apply_patch_tool_routes_through_sandbox(self):
        import os
        import subprocess
        from core.state import AgentState
        from tools.tools import apply_patch

        subprocess.run(["git", "init", "-q"], cwd=self.repo)
        subprocess.run(["git", "config", "user.email", "a@b.com"], cwd=self.repo)
        subprocess.run(["git", "config", "user.name", "a"], cwd=self.repo)
        subprocess.run(["git", "add", "."], cwd=self.repo)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=self.repo)

        (self.repo / "hello.py").write_text("print('hi')\nprint('bye')\n")
        diff = subprocess.run(
            ["git", "diff", "hello.py"], cwd=self.repo, capture_output=True, text=True
        ).stdout
        subprocess.run(["git", "checkout", "--", "hello.py"], cwd=self.repo)  # revert working copy

        os.environ["AGENT_SANDBOX_ENABLED"] = "1"
        try:
            state = AgentState(input_query="x", repo_root=str(self.repo))
            r = apply_patch(state, patch=diff)
            self.assertTrue(r.success, r.error)
            self.assertIn("bye", (self.repo / "hello.py").read_text())
            # temp patch file must not be left behind
            self.assertEqual(list(self.repo.glob(".agent_patch_*")), [])
        finally:
            os.environ.pop("AGENT_SANDBOX_ENABLED", None)
            from sandbox.docker_sandbox import close_shared_sandbox
            close_shared_sandbox(str(self.repo))


if __name__ == "__main__":
    unittest.main(verbosity=2)
