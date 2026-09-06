"""The container limits. Slow (each case is a container start), so marked."""

import shutil
import subprocess

import pytest

import sandbox

docker_available = shutil.which("docker") is not None and (
    subprocess.run(["docker", "info"], capture_output=True).returncode == 0
)

pytestmark = [
    pytest.mark.skipif(not docker_available, reason="docker not running"),
    pytest.mark.slow,
]


def test_honest_code_runs():
    result = sandbox.run_python("print(sum(range(1000)))")
    assert result.ok
    assert result.stdout.strip() == "499500"


def test_no_network():
    result = sandbox.run_python(
        "import urllib.request; urllib.request.urlopen('http://example.com', timeout=5)"
    )
    assert not result.ok


def test_host_filesystem_is_absent():
    result = sandbox.run_python("import os; print(os.path.exists('/Users'))")
    assert result.stdout.strip() == "False"


def test_filesystem_is_read_only_except_tmp():
    assert not sandbox.run_python("open('/etc/passwd','a')").ok
    assert sandbox.run_python("open('/tmp/x','w').write('ok'); print('ok')").ok


def test_memory_limit_kills():
    result = sandbox.run_python(
        "b = bytearray()\n"
        "[b.extend(b'x'*10_000_000) for _ in range(200)]"
    )
    assert result.killed_for and "memory" in result.killed_for


def test_timeout_kills_and_reports_it():
    """The limit must surface as an error, never as empty output."""
    result = sandbox.run_python("while True: pass", timeout=3)
    assert result.timed_out
    assert result.killed_for


def test_cannot_become_root():
    assert not sandbox.run_python("import os; os.setuid(0)").ok
