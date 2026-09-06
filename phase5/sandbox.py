"""Running code the model wrote, without trusting it.

An agent that can execute Python is an agent that can do anything the
process it runs in can do: read your ~/.ssh, post your database to an
endpoint, spin the CPU forever. `exec()` with a blocklist does not change
that. Blocklists on Python are a losing game - `__builtins__` is reachable
through half a dozen dunder paths, and a filter that catches all of them
today catches none of them after the next release.

So the boundary is not in Python. It is the container:

    --network none          no egress. Not "no requests library" - no route.
    --read-only             the filesystem cannot be written, except one tmpfs
    --cap-drop ALL          no Linux capabilities at all
    --security-opt no-new-privileges
                            setuid binaries cannot regain them
    --user 65534:65534      nobody, not root
    --memory / --cpus       a runaway allocation is killed, not swapped
    --pids-limit            a fork bomb cannot fork
    no volume mounts        the host filesystem is not there to find
    a timeout               with `docker kill` behind it

This is the same lesson as step 7's SQL limits, one level up: the control
has to live somewhere the code being controlled cannot reach.
"""

from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass

IMAGE = "python:3.13-slim"

MEMORY = "256m"
CPUS = "0.5"
PIDS = "64"
TIMEOUT_SECONDS = 10
OUTPUT_LIMIT = 4_000


@dataclass
class Result:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    killed_for: str | None = None

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def run_python(code: str, *, timeout: int = TIMEOUT_SECONDS) -> Result:
    """Execute `code` in a throwaway container and return what it printed."""
    name = f"phase5-sbx-{uuid.uuid4().hex[:12]}"

    argv = [
        "docker", "run", "--rm", "--name", name,
        "--network", "none",
        "--read-only",
        # Code needs somewhere to write - a small, in-memory, non-executable
        # scratch space rather than a writable filesystem.
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--user", "65534:65534",
        "--memory", MEMORY,
        # Without this, memory pressure is answered by swapping instead of
        # by killing, and the limit stops being a limit.
        "--memory-swap", MEMORY,
        "--cpus", CPUS,
        "--pids-limit", PIDS,
        "--workdir", "/tmp",
        IMAGE,
        "python", "-I", "-c", code,
    ]

    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        timed_out = False
        stdout, stderr, code_ = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        # The Python-side timeout only stops US waiting. The container is
        # still running, so it has to be killed explicitly - otherwise a
        # sandbox escape is just "outlive the caller".
        subprocess.run(["docker", "kill", name], capture_output=True)
        timed_out = True
        stdout = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        code_ = -1

    killed_for = None
    if timed_out:
        killed_for = f"exceeded {timeout}s"
    elif code_ == 137:
        # 128 + SIGKILL(9): the kernel OOM killer, i.e. --memory did its job.
        killed_for = f"exceeded {MEMORY} of memory"

    return Result(
        stdout=stdout[:OUTPUT_LIMIT],
        stderr=stderr[:OUTPUT_LIMIT],
        exit_code=code_,
        timed_out=timed_out,
        killed_for=killed_for,
    )
