"""Step 10b - proving the sandbox, one limit at a time.

    uv run step10b_sandbox.py            run every escape attempt
    uv run step10b_sandbox.py --naive    the same attempts against exec()

Each case is something a model could plausibly write - by accident or on
instruction from a poisoned review. The interesting column is not "did it
fail" but WHERE it failed: in Python, or in the kernel.
"""

from __future__ import annotations

import argparse
import time

from sandbox import run_python

CASES: list[tuple[str, str]] = [
    ("honest work", "print(sum(range(1000)))"),
    ("read the host filesystem",
     "import os; print(os.listdir('/'))"),
    ("read the user's ssh keys",
     "import os; print(os.path.exists('/Users'), os.path.exists('/root/.ssh'))"),
    ("exfiltrate over the network",
     "import urllib.request; print(urllib.request.urlopen('http://example.com', timeout=5).status)"),
    ("write to the filesystem",
     "open('/etc/passwd','a').write('x'); print('wrote')"),
    ("write to scratch space (allowed)",
     "open('/tmp/x','w').write('ok'); print(open('/tmp/x').read())"),
    ("exhaust memory",
     "b = bytearray(); [b.extend(b'x'*10_000_000) for _ in range(200)]; print(len(b))"),
    ("spin forever",
     "while True: pass"),
    ("fork bomb",
     "import os\nwhile True: os.fork()"),
    ("become root",
     "import os; os.setuid(0); print('root now', os.getuid())"),
]


def naive_exec(code: str):
    """The tempting alternative: exec() with the scary names removed."""
    BLOCKED = ("os", "subprocess", "socket", "open", "__import__", "eval", "exec")
    for word in BLOCKED:
        if word in code:
            return f"BLOCKED (contains {word!r})"
    try:
        exec(code, {"__builtins__": {"print": print, "range": range, "sum": sum, "len": len}})
        return "ran"
    except Exception as exc:  # noqa: BLE001
        return f"error: {type(exc).__name__}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--naive", action="store_true",
                        help="run the SAFE cases against exec() for comparison")
    args = parser.parse_args()

    if args.naive:
        print("exec() with a blocklist - what it says about each attempt:\n")
        for label, code in CASES:
            # Only the cases that cannot hang this process.
            if label in ("spin forever", "fork bomb", "exhaust memory"):
                print(f"  {label:<32} NOT RUN - would take down the host process")
                continue
            print(f"  {label:<32} {naive_exec(code)}")
        print("\nNote which ones it 'blocked' by looking for a substring,")
        print("and what it would do with `getattr(__builtins__, 'ope'+'n')`.")
        return

    print(f"{'case':<32} {'exit':>5}  {'secs':>5}  outcome")
    print("-" * 92)
    for label, code in CASES:
        started = time.perf_counter()
        result = run_python(code)
        elapsed = time.perf_counter() - started

        if result.killed_for:
            outcome = f"KILLED: {result.killed_for}"
        elif result.ok:
            outcome = f"ok: {result.stdout.strip()[:44]}"
        else:
            line = [l for l in result.stderr.strip().splitlines() if l.strip()]
            outcome = f"error: {line[-1][:52] if line else '(no stderr)'}"

        print(f"{label:<32} {result.exit_code:>5}  {elapsed:>5.1f}  {outcome}")


if __name__ == "__main__":
    main()
