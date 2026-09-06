"""Make the phase 5 modules importable from tests/ without installing them.

phase5 is a script directory, not a package, so `import approval` only works
when phase5/ is on the path. Adding it here (rather than with a PYTHONPATH in
the CI workflow) means `uv run pytest` behaves identically on a laptop and on
a runner - a test suite that needs environment setup to pass is a test suite
that gets skipped.
"""

import sys
from pathlib import Path

PHASE5 = Path(__file__).resolve().parent.parent
REPO = PHASE5.parent

for path in (PHASE5, REPO / "phase4", REPO / "phase4/eval"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
