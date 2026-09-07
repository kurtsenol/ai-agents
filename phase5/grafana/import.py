"""Load the dashboard into a running Grafana.

    uv run grafana/import.py

A dashboard that exists only in someone's browser is a dashboard that
disappears with their laptop. This file and dashboard.json are the real
artifact; the panels in Grafana are a rendering of them.

Edits made by hand in the Grafana UI do NOT come back here. To keep a change,
export it from the dashboard's settings (JSON Model) over dashboard.json, or
re-run this after editing the file.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

GRAFANA_URL = "http://localhost:3000"
DASHBOARD = Path(__file__).parent / "dashboard.json"


def main() -> None:
    payload = json.dumps(
        {
            "dashboard": json.loads(DASHBOARD.read_text()),
            "overwrite": True,
            "message": "imported from grafana/dashboard.json",
        }
    ).encode()

    request = urllib.request.Request(
        f"{GRAFANA_URL}/api/dashboards/db",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = json.load(response)
    except urllib.error.URLError as exc:
        sys.exit(f"Grafana unreachable at {GRAFANA_URL} ({exc}). Is phase5-lgtm up?")

    print(f"{body.get('status')}: {GRAFANA_URL}{body.get('url', '')}")


if __name__ == "__main__":
    main()
