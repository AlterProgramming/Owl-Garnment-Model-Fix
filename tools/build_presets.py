"""Generate one GLB per curated preset into out/, plus a manifest for the viewer picker.

    python3 tools/build_presets.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from avatarforge.presets import PRESETS  # noqa: E402

OUT = ROOT / "out"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for key, preset in PRESETS.items():
        name = f"preset_{key}"
        subprocess.run(
            [sys.executable, "-m", "avatarforge", "--preset", key,
             "--out", str(OUT), "--name", name, "--model-info"],
            cwd=ROOT, check=True,
        )
        entries.append({"file": f"{name}.glb", "label": preset.label, "description": preset.description})

    manifest = OUT / "manifest.json"
    manifest.write_text(json.dumps({"builds": entries}, indent=2))
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
