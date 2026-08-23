"""Gather recent Photo Booth captures into the capture folder.

Photo Booth writes into its own library and exports to wherever you drag them,
with filenames containing a narrow no-break space (U+202F) before AM/PM. That
character has already broken one literal path this session, so nothing here
constructs a filename by hand — everything comes from a directory scan.

    python3 tools/collect.py                 # last 30 minutes, into in/capture
    python3 tools/collect.py --minutes 120
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

SEARCH = [Path.home() / "Desktop", Path.home() / "Documents", Path.home() / "Downloads",
          Path.home() / "Pictures"]
SUFFIXES = {".jpg", ".jpeg", ".png", ".heic"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("in/capture"))
    ap.add_argument("--clear", action="store_true", help="empty the target folder first")
    args = ap.parse_args(argv)

    cutoff = time.time() - args.minutes * 60
    found: list[Path] = []
    for root in SEARCH:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in SUFFIXES or not path.is_file():
                continue
            try:
                if path.stat().st_mtime >= cutoff:
                    found.append(path)
            except OSError:
                continue

    found.sort(key=lambda p: p.stat().st_mtime)
    if not found:
        print(f"no images modified in the last {args.minutes} minutes under "
              f"{', '.join(str(s) for s in SEARCH)}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    if args.clear:
        for old in args.out.iterdir():
            if old.is_file():
                old.unlink()

    for i, src in enumerate(found, 1):
        dst = args.out / f"view{i:02d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        print(f"  {src.name}  ->  {dst}")

    print(f"\n{len(found)} image(s) collected into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
