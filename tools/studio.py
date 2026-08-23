"""Seamless local studio: serve the repo root and open the viewer.

    python3 -m tools.studio
    python3 -m tools.studio --port 8765 --no-browser
"""

from __future__ import annotations

import argparse
import functools
import http.server
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="don't auto-open a tab")
    args = p.parse_args(argv)

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/viewer/"

    print(f"[studio] serving {ROOT}")
    print(f"[studio] {url}")
    print("[studio] drag a .glb onto the page, or pick one from the build chips")
    print("[studio] rebuild with: python3 -m avatarforge --fbx PATH --out out --name NAME --model-info")
    print("[studio] ctrl-c to stop")

    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
