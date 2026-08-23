"""Headless Chromium screenshots of a GLB through a real Three.js renderer.

    python3 tools/owl_shots.py viewer/assets/owl.glb --out shots/ \
        --view front:0:0 --view side:90:0 --clip idle=1.0 --clip wave=0.8

Why a real browser: the software rasterizers in meshforge are fine for
shape/weight debugging, but "do the colors read right", "does the
cornea catch light", "does the clip actually move the joint in a real
skinning shader" are only answerable by the renderer that will ship.
Memory note from an earlier session: DOM assertions and a clean console
passed while the arm never moved — compare pixels, not flags.
"""
from __future__ import annotations

import argparse
import asyncio
import http.server
import json
import os
import shutil
import socket
import tempfile
import threading
from pathlib import Path

BUNDLE_CANDIDATES = [
    Path(__file__).resolve().parent / "three_bundle.js",
    Path(os.environ.get("THREE_BUNDLE", "")) if os.environ.get("THREE_BUNDLE") else None,
]


def find_bundle() -> Path:
    for p in BUNDLE_CANDIDATES:
        if p and p.exists():
            return p
    raise FileNotFoundError("three_bundle.js not found next to tools/owl_shots.py (set THREE_BUNDLE=/path/to/three_bundle.js)")


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D401
        pass


def _serve(directory: str):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    handler = lambda *a, **k: _Quiet(*a, directory=directory, **k)  # noqa: E731
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


async def shoot(glb: Path, out_dir: Path, views: list[tuple[str, float, float]], clips: dict[str, float],
                size: int = 700, zoom: float = 1.0, target=(0.0, 0.0, 0.0), bg: str = "#efe6d4",
                joint_rotations: dict[str, list[float]] | None = None, prefix: str = "") -> list[Path]:
    from playwright.async_api import async_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="owl_shots_"))
    shutil.copy(find_bundle(), stage / "three_bundle.js")
    shutil.copy(Path(__file__).resolve().parent / "owl_shots.html", stage / "index.html")
    shutil.copy(glb, stage / "model.glb")
    httpd, port = _serve(str(stage))
    written = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
            page = await browser.new_page(viewport={"width": size, "height": size}, device_scale_factor=1)
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            await page.goto(f"http://127.0.0.1:{port}/index.html?glb=model.glb&size={size}&bg={bg.replace('#', '%23')}")
            await page.wait_for_function("window.owlShot && (window.owlShot.ready || window.owlShot.error)", timeout=120000)
            err = await page.evaluate("window.owlShot.error")
            if err:
                raise RuntimeError(f"GLB failed to load: {err}")
            info = await page.evaluate("({clips: window.owlShot.clips, joints: window.owlShot.joints})")
            (out_dir / f"{prefix}info.json").write_text(json.dumps(info, indent=2))
            for name, yaw, pitch in views:
                await page.evaluate("(v) => window.owlShot.setView(v)", {
                    "yaw": yaw, "pitch": pitch, "zoom": zoom, "target": list(target), "clips": clips,
                })
                for jname, q in (joint_rotations or {}).items():
                    await page.evaluate("([n, q]) => window.owlShot.setJointRotation(n, q)", [jname, q])
                await page.wait_for_timeout(120)
                path = out_dir / f"{prefix}{name}.png"
                await page.locator("canvas").screenshot(path=str(path))
                written.append(path)
            if errors:
                (out_dir / f"{prefix}console_errors.txt").write_text("\n".join(errors))
            await browser.close()
    finally:
        httpd.shutdown()
        shutil.rmtree(stage, ignore_errors=True)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("glb", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--view", action="append", default=[], help="name:yaw:pitch (degrees); repeatable")
    ap.add_argument("--clip", action="append", default=[], help="clipname=time_seconds; repeatable")
    ap.add_argument("--size", type=int, default=700)
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--target", type=str, default="0,0,0", help="look-at offset from the model centre, x,y,z")
    ap.add_argument("--bg", type=str, default="#efe6d4")
    ap.add_argument("--prefix", type=str, default="")
    args = ap.parse_args(argv)
    views = []
    for v in args.view or ["front:0:0"]:
        name, yaw, pitch = v.split(":")
        views.append((name, float(yaw), float(pitch)))
    clips = {}
    for c in args.clip:
        k, t = c.split("=")
        clips[k] = float(t)
    target = tuple(float(x) for x in args.target.split(","))
    paths = asyncio.run(shoot(args.glb, args.out, views, clips, size=args.size, zoom=args.zoom, target=target, bg=args.bg, prefix=args.prefix))
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
