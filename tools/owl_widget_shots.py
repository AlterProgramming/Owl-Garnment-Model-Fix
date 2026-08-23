"""Pixel verification for the Babylon owl-guide corner widget.

    python3 tools/owl_widget_shots.py --out shots/widget

Serves `viewer/` over HTTP (the Babylon CDN and the environment texture are
network fetches), opens `owl_guide_test.html` in headless Chromium, waits
for the widget to report `ready`, then drives `window.__owlGuide` to
capture rest / gaze / clip frames and prints the mean absolute pixel
difference of each against rest.

The diffs are the point: a widget can log "ready", answer every DOM query
and still render a frozen owl — this repo has already shipped that exact
bug once (see the project memory note on verifying 3D by pixels).
"""
from __future__ import annotations

import argparse
import asyncio
import http.server
import json
import socket
import threading
from pathlib import Path

import numpy as np
from PIL import Image

VIEWER_DIR = Path(__file__).resolve().parent.parent / "viewer"


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D401
        pass


def serve(directory: Path):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    handler = lambda *a, **k: _Quiet(*a, directory=str(directory), **k)  # noqa: E731
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port


def mean_abs_diff(a: Path, b: Path) -> float:
    ia = np.asarray(Image.open(a).convert("RGB"), dtype=np.float32)
    ib = np.asarray(Image.open(b).convert("RGB"), dtype=np.float32)
    if ia.shape != ib.shape:
        raise ValueError(f"shape mismatch {ia.shape} vs {ib.shape}")
    return float(np.abs(ia - ib).mean())


async def run(out_dir: Path, viewer_dir: Path, keep_open: bool = False) -> dict:
    from playwright.async_api import async_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    httpd, port = serve(viewer_dir)
    results: dict = {"shots": {}, "diffs": {}, "console_errors": []}
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(args=[
                "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist",
            ])
            page = await browser.new_page(viewport={"width": 900, "height": 700}, device_scale_factor=1)
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            await page.goto(f"http://127.0.0.1:{port}/owl_guide_test.html", wait_until="load")
            await page.wait_for_function(
                "window.__owlGuideState === 'ready' || window.__owlGuideState === 'fallback'", timeout=120000,
            )
            state = await page.evaluate("window.__owlGuideState")
            results["state"] = state
            if state != "ready":
                raise RuntimeError(f"widget did not reach 'ready' (state={state}); errors={errors}")

            panel = page.locator("#owlGuidePanel")
            # Freeze the ambient sway + idle clip: otherwise every capture
            # differs from rest by the sway phase and the diffs below prove
            # nothing about the clip under test.
            await page.evaluate("window.__owlGuide.freeze(true)")

            async def shot(name: str, wait_ms: int = 400):
                await page.wait_for_timeout(wait_ms)
                path = out_dir / f"{name}.png"
                await panel.screenshot(path=str(path))
                results["shots"][name] = str(path)
                return path

            # Rest: park the gaze straight ahead and let the idle clip settle.
            await page.evaluate("window.__owlGuide.setGazeTarget(0.5, 0.5)")
            rest = await shot("rest", 1500)

            await page.evaluate("window.__owlGuide.setGazeTarget(0.02, 0.5)")
            gaze_left = await shot("gaze_left", 900)
            await page.evaluate("window.__owlGuide.setGazeTarget(0.98, 0.5)")
            gaze_right = await shot("gaze_right", 900)
            await page.evaluate("window.__owlGuide.setGazeTarget(0.5, 0.5)")
            await page.wait_for_timeout(700)

            await page.evaluate("window.__owlGuide.play('wave')")
            wave = await shot("wave", 600)
            await page.wait_for_timeout(1400)

            await page.evaluate("window.__owlGuide.play('hop')")
            hop = await shot("hop", 320)
            await page.wait_for_timeout(900)

            await page.evaluate("window.__owlGuide.play('blink')")
            blink = await shot("blink", 100)
            await page.wait_for_timeout(600)

            await page.evaluate("window.__owlGuide.play('tablet_show')")
            tablet = await shot("tablet_show", 600)

            results["clips"] = await page.evaluate("window.__owlGuide.clips")
            results["diffs"] = {
                "wave": mean_abs_diff(rest, wave),
                "hop": mean_abs_diff(rest, hop),
                "blink": mean_abs_diff(rest, blink),
                "tablet_show": mean_abs_diff(rest, tablet),
                "gaze_left_vs_right": mean_abs_diff(gaze_left, gaze_right),
            }
            results["console_errors"] = [e for e in errors if "GL Driver Message" not in e]
            if keep_open:
                await page.wait_for_timeout(5000)
            await browser.close()
    finally:
        httpd.shutdown()
    return results


THRESHOLDS = {"wave": 2.0, "hop": 2.0, "gaze_left_vs_right": 0.3}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("shots/widget"))
    ap.add_argument("--viewer", type=Path, default=VIEWER_DIR)
    args = ap.parse_args(argv)
    results = asyncio.run(run(args.out, args.viewer))
    print(json.dumps({k: results[k] for k in ("state", "clips", "diffs", "console_errors")}, indent=2, default=str))
    failures = [f"{k}: {results['diffs'][k]:.3f} < {v}" for k, v in THRESHOLDS.items() if results["diffs"].get(k, 0) < v]
    if results["console_errors"]:
        failures.append(f"{len(results['console_errors'])} console error(s)")
    for f in failures:
        print("FAIL ", f)
    print("OK" if not failures else f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
