"""Pixel verification for the owl artifact page."""
import asyncio, http.server, json, socket, sys, threading
from pathlib import Path
import numpy as np
from PIL import Image
from playwright.async_api import async_playwright

HERE = Path(__file__).resolve().parent
OUT = HERE / "shots"

class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass

def serve(d):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), lambda *a, **k: Q(*a, directory=str(d), **k))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port

def diff(a, b, box=None):
    """Mean absolute pixel difference, optionally over a fractional box
    (x0, y0, x1, y1). A whole-frame mean is scale-dependent: the wave moves
    one wing, which is a few percent of a 16:10 stage, so its whole-frame
    number sits near the noise floor while a hop (whole body) reads 15x
    higher. Measure each motion where it happens."""
    ia = np.asarray(Image.open(a).convert("RGB"), dtype=np.float32)
    ib = np.asarray(Image.open(b).convert("RGB"), dtype=np.float32)
    if box is not None:
        h, w = ia.shape[:2]
        x0, y0, x1, y1 = box
        sl = (slice(int(y0 * h), int(y1 * h)), slice(int(x0 * w), int(x1 * w)))
        ia, ib = ia[sl], ib[sl]
    return float(np.abs(ia - ib).mean())

async def main():
    OUT.mkdir(exist_ok=True)
    httpd, port = serve(HERE)
    try:
        async with async_playwright() as p:
            b = await p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
            page = await b.new_page(viewport={"width": 1200, "height": 1000}, device_scale_factor=1)
            errors = []
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            await page.goto(f"http://127.0.0.1:{port}/owl_artifact.html", wait_until="load")
            await page.wait_for_function("window.__owlReady === true", timeout=180000)
            await page.wait_for_timeout(1200)
            await page.screenshot(path=str(OUT / "page.png"), full_page=True)
            stage = page.locator("#stage")

            # freeze the idle clip so each capture isolates the clip under test
            await page.evaluate("window.__owl.setPointer(0.5, 0.5)")
            await page.wait_for_timeout(1200)
            await page.evaluate("window.__owl.freeze(true)")
            await page.wait_for_timeout(300)
            await stage.screenshot(path=str(OUT / "rest.png"))

            async def shot(name, clip, t):
                await page.evaluate("([c, t]) => window.__owl.poseClip(c, t)", [clip, t])
                await page.wait_for_timeout(160)
                await stage.screenshot(path=str(OUT / f"{name}.png"))

            await shot("wave", "wave", 0.62)
            await shot("hop", "hop", 0.34)
            await shot("blink", "blink", 0.10)
            await shot("tablet", "tablet_show", 0.62)
            await page.evaluate("window.__owl.poseClip('idle', 0)")
            await page.evaluate("window.__owl.freeze(false)")
            await page.evaluate("window.__owl.freeze(true)")
            await page.evaluate("window.__owl.setPointer(0.05, 0.5)"); await page.wait_for_timeout(1000)
            await stage.screenshot(path=str(OUT / "gaze_left.png"))
            await page.evaluate("window.__owl.setPointer(0.95, 0.5)"); await page.wait_for_timeout(1000)
            await stage.screenshot(path=str(OUT / "gaze_right.png"))
            await page.evaluate("window.__owl.toggleSkeleton()")
            await page.wait_for_timeout(300)
            await stage.screenshot(path=str(OUT / "skeleton.png"))
            stats = await page.locator("#stats").text_content()
            await b.close()
    finally:
        httpd.shutdown()

    # regions: the raised wing lives left-of-centre, the eyes in the upper middle
    WING = (0.10, 0.05, 0.50, 0.75)
    FACE = (0.30, 0.05, 0.70, 0.45)
    diffs = {n: diff(OUT / "rest.png", OUT / f"{n}.png") for n in ("wave", "hop", "blink", "tablet", "skeleton")}
    diffs["wave_wing"] = diff(OUT / "rest.png", OUT / "wave.png", WING)
    diffs["blink_face"] = diff(OUT / "rest.png", OUT / "blink.png", FACE)
    diffs["gaze_left_vs_right"] = diff(OUT / "gaze_left.png", OUT / "gaze_right.png")
    diffs["gaze_face"] = diff(OUT / "gaze_left.png", OUT / "gaze_right.png", FACE)
    print(json.dumps({"stats": stats.strip(), "diffs": diffs, "console_errors": [e for e in errors if "GL Driver" not in e]}, indent=2))
    thresholds = {"wave_wing": 2.0, "hop": 2.0, "blink_face": 2.0, "tablet": 1.0,
                  "gaze_face": 1.0, "skeleton": 0.5}
    fails = [f"{k} {diffs[k]:.2f} < {v}" for k, v in thresholds.items() if diffs[k] < v]
    if [e for e in errors if "GL Driver" not in e]: fails.append("console errors")
    for f in fails: print("FAIL", f)
    print("OK" if not fails else f"{len(fails)} failure(s)")
    return 1 if fails else 0

sys.exit(asyncio.run(main()))
