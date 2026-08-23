#!/usr/bin/env python3
"""Render deterministic evidence frames from the browser 2D puppet.

This intentionally captures the actual runtime, not concept art. The same SVG,
rig solver and animation clips shipped by puppet2d/index.html are evaluated in
Chromium, then sampled at fixed clip times.
"""
from __future__ import annotations

import argparse
import contextlib
import http.server
import socketserver
import threading
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw
from playwright.sync_api import sync_playwright


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="renders/puppet2d")
    ap.add_argument("--root", default="puppet2d")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    handler = lambda *a, **kw: QuietHandler(*a, directory=str(root), **kw)
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as server:
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                page = browser.new_page(viewport={"width": 1180, "height": 920}, device_scale_factor=1)
                page.goto(f"http://127.0.0.1:{port}/index.html", wait_until="networkidle")
                page.wait_for_selector('#stage[data-ready="true"]')

                frames = [
                    ("idle", 1.00, "idle.png"),
                    ("wave", 1.16, "wave.png"),
                    ("talk", 0.93, "talk.png"),
                    ("celebrate", 0.80, "celebrate.png"),
                ]
                for clip, t, name in frames:
                    page.evaluate("([clip,t]) => { window.owlPuppet.setClip(clip); window.owlPuppet.pause(); window.owlPuppet.seek(t); }", [clip, t])
                    page.locator("#stage-shell").screenshot(path=str(out / name))

                page.evaluate("() => { window.owlPuppet.setClip('wave'); window.owlPuppet.pause(); window.owlPuppet.seek(1.16); window.owlPuppet.setDebug(true); }")
                page.locator("#stage-shell").screenshot(path=str(out / "wave_rig_debug.png"))
                browser.close()

            make_sheet(out, [f[2] for f in frames])
        finally:
            server.shutdown()
            thread.join(timeout=2)

    print(out / "contact_sheet.png")
    return 0


def make_sheet(out: Path, names: list[str]) -> None:
    imgs = [Image.open(out / n).convert("RGB") for n in names]
    thumb_w = 520
    thumbs = []
    labels = [Path(n).stem.upper() for n in names]
    for im in imgs:
        ratio = thumb_w / im.width
        thumbs.append(im.resize((thumb_w, int(im.height * ratio)), Image.Resampling.LANCZOS))
    pad, label_h = 24, 42
    cell_h = max(im.height for im in thumbs) + label_h
    sheet = Image.new("RGB", (thumb_w * 2 + pad * 3, cell_h * 2 + pad * 3), "#f3efe8")
    draw = ImageDraw.Draw(sheet)
    for i, (im, label) in enumerate(zip(thumbs, labels)):
        col, row = i % 2, i // 2
        x = pad + col * (thumb_w + pad)
        y = pad + row * (cell_h + pad)
        sheet.paste(im, (x, y + label_h))
        draw.text((x + 4, y + 8), label, fill="#33231c")
    sheet.save(out / "contact_sheet.png", quality=94)


if __name__ == "__main__":
    raise SystemExit(main())
