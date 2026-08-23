"""Assemble the self-contained owl artifact page.

    python3 build.py [--glb PATH] [--out owl_artifact.html]

Inlines tools/three_bundle.js and base64-embeds the GLB into template.html.
Kept as a script because the model is rebuilt often and the page has to be
reassembled byte-for-byte each time.
"""
import argparse, base64, pathlib, sys

REPO = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--glb", type=pathlib.Path, default=REPO / "viewer/assets/owl.glb")
ap.add_argument("--bundle", type=pathlib.Path, default=REPO / "tools/three_bundle.js")
ap.add_argument("--out", type=pathlib.Path, default=HERE / "owl_artifact.html")
args = ap.parse_args()

template = (HERE / "template.html").read_text()
bundle = args.bundle.read_text()
b64 = base64.b64encode(args.glb.read_bytes()).decode()
for token in ("__THREE_BUNDLE__", "__GLB_B64__"):
    if token not in template:
        sys.exit(f"template is missing {token}")
html = template.replace("__THREE_BUNDLE__", bundle).replace("__GLB_B64__", b64)
args.out.write_text(html)
print(f"{args.out}  {len(html)/1e6:.2f} MB  (glb {args.glb.stat().st_size/1e6:.2f} MB, bundle {len(bundle)/1e6:.2f} MB)")
