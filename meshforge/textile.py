"""Procedural kente-weave textile generator.

Real kente cloth is not painted, it's built: narrow (roughly 10 cm) strips
of warp are woven on a horizontal loom, each strip its own repeating
sequence of solid weft-faced bands and warp-float motif bands (checker,
zigzag, cross), sewn edge to edge into cloth, with neighbouring strips
carrying different but harmonious recipes so the assembled cloth doesn't
read as one flat repeat.

`generate_weave` reproduces that structure as pixels rather than asking a
diffusion model to draw fabric and hoping the seam matches: every strip
and every block repeats on an exact pixel modulus of the output, so the
image tiles by construction in both axes — no seam-matching, no
re-rolling. See docs/superpowers/specs/2026-08-21-owl-kente-textile-design.md.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

Color = tuple[float, float, float]  # sRGB in [0, 1], matches meshforge.paint.to_image


@dataclasses.dataclass(frozen=True)
class Colorway:
    name: str
    ground: tuple[Color, ...]  # 2+ base strip colors, rotated across strip recipes
    accent: Color              # thin thread line between blocks
    motif: Color                # the warp-float motif's contrasting color


ASANTE_GOLD = Colorway(
    name="asante_gold",
    ground=((0.80, 0.62, 0.05), (0.62, 0.05, 0.05), (0.05, 0.35, 0.10)),  # gold, deep red, forest green
    accent=(0.95, 0.95, 0.90),
    motif=(0.05, 0.05, 0.05),
)

EWE_ADANUDO = Colorway(
    name="ewe_adanudo",
    ground=((0.05, 0.15, 0.45), (0.62, 0.05, 0.05), (0.80, 0.62, 0.05)),  # indigo, red, gold
    accent=(0.95, 0.95, 0.90),
    motif=(0.05, 0.05, 0.05),
)

COLORWAYS = {c.name: c for c in (ASANTE_GOLD, EWE_ADANUDO)}

MOTIFS: tuple[str, ...] = ("checker", "nkyimkyim", "cross", "babadua")
# Named after real Asante/Ewe weft-float motifs, not generic geometry —
# verified against secondary sources (Wikipedia's Kente cloth entry;
# essenceoftheroadart.online's motif glossary), not assumed from memory:
#   nkyimkyim — zigzag; "life's twists and the adaptability they demand"
#   babadua   — tight horizontal bars stacked like bamboo joints (the reed
#               it's named for); cooperation and resilience of bound things
# "checker" and "cross" are common kente weft-float shapes but weren't
# matched to a specific named motif in what was checked, so they stay
# generic rather than claim a name that wasn't verified.

# (kind, fraction) — fractions sum to 1.0; a "solid"/"accent" band's color
# alternates by index parity, a "motif" band draws one of MOTIFS.
_BLOCK_LAYOUT: tuple[tuple[str, float], ...] = (
    ("solid", 0.16),
    ("motif", 0.44),
    ("accent", 0.06),
    ("solid", 0.16),
    ("accent", 0.06),
    ("solid", 0.12),
)


@dataclasses.dataclass(frozen=True)
class WeaveParams:
    colorway: Colorway
    strip_px: int = 64        # strip width in pixels
    strip_cycle: int = 3      # distinct strip recipes before the pattern repeats horizontally
    block_px: int = 96        # vertical repeat unit
    motifs: Sequence[str] = MOTIFS
    seed: int = 0


def _recipe(colorway: Colorway, motifs: Sequence[str], seed: int, strip_index: int) -> dict:
    rng = np.random.default_rng(seed + 1_000_003 * strip_index)
    ground = colorway.ground
    a = int(rng.integers(0, len(ground)))
    b = (a + 1 + int(rng.integers(0, max(1, len(ground) - 1)))) % len(ground)
    return {
        "color_a": np.array(ground[a], dtype=np.float64),
        "color_b": np.array(ground[b], dtype=np.float64),
        "motif": motifs[int(rng.integers(0, len(motifs)))],
    }


def _motif_mask(kind: str, xs: np.ndarray, ym: np.ndarray, strip_px: int, motif_px: int) -> np.ndarray:
    """Boolean mask (True = motif color) over pixel-integer coordinates
    `xs` (0..strip_px-1, position within the strip) and `ym` (0..motif_px-1,
    position within the motif band) — both periodic by construction."""
    if kind == "checker":
        cell = max(1, strip_px // 4)
        return ((xs // cell) + (ym // max(1, motif_px // 4))) % 2 == 0
    if kind == "nkyimkyim":
        # solid interlocking triangles — a zigzag, not a thin outlined
        # squiggle: several short periods across the strip so it reads as
        # a woven chevron texture rather than one big glyph.
        period = max(2, strip_px // 4)
        phase = xs % period
        tri = np.minimum(phase, period - phase)  # triangle wave, period `period`
        boundary = (tri * motif_px * 2) // max(1, period)
        return ym < np.minimum(boundary, motif_px)
    if kind == "cross":
        # isolated plus-signs on a grid, not a plaid of full-length lines:
        # only mark pixels within `arm` of a grid intersection.
        cols, rows = 4, max(1, motif_px // max(4, strip_px // 4))
        cell_x = max(2, strip_px // cols)
        cell_y = max(2, motif_px // rows)
        dx = np.abs((xs + cell_x // 2) % cell_x - cell_x // 2)
        dy = np.abs((ym + cell_y // 2) % cell_y - cell_y // 2)
        arm = max(1, min(cell_x, cell_y) // 3)
        thick = max(1, arm // 3)
        return ((dx < thick) & (dy < arm)) | ((dy < thick) & (dx < arm))
    if kind == "babadua":
        # tight horizontal bars stacked like bamboo joints, per its
        # namesake reed — alternating bands stacked along the strip's
        # *length* (ym), not vertical stripes across its width (an earlier
        # version of this drew vertical stripes here, which is a
        # different, unnamed pattern; babadua is specifically horizontal).
        cell = max(2, motif_px // 6)
        return (ym // cell) % 2 == 0
    raise ValueError(f"unknown motif {kind!r}")


_BAND_HEIGHT = {"solid": 0.35, "accent": 0.55, "ground": 0.35, "motif": 0.80}
# Real kente weft-float motifs sit physically raised over the ground
# weave — the supplementary weft thread floats over the surface to make
# the pattern, then dips under for the ground — which flat base color
# alone can never convey under lighting. This is a coarse approximation
# of that relief (not a measured fabric profile), used to derive a normal
# map so the motif reads as woven rather than printed.


def weave_height_at(x: np.ndarray, y: np.ndarray, params: WeaveParams) -> np.ndarray:
    """Grayscale height in [0, 1], shape (*x.shape) — same band/strip
    structure and periodicity contract as `weave_color_at` (arbitrary
    real-valued, arbitrarily-shaped `x`/`y`), describing physical relief
    instead of color. Feed this to a Sobel/central-difference normal-map
    step, not a lighting model directly."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    yb = yi % params.block_px

    out = np.full(x.shape, _BAND_HEIGHT["ground"], dtype=np.float64)
    band_starts = np.cumsum([0.0, *[f for _, f in _BLOCK_LAYOUT]]) * params.block_px
    band_starts = band_starts.round().astype(int)

    for i, (kind, _frac) in enumerate(_BLOCK_LAYOUT):
        lo, hi = band_starts[i], band_starts[i + 1]
        band_mask = (yb >= lo) & (yb < hi)
        if not band_mask.any():
            continue
        if kind == "motif":
            strip_index = (xi // params.strip_px) % params.strip_cycle
            xs = xi % params.strip_px
            for k in range(params.strip_cycle):
                sub = band_mask & (strip_index == k)
                if not sub.any():
                    continue
                recipe = _recipe(params.colorway, params.motifs, params.seed, k)
                ym = yb[sub] - lo
                m = _motif_mask(recipe["motif"], xs[sub], ym, params.strip_px, hi - lo)
                out[sub] = np.where(m, _BAND_HEIGHT["motif"], _BAND_HEIGHT["ground"])
        else:
            out[band_mask] = _BAND_HEIGHT[kind]
    return out


def weave_color_at(x: np.ndarray, y: np.ndarray, params: WeaveParams) -> np.ndarray:
    """RGBA float in [0, 1], shape (*x.shape, 4) — `x` and `y` broadcast
    together first. `x`/`y` are pixel-unit coordinates and may be *any*
    real values in *any* shape: a dense grid (what `generate_weave` below
    builds), or an arbitrary scattered set (what `kente.py` evaluates at
    real 3D surface positions when baking onto a UV atlas it doesn't
    control the layout of). Only the integer floor and modulus of each
    coordinate matter, so the weave is exactly as periodic here as in
    `generate_weave` — same strips, same blocks, same motifs, just not
    tied to being called on a grid."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    strip_index = (xi // params.strip_px) % params.strip_cycle
    xs = xi % params.strip_px
    yb = yi % params.block_px

    out = np.zeros((*x.shape, 4), dtype=np.float64)
    out[..., 3] = 1.0  # opaque

    band_starts = np.cumsum([0.0, *[f for _, f in _BLOCK_LAYOUT]]) * params.block_px
    band_starts = band_starts.round().astype(int)

    for k in range(params.strip_cycle):
        strip_mask = strip_index == k
        if not strip_mask.any():
            continue
        recipe = _recipe(params.colorway, params.motifs, params.seed, k)
        for i, (kind, _frac) in enumerate(_BLOCK_LAYOUT):
            lo, hi = band_starts[i], band_starts[i + 1]
            band_mask = strip_mask & (yb >= lo) & (yb < hi)
            if not band_mask.any():
                continue
            if kind == "accent":
                out[band_mask] = np.append(np.asarray(params.colorway.accent, dtype=np.float64), 1.0)
            elif kind == "solid":
                color = recipe["color_a"] if i % 4 == 0 else recipe["color_b"]
                out[band_mask] = np.append(color, 1.0)
            elif kind == "motif":
                ym = yb[band_mask] - lo
                m = _motif_mask(recipe["motif"], xs[band_mask], ym, params.strip_px, hi - lo)
                motif_c = np.append(np.asarray(params.colorway.motif, dtype=np.float64), 1.0)
                ground_c = np.append(recipe["color_a"], 1.0)
                out[band_mask] = np.where(m[:, None], motif_c, ground_c)

    return out


def generate_weave(width: int, height: int, params: WeaveParams) -> np.ndarray:
    """RGBA float array in [0, 1], shape (height, width, 4).

    Tiles exactly when `width` is a multiple of
    `params.strip_px * params.strip_cycle` and `height` is a multiple of
    `params.block_px` — both axes are built from pixel-integer modulo, so
    there is no floating-point seam to chase.
    """
    period_x = params.strip_px * params.strip_cycle
    if width % period_x != 0 or height % params.block_px != 0:
        raise ValueError(
            f"{width}x{height} is not a multiple of the tile period "
            f"{period_x}x{params.block_px}"
        )
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    return weave_color_at(x, y, params)


def to_image(color: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(color, 0, 1) * 255 + 0.5).astype(np.uint8))


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/kente_previews")
    ap.add_argument("--colorway", choices=sorted(COLORWAYS), default="asante_gold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strip-px", type=int, default=64)
    ap.add_argument("--strip-cycle", type=int, default=3)
    ap.add_argument("--block-px", type=int, default=96)
    ap.add_argument("--tiles", type=int, default=4, help="repeats per axis in the preview")
    args = ap.parse_args()

    params = WeaveParams(
        colorway=COLORWAYS[args.colorway],
        strip_px=args.strip_px,
        strip_cycle=args.strip_cycle,
        block_px=args.block_px,
        seed=args.seed,
    )
    width = params.strip_px * params.strip_cycle * args.tiles
    height = params.block_px * args.tiles
    arr = generate_weave(width, height, params)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.colorway}_seed{args.seed}.png"
    to_image(arr).save(out_dir / name)
    print(f"wrote {out_dir / name} ({width}x{height})")


if __name__ == "__main__":
    _main()
