"""Collar repair: flatten the embossed hallucinated lettering, measure the
band, and print "AI-CCORE" on it.

TRELLIS did not just paint "ALCOPE" on the collar — it raised the letters
as geometry, so repainting the texture alone leaves ghost letters in the
shading. `flatten_collar` fits a smooth base surface to the band's *red*
(non-letter) vertices and drops the white ones onto it; `despike_band`
clamps the few ridges that survives that.

The lettering is a decal primitive rather than texels in the body atlas:
the atlas gives the band ~350 texels across the whole text, while a
dedicated image keeps letter edges crisp at any zoom. The decal reuses
the collar's own triangles (offset along the normal) and the same skin
weights, so it deforms with the neck exactly like the surface under it.

Both of those need to know where the band *is*, and the answer that ships
with the region map is not good enough for typesetting — see `BandFrame`
for what was measured and why. `measure_band_frame` replaces it with
polynomial rims fitted to the red facing; the decal's UVs, its patch, and
the texture fill in `meshforge.paint` all ride that frame.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import trimesh
from PIL import Image

from meshforge.bake import render_text_image
from meshforge.paint import DEFAULT_FONTS
from meshforge.regions import RegionMap, collar_bounds_at
from meshforge.rigexport import MaterialSpec, PrimitiveSpec
from meshforge.weights import vertex_adjacency


def collar_coordinates(points: np.ndarray, region_map: RegionMap, bounds: np.ndarray):
    """(theta, v_centered, lo, hi) for world points, in the collar's
    bbox-fraction parameterization. v_centered is the height above the
    band centreline in bbox-fraction units."""
    bmin, bmax = np.asarray(bounds[0]), np.asarray(bounds[1])
    size = bmax - bmin
    f = (np.asarray(points, dtype=np.float64) - bmin) / size
    cx, cz = region_map.collar["centre_xz"]
    theta = np.arctan2(f[:, 0] - cx, f[:, 2] - cz)
    lo, hi = collar_bounds_at(region_map.collar, theta)
    v = f[:, 1] - (lo + hi) / 2
    return theta, v, lo, hi, f


def collar_interior_mask(mesh: trimesh.Trimesh, region_map: RegionMap, rim: float = 0.010,
                         labelled_only: bool = True) -> np.ndarray:
    """Vertices strictly between the collar's two rims. `labelled_only`
    also requires the region map to call them "neck"; pass False to reach
    band vertices the region map assigned elsewhere — the folded wing
    crosses the band on the +X side, and its letter relief survived the
    first flattening pass because of exactly that (measured: of the
    vertices still standing proud after v2, 210 were labelled
    `wing_right` and only 33 `neck`)."""
    theta, v, lo, hi, f = collar_coordinates(mesh.vertices, region_map, np.array(mesh.bounds))
    inside = (f[:, 1] > lo + rim) & (f[:, 1] < hi - rim)
    return inside & (region_map.labels == "neck") if labelled_only else inside


class BandFrame(NamedTuple):
    """Smooth per-angle description of the collar band's red facing.

    `regions.measure_collar` locates the band by taking the 8th/92nd
    percentile of every *dark* vertex in a broad height window, in 10°
    bins, and linearly interpolating between bin centres. Measured on the
    built asset that is wrong twice over: the dark set also catches the
    tablet, the mouth and the wing piping, so the reported band is 0.097
    bbox-fraction tall where the real facing is 0.044-0.067; and the
    piecewise-linear interpolation of noisy per-bin percentiles zigzags,
    which a debug grid baked into the decal showed as a visible chevron
    kink across the middle of the band. Text parameterized on that frame
    drifts ~0.45 band-heights relative to the band across the span, which
    is what "the writing sits oddly" looks like.

    This frame instead measures the *red* facing (the thing the letters
    are printed on) in fine bins and fits a low-order polynomial per rim,
    so the centreline is smooth by construction.
    """
    lo_coef: np.ndarray          # np.polyval coefficients, lower rim (bbox-fraction y) vs theta
    hi_coef: np.ndarray          # ... upper rim
    span: float                  # radians; the fit is clamped outside ±span
    height: float                # median band height (bbox-fraction y)
    radius: float                # median band radius (bbox-fraction xz)
    diagnostics: dict

    def bounds(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = np.clip(np.asarray(theta, dtype=np.float64), -self.span, self.span)
        return np.polyval(self.lo_coef, t), np.polyval(self.hi_coef, t)

    def mid(self, theta: np.ndarray) -> np.ndarray:
        lo, hi = self.bounds(theta)
        return 0.5 * (lo + hi)


def _red_mask(colors: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(colors, dtype=np.float64), 0, 1)
    mx, mn = c.max(axis=1), c.min(axis=1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return (sat > 0.35) & (mx > 0.25) & (c[:, 0] >= c[:, 1]) & (c[:, 0] >= c[:, 2])


def measure_band_frame(mesh: trimesh.Trimesh, region_map: RegionMap, colors: np.ndarray,
                       span_deg: float = 70.0, n_bins: int = 28, degree: int = 4,
                       min_per_bin: int = 12) -> BandFrame:
    """Fit smooth rims to the collar band's front red facing.

    Only vertices the region map calls "neck" are considered, which keeps
    the folded right wing (same red, crosses the band on +X) out of the
    fit; the white lettering is excluded automatically because the rims
    are read as extreme percentiles of the *red* set."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    bounds = np.array(mesh.bounds)
    theta, _v, lo0, hi0, f = collar_coordinates(V, region_map, bounds)
    span = np.radians(span_deg)
    cand = (region_map.labels == "neck") & (np.abs(theta) < span) & _red_mask(colors)
    cx, cz = region_map.collar["centre_xz"]
    radius = np.sqrt((f[:, 0] - cx) ** 2 + (f[:, 2] - cz) ** 2)

    edges = np.linspace(-span, span, n_bins + 1)
    which = np.digitize(theta, edges) - 1
    cen, los, his, wts = [], [], [], []
    for b in range(n_bins):
        sel = cand & (which == b)
        if sel.sum() < min_per_bin:
            continue
        ys = f[sel, 1]
        cen.append((edges[b] + edges[b + 1]) / 2)
        los.append(np.percentile(ys, 3))
        his.append(np.percentile(ys, 97))
        wts.append(np.sqrt(sel.sum()))
    if len(cen) < degree + 2:
        # fall back to the coarse dark-rim frame rather than fabricate one
        flat_lo = float(np.median(lo0)) if np.ndim(lo0) else float(lo0)
        flat_hi = float(np.median(hi0)) if np.ndim(hi0) else float(hi0)
        return BandFrame(np.array([flat_lo]), np.array([flat_hi]), span, flat_hi - flat_lo,
                         float(np.median(radius[cand])) if cand.any() else 0.3,
                         {"bins_used": len(cen), "fallback": True})

    cen = np.asarray(cen); los = np.asarray(los); his = np.asarray(his); wts = np.asarray(wts)
    deg = min(degree, len(cen) - 2)
    lo_coef = np.polyfit(cen, los, deg, w=wts)
    hi_coef = np.polyfit(cen, his, deg, w=wts)
    resid = float(np.sqrt(np.mean((np.polyval(lo_coef, cen) - los) ** 2
                                  + (np.polyval(hi_coef, cen) - his) ** 2)))
    height = float(np.median(his - los))
    diag = {"bins_used": int(len(cen)), "degree": int(deg), "residual": resid,
            "height": height, "coarse_height": float(np.median(hi0 - lo0)),
            "vertices": int(cand.sum())}
    return BandFrame(lo_coef, hi_coef, span, height,
                     float(np.median(radius[cand])), diag)


def flatten_collar(mesh: trimesh.Trimesh, region_map: RegionMap, vertex_colors: np.ndarray | None = None,
                   rim: float = 0.010, smooth_iterations: int = 8, theta_span_deg: float = 95.0,
                   letter_mask: np.ndarray | None = None,
                   frame: "BandFrame | None" = None, frame_rim: float = 0.04,
                   exclude_labels: tuple[str, ...] = ("wing_left", "wing_right")) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Remove the raised lettering from the collar's front interior.

    Measured on the asset: the white letter vertices sit 13 mm (median)
    proud of the red band along the surface normal, and they are 42 % of
    the band's front vertices — too many for any "robust fit that
    discards outliers" to find the band (attempts one and two: Laplacian
    relaxation averaged letters with letters; a local percentile opening
    flattened the band's own bulge). The letters are *white*, though, so
    we use the color: fit a smooth base surface r(theta, y) to the red
    (non-letter) vertices only, then drop every letter vertex onto that
    base along its radial direction, shave any red vertex still above the
    base (letter-edge ramps), and smooth the moved patch radially.

    Pass a `BandFrame`. Without one the movable set comes from
    `collar_interior_mask`, i.e. from the coarse dark-rim bounds, which on
    this asset are about twice the band's real height — so the "interior"
    reaches well past the band's top rim into the chest, and pulling those
    vertices down onto the band's radius tears the seam open. That tear is
    visible in a flat-shaded render of the flattened mesh and absent from
    the same render of the decimated one, which is how it was found.

    `vertex_colors` (sRGB, per vertex) or an explicit `letter_mask` selects
    the letter vertices. Returns (new mesh, moved-vertex mask)."""
    from scipy.spatial import cKDTree

    V = np.asarray(mesh.vertices, dtype=np.float64).copy()
    bounds = np.array(mesh.bounds)
    size = bounds[1] - bounds[0]
    theta, v, lo, hi, f = collar_coordinates(V, region_map, bounds)
    if frame is not None:
        lo, hi = frame.bounds(theta)
        inset = frame_rim * (hi - lo)
        interior = (f[:, 1] > lo + inset) & (f[:, 1] < hi - inset)
        span = min(np.radians(theta_span_deg), frame.span)
        feather = 0.10 * frame.height
    else:
        interior = collar_interior_mask(mesh, region_map, rim)
        inset = rim
        span = np.radians(theta_span_deg)
        feather = 0.018
    in_span = np.abs(theta) < span
    movable = interior & in_span
    if frame is not None:
        # The wing (and the tablet it holds, which the region map lumps in
        # with it) crosses the band near the +X end. Dragging those onto the
        # band's radius shatters them; the `stray` pass below still rescues
        # any *lettering* they own, which is the only part that must move.
        for name in exclude_labels:
            movable &= region_map.labels != name
    if not movable.any():
        return mesh, movable
    # blend weight: 0 at the edge of the movable band (so the rims and the
    # band's own edge geometry are untouched), 1 a little inside
    edge_lo = (f[:, 1] - (lo + inset))
    edge_hi = ((hi - inset) - f[:, 1])
    edge_dist = np.minimum(edge_lo, edge_hi)           # bbox-fraction units
    blend = np.clip(edge_dist / feather, 0.0, 1.0)
    blend = blend * blend * (3 - 2 * blend)

    cx, cz = region_map.collar["centre_xz"]
    axis_x = bounds[0][0] + cx * size[0]
    axis_z = bounds[0][2] + cz * size[2]
    radial = np.stack([V[:, 0] - axis_x, np.zeros(len(V)), V[:, 2] - axis_z], axis=1)
    dist = np.linalg.norm(radial, axis=1)
    rhat = radial / np.maximum(dist[:, None], 1e-9)

    idx = np.where(movable)[0]
    if letter_mask is None:
        if vertex_colors is None:
            raise ValueError("flatten_collar: pass vertex_colors or letter_mask to identify the lettering")
        c = np.clip(np.asarray(vertex_colors, dtype=np.float64), 0, 1)
        mx = c.max(axis=1)
        sat = np.where(mx > 0, (mx - c.min(axis=1)) / np.maximum(mx, 1e-6), 0.0)
        letter_mask = (sat < 0.30) & (mx > 0.6)
    # letter vertices anywhere in the geometric band join the movable set,
    # whatever region owns them (see collar_interior_mask's docstring)
    geometric = interior if frame is not None else collar_interior_mask(mesh, region_map, rim, labelled_only=False)
    stray = geometric & in_span & letter_mask & ~movable
    if stray.any():
        movable = movable | stray
        idx = np.where(movable)[0]
    letters = letter_mask[idx]
    th = theta[idx]
    yy = V[idx, 1]
    th_n = (th - th.mean()) / max(th.std(), 1e-6)
    y_n = (yy - yy.mean()) / max(yy.std(), 1e-6)
    A = np.stack([np.ones_like(th_n), th_n, th_n ** 2, th_n ** 3, th_n ** 4, y_n, y_n ** 2, y_n ** 3,
                  th_n * y_n, th_n ** 2 * y_n, th_n * y_n ** 2, th_n ** 2 * y_n ** 2, th_n ** 3 * y_n], axis=1)
    r = dist[idx]
    keep = ~letters
    if keep.sum() < 30:
        keep = np.ones(len(idx), dtype=bool)
    coef = None
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(A[keep], r[keep], rcond=None)
        resid = r - A @ coef
        # one more robust pass on the non-letter set: drop red vertices on letter-edge ramps
        thresh = np.percentile(resid[keep], 80)
        keep = (~letters) & (resid <= thresh)
        if keep.sum() < 30:
            break
    base_r = A @ coef
    # replace the whole front interior by the smooth base: the real band is
    # a smooth bib, so nothing of value is lost, and letter-edge ramps
    # (red vertices half-way up a stroke) disappear with the letters.
    target = base_r.copy()
    fit_rms = float(np.sqrt(np.mean((r[keep] - base_r[keep]) ** 2))) if keep.any() else float("nan")
    flatten_collar.last_fit_rms = fit_rms
    w = blend[idx]
    target = w * target + (1 - w) * r
    V[idx] = np.stack([axis_x, 0, axis_z]) * np.array([1, 0, 1]) + rhat[idx] * target[:, None]
    V[idx, 1] = mesh.vertices[idx, 1]  # keep heights

    # Collapsing 13 mm letters onto the band leaves their wall triangles as
    # folded slivers (they rendered as dark outlines). Relax the patch
    # tangentially so vertices spread out evenly, then re-project every
    # vertex radially onto the fitted base evaluated at its new (theta, y).
    adj = vertex_adjacency(len(V), mesh.faces)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    deg[deg == 0] = 1
    th_mu, th_sd = th.mean(), max(th.std(), 1e-6)
    y_mu, y_sd = yy.mean(), max(yy.std(), 1e-6)

    def base_radius(points):
        t = np.arctan2(points[:, 0] - axis_x, points[:, 2] - axis_z)
        tn = (t - th_mu) / th_sd
        yn = (points[:, 1] - y_mu) / y_sd
        A2 = np.stack([np.ones_like(tn), tn, tn ** 2, tn ** 3, tn ** 4, yn, yn ** 2, yn ** 3,
                       tn * yn, tn ** 2 * yn, tn * yn ** 2, tn ** 2 * yn ** 2, tn ** 3 * yn], axis=1)
        return A2 @ coef

    orig_r = dist[idx]
    for _ in range(max(smooth_iterations, 1) * 3):
        avg = (adj @ V) / deg[:, None]
        V[idx] = V[idx] + 0.5 * w[:, None] * (avg[idx] - V[idx])
        rad = np.stack([V[idx, 0] - axis_x, np.zeros(len(idx)), V[idx, 2] - axis_z], axis=1)
        rn = rad / np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-9)
        rr = w * base_radius(V[idx]) + (1 - w) * orig_r
        V[idx, 0] = axis_x + rn[:, 0] * rr
        V[idx, 2] = axis_z + rn[:, 2] * rr
    out = trimesh.Trimesh(vertices=V, faces=mesh.faces.copy(), process=False)
    return out, movable


def despike_band(mesh: trimesh.Trimesh, region_map: RegionMap, frame: BandFrame,
                 threshold: float = 0.0035, iterations: int = 6,
                 span_deg: float = 75.0) -> tuple[trimesh.Trimesh, int]:
    """Shave the last emboss ridges left inside the band.

    `flatten_collar` clears the letters it can identify by colour, but a
    handful of ridge vertices survive — measured on the built asset, 12
    band vertices carry 5-9× the mesh's typical Laplacian relief, and they
    render as dark chips on the red beside the new lettering. Clamp any
    band vertex whose offset from its neighbourhood along the normal
    exceeds `threshold`; everything below it is untouched, so the band
    keeps its own gentle bulge."""
    V = np.asarray(mesh.vertices, dtype=np.float64).copy()
    N = np.asarray(mesh.vertex_normals, dtype=np.float64)
    theta, _v, _lo0, _hi0, f = collar_coordinates(V, region_map, np.array(mesh.bounds))
    lo, hi = frame.bounds(theta)
    sel = (f[:, 1] > lo) & (f[:, 1] < hi) & (np.abs(theta) < np.radians(span_deg))
    adj = vertex_adjacency(len(V), mesh.faces)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    deg[deg == 0] = 1
    moved = np.zeros(len(V), dtype=bool)
    for _ in range(max(iterations, 1)):
        avg = (adj @ V) / deg[:, None]
        d = np.einsum("ij,ij->i", V - avg, N)
        hot = sel & (np.abs(d) > threshold)
        if not hot.any():
            break
        excess = (np.abs(d[hot]) - threshold) * np.sign(d[hot])
        V[hot] -= N[hot] * excess[:, None]
        moved |= hot
    return trimesh.Trimesh(vertices=V, faces=mesh.faces.copy(), process=False), int(moved.sum())


class CollarDecal(NamedTuple):
    primitive: PrimitiveSpec
    source_vertex_index: np.ndarray     # (n_decal,) index into the source mesh vertices (for weights)
    info: dict


def build_collar_decal(mesh: trimesh.Trimesh, region_map: RegionMap, text: str = "AI-CCORE",
                       theta_span_deg: float = 62.0, cap_frac: float = 0.50, width_frac: float = 0.84,
                       offset: float = 0.0012,
                       rim: float = 0.010, font_paths: list[str] | None = None,
                       image_height: int = 512, text_color=(255, 255, 255),
                       colors: np.ndarray | None = None, frame: BandFrame | None = None,
                       inset: float = 0.06, tracking: float = 0.05,
                       plate_color=None, plate_shade: tuple[float, float] = (0.90, 1.02),
                       plate_feather: float = 0.14,
                       exclude_labels: tuple[str, ...] = ("wing_left", "wing_right")) -> CollarDecal:
    """Build the lettering decal over the collar's front red facing.

    The patch and its UVs ride a `BandFrame` — smooth polynomial rims
    fitted to the red facing — so the text follows the band's curve and
    keeps a constant letter height.

    The decal image *is* the band: v spans rim to rim and the image is
    sized to the band's true arc/height aspect, so `cap_frac` and
    `width_frac` are read directly off the reference art (cap height and
    text width as fractions of the band) instead of being back-computed
    through a padded canvas. `inset` keeps the patch clear of the rims by
    that fraction of the band height."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    bounds = np.array(mesh.bounds)
    size = bounds[1] - bounds[0]
    theta, v, lo, hi, f = collar_coordinates(V, region_map, bounds)
    if frame is None:
        if colors is None:
            raise ValueError("build_collar_decal: pass `colors` or a prebuilt `frame`")
        frame = measure_band_frame(mesh, region_map, colors)
    span = np.radians(theta_span_deg)
    lo_b, hi_b = frame.bounds(theta)
    band_h = hi_b - lo_b
    # the patch is defined by the fitted rims, not by the region label: the
    # label boundary is a ragged slab cut and gave the decal a chevron edge.
    inside = (f[:, 1] > lo_b + inset * band_h) & (f[:, 1] < hi_b - inset * band_h)
    sel_v = inside & (np.abs(theta) < span)
    for name in exclude_labels:
        sel_v &= region_map.labels != name
    face_mask = sel_v[mesh.faces].all(axis=1)
    if face_mask.sum() < 10:
        raise ValueError("build_collar_decal: collar front has too few faces")
    faces = mesh.faces[face_mask]
    used = np.unique(faces)
    remap = -np.ones(len(V), dtype=np.int64)
    remap[used] = np.arange(len(used))
    new_faces = remap[faces]
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)[used]
    verts = V[used] + normals * offset

    # uv: u along the arc, v from the upper rim (0) to the lower rim (1) of
    # the *fitted* band (the coarse frame's centreline drifts ~0.45
    # band-heights across the span, which is what read as crooked text)
    h_box = frame.height                                          # in bbox-fraction units of y
    # centre the image on the patch that survived, not on theta=0: the
    # collar axis is a centroid, the owl is not perfectly square to it and
    # the folded right wing eats the +X end, so a span symmetric in theta
    # pushed the final E off the band's right edge and onto the wing.
    th_lo, th_hi = np.percentile(theta[used], [1.5, 98.5])
    th_c, th_half = 0.5 * (th_lo + th_hi), max(0.5 * (th_hi - th_lo), 1e-6)
    u = 0.5 + (theta[used] - th_c) / (2 * th_half)
    vv = 0.5 - (f[used, 1] - frame.mid(theta[used])) / h_box      # image row fraction (0 = top)
    uvs = np.stack([u, vv], axis=1)

    # image sized to the band's true arc/height aspect: no padding, so the
    # fit fractions below are literally cap height and text width on the band
    arc_len = 2 * th_half * frame.radius * float(np.mean(size[[0, 2]]))
    box_h = h_box * size[1]
    aspect = max(arc_len / max(box_h, 1e-6), 1.0)
    h = int(image_height)
    w = int(round(h * aspect))
    img = render_text_image(text, w, h, font_paths or DEFAULT_FONTS, fill=(*text_color, 255),
                            tracking=tracking, fit=(width_frac, cap_frac))
    if plate_color is not None:
        # Print the lettering on an opaque panel of the band's own colour
        # rather than on transparent film. The baked band still carries
        # chips and ridges the flatten could not reach — one sits right of
        # the final E — and a panel simply covers them, which is also what
        # a screen-printed scarf actually looks like. Alpha feathers at all
        # four edges so the panel has no visible seam.
        r = np.linspace(0, 1, h)[:, None]                      # 0 = top = upper rim
        shade = plate_shade[0] + (plate_shade[1] - plate_shade[0]) * (1.0 - r)
        rgb = np.clip(np.asarray(plate_color, dtype=np.float64)[None, None, :] * shade[..., None], 0, 1)
        rgb = np.repeat(rgb, w, axis=1)
        # fade to fully transparent at the *patch's* own UV limits, not at
        # the image border: the patch only reaches v≈0.04-0.94, so a border-
        # referenced ramp still had ~0.3 alpha where the geometry stops and
        # drew a visible ledge along the band.
        u0, u1 = float(u.min()), float(u.max())
        v0, v1 = float(vv.min()), float(vv.max())
        fw = max(plate_feather, 1e-6)
        fwu = max(plate_feather * 0.5, 1e-6)
        fy = np.clip((r - v0) / fw, 0, 1) * np.clip((v1 - r) / fw, 0, 1)
        cx_ = np.linspace(0, 1, w)[None, :]
        fx = np.clip((cx_ - u0) / fwu, 0, 1) * np.clip((u1 - cx_) / fwu, 0, 1)
        a = (fy * fx)
        a = a * a * (3 - 2 * a)
        plate = Image.fromarray(np.concatenate(
            [(rgb * 255).round().astype(np.uint8),
             (np.broadcast_to(a, (h, w))[..., None] * 255).round().astype(np.uint8)], axis=2), "RGBA")
        img = Image.alpha_composite(plate, img)

    mat = MaterialSpec(name="owl_collar_text", base_color_image=img, image_format="PNG", alpha_mode="BLEND",
                       roughness=0.9, metallic=0.0, double_sided=False)
    prim = PrimitiveSpec(name="collar_text", vertices=verts, faces=new_faces, normals=normals, uvs=uvs, material=mat)
    info = {"faces": int(len(new_faces)), "aspect": round(float(aspect), 3), "image": img.size,
            "h_box_frac": round(float(h_box), 5), "band": frame.diagnostics,
            "theta_deg": [round(float(np.degrees(th_lo)), 1), round(float(np.degrees(th_hi)), 1)],
            "v_range": [round(float(vv.min()), 3), round(float(vv.max()), 3)],
            "plate": plate_color is not None}
    return CollarDecal(primitive=prim, source_vertex_index=used, info=info)
