"""Stylized eye set v2: textured eyeball + glossy cornea + upper/lower
lids per eye, sized and placed from the painted eye discs measured on the
texture.

What changed versus meshforge.eyes (v1), and versus the first v2 draft —
every item was visible in screenshots:

* v1 painted a black pupil on pure white with *vertex colors*. Under a
  close-up the iris was a blotch of a dozen interpolated triangles. The
  eyeball now carries a procedurally drawn 512² texture (sclera shading,
  iris gradient + fibres + limbal ring, pupil, two catchlights) mapped by
  orthographic projection along the gaze axis, so circles stay circles
  and there is no pole singularity where it matters.
* v1 cut lids out of an icosphere by vertex selection, which leaves a
  zig-zag edge. Lids and the cornea are now parametric caps (rings ×
  segments) with a clean circular rim, and the upper lid has an inner
  layer + rim strip so it reads as a lid with thickness, not a paper
  shell.
* Placement: the visible cap of the sphere is centred on the socket
  normal (so the opening, lids and the blanked disc are concentric); the
  iris is painted toward a gaze direction blended between that normal and
  straight ahead, and that direction is exported per eye as
  `extras.gaze_forward` for the runtime look-at.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter
from scipy.spatial.transform import Rotation

from meshforge.paint import EyeDisc
from meshforge.rigexport import MaterialSpec, PrimitiveSpec


class EyeParams(NamedTuple):
    visible_half_angle_deg: float = 48.0   # cap of the sphere that shows through the socket
    iris_frac: float = 0.70                # painted iris radius / white disc radius (reference art)
    pupil_frac: float = 0.40               # pupil radius / iris radius
    gaze_blend: float = 0.6                # 0 = look along the socket normal, 1 = straight ahead (+Z)
    # The lids are parked just OUTSIDE the visible aperture at rest: the
    # reference art shows a clean round eye with a painted brow above it,
    # and a lid drawn inside the aperture reads as a cream blob sitting on
    # the eyeball (measured in the v2 screenshots). Blink still works —
    # the closing rotation sweeps the lid across the whole aperture.
    # Sizing rule, with `edge` = lid_edge_deg, `cap` = lid_cap_deg,
    # `aperture` = visible_half_angle_deg and `close` = the blink rotation:
    # at rest the shell spans [edge, edge + 2*cap] from the opening axis, so
    # edge >= aperture hides it; closed it spans [edge - close, edge + 2*cap
    # - close], so covering the whole aperture needs close >= edge + aperture
    # and 2*cap >= close + aperture - edge. The v3 screenshots showed the
    # lower quarter of the eye still visible mid-blink because cap was too
    # small for the travel.
    lid_cap_deg: float = 56.0              # half-angle of the upper lid shell
    lid_edge_deg: float = 50.0             # rest position of the upper lid edge, from the opening axis
    lower_lid_cap_deg: float = 24.0
    lower_lid_edge_deg: float = 50.0
    lid_scale: float = 1.04                # outer lid shell radius / eyeball radius
    lid_thickness: float = 0.02            # inner shell at (lid_scale - thickness) × R
    cornea_scale: float = 1.012
    blink_overshoot_deg: float = 52.0      # how far past the opening axis the lid edge travels when closed
    subdivisions: int = 4
    texture_size: int = 512


class EyeSet(NamedTuple):
    primitives: list[PrimitiveSpec]           # each with .joints/.weights left None — the caller binds them
    bindings: list[str]                       # joint name each primitive is rigidly bound to
    pivots: dict[str, np.ndarray]             # eye_left/eye_right/lid_left/lid_right -> sphere centre
    gaze_forward: dict[str, np.ndarray]       # per eye joint: local forward axis the pupil is painted on
    blink_axes: dict[str, np.ndarray]         # per lid joint: rotation axis that closes it
    blink_close_deg: float
    radius: dict[str, float]
    info: dict


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / max(np.linalg.norm(v), 1e-12)


def _frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(right, up, forward) orthonormal frame around `axis`, up as close to +Y as possible."""
    g = _unit(axis)
    up = np.array([0.0, 1.0, 0.0])
    up = _unit(up - g * np.dot(up, g))
    right = _unit(np.cross(up, g))
    return right, up, g


# --------------------------------------------------------------------------
# eyeball texture
# --------------------------------------------------------------------------

def eye_texture(size: int, iris_deg: float, pupil_deg: float, supersample: int = 4) -> Image.Image:
    """Orthographic eye texture: the gaze pole sits at the image centre and a
    direction at angle a from it lands at radius sin(a) × (size/2)."""
    S = size * supersample
    c = S / 2
    rad = lambda deg: np.sin(np.radians(deg)) * c  # noqa: E731
    img = Image.new("RGB", (S, S), (245, 241, 233))
    draw = ImageDraw.Draw(img)

    # sclera: warm white with a soft darkening toward the edge (socket shadow)
    sclera = np.zeros((S, S, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:S, 0:S]
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / c
    base = np.array([0.975, 0.965, 0.945])
    edge = np.array([0.84, 0.82, 0.80])
    t = np.clip((r - 0.72) / 0.28, 0, 1)[..., None]
    sclera = base * (1 - t) + edge * t
    # faint pink/vein tint toward the corners
    img = Image.fromarray((np.clip(sclera, 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # iris: radial gradient drawn as concentric rings + fibres
    ri = rad(iris_deg)
    inner = np.array([0.30, 0.16, 0.09])
    outer = np.array([0.50, 0.29, 0.15])
    steps = 64
    for i in range(steps, 0, -1):
        tt = i / steps
        col = inner * (1 - tt) + outer * tt
        rr = ri * tt
        draw.ellipse([c - rr, c - rr, c + rr, c + rr], fill=tuple(int(v * 255) for v in col))
    # fibres: thin darker/lighter spokes
    fibre = Image.new("L", (S, S), 0)
    fd = ImageDraw.Draw(fibre)
    n_fib = 96
    rng = np.random.default_rng(7)
    for k in range(n_fib):
        ang = 2 * np.pi * k / n_fib + rng.uniform(-0.02, 0.02)
        r0 = ri * rng.uniform(0.35, 0.5)
        r1 = ri * rng.uniform(0.9, 1.0)
        fd.line([(c + r0 * np.cos(ang), c + r0 * np.sin(ang)), (c + r1 * np.cos(ang), c + r1 * np.sin(ang))],
                fill=int(255 * rng.uniform(0.5, 1.0)), width=max(1, int(S / 400)))
    fibre = fibre.filter(ImageFilter.GaussianBlur(S / 600))
    fib = np.asarray(fibre, dtype=np.float32) / 255.0
    arr = np.asarray(img, dtype=np.float32) / 255.0
    iris_mask = (r <= np.sin(np.radians(iris_deg)))[..., None]
    arr = np.where(iris_mask, arr * (1 - 0.22 * fib[..., None]) + 0.06 * fib[..., None], arr)
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # limbal ring (soft dark rim around the iris)
    ring_w = ri * 0.10
    for i in range(8):
        rr = ri - ring_w * i / 8
        a = 1 - i / 8
        col = (int(20 + 60 * (1 - a)), int(11 + 35 * (1 - a)), int(7 + 20 * (1 - a)))
        draw.ellipse([c - rr, c - rr, c + rr, c + rr], outline=col, width=max(1, int(ring_w / 8) + 1))

    # pupil
    rp = rad(pupil_deg)
    draw.ellipse([c - rp, c - rp, c + rp, c + rp], fill=(9, 8, 10))
    # subtle pupil edge softening
    img = img.filter(ImageFilter.GaussianBlur(S / 1024))
    draw = ImageDraw.Draw(img)

    # catchlights: big upper-left, small lower-right (viewer's perspective: +u is viewer-right)
    def catch(du_deg, dv_deg, radius_deg, alpha):
        hx = c + rad(du_deg)
        hy = c - rad(dv_deg)
        rr = rad(radius_deg)
        layer = Image.new("L", (S, S), 0)
        ImageDraw.Draw(layer).ellipse([hx - rr, hy - rr, hx + rr, hy + rr], fill=int(255 * alpha))
        layer = layer.filter(ImageFilter.GaussianBlur(rr * 0.12))
        white = Image.new("RGB", (S, S), (255, 255, 255))
        return Image.composite(white, img, layer)

    img = catch(-12.0, 13.0, 7.0, 1.0)
    img = catch(9.0, -11.0, 3.0, 0.9)
    return img.resize((size, size), Image.LANCZOS)


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------


def _cap_res(subdivisions: int) -> tuple[int, int, int, int]:
    """(cornea rings, cap rings, cap segments, lid segments) for a given
    `EyeParams.subdivisions`. At the default 4 these are the values the
    eyes were authored with; lower settings scale the whole eye assembly
    down together. Worth having: at subdivisions=2 the eyeballs shrank but
    the corneas and lids did not, and the two eyes still carried more
    triangles (14,592) than the entire decimated body (13,964).
    """
    sub = max(1, int(subdivisions))
    return (max(4, int(round(2.5 * sub))), max(4, 3 * sub),
            max(16, 16 * sub), max(18, 18 * sub))


def _cap(radius: float, center_dir: np.ndarray, half_angle_deg: float, rings: int = 12, segments: int = 64,
         start_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parametric spherical cap around `center_dir`: returns (points, faces,
    angle_from_center_deg). `start_deg > 0` makes an annulus."""
    right, up, g = _frame(center_dir)
    angles = np.linspace(start_deg, half_angle_deg, rings + 1)
    pts = []
    ang_of = []
    if start_deg == 0.0:
        pts.append(g * radius)
        ang_of.append(0.0)
        angles = angles[1:]
    ring_start = len(pts)
    phis = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    for a in angles:
        ca, sa = np.cos(np.radians(a)), np.sin(np.radians(a))
        for p in phis:
            d = g * ca + (right * np.cos(p) + up * np.sin(p)) * sa
            pts.append(d * radius)
            ang_of.append(a)
    pts = np.array(pts)
    faces = []
    n_rings = len(angles)
    if start_deg == 0.0:
        for s in range(segments):
            faces.append([0, ring_start + s, ring_start + (s + 1) % segments])
    for r in range(n_rings - 1):
        a0 = ring_start + r * segments
        a1 = ring_start + (r + 1) * segments
        for s in range(segments):
            s1 = (s + 1) % segments
            faces.append([a0 + s, a1 + s, a1 + s1])
            faces.append([a0 + s, a1 + s1, a0 + s1])
    return pts, np.array(faces, dtype=np.int64), np.array(ang_of)


def _lid_shell(R_outer: float, R_inner: float, center_dir: np.ndarray, half_angle_deg: float, cream: np.ndarray,
               rings: int = 12, segments: int = 72) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Double-layer lid: outer cap, inner cap, and a rim strip joining their
    edges. Returns (vertices, faces, normals, colors)."""
    po, fo, ao = _cap(R_outer, center_dir, half_angle_deg, rings, segments)
    pi_, fi, ai = _cap(R_inner, center_dir, half_angle_deg, rings, segments)
    n_o = len(po)
    fi = fi[:, ::-1] + n_o  # flip winding: inner layer faces the eyeball
    verts = np.vstack([po, pi_])
    faces = [fo, fi]
    # rim strip between the last rings of the two layers
    last_o = np.arange(n_o - segments, n_o)
    last_i = np.arange(2 * n_o - segments, 2 * n_o)
    strip = []
    for s in range(segments):
        s1 = (s + 1) % segments
        strip.append([last_o[s], last_i[s], last_i[s1]])
        strip.append([last_o[s], last_i[s1], last_o[s1]])
    faces.append(np.array(strip, dtype=np.int64))
    faces = np.vstack(faces)
    normals = np.vstack([po / R_outer, pi_ / R_inner])  # radial (good enough for a thin shell)
    ang = np.concatenate([ao, ai])
    lash = cream * 0.5
    t_lash = np.clip((ang - (half_angle_deg - 4.0)) / 3.0, 0, 1)[:, None]
    t_crease = np.clip((ang - (half_angle_deg - 16)) / 12.0, 0, 1)[:, None]
    col = cream * (1 - 0.12 * t_crease)
    col = col * (1 - t_lash) + lash * t_lash
    return verts, faces, normals, col


# --------------------------------------------------------------------------
# the set
# --------------------------------------------------------------------------

def build_eye_set(discs: dict[str, EyeDisc], face_colors: dict[str, np.ndarray],
                  params: EyeParams = EyeParams()) -> EyeSet:
    """`discs` maps side ('left'/'right') to the measured painted disc;
    `face_colors` maps side to the sRGB cream sampled around that socket
    (used for the lids)."""
    prims: list[PrimitiveSpec] = []
    bindings: list[str] = []
    pivots: dict[str, np.ndarray] = {}
    gaze_forward: dict[str, np.ndarray] = {}
    blink_axes: dict[str, np.ndarray] = {}
    radius: dict[str, float] = {}
    info: dict = {}

    cornea_mat = MaterialSpec(name="owl_cornea", base_color_factor=(1.0, 1.0, 1.0, 0.14), roughness=0.04,
                              metallic=0.0, alpha_mode="BLEND")
    lid_mat = MaterialSpec(name="owl_lid", roughness=0.85, metallic=0.0, double_sided=True)

    vis = np.radians(params.visible_half_angle_deg)
    for side in ("left", "right"):
        disc = discs[side]
        R = disc.disc_radius / np.sin(vis)
        n = _unit(disc.normal)
        straight = np.array([0.0, 0.0, 1.0])
        gaze = _unit(n * (1 - params.gaze_blend) + straight * params.gaze_blend)
        center = disc.center - n * (R * np.cos(vis))
        iris_deg = np.degrees(np.arcsin(np.clip(params.iris_frac * disc.disc_radius / R, 0, 1)))
        pupil_deg = np.degrees(np.arcsin(np.clip(params.pupil_frac * params.iris_frac * disc.disc_radius / R, 0, 1)))
        g_right, g_up, g = _frame(gaze)
        o_right, o_up, o = _frame(n)

        # eyeball: icosphere + orthographic uv along the gaze axis + texture
        ball = trimesh.creation.icosphere(subdivisions=params.subdivisions, radius=R)
        dirs = ball.vertices / R
        x = dirs @ g_right
        y = dirs @ g_up
        front = dirs @ g >= 0
        # back hemisphere folds onto the same disc (never visible through the socket)
        uvs = np.stack([0.5 + 0.5 * x, 0.5 - 0.5 * y], axis=1)
        uvs = np.clip(uvs, 0.0, 1.0)
        tex = eye_texture(params.texture_size, iris_deg, pupil_deg)
        eye_mat = MaterialSpec(name=f"owl_eye_{side}", base_color_image=tex, image_format="PNG", roughness=0.32, metallic=0.0)
        ball.apply_translation(center)
        prims.append(PrimitiveSpec(name=f"eye_{side}", vertices=ball.vertices.copy(), faces=ball.faces.copy(),
                                   normals=np.asarray(ball.vertex_normals), material=eye_mat, uvs=uvs))
        bindings.append(f"eye_{side}")

        # cornea: parametric front cap, glossy + transparent
        c_rings, cap_rings, cap_segs, lid_segs = _cap_res(params.subdivisions)
        cp, cf, _ = _cap(R * params.cornea_scale, o, params.visible_half_angle_deg + 14,
                         rings=c_rings, segments=cap_segs)
        prims.append(PrimitiveSpec(name=f"cornea_{side}", vertices=cp + center, faces=cf,
                                   normals=cp / np.linalg.norm(cp, axis=1, keepdims=True), material=cornea_mat))
        bindings.append(f"eye_{side}")

        cream = np.asarray(face_colors[side], dtype=np.float64)

        # upper lid at rest: centred `lid_edge + lid_cap` degrees above the opening axis
        rest_angle = params.lid_edge_deg + params.lid_cap_deg
        lid_dir = Rotation.from_rotvec(o_right * np.radians(-rest_angle)).apply(o)
        lv, lf, ln, lc = _lid_shell(R * params.lid_scale, R * (params.lid_scale - params.lid_thickness), lid_dir,
                                    params.lid_cap_deg, cream, rings=cap_rings, segments=lid_segs)
        prims.append(PrimitiveSpec(name=f"lid_{side}", vertices=lv + center, faces=lf, normals=ln, material=lid_mat, colors=lc))
        bindings.append(f"lid_{side}")

        # lower lid: static, bound to the head
        low_angle = params.lower_lid_edge_deg + params.lower_lid_cap_deg
        low_dir = Rotation.from_rotvec(o_right * np.radians(low_angle)).apply(o)
        wv, wf, wn, wc = _lid_shell(R * params.lid_scale, R * (params.lid_scale - params.lid_thickness), low_dir,
                                    params.lower_lid_cap_deg, cream,
                                    rings=max(3, cap_rings * 2 // 3), segments=lid_segs)
        prims.append(PrimitiveSpec(name=f"lowerlid_{side}", vertices=wv + center, faces=wf, normals=wn, material=lid_mat, colors=wc))
        bindings.append("head")

        pivots[f"eye_{side}"] = center.copy()
        pivots[f"lid_{side}"] = center.copy()
        gaze_forward[f"eye_{side}"] = g.copy()
        # a positive rotation about o_right moves +up toward +forward: the lid sweeps down to close
        blink_axes[f"lid_{side}"] = o_right.copy()
        radius[f"eye_{side}"] = float(R)
        info[side] = {"R": float(R), "iris_deg": float(iris_deg), "pupil_deg": float(pupil_deg),
                      "center": center.tolist(), "gaze": g.tolist(), "opening_axis": o.tolist()}

    close_deg = params.lid_edge_deg + params.blink_overshoot_deg
    return EyeSet(primitives=prims, bindings=bindings, pivots=pivots, gaze_forward=gaze_forward,
                  blink_axes=blink_axes, blink_close_deg=close_deg, radius=radius, info=info)
