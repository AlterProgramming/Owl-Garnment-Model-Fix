"""A beaded anklet: small alternating-color spheres ringing the leg, rigid
to a single joint.

The original ask was "a Kente cloth outfit, with perhaps even beads" — the
cloth (`meshforge.kente`) got the depth of this session's attention and
the beads were never actually built. Placement is a real constraint, not
an arbitrary choice: the neck is already fully covered by the AI-CCORE
collar decal (a necklace there would be invisible under it), and the legs
are bare and visible from every angle already rendered for the kente work.
So: anklets, one per leg, positioned from the leg region's own measured
geometry (`regions.classify` already labels `leg_left`/`leg_right`),
not a guessed height/radius.

Unlike the kente decal, beads are individual rigid objects, not a surface
that has to deform smoothly with the body — trade beads on a string don't
stretch with the skin underneath them. So each bead is skinned rigidly to
one joint (the same pattern `owl_pipeline.build_owl` already uses for the
eye parts), not blended weights, and there is no UV/texture bake at all:
a small sphere in one flat color per bead is closer to a real glass trade
bead than a textured surface would be, and needs none of the position-bake
machinery `kente.py` needed for a UV atlas.
"""
from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import trimesh

from meshforge.regions import RegionMap
from meshforge.rigexport import MaterialSpec, PrimitiveSpec

# Amber, cobalt blue, white — the trade-bead palette settled on during this
# session's concept-art pass, carried through to the real geometry.
DEFAULT_BEAD_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.80, 0.50, 0.05),
    (0.05, 0.20, 0.55),
    (0.92, 0.90, 0.85),
)


class BeadRing(NamedTuple):
    primitive: PrimitiveSpec
    joint_name: str
    info: dict


def _leg_geometry(mesh: trimesh.Trimesh, region_map: RegionMap, leg_label: str,
                  height_percentile: float) -> tuple[np.ndarray, float, float]:
    """(centre_xz, radius, world_y) measured from the leg region's own
    vertices near its lower end (close to the ankle, above the foot) —
    not an assumed proportion of the whole body."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    sel = region_map.labels == leg_label
    if sel.sum() < 6:
        raise ValueError(f"_leg_geometry: {leg_label!r} has too few vertices to measure")
    pts = V[sel]
    y_cut = np.percentile(pts[:, 1], height_percentile)
    band = pts[np.abs(pts[:, 1] - y_cut) < 0.02]
    if len(band) < 4:
        band = pts[np.argsort(np.abs(pts[:, 1] - y_cut))[:12]]
    centre_xz = band[:, [0, 2]].mean(axis=0)
    radius = float(np.linalg.norm(band[:, [0, 2]] - centre_xz, axis=1).mean())
    world_y = float(band[:, 1].mean())
    return centre_xz, radius, world_y


def build_bead_ring(
    mesh: trimesh.Trimesh,
    region_map: RegionMap,
    leg_label: str,
    joint_name: str,
    bead_count: int = 14,
    bead_radius_frac: float = 0.22,
    bead_elongation: float = 1.6,
    ring_radius_scale: float = 1.05,
    height_percentile: float = 12.0,
    colors: Sequence[tuple[float, float, float]] = DEFAULT_BEAD_COLORS,
    seed: int = 0,
    material_name: str = "owl_beads",
    radius_override: float | None = None,
    subdivisions: int = 2,
) -> BeadRing:
    """A ring of `bead_count` small icospheres around `leg_label`'s
    measured circumference near its ankle end, alternating through
    `colors` with a slight per-bead size jitter (real beads on a string
    are never perfectly uniform). Rigid-bound to `joint_name` — the
    caller sets `.joints`/`.weights` the same way `owl_pipeline.py`
    already does for the eye parts.

    `radius_override`, if given, replaces the *measured* radius (position
    still comes from this leg's own geometry) — needed on the real owl:
    `regions.classify` labels noticeably more geometry `leg_right` than
    `leg_left` at every height percentile checked (2-3x the radius,
    consistently, not a small cluster of outliers to filter), the same
    asymmetry `collar.py` already documents for `wing_right` vs
    `wing_left` and attributes to the tablet-holding wing folding down
    near that side. The body itself is meant to be bilaterally symmetric
    even where the labels aren't, so `build_symmetric_bead_rings` below
    measures both legs and uses the smaller (less contaminated) one's
    radius for both, via this override, rather than trust a measurement
    known to run wide on one side."""
    centre_xz, measured_radius, world_y = _leg_geometry(mesh, region_map, leg_label, height_percentile)
    radius = measured_radius if radius_override is None else radius_override
    ring_radius = radius * ring_radius_scale
    bead_radius = radius * bead_radius_frac

    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, bead_count, endpoint=False)
    angles = angles + rng.uniform(-0.15, 0.15, size=bead_count) / max(bead_count, 1)
    jitter = rng.uniform(0.85, 1.15, size=bead_count)

    all_verts, all_faces, all_colors = [], [], []
    offset = 0
    for i, theta in enumerate(angles):
        cx = centre_xz[0] + ring_radius * np.cos(theta)
        cz = centre_xz[1] + ring_radius * np.sin(theta)
        r = bead_radius * jitter[i]
        sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=r)   # 2 -> 320 faces a bead; 1 for the guide
        # real trade beads are typically oval, elongated along the string
        # they're threaded on — here, the ring's own local tangent
        # direction at this bead's angle, not an arbitrary fixed axis, so
        # every bead's long axis follows the loop instead of all pointing
        # the same way regardless of where they sit around it.
        tangent = np.array([-np.sin(theta), 0.0, np.cos(theta)])
        component = sphere.vertices @ tangent
        sv = sphere.vertices + (bead_elongation - 1.0) * component[:, None] * tangent[None, :]
        verts = sv + np.array([cx, world_y, cz])
        all_verts.append(verts)
        all_faces.append(sphere.faces + offset)
        color = np.asarray(colors[i % len(colors)], dtype=np.float64)
        all_colors.append(np.broadcast_to(color, (len(verts), 3)))
        offset += len(verts)

    verts = np.concatenate(all_verts, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    vcolors = np.concatenate(all_colors, axis=0)
    mesh_beads = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    normals = np.asarray(mesh_beads.vertex_normals, dtype=np.float64)

    mat = MaterialSpec(name=material_name, roughness=0.3, metallic=0.05, double_sided=False)
    prim = PrimitiveSpec(name=f"beads_{leg_label}", vertices=verts, faces=faces, normals=normals,
                         colors=vcolors, material=mat)
    info = {"leg": leg_label, "beads": bead_count, "ring_radius": round(ring_radius, 4),
            "bead_radius": round(bead_radius, 4), "world_y": round(world_y, 4),
            "centre_xz": [round(float(c), 4) for c in centre_xz],
            "measured_radius": round(measured_radius, 4), "radius_overridden": radius_override is not None}
    return BeadRing(primitive=prim, joint_name=joint_name, info=info)


def build_symmetric_bead_rings(
    mesh: trimesh.Trimesh,
    region_map: RegionMap,
    left: tuple[str, str] = ("leg_left", "leg_left"),
    right: tuple[str, str] = ("leg_right", "leg_right"),
    **kwargs,
) -> tuple[BeadRing, BeadRing]:
    """Build both anklets using ONE shared radius — whichever leg's own
    `_leg_geometry` measurement comes back smaller, on the assumption that
    a real owl's two legs are the same thickness and the smaller reading
    is the less-contaminated one (see `build_bead_ring`'s docstring on
    `radius_override` for why: measured on the real owl, `leg_right` reads
    2-3x wider than `leg_left` at every height checked, matching a labelling
    asymmetry `collar.py` already documents on the same side)."""
    left_label, left_joint = left
    right_label, right_joint = right
    _, r_left, _ = _leg_geometry(mesh, region_map, left_label, kwargs.get("height_percentile", 12.0))
    _, r_right, _ = _leg_geometry(mesh, region_map, right_label, kwargs.get("height_percentile", 12.0))
    shared_radius = min(r_left, r_right)
    ring_left = build_bead_ring(mesh, region_map, left_label, left_joint, radius_override=shared_radius, **kwargs)
    ring_right = build_bead_ring(mesh, region_map, right_label, right_joint, radius_override=shared_radius, **kwargs)
    return ring_left, ring_right
