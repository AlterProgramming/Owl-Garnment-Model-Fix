"""Author real eyeball geometry for the owl's face.

The source mesh has no separate eye geometry at all — confirmed this
session by connected-component analysis (one 149,334-face body surface,
plus only a small tassel component, no eye mesh island) — so the pupils
are just circles painted into the skull's vertex colors. A painted circle
has no depth: no parallax as the camera orbits, no specular catchlight,
and critically no way to rotate for gaze. This module authors two small
spheres to sit in the sockets instead, each riggable with its own bone.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import trimesh


class EyeSocket(NamedTuple):
    centroid: np.ndarray
    normal: np.ndarray


def _unit(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=np.float64)
    return a / np.linalg.norm(a)


# Empirically located against the real owl-mascot.glb: vertex color
# thresholded for dark, face-height clusters, then confirmed by rendering
# the candidate clusters in bright colors directly on the face and
# actually looking at the image (same discipline as WING_REGION in
# cli.py) — not derived from assumed left/right symmetry, since the
# source photo's lighting was not perfectly even, so the two clusters'
# raw sizes and offsets from x=0 differ even though both sit correctly
# centered in their own eye.
RIGHT_EYE_SOCKET = EyeSocket(centroid=np.array([0.246, 0.352, 0.331]), normal=_unit([0.256, 0.138, 0.957]))
LEFT_EYE_SOCKET = EyeSocket(centroid=np.array([-0.097, 0.354, 0.334]), normal=_unit([-0.263, 0.203, 0.943]))


def neutralize_old_eye_paint(
    mesh: trimesh.Trimesh,
    sockets: list[EyeSocket],
    radius: float = 0.145,
    fill_color: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """Blank the vertex-painted pupil/socket-shading under each socket to
    a flat off-white, so the new eyeball geometry doesn't double up with
    old paint peeking out past its edges (visible as a discolored halo
    ring around the new eye if skipped — caught by rendering the first
    prototype pass). `radius` needs to comfortably exceed the eyeball's
    own silhouette radius, not just match it, since the blanked patch
    follows the head's curved surface while the eyeball's silhouette is
    a flat projected circle — a radius equal to the eyeball's own is not
    enough and leaves a visible ring. Returns a new mesh; `mesh` is not
    modified in place.
    """
    if fill_color is None:
        fill_color = np.array([0.95, 0.94, 0.90])
    if mesh.visual.kind == "vertex":
        colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
    else:
        colors = np.tile(np.array([0.9, 0.9, 0.9]), (len(mesh.vertices), 1))
    colors = colors.copy()
    for socket in sockets:
        dist = np.linalg.norm(mesh.vertices - socket.centroid, axis=1)
        colors[dist < radius] = fill_color

    result = mesh.copy()
    rgba = np.concatenate([colors, np.ones((len(colors), 1))], axis=1)
    result.visual = trimesh.visual.ColorVisuals(mesh=result, vertex_colors=rgba)
    return result


def _rest_lid_direction(rest_up_deg: float) -> np.ndarray:
    up = np.radians(rest_up_deg)
    d = np.array([0.0, np.sin(up), np.cos(up)])
    return d / np.linalg.norm(d)


def eyelid_blink_axis_angle(rest_up_deg: float = 72.0) -> tuple[np.ndarray, float]:
    """The (axis, angle_rad) that rotates an eyelid built by build_eyelid
    from its authored rest pose down onto the pupil's forward pole (full
    coverage), derived by cross product from the same two directions
    build_eyelid used — so the sign and magnitude always match the actual
    geometry instead of being a separately hand-picked constant that can
    drift out of sync with it if `rest_up_deg` is ever tuned.
    """
    rest_dir = _rest_lid_direction(rest_up_deg)
    forward = np.array([0.0, 0.0, 1.0])
    axis = np.cross(rest_dir, forward)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        raise ValueError("eyelid_blink_axis_angle: rest_up_deg is 0 or 180 — rotation axis is undefined")
    axis = axis / axis_norm
    angle = float(np.arccos(np.clip(np.dot(rest_dir, forward), -1.0, 1.0)))
    return axis, angle


def build_eyelid(
    socket: EyeSocket,
    radius: float = 0.115,
    lid_radius_scale: float = 1.05,
    cap_deg: float = 34.0,
    rest_up_deg: float = 72.0,
    bulge: float = 0.022,
    color: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """Build one eyelid patch at its REST (open) pose: a small spherical
    cap, concentric with the eyeball but a hair larger in radius (so it
    sits just outside the eyeball's surface, not z-fighting with it),
    centered on a direction tilted `rest_up_deg` above the pupil's
    forward pole. That tilt is what keeps it a thin, barely-visible
    crease at the top rim at rest instead of drooping over the pupil —
    the first prototype pass used too small a tilt and read as a
    permanently sleepy/closed eye.

    A blink then rotates this same rigid piece about the eye's own
    center (see eyelid_blink_axis_angle) to sweep its center down onto
    the pupil pole, fully covering it — the geometry itself never
    changes shape, only its orientation, so `cap_deg` must be generous
    enough relative to build_eyeball's `pupil_deg` to fully cover the
    pupil once rotated there (checked at call sites, not enforced here,
    since this function has no way to know the paired eyeball's
    pupil_deg).
    """
    if radius <= 0:
        raise ValueError(f"build_eyelid: radius must be > 0, got {radius}")
    if not (0 < cap_deg < 90):
        raise ValueError(f"build_eyelid: cap_deg must be in (0, 90), got {cap_deg}")
    if color is None:
        color = np.array([0.94, 0.86, 0.82])

    r = radius * lid_radius_scale
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=r)
    local_dirs = sphere.vertices / r
    center_dir = _rest_lid_direction(rest_up_deg)
    ang = np.degrees(np.arccos(np.clip(local_dirs @ center_dir, -1, 1)))
    vertex_keep = ang < cap_deg
    face_keep = vertex_keep[sphere.faces].any(axis=1)
    lid = sphere.submesh([face_keep], append=True)

    colors = np.tile(np.concatenate([color, [1.0]]), (len(lid.vertices), 1))
    lid.visual = trimesh.visual.ColorVisuals(mesh=lid, vertex_colors=colors)
    lid.apply_translation(socket.centroid - socket.normal * (radius - bulge))
    return lid


def eye_pivot(socket: EyeSocket, radius: float = 0.115, bulge: float = 0.022) -> np.ndarray:
    """The bone pivot for an eyeball built with the same radius/bulge —
    must match build_eyeball's translation exactly, or gaze rotation will
    orbit around the wrong center."""
    return socket.centroid - socket.normal * (radius - bulge)


def build_eyeball(
    socket: EyeSocket,
    radius: float = 0.115,
    bulge: float = 0.022,
    pupil_deg: float = 29.0,
    soften_deg: float = 2.5,
    subdivisions: int = 4,
) -> trimesh.Trimesh:
    """Build one vertex-colored eyeball sphere.

    The dark pupil is painted at the sphere's own local +Z pole — the
    rest/bind gaze direction. A later runtime "look-at" rig rotates this
    joint away from identity to aim gaze; keeping the pupil fixed to the
    joint's own local forward axis (rather than, say, the socket's
    tilted surface normal) is what makes that rotation aim correctly
    instead of aiming a fixed offset away from wherever it's told to
    look. `soften_deg` blends the pupil/sclera boundary over a few
    degrees so it reads as a round pupil rather than the visibly
    faceted polygon edge a hard color cutoff produces at practical
    subdivision levels. The small fixed highlight is the standard
    "painted-on catchlight" trick stylized character eyes use to read as
    wet/alive even without a dynamic specular light hitting the model.
    `bulge` is how far the sphere's front pole pokes past the socket
    centroid along `socket.normal`, so the eye sits IN the socket rather
    than floating in front of it or sinking behind it.
    """
    if radius <= 0:
        raise ValueError(f"build_eyeball: radius must be > 0, got {radius}")
    if not (0 < pupil_deg < 90):
        raise ValueError(f"build_eyeball: pupil_deg must be in (0, 90), got {pupil_deg}")

    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    local_dirs = sphere.vertices / radius
    ang = np.degrees(np.arccos(np.clip(local_dirs[:, 2], -1, 1)))

    black = np.array([0.04, 0.04, 0.05])
    white = np.array([0.98, 0.98, 0.98])
    t = np.clip((ang - (pupil_deg - soften_deg)) / (2 * soften_deg), 0, 1)[:, None]
    colors = black * (1 - t) + white * t

    highlight_dir = _unit([0.0, 1.0, 1.0])
    highlight_ang = np.degrees(np.arccos(np.clip(local_dirs @ highlight_dir, -1, 1)))
    highlight_t = np.clip(1 - highlight_ang / (pupil_deg * 0.4), 0, 1)[:, None]
    colors = colors * (1 - highlight_t) + np.array([1.0, 1.0, 1.0]) * highlight_t

    rgba = np.concatenate([colors, np.ones((len(colors), 1))], axis=1)
    sphere.visual = trimesh.visual.ColorVisuals(mesh=sphere, vertex_colors=rgba)
    sphere.apply_translation(socket.centroid - socket.normal * (radius - bulge))
    return sphere
