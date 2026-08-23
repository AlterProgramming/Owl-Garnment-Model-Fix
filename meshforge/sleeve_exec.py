from __future__ import annotations

import dataclasses

import numpy as np
import trimesh


def shape_sleeve(sleeve, sag: float = 0.095, ease: float = 0.055):
    """Turn the measured wing tube into a loose shoulder sleeve.

    The underlying scan still guarantees a real arm opening and wing/body
    clearance.  This pass only adds room away from the wing on the lower
    side of the free span, with zero displacement at the root and cuff.
    """
    prim = sleeve.primitive
    P = np.asarray(prim.vertices, dtype=np.float64).copy()
    frame = sleeve.frame
    s0 = float(sleeve.info["s_root"])
    s1 = float(sleeve.info["s_cuff"])
    span = max(s1 - s0, 1e-9)
    s = (P - frame.pivot) @ frame.u
    t = np.clip((s - s0) / span, 0.0, 1.0)
    bell = np.sin(np.pi * t) ** 1.35

    cw = np.interp(s, sleeve.scan.s, sleeve.scan.centre[:, 0])
    cn = np.interp(s, sleeve.scan.s, sleeve.scan.centre[:, 1])
    centre = frame.pivot + s[:, None] * frame.u + cw[:, None] * frame.w + cn[:, None] * frame.n
    radial = P - centre
    radial -= (radial @ frame.u)[:, None] * frame.u
    rlen = np.maximum(np.linalg.norm(radial, axis=1), 1e-9)
    rhat = radial / rlen[:, None]

    world_down = np.array([0.0, -1.0, 0.0])
    down = world_down - float(world_down @ frame.u) * frame.u
    dn = float(np.linalg.norm(down))
    if dn > 1e-6:
        down /= dn
        lower = np.clip(rhat @ down, 0.0, 1.0) ** 1.6
        P += down[None, :] * ((sag * span) * bell * lower)[:, None]

    # Mid-sleeve ease makes the opening read as cloth around an arm rather
    # than as a skin-tight second shell.  Root and cuff stay registered.
    P += rhat * ((ease * span) * bell)[:, None]

    mesh = trimesh.Trimesh(vertices=P, faces=np.asarray(prim.faces), process=False)
    N = np.asarray(mesh.vertex_normals, dtype=np.float64)
    bad = np.linalg.norm(N, axis=1) < 0.5
    if bad.any():
        N[bad] = rhat[bad]
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)

    shaped = dataclasses.replace(prim, vertices=P, normals=N)
    info = dict(sleeve.info)
    info["concept_shape"] = {"gravity_sag": float(sag), "mid_ease": float(ease)}
    return sleeve._replace(primitive=shaped, info=info)
