"""Garment diagnosis harness — research volume Stage 0 ("observe, do not edit").

Reads a *shipped* rigged GLB (the thing that will actually render) and
produces the evidence the connected-fabric volume asks for, without
touching the build:

    python3 tools/garment_diag.py out/owl_kente/owl-kente-tunic.glb --out out/owl_kente/diag \
        --report out/owl_kente/owl-kente-tunic.report.json --stage all

  metrics   per-vertex garment -> body signed distance (dense surface
            samples + face normals, ray-parity confirmed), contact classes,
            per-zone (sector x pattern band) statistics, boundary loops,
            stretch ratio l/l0 and area strain from the pattern the UV
            chart *is*, dihedral (bend) per edge, width/silhouette profile
            garment vs naked body, the same through the baked clips (LBS
            evaluated from the GLB itself), and a motion-coherence read.
  maps      false-colour flat renders (meshforge.fastpreview) of distance
            class, stretch, area strain, zone id — gray body, no texture.
  renders   gray-material turntables through the real Three.js renderer
            (tools/owl_shots.py): garment gray / everything gray / naked.

Conventions (all JSON numbers are in mesh units unless the key says frac):
  H      = body bbox height (prim `owl_body`), y/H measured from its bottom
  theta  = atan2(x - ax, z - az) about the torso axis, 0 = +Z (front),
           +90 = +X (the folded wing / hand / tablet side), -90 = -X (raised wing)
  sector = front [-45,45) | right [45,135) | back [135,225) | left [225,315)
  band   = pattern v (cloth units down from the neckline), from the chart:
           neck <0.04 | yoke 0.04-0.14 | chest 0.14-0.30 | skirt 0.30-0.55 | hem >=0.55
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh
from pygltflib import GLTF2
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

COMP = {5120: ("i1", 1), 5121: ("u1", 1), 5122: ("i2", 2), 5123: ("u2", 2), 5125: ("u4", 4), 5126: ("f4", 4)}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
SECTORS = ("front", "right", "back", "left")
BANDS = (("neck", 0.0, 0.04), ("yoke", 0.04, 0.14), ("chest", 0.14, 0.30), ("skirt", 0.30, 0.55), ("hem", 0.55, 9.0))
CONTACT = 0.025     # <= this to the body surface counts as contact (solver offset 0.014-0.022 + 3 mm)
NEAR = 0.05         # <= this is near-contact; beyond is free cloth


# --------------------------------------------------------------------------
# GLB reading + linear blend skinning straight from the file
# --------------------------------------------------------------------------

def read_accessor(g: GLTF2, idx: int, blob: bytes) -> np.ndarray:
    acc = g.accessors[idx]
    bv = g.bufferViews[acc.bufferView]
    fmt, size = COMP[acc.componentType]
    n = NCOMP[acc.type]
    start = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    stride = bv.byteStride or size * n
    dt = np.dtype(fmt)
    if stride == size * n:
        arr = np.frombuffer(blob, dtype=dt, count=acc.count * n, offset=start).reshape(acc.count, n)
    else:
        arr = np.stack([np.frombuffer(blob, dtype=dt, count=n, offset=start + i * stride) for i in range(acc.count)])
    if fmt == "f4":
        return arr.astype(np.float64)
    out = arr.astype(np.int64)
    if acc.normalized:
        return out / float(np.iinfo(dt).max)
    return out


class Scene:
    def __init__(self, path: Path):
        self.g = GLTF2().load(str(path))
        self.blob = self.g.binary_blob()
        g = self.g
        self.prims = []
        mesh_node = next(i for i, n in enumerate(g.nodes) if n.mesh is not None)
        self.mesh_node = mesh_node
        for mi, m in enumerate(g.meshes):
            for pi, p in enumerate(m.primitives):
                d = {"mesh": mi, "prim": pi, "material": g.materials[p.material].name,
                     "V": read_accessor(g, p.attributes.POSITION, self.blob),
                     "F": read_accessor(g, p.indices, self.blob).reshape(-1, 3)}
                if p.attributes.TEXCOORD_0 is not None:
                    d["UV"] = read_accessor(g, p.attributes.TEXCOORD_0, self.blob)
                if p.attributes.JOINTS_0 is not None:
                    d["J"] = read_accessor(g, p.attributes.JOINTS_0, self.blob)
                    d["W"] = read_accessor(g, p.attributes.WEIGHTS_0, self.blob)
                self.prims.append(d)
        self.skin = g.skins[0]
        self.ibm = read_accessor(g, self.skin.inverseBindMatrices, self.blob).reshape(-1, 4, 4).transpose(0, 2, 1)  # column-major -> row-major
        self.joint_nodes = list(self.skin.joints)
        self.joint_names = [g.nodes[j].name for j in self.joint_nodes]
        self.parent = {}
        for i, n in enumerate(g.nodes):
            for c in (n.children or []):
                self.parent[c] = i
        self.anims = {a.name: a for a in g.animations}

    def prim(self, material: str) -> dict:
        return next(p for p in self.prims if p["material"] == material)

    # -- skinning ----------------------------------------------------------
    @staticmethod
    def _local(node, t=None, r=None):
        T = np.asarray(node.translation if t is None else t, dtype=np.float64) if (node.translation or t is not None) else np.zeros(3)
        R = np.asarray(node.rotation if r is None else r, dtype=np.float64) if (node.rotation or r is not None) else np.array([0.0, 0.0, 0.0, 1.0])
        S = np.asarray(node.scale, dtype=np.float64) if node.scale else np.ones(3)
        x, y, z, w = R / max(np.linalg.norm(R), 1e-12)
        Rm = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                       [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                       [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        M = np.eye(4)
        M[:3, :3] = Rm @ np.diag(S)
        M[:3, 3] = T
        return M

    def clip_overrides(self, name: str, time: float) -> dict:
        a = self.anims[name]
        over = {}
        for ch in a.channels:
            s = a.samplers[ch.sampler]
            times = read_accessor(self.g, s.input, self.blob).ravel()
            vals = read_accessor(self.g, s.output, self.blob)
            t = float(np.clip(time, times[0], times[-1]))
            k = int(np.searchsorted(times, t, side="right") - 1)
            k = max(0, min(k, len(times) - 2)) if len(times) > 1 else 0
            if len(times) == 1:
                v = vals[0]
            else:
                u = (t - times[k]) / max(times[k + 1] - times[k], 1e-9)
                a0, a1 = vals[k], vals[k + 1]
                if ch.target.path == "rotation" and np.dot(a0, a1) < 0:
                    a1 = -a1
                v = (1 - u) * a0 + u * a1
                if ch.target.path == "rotation":
                    v = v / max(np.linalg.norm(v), 1e-12)
            over.setdefault(ch.target.node, {})[ch.target.path] = v
        return over

    def world_matrices(self, over: dict | None = None) -> dict:
        over = over or {}
        cache = {}

        def world(i):
            if i in cache:
                return cache[i]
            n = self.g.nodes[i]
            o = over.get(i, {})
            L = self._local(n, o.get("translation"), o.get("rotation"))
            M = (world(self.parent[i]) @ L) if i in self.parent else L
            cache[i] = M
            return M
        for i in range(len(self.g.nodes)):
            world(i)
        return cache

    def pose(self, prim: dict, over: dict | None = None) -> np.ndarray:
        W = self.world_matrices(over)
        inv_mesh = np.linalg.inv(W[self.mesh_node])
        skin = np.stack([inv_mesh @ W[j] @ self.ibm[k] for k, j in enumerate(self.joint_nodes)])
        V = np.c_[prim["V"], np.ones(len(prim["V"]))]
        out = np.zeros((len(V), 3))
        for k in range(prim["J"].shape[1]):
            jk, wk = prim["J"][:, k], prim["W"][:, k]
            M = skin[jk]                                   # (n, 4, 4)
            out += wk[:, None] * np.einsum("nij,nj->ni", M[:, :3, :], V)
        return out

    def clip_duration(self, name: str) -> float:
        a = self.anims[name]
        return max(float(read_accessor(self.g, s.input, self.blob).max()) for s in a.samplers)


# --------------------------------------------------------------------------
# measures
# --------------------------------------------------------------------------

def body_frame(body_V: np.ndarray) -> dict:
    bmin, bmax = body_V.min(0), body_V.max(0)
    H = float(bmax[1] - bmin[1])
    band = (body_V[:, 1] > bmin[1] + 0.15 * H) & (body_V[:, 1] < bmin[1] + 0.45 * H)
    ax = float(0.5 * (bmin[0] + bmax[0]))
    az = float(body_V[band][:, 2].mean())
    return {"y0": float(bmin[1]), "H": H, "axis_xz": [ax, az], "bbox": [bmin.tolist(), bmax.tolist()]}


def theta_of(P: np.ndarray, frame: dict) -> np.ndarray:
    ax, az = frame["axis_xz"]
    return np.degrees(np.arctan2(P[:, 0] - ax, P[:, 2] - az))


def sector_of(theta_deg: np.ndarray) -> np.ndarray:
    t = (theta_deg + 45.0) % 360.0
    return np.array(SECTORS)[(t // 90).astype(int)]


def band_of(v_cloth: np.ndarray) -> np.ndarray:
    out = np.empty(len(v_cloth), dtype=object)
    for name, lo, hi in BANDS:
        out[(v_cloth >= lo) & (v_cloth < hi)] = name
    return out


class Surface:
    """Dense samples of a body with their face normals + a ray-parity check."""

    def __init__(self, V: np.ndarray, F: np.ndarray, n: int = 1_000_000, seed: int = 0):
        self.mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
        pts, fi = trimesh.sample.sample_surface(self.mesh, n, seed=seed)
        self.bary = trimesh.triangles.points_to_barycentric(self.mesh.triangles[fi], pts)
        self.face = np.asarray(fi)
        self.set_vertices(V)

    def set_vertices(self, V: np.ndarray):
        tri = np.asarray(V)[self.mesh.faces[self.face]]
        self.S = np.einsum("nk,nkj->nj", self.bary, tri)
        N = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        self.N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        self.tree = cKDTree(self.S)
        self.posed = trimesh.Trimesh(vertices=np.asarray(V), faces=self.mesh.faces, process=False)

    def signed(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d, i = self.tree.query(P, workers=-1)
        behind = np.einsum("ij,ij->i", P - self.S[i], self.N[i]) < 0
        inside = behind.copy()
        if behind.any():
            inside[behind] = np.asarray(self.posed.contains(P[behind]), dtype=bool)
        return np.where(inside, -d, d), behind


def stats(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "min": float(x.min()), "p10": float(np.percentile(x, 10)), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "max": float(x.max())}


def contact_classes(sd: np.ndarray) -> dict:
    n = max(len(sd), 1)
    return {"penetrated": float((sd < 0).mean()), "contact": float(((sd >= 0) & (sd <= CONTACT)).mean()),
            "near": float(((sd > CONTACT) & (sd <= NEAR)).mean()), "free": float((sd > NEAR).mean()), "n": int(n)}


def zone_table(sd: np.ndarray, sector: np.ndarray, band: np.ndarray, extra: dict | None = None) -> dict:
    out = {}
    for b, *_ in BANDS:
        for s in SECTORS:
            m = (sector == s) & (band == b)
            if m.sum() < 3:
                continue
            row = {"distance": stats(sd[m]), "classes": contact_classes(sd[m])}
            for k, arr in (extra or {}).items():
                row[k] = stats(arr[m])
            out[f"{b}/{s}"] = row
    return out


def pattern_coords(tunic: dict, report: dict) -> tuple[np.ndarray, np.ndarray, callable]:
    """phi (around, [0,1]) and v (down, cloth units) per vertex, read off the
    chart, plus the rest circumference profile circ(v) rebuilt from
    drape.draft's formula and the report's numbers."""
    k = report["kente"]
    length = k["pattern"]["length"]
    if "profile" in k["pattern"]:
        # builds since 2026-08-22 evening ship the rest profile itself
        pv = np.asarray(k["pattern"]["profile"]["v"], dtype=np.float64)
        pc = np.asarray(k["pattern"]["profile"]["circ"], dtype=np.float64)

        def circ(v):
            return np.interp(np.asarray(v, dtype=np.float64), pv, pc)
    else:
        # the first tunic's two-point profile, rebuilt from drape.draft's formula of that day
        c_neck, c_hem = k["pattern"]["circumference"]
        H = report["_H"]
        hem_ease = 0.06
        chest_ease = 0.20
        girth = c_hem / (1 + hem_ease)
        c_body = girth * (1 + chest_ease)
        yoke = max(0.075 * H, 1e-6)

        def circ(v):
            ty = np.clip(v / yoke, 0, 1)
            c = c_neck + (c_body - c_neck) * (ty * ty * (3 - 2 * ty))
            tt = np.clip((v - yoke) / max(length - yoke, 1e-6), 0, 1)
            return c + (c_hem - c) * (tt * tt * (3 - 2 * tt))
    uv = tunic["UV"]
    return uv[:, 0].copy(), uv[:, 1] * length, circ


def wrap(d):
    return (d + 0.5) % 1.0 - 0.5


def edges_of(F: np.ndarray) -> np.ndarray:
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def boundary_loops(F: np.ndarray) -> list[np.ndarray]:
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    es = np.sort(e, axis=1)
    _, idx, cnt = np.unique(es, axis=0, return_index=True, return_counts=True)
    bnd = es[idx[cnt == 1]]
    adj = {}
    for a, b in bnd:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen, loops = set(), []
    for start in adj:
        if start in seen:
            continue
        loop, cur, prev = [start], start, None
        seen.add(start)
        while True:
            nxt = [n for n in adj[cur] if n != prev and n not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            seen.add(cur)
            loop.append(cur)
        loops.append(np.array(loop))
    return loops


def strain(tunic: dict, phi: np.ndarray, v: np.ndarray, circ) -> dict:
    V, F = tunic["V"], tunic["F"]
    E = edges_of(F)
    a, b = E[:, 0], E[:, 1]
    dphi = wrap(phi[b] - phi[a])
    dv = v[b] - v[a]
    cm = circ(0.5 * (v[a] + v[b]))
    l0 = np.hypot(dphi * cm, dv)
    l1 = np.linalg.norm(V[b] - V[a], axis=1)
    ok = l0 > 1e-6
    ratio = np.where(ok, l1 / np.maximum(l0, 1e-9), 1.0)
    ang = np.degrees(np.arctan2(np.abs(dv), np.abs(dphi * cm) + 1e-12))
    kind = np.where(ang > 67.5, "warp", np.where(ang < 22.5, "weft", "diag"))
    # area strain per face
    def flat(i):
        return np.stack([phi[i], v[i]], axis=1)
    p0, p1, p2 = flat(F[:, 0]), flat(F[:, 1]), flat(F[:, 2])
    cmf = circ((v[F[:, 0]] + v[F[:, 1]] + v[F[:, 2]]) / 3)
    d1 = np.stack([wrap(p1[:, 0] - p0[:, 0]) * cmf, p1[:, 1] - p0[:, 1]], axis=1)
    d2 = np.stack([wrap(p2[:, 0] - p0[:, 0]) * cmf, p2[:, 1] - p0[:, 1]], axis=1)
    a0 = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    a1 = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    area_ratio = np.where(a0 > 1e-9, a1 / np.maximum(a0, 1e-12), 1.0)
    # per-vertex aggregates
    n = len(V)
    vs = np.zeros(n)
    vc = np.zeros(n)
    np.add.at(vs, a, ratio)
    np.add.at(vs, b, ratio)
    np.add.at(vc, a, 1)
    np.add.at(vc, b, 1)
    v_ratio = np.where(vc > 0, vs / np.maximum(vc, 1), 1.0)
    fa = np.zeros(n)
    fc = np.zeros(n)
    for k in range(3):
        np.add.at(fa, F[:, k], area_ratio)
        np.add.at(fc, F[:, k], 1)
    v_area = np.where(fc > 0, fa / np.maximum(fc, 1), 1.0)
    # dihedral angle per interior edge (bend): 0 = flat
    tm = trimesh.Trimesh(vertices=V, faces=F, process=False)
    fa_idx = tm.face_adjacency
    dihed = np.degrees(tm.face_adjacency_angles)
    vb = np.zeros(n)
    vbc = np.zeros(n)
    ed = tm.face_adjacency_edges
    np.add.at(vb, ed[:, 0], dihed)
    np.add.at(vb, ed[:, 1], dihed)
    np.add.at(vbc, ed[:, 0], 1)
    np.add.at(vbc, ed[:, 1], 1)
    v_bend = np.where(vbc > 0, vb / np.maximum(vbc, 1), 0.0)
    return {"edge_ratio": ratio, "edge_kind": kind, "edge_l0": l0, "edges": E, "face_area_ratio": area_ratio,
            "v_ratio": v_ratio, "v_area": v_area, "v_bend": v_bend, "dihedral": dihed}


def width_profile(sets: dict, frame: dict, step: float = 0.025) -> dict:
    """x- and z-extent per height band, for each named vertex set."""
    y0, H = frame["y0"], frame["H"]
    fr = np.arange(0.075, 0.60 + 1e-9, step)
    out = {"y_frac": [round(float(f), 3) for f in fr]}
    for name, V in sets.items():
        xs, zs = [], []
        for f in fr:
            y = y0 + f * H
            m = np.abs(V[:, 1] - y) < 0.0125 * H
            if m.sum() > 3:
                xs.append(round(float(V[m, 0].max() - V[m, 0].min()), 3))
                zs.append(round(float(V[m, 2].max() - V[m, 2].min()), 3))
            else:
                xs.append(None)
                zs.append(None)
        out[name] = {"x_extent": xs, "z_extent": zs}
    return out


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def stage_metrics(sc: Scene, report: dict, out: Path) -> dict:
    body = sc.prim("owl_body")
    tunic = sc.prim("owl_kente_tunic")
    sleeve = sc.prim("owl_kente_sleeve")
    frame = body_frame(body["V"])
    report["_H"] = frame["H"]
    H, y0 = frame["H"], frame["y0"]
    R = {"frame": frame, "conventions": __doc__.split("Conventions")[1].strip(),
         "contact_thresholds": {"contact": CONTACT, "near": NEAR}}

    surf = Surface(body["V"], body["F"])
    T = tunic["V"]
    th = theta_of(T, frame)
    sec = sector_of(th)
    phi, v, circ = pattern_coords(tunic, report)
    band = band_of(v)
    sd, suspects = surf.signed(T)
    st = strain(tunic, phi, v, circ)
    yfrac = (T[:, 1] - y0) / H

    R["tunic"] = {
        "vertices": int(len(T)), "faces": int(len(tunic["F"])),
        "pattern": {"phi_range": [float(phi.min()), float(phi.max())], "v_range": [float(v.min()), float(v.max())],
                    "circ_at_v": {f"{x:.2f}": round(float(circ(np.array([x]))[0]), 4) for x in (0.0, 0.05, 0.1, 0.14, 0.2, 0.3, 0.45, 0.6, 0.74)}},
        "signed_distance": stats(sd), "classes": contact_classes(sd),
        "inside_count": int((sd < 0).sum()), "suspects_before_parity": int(suspects.sum()),
        "by_zone": zone_table(sd, sec, band, {"stretch": st["v_ratio"], "area_ratio": st["v_area"], "bend_deg": st["v_bend"], "y_frac": yfrac}),
        "by_yfrac": {},
        "stretch": {
            "all_edges": stats(st["edge_ratio"]),
            "by_kind": {k: stats(st["edge_ratio"][st["edge_kind"] == k]) for k in ("warp", "weft", "diag")},
            "fraction_over_1.03": float((st["edge_ratio"] > 1.03).mean()),
            "fraction_over_1.10": float((st["edge_ratio"] > 1.10).mean()),
            "fraction_under_0.97": float((st["edge_ratio"] < 0.97).mean()),
            "fraction_under_0.90": float((st["edge_ratio"] < 0.90).mean()),
        },
        "area_ratio": stats(st["face_area_ratio"]),
        "dihedral_deg": {"all": stats(st["dihedral"]), "fraction_over_20": float((st["dihedral"] > 20).mean()),
                         "fraction_under_3": float((st["dihedral"] < 3).mean())},
    }
    for lo in np.arange(0.10, 0.56, 0.05):
        m = (yfrac >= lo) & (yfrac < lo + 0.05)
        if m.sum() > 3:
            R["tunic"]["by_yfrac"][f"{lo:.2f}-{lo + 0.05:.2f}"] = {
                "distance": stats(sd[m]), "classes": contact_classes(sd[m]),
                "per_sector_median": {s: (round(float(np.median(sd[m & (sec == s)])), 4) if (m & (sec == s)).sum() > 2 else None) for s in SECTORS}}

    # the worst stretch / compression spots, located
    order = np.argsort(-st["edge_ratio"])[:12]
    R["tunic"]["stretch_peaks"] = [{"ratio": round(float(st["edge_ratio"][i]), 3), "kind": str(st["edge_kind"][i]),
                                    "y_frac": round(float(yfrac[st["edges"][i, 0]]), 3), "theta_deg": round(float(th[st["edges"][i, 0]]), 1),
                                    "band": str(band[st["edges"][i, 0]])} for i in order]
    order = np.argsort(st["edge_ratio"])[:12]
    R["tunic"]["compression_peaks"] = [{"ratio": round(float(st["edge_ratio"][i]), 3), "kind": str(st["edge_kind"][i]),
                                        "y_frac": round(float(yfrac[st["edges"][i, 0]]), 3), "theta_deg": round(float(th[st["edges"][i, 0]]), 1),
                                        "band": str(band[st["edges"][i, 0]])} for i in order]

    # boundary loops: what is open, and where
    loops = []
    for L in boundary_loops(tunic["F"]):
        P = T[L]
        seg = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1).sum()
        loops.append({"vertices": int(len(L)), "length": round(float(seg), 3),
                      "y_frac": [round(float(((P[:, 1] - y0) / H).min()), 3), round(float(((P[:, 1] - y0) / H).max()), 3)],
                      "v_cloth": [round(float(v[L].min()), 3), round(float(v[L].max()), 3)],
                      "theta_deg": [round(float(th[L].min()), 1), round(float(th[L].max()), 1)],
                      "sectors": {s: int((sec[L] == s).sum()) for s in SECTORS},
                      "mean_distance_to_body": round(float(np.mean(sd[L])), 4)})
    loops.sort(key=lambda d: -d["vertices"])
    R["tunic"]["boundary_loops"] = loops
    # hem height around the figure: the lowest tunic vertex per 10-degree sector
    hem = {}
    for t0 in range(-180, 180, 10):
        m = (th >= t0) & (th < t0 + 10)
        if m.any():
            hem[str(t0)] = round(float(yfrac[m].min()), 3)
    R["tunic"]["hem_y_frac_by_theta"] = hem

    # sleeve vs the wing it wraps
    S = sleeve["V"]
    ssd, _ = surf.signed(S)
    piv, axis = np.array(report["sleeve"]["pivot"]), np.array(report["sleeve"]["axis"])
    s = (S - piv) @ axis
    s_root, s_sep, s_cuff = report["sleeve"]["s_root"], report["sleeve"]["s_separation"], report["sleeve"]["s_cuff"]
    seg = np.where(s < s_sep, "root_under_tunic", np.where(s < s_sep + 0.5 * (s_cuff - s_sep), "upper", "lower_to_cuff"))
    R["sleeve"] = {"vertices": int(len(S)), "signed_distance": stats(ssd), "classes": contact_classes(ssd),
                   "inside_count": int((ssd < 0).sum()),
                   "by_segment": {k: {"distance": stats(ssd[seg == k]), "classes": contact_classes(ssd[seg == k])} for k in np.unique(seg)},
                   "s_range": [float(s.min()), float(s.max())], "s_root": s_root, "s_separation": s_sep, "s_cuff": s_cuff}
    # tunic over the sleeve root: is there tunic cloth *beyond* each root-ring
    # vertex on the ray from the wing axis through it (a cover test), and
    # how far is the nearest cloth
    root = S[s < s_sep]
    if len(root):
        tm = trimesh.Trimesh(vertices=T, faces=tunic["F"], process=False)
        origins = piv + np.outer((root - piv) @ axis, axis)
        dirs = root - origins
        L = np.linalg.norm(dirs, axis=1)
        dirs /= np.maximum(L[:, None], 1e-12)
        locs, ray_ids, _ = tm.ray.intersects_location(origins, dirs, multiple_hits=True)
        covered = np.zeros(len(root), dtype=bool)
        above = np.full(len(root), np.nan)
        for loc, rid in zip(locs, ray_ids):
            t = float((loc - origins[rid]) @ dirs[rid])
            if t > L[rid] + 1e-6:
                covered[rid] = True
                above[rid] = t - L[rid] if np.isnan(above[rid]) else min(above[rid], t - L[rid])
        nearest, _ = cKDTree(T).query(root, workers=-1)
        R["sleeve"]["root_vs_tunic"] = {"n": int(len(root)), "fraction_covered": float(covered.mean()),
                                        "cloth_above_covered": stats(above[covered]) if covered.any() else {"n": 0},
                                        "nearest_cloth": stats(nearest)}

    # silhouette: garment vs naked body
    R["width_profile"] = width_profile({"body": body["V"], "tunic": T, "sleeve": S,
                                        "clothed": np.vstack([body["V"], T, S])}, frame)

    # skin weights: what holds the cloth when the body moves
    def wmass(prim):
        names = np.array(sc.joint_names)
        tot = {}
        for k in range(prim["J"].shape[1]):
            for j in np.unique(prim["J"][:, k]):
                m = prim["J"][:, k] == j
                tot[str(names[j])] = tot.get(str(names[j]), 0.0) + float(prim["W"][m, k].sum())
        tot = {k: round(x / len(prim["V"]), 4) for k, x in tot.items() if x > 0}
        return dict(sorted(tot.items(), key=lambda kv: -kv[1]))
    R["weights"] = {"tunic": wmass(tunic), "sleeve": wmass(sleeve)}
    # how the cloth's weights compare with the nearest body vertex's (kinematic connection)
    bt = cKDTree(body["V"])
    _, nb = bt.query(T, workers=-1)
    nj = len(sc.joint_names)

    def dense(prim, idx=None):
        n = len(prim["V"]) if idx is None else len(idx)
        Wd = np.zeros((n, nj))
        J = prim["J"] if idx is None else prim["J"][idx]
        W = prim["W"] if idx is None else prim["W"][idx]
        for k in range(J.shape[1]):
            np.add.at(Wd, (np.arange(n), J[:, k]), W[:, k])
        return Wd
    Wt, Wb = dense(tunic), dense(body, nb)
    wdiff = 0.5 * np.abs(Wt - Wb).sum(1)       # 0 = same skin as the body under it, 1 = entirely different joints
    R["tunic"]["weight_mismatch_vs_nearest_body"] = {"all": stats(wdiff),
                                                      "by_band": {b: stats(wdiff[band == b]) for b, *_ in BANDS if (band == b).sum() > 3},
                                                      "by_sector": {s: stats(wdiff[sec == s]) for s in SECTORS}}

    # through the clips: LBS from the GLB itself
    posed = {}
    for clip in sc.anims:
        dur = sc.clip_duration(clip)
        worst = None
        for tm in np.linspace(0, dur, 6):
            over = sc.clip_overrides(clip, float(tm))
            Bp = sc.pose(body, over)
            Tp = sc.pose(tunic, over)
            Sp = sc.pose(sleeve, over)
            surf.set_vertices(Bp)
            psd, _ = surf.signed(Tp)
            pss, _ = surf.signed(Sp)
            # motion coherence: garment displacement vs the nearest body vertex's
            dT = Tp - T
            dB = (Bp - body["V"])[nb]
            mag = np.linalg.norm(dT, axis=1), np.linalg.norm(dB, axis=1)
            moving = mag[1] > 0.003
            coh = np.full(len(T), np.nan)
            if moving.any():
                coh[moving] = np.einsum("ij,ij->i", dT[moving], dB[moving]) / np.maximum(mag[0][moving] * mag[1][moving], 1e-9)
            rec = {"time": round(float(tm), 3), "tunic": {"distance": stats(psd), "inside": int((psd < 0).sum()), "classes": contact_classes(psd),
                                                          "by_band_min": {b: round(float(psd[band == b].min()), 4) for b, *_ in BANDS if (band == b).any()}},
                   "sleeve": {"distance": stats(pss), "inside": int((pss < 0).sum())},
                   "coherence": {"body_vertices_moving_fraction": float(moving.mean()),
                                 "cos_by_band": {b: (round(float(np.nanmean(coh[(band == b) & moving])), 3) if ((band == b) & moving).sum() > 3 else None) for b, *_ in BANDS},
                                 "mag_ratio_by_band": {b: (round(float(np.median(mag[0][(band == b) & moving] / np.maximum(mag[1][(band == b) & moving], 1e-9))), 3) if ((band == b) & moving).sum() > 3 else None) for b, *_ in BANDS}}}
            if worst is None or rec["tunic"]["distance"]["min"] < worst["tunic"]["distance"]["min"]:
                worst = rec
        posed[clip] = {"duration": round(dur, 3), "worst_frame": worst}
        surf.set_vertices(body["V"])
    R["posed"] = posed

    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(R, indent=1))
    np.savez(out / "per_vertex.npz", tunic_sd=sd, tunic_theta=th, tunic_v=v, tunic_phi=phi, tunic_yfrac=yfrac,
             tunic_stretch=st["v_ratio"], tunic_area=st["v_area"], tunic_bend=st["v_bend"], sleeve_sd=ssd,
             sector=sec.astype("U8"), band=band.astype("U8"))
    return R


def _ramp(x, stops):
    """piecewise-linear colour ramp; stops = [(t, (r,g,b)), ...] sorted by t."""
    x = np.asarray(x, dtype=np.float64)
    ts = np.array([s[0] for s in stops])
    cs = np.array([s[1] for s in stops])
    out = np.zeros((len(x), 3))
    for k in range(3):
        out[:, k] = np.interp(x, ts, cs[:, k])
    return out


def stage_maps(sc: Scene, out: Path) -> list[Path]:
    from meshforge.fastpreview import render_flat
    from PIL import Image, ImageDraw
    pv = np.load(out / "per_vertex.npz")
    body = sc.prim("owl_body")
    tunic = sc.prim("owl_kente_tunic")
    sleeve = sc.prim("owl_kente_sleeve")
    Vb, Fb = body["V"], body["F"]
    Vt, Ft = tunic["V"], tunic["F"]
    Vs, Fs = sleeve["V"], sleeve["F"]
    V = np.vstack([Vb, Vt, Vs])
    F = np.vstack([Fb, Ft + len(Vb), Fs + len(Vb) + len(Vt)])
    bounds = (V.min(0), V.max(0))
    gray_body = np.full((len(Vb), 3), 0.82)
    sd, ssd = pv["tunic_sd"], pv["sleeve_sd"]

    def dist_colors(d):
        c = np.zeros((len(d), 3))
        c[d < 0] = (1.0, 0.0, 1.0)                                   # penetrated: magenta
        m = (d >= 0) & (d <= CONTACT); c[m] = (0.85, 0.1, 0.1)        # contact: red
        m = (d > CONTACT) & (d <= NEAR); c[m] = (0.95, 0.6, 0.1)      # near: orange
        m = d > NEAR
        c[m] = _ramp(np.clip((d[m] - NEAR) / 0.15, 0, 1), [(0, (0.2, 0.75, 0.2)), (1, (0.1, 0.2, 0.9))])  # free: green -> blue with distance
        return c
    maps = {
        "distance": (dist_colors(sd), dist_colors(ssd), "magenta=inside  red=contact<=0.025  orange=near<=0.05  green->blue=free (0.05->0.20)"),
        "stretch": (_ramp(pv["tunic_stretch"], [(0.90, (0.1, 0.2, 0.9)), (0.97, (0.6, 0.75, 1.0)), (1.0, (0.92, 0.92, 0.92)), (1.03, (1.0, 0.7, 0.6)), (1.12, (0.8, 0.0, 0.0))]),
                    np.full((len(Vs), 3), 0.6), "edge length / pattern rest length: blue<0.97 compressed  white=1  red>1.03 stretched (sat 0.90/1.12)"),
        "area": (_ramp(pv["tunic_area"], [(0.80, (0.1, 0.2, 0.9)), (0.95, (0.6, 0.75, 1.0)), (1.0, (0.92, 0.92, 0.92)), (1.05, (1.0, 0.7, 0.6)), (1.25, (0.8, 0.0, 0.0))]),
                 np.full((len(Vs), 3), 0.6), "triangle area / pattern rest area: blue<0.95  white=1  red>1.05 (sat 0.80/1.25)"),
        "bend": (_ramp(pv["tunic_bend"], [(0, (0.95, 0.95, 0.95)), (5, (0.9, 0.85, 0.4)), (15, (0.9, 0.4, 0.1)), (40, (0.5, 0.0, 0.3))]),
                 np.full((len(Vs), 3), 0.6), "mean dihedral angle at the vertex, degrees: white=flat  yellow 5  orange 15  plum 40+"),
    }
    # zone id map
    zc = np.zeros((len(Vt), 3))
    band_col = {"neck": (0.9, 0.1, 0.1), "yoke": (0.95, 0.55, 0.1), "chest": (0.9, 0.85, 0.2), "skirt": (0.2, 0.7, 0.3), "hem": (0.2, 0.3, 0.9)}
    sec_mul = {"front": 1.0, "right": 0.75, "back": 0.5, "left": 0.6}
    for i in range(len(Vt)):
        zc[i] = np.array(band_col[str(pv["band"][i])]) * sec_mul[str(pv["sector"][i])]
    maps["zones"] = (zc, np.full((len(Vs), 3), 0.6), "bands: red neck, orange yoke, yellow chest, green skirt, blue hem; darker = right/back/left sector")
    written = []
    for name, (ct, cs, legend) in maps.items():
        C = np.vstack([gray_body, ct, cs])
        tiles = []
        for vname, yaw, pitch in (("front", 0, 0), ("right", 90, 0), ("back", 180, 0), ("left", 270, 0), ("front_up", 0, 35), ("hem_low", 0, -25)):
            img = render_flat(V, F, C, yaw_deg=yaw, pitch_deg=pitch, resolution=640, bounds=bounds, background=(245, 242, 235))
            img = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img))
            d = ImageDraw.Draw(img)
            d.text((8, 8), f"{name} — {vname} (yaw {yaw}, pitch {pitch})", fill=(20, 20, 20))
            tiles.append(img)
        W, Hh = tiles[0].size
        sheet = Image.new("RGB", (W * 3, Hh * 2 + 24), (245, 242, 235))
        for k, t in enumerate(tiles):
            sheet.paste(t, ((k % 3) * W, (k // 3) * Hh))
        ImageDraw.Draw(sheet).text((8, Hh * 2 + 6), legend, fill=(20, 20, 20))
        p = out / f"map_{name}.png"
        sheet.save(p)
        written.append(p)
    return written


def make_variants(src: Path, out: Path) -> dict:
    """Three GLBs for the renderer: garment gray (body as shipped), all
    gray, and the naked body (garment primitives removed)."""
    variants = {}

    def load():
        return GLTF2().load(str(src))

    def gray(mat, level, rough=0.9):
        pbr = mat.pbrMetallicRoughness
        pbr.baseColorTexture = None
        pbr.baseColorFactor = [level, level, level, 1.0]
        pbr.metallicFactor = 0.0
        pbr.roughnessFactor = rough
        pbr.metallicRoughnessTexture = None
        mat.normalTexture = None
        mat.emissiveTexture = None
        mat.emissiveFactor = [0.0, 0.0, 0.0]

    g = load()
    for m in g.materials:
        if m.name.startswith("owl_kente"):
            gray(m, 0.58)
    p = out / "variant_garment_gray.glb"
    g.save(str(p))
    variants["garment_gray"] = p

    g = load()
    for m in g.materials:
        if m.name.startswith("owl_kente"):
            gray(m, 0.45)
        elif m.name in ("owl_body", "owl_collar_text", "owl_beads"):
            gray(m, 0.80)
    p = out / "variant_all_gray.glb"
    g.save(str(p))
    variants["all_gray"] = p

    g = load()
    drop = {i for i, m in enumerate(g.materials) if m.name.startswith("owl_kente")}
    for m in g.meshes:
        m.primitives = [pr for pr in m.primitives if pr.material not in drop]
    for m in g.materials:
        if m.name in ("owl_body", "owl_collar_text", "owl_beads"):
            gray(m, 0.80)
    p = out / "variant_naked_gray.glb"
    g.save(str(p))
    variants["naked_gray"] = p
    return variants


def _shots(glb: Path, out: Path, views: list[str], clips: list[str] = (), prefix: str = "", zoom: float = 1.0,
           target: str = "0,0,0", size: int = 640):
    cmd = [sys.executable, str(ROOT / "tools" / "owl_shots.py"), str(glb), "--out", str(out), "--size", str(size),
           "--zoom", str(zoom), f"--target={target}", "--prefix", prefix, "--bg", "#e9e4d8"]   # '=' form: the target may start with '-'
    for v in views:
        cmd += ["--view", v]
    for c in clips:
        cmd += ["--clip", c]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def contact_sheet(paths: list[Path], dest: Path, cols: int = 6, label: str = ""):
    from PIL import Image, ImageDraw
    imgs = [Image.open(p).convert("RGB") for p in paths]
    if not imgs:
        return
    w, h = imgs[0].size
    rows = math.ceil(len(imgs) / cols)
    sheet = Image.new("RGB", (w * cols, h * rows + 22), (233, 228, 216))
    for k, (im, p) in enumerate(zip(imgs, paths)):
        sheet.paste(im, ((k % cols) * w, (k // cols) * h))
        ImageDraw.Draw(sheet).text(((k % cols) * w + 6, (k // cols) * h + 6), p.stem, fill=(30, 30, 30))
    ImageDraw.Draw(sheet).text((6, h * rows + 5), label, fill=(30, 30, 30))
    sheet.save(dest)


def _zoom_plan(src: Path):
    """Close-up crops, aimed at where the garment's parts actually ended up.

    The wrap's knot and hanging tail are placed by a measurement of the
    body, not by stored coordinates, so the camera has to be aimed the same
    way: load the build, find the garment primitives by material, and point
    each crop at that part's own centre. Falls back to the tunic-era crops
    (armscye, hand slit, neckline) for builds without a wrap."""
    scene = trimesh.load(str(src), process=False)
    try:
        centre = np.asarray(scene.dump(concatenate=True).bounds).mean(axis=0)
    except Exception:
        centre = np.zeros(3)
    parts = {}
    for g in getattr(scene, "geometry", {}).values():
        mat = getattr(getattr(g, "visual", None), "material", None)
        name = getattr(mat, "name", "") or ""
        if name.startswith("owl_kente"):
            parts.setdefault(name, []).append((len(g.vertices), np.asarray(g.bounds).mean(axis=0)))
    wrap = sorted(parts.get("owl_kente_wrap", []))
    if len(wrap) < 2:
        return ([("armscye_left", "-75:5", 2.2, "-0.30,0.15,-0.05"), ("armscye_left_back", "-140:8", 2.2, "-0.30,0.15,-0.05"),
                 ("shoulder_right", "60:8", 2.2, "0.32,0.12,0.05"), ("hand_slit", "30:0", 2.0, "0.30,0.0,0.25"),
                 ("hem_front", "0:-12", 1.7, "0,-0.55,0"), ("hem_side", "90:-12", 1.7, "0,-0.55,0"),
                 ("neckline", "0:30", 2.0, "0,0.35,0.1")],
                "garment gray zooms: left armscye (side, back), right shoulder, hand slit, hem front/side, neckline from above")
    # the wrap shares one material between the sheet and the knot; the knot
    # is the small primitive, the sheet the large one
    knot, sheet = wrap[0][1], wrap[-1][1]
    tail = parts.get("owl_kente_wrap_tail", [(0, sheet)])[-1][1]

    def aim(p, dy=0.0, dyaw=0.0):
        t = np.asarray(p, dtype=float) - centre + np.array([0.0, dy, 0.0])
        yaw = math.degrees(math.atan2(t[0], t[2])) + dyaw
        return f"{t[0]:.3f},{t[1]:.3f},{t[2]:.3f}", round(yaw)

    kt, ky = aim(knot)
    ft, fy = aim(knot, dy=-0.16, dyaw=-40)
    tt, ty = aim(tail)
    zooms = [("knot", f"{ky}:8", 3.4, kt), ("knot_front", "0:8", 2.6, kt), ("knot_high", f"{ky}:38", 3.0, kt),
             ("flank", f"{fy}:-4", 2.4, ft), ("tail", f"{ty}:-6", 2.2, tt),
             ("hem_front", "0:-12", 1.7, "0,-0.55,0"), ("hem_back_left", "225:-12", 1.7, "0,-0.55,0")]
    return zooms, ("garment gray zooms (aimed at the measured parts): knot from its own side / front / above, "
                   "the wing-base flank, the hanging tail, hem front and back-left")


def stage_renders(src: Path, out: Path) -> list[Path]:
    rd = out / "renders"
    rd.mkdir(parents=True, exist_ok=True)
    var = make_variants(src, out)
    turn = [f"t{yaw:03d}:{yaw}:4" for yaw in range(0, 360, 30)]
    high = [f"h{yaw:03d}:{yaw}:35" for yaw in (0, 90, 180, 270)]
    low = [f"l{yaw:03d}:{yaw}:-20" for yaw in (0, 90, 180, 270)]
    sheets = []
    for name in ("garment_gray", "all_gray", "naked_gray"):
        d = rd / name
        _shots(var[name], d, turn + high + low, prefix="rest_")
        contact_sheet([d / f"rest_{v.split(':')[0]}.png" for v in turn], rd / f"sheet_{name}_turntable.png", label=f"{name}: rest, yaw every 30 deg, pitch 4")
        contact_sheet([d / f"rest_{v.split(':')[0]}.png" for v in high + low], rd / f"sheet_{name}_high_low.png", cols=4, label=f"{name}: rest, pitch +35 (top row) and -20 (bottom row)")
        sheets += [rd / f"sheet_{name}_turntable.png", rd / f"sheet_{name}_high_low.png"]
    # posed, garment gray
    poses = {"wave": 0.55, "wave_peak": ("wave", 0.95), "tablet_show": 0.7, "tablet_056": ("tablet_show", 0.56),
             "hop": 0.45, "nod": 0.45, "idle": 1.0}
    d = rd / "garment_gray"
    posed_paths = []
    for key, spec in poses.items():
        clip, t = (spec if isinstance(spec, tuple) else (key, spec))
        views = [f"{key}_front:0:4", f"{key}_right:90:4", f"{key}_back:180:4", f"{key}_left:270:4"]
        _shots(var["garment_gray"], d, views, clips=[f"{clip}={t}"], prefix="pose_")
        posed_paths += [d / f"pose_{v.split(':')[0]}.png" for v in views]
    contact_sheet(posed_paths, rd / "sheet_garment_gray_posed.png", cols=4, label="garment gray through the clips: wave 0.55 / wave 0.95 / tablet_show 0.7 / hop 0.45 / nod 0.45 / idle 1.0; front right back left")
    sheets.append(rd / "sheet_garment_gray_posed.png")
    # zooms on the parts the build measured for itself
    zooms, zoom_label = _zoom_plan(src)
    zp = []
    for name, yp, zoom, target in zooms:
        _shots(var["garment_gray"], d, [f"{name}:{yp}"], prefix="zoom_", zoom=zoom, target=target)
        zp.append(d / f"zoom_{name}.png")
    contact_sheet(zp, rd / "sheet_garment_gray_zooms.png", cols=4, label=zoom_label)
    sheets.append(rd / "sheet_garment_gray_zooms.png")
    # the shipped look, same views, for reference
    d = rd / "shipped"
    _shots(src, d, turn, prefix="rest_")
    contact_sheet([d / f"rest_{v.split(':')[0]}.png" for v in turn], rd / "sheet_shipped_turntable.png", label="as shipped (kente texture): rest turntable")
    sheets.append(rd / "sheet_shipped_turntable.png")
    return sheets


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("glb", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True, help="the build's .report.json (pattern numbers, sleeve frame)")
    ap.add_argument("--stage", default="all", choices=("metrics", "maps", "renders", "all"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.report.read_text())
    sc = Scene(args.glb)
    if args.stage in ("metrics", "all"):
        R = stage_metrics(sc, report, args.out)
        t = R["tunic"]
        print(json.dumps({"tunic_signed_distance": t["signed_distance"], "classes": t["classes"], "inside": t["inside_count"],
                          "stretch": t["stretch"]["all_edges"], "loops": [(l["vertices"], l["y_frac"], l["v_cloth"]) for l in t["boundary_loops"]],
                          "sleeve": R["sleeve"]["signed_distance"]}, indent=1))
    if args.stage in ("maps", "all"):
        for p in stage_maps(sc, args.out):
            print(p)
    if args.stage in ("renders", "all"):
        for p in stage_renders(args.glb, args.out):
            print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
