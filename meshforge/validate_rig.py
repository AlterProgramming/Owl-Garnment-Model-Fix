"""Structural validator for GLBs written by meshforge.rigexport (multi-
primitive skin, N-joint hierarchy, several clips with rotation and
translation tracks). The rigging guide's "Level 1: hard structural
invariants", checked against the exported bytes, not the in-memory
arrays that produced them.

    python3 -m meshforge.validate_rig viewer/assets/owl.glb
"""
from __future__ import annotations

import sys

import numpy as np
import pygltflib

_COMPONENT_DTYPES = {
    5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32,
}
_TYPE_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read_accessor(gltf: pygltflib.GLTF2, blob: bytes, index: int) -> np.ndarray:
    acc = gltf.accessors[index]
    view = gltf.bufferViews[acc.bufferView]
    dtype = _COMPONENT_DTYPES[acc.componentType]
    width = _TYPE_WIDTH[acc.type]
    start = (view.byteOffset or 0) + (acc.byteOffset or 0)
    count = acc.count * width
    arr = np.frombuffer(blob, dtype=dtype, count=count, offset=start)
    arr = arr.reshape(acc.count, width) if width > 1 else arr
    if acc.normalized:
        arr = arr.astype(np.float64) / np.iinfo(dtype).max
    return arr


def _node_world_rest(gltf: pygltflib.GLTF2, index: int, cache: dict, parents: dict) -> np.ndarray:
    if index in cache:
        return cache[index]
    node = gltf.nodes[index]
    local = np.eye(4)
    if node.matrix:
        local = np.array(node.matrix, dtype=np.float64).reshape(4, 4).T
    else:
        if node.rotation:
            x, y, z, w = node.rotation
            local[:3, :3] = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ])
        if node.scale:
            local[:3, :3] = local[:3, :3] @ np.diag(node.scale)
        if node.translation:
            local[:3, 3] = node.translation
    parent = parents.get(index)
    world = local if parent is None else _node_world_rest(gltf, parent, cache, parents) @ local
    cache[index] = world
    return world


def validate(path: str, expect_clips: list[str] | None = None, expect_joints: list[str] | None = None,
             verbose: bool = True) -> list[str]:
    """Returns the list of failure messages (empty = pass)."""
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = ""):
        if verbose:
            print(f"{'PASS' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail and not ok else ''}")
        if not ok:
            failures.append(f"{name}{(' — ' + detail) if detail else ''}")

    gltf = pygltflib.GLTF2().load(path)
    blob = gltf.binary_blob()

    parents = {}
    for i, node in enumerate(gltf.nodes):
        for c in node.children or []:
            check(f"node {c} has a single parent", c not in parents, f"also child of {parents.get(c)}")
            parents[c] = i
    # cycle check
    for i in range(len(gltf.nodes)):
        seen = set()
        cur = i
        while cur is not None and cur not in seen:
            seen.add(cur)
            cur = parents.get(cur)
        check(f"node {i} ancestry is acyclic", cur is None)

    names = [n.name for n in gltf.nodes]
    check("node names are unique", len(set(names)) == len(names))

    mesh_nodes = [n for n in gltf.nodes if n.mesh is not None]
    check("exactly one mesh-bearing node", len(mesh_nodes) == 1, f"{len(mesh_nodes)}")
    check("at least one skin", len(gltf.skins) >= 1)
    if not gltf.skins or not mesh_nodes:
        return failures
    skin = gltf.skins[0]
    check("mesh node references the skin", mesh_nodes[0].skin == 0)
    joints = skin.joints
    n_joints = len(joints)
    joint_names = [gltf.nodes[j].name for j in joints]
    if expect_joints:
        missing = [j for j in expect_joints if j not in joint_names]
        check("expected joints present", not missing, f"missing {missing}")

    # inverse bind × world bind ≈ identity
    ibm = read_accessor(gltf, blob, skin.inverseBindMatrices).reshape(-1, 4, 4)
    check("one inverse bind matrix per joint", len(ibm) == n_joints)
    cache: dict = {}
    worst = 0.0
    for k, j in enumerate(joints):
        world = _node_world_rest(gltf, j, cache, parents)
        inv = ibm[k].T  # glTF column-major -> row-major
        err = float(np.abs(world @ inv - np.eye(4)).max())
        worst = max(worst, err)
    check("world_bind × inverse_bind ≈ identity for every joint", worst < 1e-4, f"max error {worst:.2e}")

    mesh = gltf.meshes[mesh_nodes[0].mesh]
    check("mesh has primitives", len(mesh.primitives) >= 1)
    has_texture = False
    for p_idx, prim in enumerate(mesh.primitives):
        a = prim.attributes
        pos = read_accessor(gltf, blob, a.POSITION)
        check(f"prim {p_idx}: positions finite", bool(np.isfinite(pos).all()))
        check(f"prim {p_idx}: has NORMAL", a.NORMAL is not None)
        if a.NORMAL is not None:
            nrm = read_accessor(gltf, blob, a.NORMAL)
            lens = np.linalg.norm(nrm, axis=1)
            check(f"prim {p_idx}: normals unit length", bool(np.all(np.abs(lens - 1) < 1e-2)), f"range {lens.min():.3f}..{lens.max():.3f}")
        idx = read_accessor(gltf, blob, prim.indices)
        check(f"prim {p_idx}: indices in range", int(idx.max()) < len(pos), f"max {int(idx.max())} verts {len(pos)}")
        check(f"prim {p_idx}: index count multiple of 3", len(idx) % 3 == 0)
        if a.JOINTS_0 is not None or a.WEIGHTS_0 is not None:
            check(f"prim {p_idx}: JOINTS_0 and WEIGHTS_0 both present", a.JOINTS_0 is not None and a.WEIGHTS_0 is not None)
            if a.JOINTS_0 is not None and a.WEIGHTS_0 is not None:
                jn = read_accessor(gltf, blob, a.JOINTS_0)
                w = read_accessor(gltf, blob, a.WEIGHTS_0).astype(np.float64)
                check(f"prim {p_idx}: joint indices < joint count", int(jn.max()) < n_joints, f"max {int(jn.max())}")
                check(f"prim {p_idx}: weights finite and non-negative", bool(np.isfinite(w).all() and (w >= -1e-6).all()))
                sums = w.sum(axis=1)
                check(f"prim {p_idx}: weights sum to 1", bool(np.all(np.abs(sums - 1) < 1e-3)), f"range {sums.min():.4f}..{sums.max():.4f}")
        mat = gltf.materials[prim.material] if prim.material is not None else None
        check(f"prim {p_idx}: has a material", mat is not None)
        if mat and mat.pbrMetallicRoughness and mat.pbrMetallicRoughness.baseColorTexture is not None:
            has_texture = True
            check(f"prim {p_idx}: textured primitive has TEXCOORD_0", a.TEXCOORD_0 is not None)
    check("at least one baseColorTexture", has_texture)
    for im_idx, image in enumerate(gltf.images):
        check(f"image {im_idx} embedded", image.bufferView is not None)

    clip_names = [anim.name for anim in gltf.animations]
    check("at least one animation", len(gltf.animations) >= 1)
    if expect_clips:
        missing = [c for c in expect_clips if c not in clip_names]
        check("expected clips present", not missing, f"missing {missing}")
    for anim in gltf.animations:
        for ch in anim.channels:
            check(f"clip {anim.name}: channel targets a real node", 0 <= ch.target.node < len(gltf.nodes))
            check(f"clip {anim.name}: channel path valid", ch.target.path in ("rotation", "translation", "scale"))
            sampler = anim.samplers[ch.sampler]
            times = read_accessor(gltf, blob, sampler.input)
            values = read_accessor(gltf, blob, sampler.output)
            check(f"clip {anim.name}/{gltf.nodes[ch.target.node].name}: times monotonic", bool(np.all(np.diff(times) >= 0)))
            check(f"clip {anim.name}/{gltf.nodes[ch.target.node].name}: values finite", bool(np.isfinite(values).all()))
            check(f"clip {anim.name}/{gltf.nodes[ch.target.node].name}: key counts match", len(times) == len(values))
            if ch.target.path == "rotation":
                lens = np.linalg.norm(values, axis=1)
                check(f"clip {anim.name}/{gltf.nodes[ch.target.node].name}: unit quaternions", bool(np.all(np.abs(lens - 1) < 1e-3)))
    return failures


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    failures = validate(argv[0])
    print(f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
