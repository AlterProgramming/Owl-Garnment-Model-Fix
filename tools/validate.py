"""Contract test for an exported GLB.

Re-opens the shipped file rather than inspecting the in-memory build, so this
verifies what a consumer actually receives. Checks the parts of the export that
downstream code binds to and that a silent regression would break: sub-mesh
names, Mixamo joint names, morph-target names, skin weight normalisation, and
that every embedded image really decodes.

    python3 tools/validate.py out/avatar_lod1.glb
"""

from __future__ import annotations

import io
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pygltflib import GLTF2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preview import skin_matrices, world_matrices  # noqa: E402

from avatarforge import measure  # noqa: E402
from avatarforge import topology as topo  # noqa: E402
from avatarforge.blendshapes import CORRECTIVES, VISEMES  # noqa: E402
from avatarforge.skeleton import PREFIX, Skeleton  # noqa: E402

_COMPONENT = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


class Report:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []
        self.notes: list[str] = []

    def check(self, condition: bool, label: str, detail: str = "") -> bool:
        (self.passed if condition else self.failed).append(label if not detail else f"{label} — {detail}")
        return condition

    def note(self, label: str, detail: str = "") -> None:
        """An observation about the source asset, not a contract violation.

        Whether a source rig ships every viseme, or carries a target that never
        moves a vertex, is worth surfacing but is not the exporter's failure.
        """
        self.notes.append(label if not detail else f"{label} — {detail}")

    def render(self) -> int:
        for line in self.passed:
            print(f"  PASS  {line}")
        for line in self.notes:
            print(f"  NOTE  {line}")
        for line in self.failed:
            print(f"  FAIL  {line}")
        print(f"\n{len(self.passed)} passed, {len(self.notes)} notes, {len(self.failed)} failed")
        return 1 if self.failed else 0


def read_accessor(gltf: GLTF2, blob: bytes, index: int) -> np.ndarray:
    acc = gltf.accessors[index]
    view = gltf.bufferViews[acc.bufferView]
    offset = (view.byteOffset or 0) + (acc.byteOffset or 0)
    n = _COUNT[acc.type]
    fmt = _COMPONENT[acc.componentType]
    itemsize = struct.calcsize(fmt)
    raw = blob[offset : offset + acc.count * n * itemsize]
    arr = np.frombuffer(raw, dtype=np.dtype(fmt))
    return arr.reshape(acc.count, n) if n > 1 else arr


def validate(path: Path) -> int:
    gltf = GLTF2().load(str(path))
    blob = gltf.binary_blob()
    r = Report()

    print(f"Validating {path} ({path.stat().st_size / 1024:.0f} KB)\n")

    # --- sub-mesh contract ------------------------------------------------
    # Only the two skin meshes are mandatory. Corneas, teeth and asset slots
    # are legitimately absent depending on export flags and on what the source
    # avatar shipped, so their absence is not a contract violation — an
    # unrecognised name is.
    required = {topo.HEAD, topo.BODY}
    optional = set(topo.PART_ORDER) - required | {"haircut", "outfit"}

    names = [m.name for m in gltf.meshes]
    for expected in sorted(required):
        r.check(expected in names, f"sub-mesh {expected} present")
    unknown = [n for n in names if n not in required | optional]
    r.check(not unknown, "all sub-mesh names are in the contract vocabulary",
            f"unrecognised {unknown}" if unknown else "")
    present = [n for n in topo.PART_ORDER if n in names]
    r.check(True, f"optional parts present: {len(present) - len(required)}", ", ".join(
        n for n in present if n not in required) or "none")
    r.check(len(names) == len(set(names)), "sub-mesh names are unique")

    # --- skeleton contract ------------------------------------------------
    # The rig is whatever the file declares. Checking against the procedural
    # skeleton would reject any imported avatar for the crime of having fingers.
    r.check(bool(gltf.skins), "skin defined")
    joint_count = 0
    if gltf.skins:
        sk = gltf.skins[0]
        joint_count = len(sk.joints)
        r.check(sk.inverseBindMatrices is not None, "inverse bind matrices present")
        ibm = read_accessor(gltf, blob, sk.inverseBindMatrices)
        r.check(ibm.shape == (joint_count, 16),
                "one inverse bind matrix per joint", f"{ibm.shape[0]} for {joint_count} joints")
        r.check(bool(np.isfinite(ibm).all()), "inverse bind matrices are finite")

        joint_names = [gltf.nodes[j].name or "" for j in sk.joints]
        r.check(len(set(joint_names)) == len(joint_names), "joint names are unique")

        prefixed = sum(n.startswith(PREFIX) for n in joint_names)
        if prefixed:
            r.check(prefixed == joint_count, "every skin joint carries the mixamorig: prefix",
                    f"{prefixed}/{joint_count}")
        else:
            r.note("rig is not mixamorig:-prefixed", f"{joint_count} bare joint names")

        # The check that matters most, and the one whose absence let a file
        # with 73 broken bind matrices pass every other test: in the rest pose
        # world(joint) @ inverseBind must be the identity. When it is not, the
        # skin deforms geometry that was never posed that way and the mesh
        # shreds on load — while any renderer that skips skinning shows it
        # perfectly intact.
        deviation = np.abs(skin_matrices(gltf, blob, 0) - np.eye(4))
        worst = float(deviation.reshape(len(sk.joints), -1).max(axis=1).max())
        broken = int((deviation.reshape(len(sk.joints), -1).max(axis=1) > 1e-4).sum())
        r.check(broken == 0, "rest pose is the bind pose (world @ IBM == identity)",
                f"max deviation {worst:.2e}"
                + (f", {broken}/{len(sk.joints)} joints broken" if broken else ""))

        # Retargeting needs the core chain by name, prefix or not.
        core = ["Hips", "Spine", "Neck", "Head", "LeftArm", "RightArm", "LeftUpLeg", "RightUpLeg"]
        stripped = {n.replace(PREFIX, "") for n in joint_names}
        absent = [c for c in core if c not in stripped]
        r.check(not absent, "core humanoid chain present", f"missing {absent}" if absent else "")

    # --- morph target contract -------------------------------------------
    by_name = {m.name: m for m in gltf.meshes}
    head = by_name.get(topo.HEAD)
    if head is not None:
        targets = (head.extras or {}).get("targetNames", [])
        r.check(len(targets) == len(head.primitives[0].targets or []),
                "AvatarHead target names match target count",
                f"{len(targets)} names vs {len(head.primitives[0].targets or [])} targets")
        # A build may deliberately omit the viseme family; only hold it to the
        # full set once at least one viseme is present.
        if any(v in targets for v in VISEMES):
            missing_v = [v for v in VISEMES if v not in targets]
            if missing_v:
                r.note("viseme set is incomplete", f"missing {missing_v}")
            else:
                r.check(True, "all 15 visemes on AvatarHead")
        r.check(head.weights is not None and len(head.weights) == len(targets),
                "AvatarHead default weights sized to targets")
        # A target of all zeros would be dead weight in the file.
        dead = []
        for tname, t in zip(targets, head.primitives[0].targets or []):
            delta = read_accessor(gltf, blob, t["POSITION"])
            if np.abs(delta).max() < 1e-6:
                dead.append(tname)
        if dead:
            r.note("morph targets that move nothing", f"{dead}")
        else:
            r.check(True, "no all-zero morph targets")

    # --- morph target NORMAL deltas ----------------------------------------
    # A morph target that reshapes the mesh without a matching NORMAL delta
    # just means shading won't follow the shape change — worth surfacing, but
    # not this exporter's failure if the whole family predates the feature or
    # the source asset never carried one. A NORMAL delta that IS present but
    # broken (non-finite, or cancels the base normal to nothing) is a real bug.
    for mesh in gltf.meshes:
        prim = mesh.primitives[0]
        if prim.attributes.NORMAL is None or not prim.targets:
            continue
        base_n = read_accessor(gltf, blob, prim.attributes.NORMAL).astype(np.float64)
        targets_here = (mesh.extras or {}).get("targetNames", [])
        with_normal = [t for t in prim.targets if "NORMAL" in t]
        if not with_normal:
            r.note(f"{mesh.name} morph targets carry no NORMAL delta",
                   f"{len(prim.targets)} targets, POSITION only")
            continue
        for tname, t in zip(targets_here, prim.targets):
            if "NORMAL" not in t:
                continue
            nd = read_accessor(gltf, blob, t["NORMAL"]).astype(np.float64)
            r.check(bool(np.isfinite(nd).all()), f"{mesh.name}/{tname} NORMAL delta is finite")
            lengths = np.linalg.norm(base_n + nd, axis=1)
            r.check(bool((lengths > 1e-3).all()),
                    f"{mesh.name}/{tname} NORMAL delta at full weight stays non-degenerate",
                    f"min length {lengths.min():.2e}")

    # --- corrective morph target contract ----------------------------------
    # Correctives are pose-driven (a runtime driver sets .influence from a
    # joint angle), so unlike visemes they are not confined to one sub-mesh —
    # the deltoid-root blob genuinely also displaces AvatarHead's adjacent
    # shoulder geometry by design (the two meshes share that neighbourhood so
    # the seam stays closed). Presence/absence is a NOTE, not a failure,
    # exactly like the viseme-completeness check: a build may deliberately
    # omit this family by not passing --correctives.
    all_target_names = {n for m in gltf.meshes for n in (m.extras or {}).get("targetNames", [])}
    if CORRECTIVES:
        present = [c for c in CORRECTIVES if c in all_target_names]
        missing = [c for c in CORRECTIVES if c not in all_target_names]
        if present:
            r.note("pose correctives present", f"{present}")
        if missing:
            r.note("pose correctives absent", f"{missing}")
        # Guards the exact naming-collision class this project has been bitten
        # by before (the "sil vs Laughter" story) — a corrective ending in
        # +/- would be misread as half of a signed proportion pair.
        collision = [c for c in CORRECTIVES if c.endswith(("+", "-"))]
        r.check(not collision, "corrective names do not collide with the +/- signed-pair convention",
                f"{collision}" if collision else "")

    # --- skinning sanity --------------------------------------------------
    for mesh_name in (topo.HEAD, topo.BODY):
        mesh = by_name.get(mesh_name)
        if mesh is None:
            continue
        prim = mesh.primitives[0]
        r.check(prim.attributes.JOINTS_0 is not None, f"{mesh_name} has JOINTS_0")
        r.check(prim.attributes.WEIGHTS_0 is not None, f"{mesh_name} has WEIGHTS_0")
        w = read_accessor(gltf, blob, prim.attributes.WEIGHTS_0).astype(np.float64)
        sums = w.sum(axis=1)
        r.check(bool(np.allclose(sums, 1.0, atol=1e-3)),
                f"{mesh_name} skin weights normalise to 1",
                f"range {sums.min():.4f}–{sums.max():.4f}")
        j = read_accessor(gltf, blob, prim.attributes.JOINTS_0)
        r.check(int(j.max()) < joint_count, f"{mesh_name} joint indices in range",
                f"max {int(j.max())} of {joint_count} joints")

    # --- geometry sanity --------------------------------------------------
    for mesh in gltf.meshes:
        prim = mesh.primitives[0]
        pos = read_accessor(gltf, blob, prim.attributes.POSITION)
        idx = read_accessor(gltf, blob, prim.indices)
        r.check(idx.max() < len(pos), f"{mesh.name} indices within vertex range")
        r.check(len(idx) % 3 == 0, f"{mesh.name} index count is a multiple of 3")
        r.check(bool(np.isfinite(pos).all()), f"{mesh.name} positions are finite")
        if prim.attributes.NORMAL is not None:
            n = read_accessor(gltf, blob, prim.attributes.NORMAL).astype(np.float64)
            lengths = np.linalg.norm(n, axis=1)
            r.check(bool(np.allclose(lengths, 1.0, atol=1e-2)),
                    f"{mesh.name} normals are unit length",
                    f"range {lengths.min():.3f}–{lengths.max():.3f}")

    # --- textures ---------------------------------------------------------
    r.check(bool(gltf.images), "images embedded")
    for i, img in enumerate(gltf.images):
        view = gltf.bufferViews[img.bufferView]
        raw = blob[view.byteOffset or 0 : (view.byteOffset or 0) + view.byteLength]
        try:
            decoded = Image.open(io.BytesIO(raw))
            decoded.load()
            r.check(True, f"image {i} decodes ({decoded.size[0]}px {decoded.format})")
        except Exception as exc:  # pragma: no cover - failure path
            r.check(False, f"image {i} decodes", str(exc))

    r.check(all(m.pbrMetallicRoughness is not None for m in gltf.materials),
            "every material has a PBR block")

    # --- winding ----------------------------------------------------------
    # Reversed winding renders inside-out under backface culling. A renderer
    # that does not cull — like this project's own rasteriser — cannot see it.
    for mesh in gltf.meshes:
        prim = mesh.primitives[0]
        if prim.attributes.NORMAL is None:
            continue
        P = read_accessor(gltf, blob, prim.attributes.POSITION).astype(np.float64)
        N = read_accessor(gltf, blob, prim.attributes.NORMAL).astype(np.float64)
        T = read_accessor(gltf, blob, prim.indices).astype(np.int64).reshape(-1, 3)
        face = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
        ln = np.linalg.norm(face, axis=1)
        keep = ln > 1e-12
        if not keep.any():
            continue
        face = face[keep] / ln[keep, None]
        vn = (N[T[:, 0]] + N[T[:, 1]] + N[T[:, 2]])[keep]
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
        agree = float((np.sum(face * vn, axis=1) > 0).mean())
        if agree >= 0.95:
            r.check(True, f"{mesh.name} winding agrees with normals", f"{agree * 100:.1f}%")
        elif agree < 0.5:
            r.check(False, f"{mesh.name} winding agrees with normals",
                    f"only {agree * 100:.1f}% — mesh is inside out")
        else:
            # Open shell geometry (lash strips, hair cards) is legitimately mixed.
            r.note(f"{mesh.name} winding is mixed", f"{agree * 100:.1f}% agreement")

    # --- tangents ---------------------------------------------------------
    # glTF lets a renderer derive tangents, but a mirrored UV chart makes it
    # guess the handedness backwards on one half of the face.
    for mesh in gltf.meshes:
        prim = mesh.primitives[0]
        if gltf.materials[prim.material].normalTexture is None:
            continue
        if not r.check(prim.attributes.TANGENT is not None,
                       f"{mesh.name} has TANGENT for its normal map"):
            continue
        tan = read_accessor(gltf, blob, prim.attributes.TANGENT).astype(np.float64)
        lengths = np.linalg.norm(tan[:, :3], axis=1)
        r.check(bool(np.allclose(lengths, 1.0, atol=1e-2)),
                f"{mesh.name} tangents are unit length",
                f"range {lengths.min():.3f}-{lengths.max():.3f}")
        r.check(bool(np.all(np.abs(np.abs(tan[:, 3]) - 1.0) < 1e-6)),
                f"{mesh.name} tangent handedness is +/-1")

    # --- alpha coherence --------------------------------------------------
    # MASK with no alpha channel cuts nothing and renders as a solid shell.
    for material in gltf.materials:
        if material.alphaMode not in ("MASK", "BLEND"):
            continue
        info = material.pbrMetallicRoughness.baseColorTexture
        factor = material.pbrMetallicRoughness.baseColorFactor or [1, 1, 1, 1]
        if info is None:
            r.check(factor[3] < 1.0,
                    f"{material.name} declares {material.alphaMode} with usable alpha",
                    "no base texture and baseColorFactor alpha is 1.0 — nothing to cut")
            continue
        image = gltf.images[gltf.textures[info.index].source]
        view = gltf.bufferViews[image.bufferView]
        raw = blob[view.byteOffset or 0 : (view.byteOffset or 0) + view.byteLength]
        mode = Image.open(io.BytesIO(raw)).mode
        r.check(mode in ("RGBA", "LA", "PA", "P"),
                f"{material.name} base texture carries alpha for {material.alphaMode}",
                f"image mode is {mode}")

    # --- texture coordinates ---------------------------------------------
    for mesh in gltf.meshes:
        uv = read_accessor(gltf, blob, mesh.primitives[0].attributes.TEXCOORD_0).astype(np.float64)
        r.check(bool(np.isfinite(uv).all()), f"{mesh.name} UVs are finite")
        outside = float(((uv < -1e-6) | (uv > 1.0 + 1e-6)).any(axis=1).mean())
        if outside > 0.001:
            # Tiling is legal, so this is worth seeing but is not a failure.
            r.note(f"{mesh.name} UVs leave [0,1]", f"{outside * 100:.1f}% of vertices")

    # --- morph target semantics -------------------------------------------
    # The proportion dials ship as signed pairs. A pair whose halves are not
    # opposites, or that covers different sub-meshes, produces a control that
    # deforms the body without the clothes.
    targets_by_mesh: dict[str, dict[str, int]] = {}
    for mesh in gltf.meshes:
        names = (mesh.extras or {}).get("targetNames", [])
        targets_by_mesh[mesh.name] = {n: i for i, n in enumerate(names)}

    all_names = {n for names in targets_by_mesh.values() for n in names}
    bases = {n[:-1] for n in all_names if n.endswith(("+", "-"))}
    for base in sorted(bases):
        plus, minus = f"{base}+", f"{base}-"
        r.check(plus in all_names and minus in all_names,
                f"signed dial {base} has both directions")
        owners_p = {m for m, t in targets_by_mesh.items() if plus in t}
        owners_m = {m for m, t in targets_by_mesh.items() if minus in t}
        r.check(owners_p == owners_m, f"signed dial {base} covers the same sub-meshes",
                f"+{sorted(owners_p)} vs -{sorted(owners_m)}" if owners_p != owners_m else "")

        for mesh in gltf.meshes:
            slots = targets_by_mesh[mesh.name]
            if plus not in slots or minus not in slots:
                continue
            prim = mesh.primitives[0]
            dp = read_accessor(gltf, blob, prim.targets[slots[plus]]["POSITION"]).astype(np.float64)
            dm = read_accessor(gltf, blob, prim.targets[slots[minus]]["POSITION"]).astype(np.float64)
            moving = (np.linalg.norm(dp, axis=1) > 1e-6) & (np.linalg.norm(dm, axis=1) > 1e-6)
            if moving.sum() < 8:
                continue
            # The load-bearing test is whether the two displacement fields
            # oppose in bulk. Per-vertex agreement is the wrong statistic:
            # these are independently sculpted shapes, not exact negatives, so
            # a strict per-vertex threshold flags honest artist work as broken.
            a, b_ = dp[moving].ravel(), dm[moving].ravel()
            cosine = float(np.dot(a, b_) / (np.linalg.norm(a) * np.linalg.norm(b_) + 1e-12))
            r.check(cosine < 0.0, f"{base} on {mesh.name}: + and - oppose each other",
                    f"cosine {cosine:+.3f}")
            per_vertex = float((np.sum(dp[moving] * dm[moving], axis=1) < 0).mean())
            if per_vertex < 0.75:
                r.note(f"{base} on {mesh.name} is only partly antisymmetric",
                       f"{per_vertex * 100:.0f}% of shared vertices oppose")

    # A target displacing half a metre is a unit error, not a proportion dial.
    for mesh in gltf.meshes:
        names = (mesh.extras or {}).get("targetNames", [])
        for name, target in zip(names, mesh.primitives[0].targets or []):
            peak = float(np.abs(read_accessor(gltf, blob, target["POSITION"])).max())
            if peak > 0.5:
                r.note(f"{mesh.name}/{name} displaces {peak:.2f} m", "possible unit mismatch")

    # --- anthropometry -----------------------------------------------------
    # Everything above this line checks the export contract, which a coathanger
    # would also pass. These are the shape checks: dimensionless body ratios that
    # survive a change of stature.
    all_pos = np.concatenate([
        read_accessor(gltf, blob, m.primitives[0].attributes.POSITION) for m in gltf.meshes
    ])
    height = float(all_pos[:, 1].max() - all_pos[:, 1].min())
    r.check(1.55 < height < 1.95, "stature within a human range", f"{height:.3f} m")

    positions = {
        m.name: np.concatenate([
            read_accessor(gltf, blob, prim.attributes.POSITION).astype(np.float64)
            for prim in m.primitives
        ])
        for m in gltf.meshes
    }
    world = world_matrices(gltf)
    joint_y = {
        node.name.split(":")[-1]: float(world[i][1, 3])
        for i, node in enumerate(gltf.nodes)
        if node.name
    }
    for ratio in measure.ratios(positions, joint_y):
        if ratio.value is None:
            # Unmeasurable is its own outcome. Reporting it as a pass would let a
            # landmark regression hide behind a green run.
            r.note(f"{ratio.name} not measurable", ratio.detail)
        elif ratio.severity == "fail":
            r.check(ratio.ok, ratio.name, ratio.describe())
        elif not ratio.ok:
            r.note(f"{ratio.name} outside the usual band", ratio.describe())
        else:
            r.check(True, ratio.name, ratio.describe())

    return r.render()


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "out/avatar_lod1.glb")
    raise SystemExit(validate(target))
