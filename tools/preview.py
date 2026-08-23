"""Headless preview renderer.

A small z-buffered software rasteriser so the build can be looked at without a
GPU, a browser, or a display. It exists to make geometry regressions visible in
CI-style runs: a mesh that validates structurally can still be inside out, and
no assertion catches that as fast as a picture.

    python3 tools/preview.py out/avatar_lod1.glb --view face --out out/face.png
    python3 tools/preview.py out/avatar_lod1.glb --morph aa=1.0 --view face
"""

from __future__ import annotations

import argparse
import io
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pygltflib import GLTF2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from avatarforge.skeleton import PREFIX  # noqa: E402

_COMPONENT = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

def node_parents(gltf: GLTF2) -> dict[int, int]:
    parent: dict[int, int] = {}
    for i, node in enumerate(gltf.nodes):
        for child in (node.children or []):
            parent[child] = i
    return parent


def local_matrix(node) -> np.ndarray:
    if getattr(node, "matrix", None):
        return np.array(node.matrix, dtype=np.float64).reshape(4, 4).T
    m = np.eye(4)
    if node.rotation:
        x, y, z, w = node.rotation
        m[:3, :3] = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
    if node.scale:
        m[:3, :3] = m[:3, :3] * np.array(node.scale, dtype=np.float64)
    if node.translation:
        m[:3, 3] = np.array(node.translation, dtype=np.float64)
    return m


def world_matrices(gltf: GLTF2) -> list[np.ndarray]:
    parent = node_parents(gltf)
    cache: dict[int, np.ndarray] = {}

    def world(i: int) -> np.ndarray:
        if i in cache:
            return cache[i]
        m = local_matrix(gltf.nodes[i])
        p = parent.get(i)
        if p is not None:
            m = world(p) @ m
        cache[i] = m
        return m

    return [world(i) for i in range(len(gltf.nodes))]


def apply_poses(gltf: GLTF2, poses: dict[str, float], axis: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> None:
    """Rotate named joints away from T-pose rest, for manual pose testing.

    Mutates node.rotation in place, which local_matrix()/world_matrices()
    already read — nothing about the transform pipeline itself changes.
    Rotation is a single scalar in degrees about `axis` (default local Z,
    matching this rig's straight-out-along-X rest arms). Not an animation or
    IK system: exactly the manually-supplied rotation the caller asks for.
    """
    if not poses:
        return
    by_name = {(node.name or ""): i for i, node in enumerate(gltf.nodes)}
    ax = np.array(axis, dtype=np.float64)
    ax /= np.linalg.norm(ax)
    for joint_name, degrees in poses.items():
        idx = by_name.get(PREFIX + joint_name, by_name.get(joint_name))
        if idx is None:
            raise ValueError(f"unknown joint {joint_name!r} for --pose")
        angle = np.radians(degrees)
        s = np.sin(angle / 2)
        gltf.nodes[idx].rotation = [float(ax[0] * s), float(ax[1] * s), float(ax[2] * s), float(np.cos(angle / 2))]


def skin_matrices(gltf: GLTF2, blob: bytes, skin_index: int) -> np.ndarray:
    """joint slot -> world(joint) @ inverseBindMatrix.

    For a correct bind pose every one of these is the identity, so applying
    them to the rest-pose vertices is a no-op. That is exactly why rendering
    *through* the skin is worth the cost: a renderer that skips it cannot
    distinguish a valid rig from a broken one.
    """
    skin = gltf.skins[skin_index]
    ibm = read_accessor(gltf, blob, skin.inverseBindMatrices).astype(np.float64)
    ibm = ibm.reshape(-1, 4, 4).transpose(0, 2, 1)  # stored column-major
    worlds = world_matrices(gltf)
    return np.stack([worlds[node] @ ibm[slot] for slot, node in enumerate(skin.joints)])


def apply_skin(
    positions: np.ndarray,
    normals: np.ndarray,
    joints: np.ndarray,
    weights: np.ndarray,
    matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear blend skinning, the same maths the engine runs."""
    out_p = np.zeros_like(positions)
    out_n = np.zeros_like(normals)
    homo = np.concatenate([positions, np.ones((len(positions), 1))], axis=1)

    for slot in range(joints.shape[1]):
        w = weights[:, slot]
        if not np.any(w):
            continue
        m = matrices[joints[:, slot]]
        out_p += w[:, None] * np.einsum("vij,vj->vi", m[:, :3, :], homo)
        out_n += w[:, None] * np.einsum("vij,vj->vi", m[:, :3, :3], normals)

    lengths = np.linalg.norm(out_n, axis=1, keepdims=True)
    lengths[lengths < 1e-9] = 1.0
    return out_p, out_n / lengths


VIEWS = {
    # (eye, target, vertical fov in degrees)
    "face": ((0.28, 1.70, 0.62), (0.0, 1.64, 0.02), 26.0),
    "front": ((0.0, 1.05, 3.1), (0.0, 0.95, 0.0), 34.0),
    "body": ((1.5, 1.25, 2.4), (0.0, 0.95, 0.0), 34.0),
    "profile": ((1.05, 1.68, 0.20), (0.0, 1.64, 0.0), 26.0),
    # Straight-on framing, used by the identity check so a face detector has
    # something it can actually find.
    "portrait": ((0.0, 1.655, 0.62), (0.0, 1.645, 0.0), 24.0),
}


def read_accessor(gltf: GLTF2, blob: bytes, index: int) -> np.ndarray:
    acc = gltf.accessors[index]
    view = gltf.bufferViews[acc.bufferView]
    offset = (view.byteOffset or 0) + (acc.byteOffset or 0)
    n = _COUNT[acc.type]
    fmt = _COMPONENT[acc.componentType]
    raw = blob[offset : offset + acc.count * n * struct.calcsize(fmt)]
    arr = np.frombuffer(raw, dtype=np.dtype(fmt))
    return arr.reshape(acc.count, n) if n > 1 else arr


def load_texture(gltf: GLTF2, blob: bytes, material_index: int) -> np.ndarray | None:
    mat = gltf.materials[material_index]
    info = mat.pbrMetallicRoughness.baseColorTexture if mat.pbrMetallicRoughness else None
    if info is None:
        return None
    image = gltf.images[gltf.textures[info.index].source]
    view = gltf.bufferViews[image.bufferView]
    raw = blob[view.byteOffset or 0 : (view.byteOffset or 0) + view.byteLength]
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    # Rows are the camera basis; camera looks down -Z in view space.
    return np.stack([right, up, -forward])


def rasterise(
    tris: np.ndarray,
    uvs: np.ndarray,
    normals: np.ndarray,
    texture: np.ndarray | None,
    base_colour: np.ndarray,
    colour_buf: np.ndarray,
    depth_buf: np.ndarray,
    light_dirs: list[tuple[np.ndarray, float]],
    alpha: float = 1.0,
) -> None:
    height, width = depth_buf.shape

    for tri, uv, nrm in zip(tris, uvs, normals):
        z = tri[:, 2]
        if np.any(z >= -1e-4):  # behind or on the camera plane
            continue

        xs, ys = tri[:, 0], tri[:, 1]
        x0 = max(int(np.floor(xs.min())), 0)
        x1 = min(int(np.ceil(xs.max())) + 1, width)
        y0 = max(int(np.floor(ys.min())), 0)
        y1 = min(int(np.ceil(ys.max())) + 1, height)
        if x0 >= x1 or y0 >= y1:
            continue

        area = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0])
        if abs(area) < 1e-9:
            continue

        px = np.arange(x0, x1) + 0.5
        py = np.arange(y0, y1) + 0.5
        gx, gy = np.meshgrid(px, py)

        w0 = ((xs[1] - xs[0]) * (gy - ys[0]) - (gx - xs[0]) * (ys[1] - ys[0])) / area
        w1 = ((gx - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (gy - ys[0])) / area
        b1, b2 = w1, w0
        b0 = 1.0 - b1 - b2
        inside = (b0 >= 0) & (b1 >= 0) & (b2 >= 0)
        if not inside.any():
            continue

        # Perspective-correct interpolation via reciprocal depth.
        inv_z = b0 / z[0] + b1 / z[1] + b2 / z[2]
        depth = 1.0 / inv_z
        window = depth_buf[y0:y1, x0:x1]
        visible = inside & (depth > window)  # larger (less negative) is nearer
        if not visible.any():
            continue

        n = (b0[..., None] * nrm[0] + b1[..., None] * nrm[1] + b2[..., None] * nrm[2])
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)

        shade = np.zeros(n.shape[:2], dtype=np.float32)
        for direction, intensity in light_dirs:
            shade += intensity * np.clip(n @ direction, 0.0, None)

        if texture is not None:
            u = (b0 / z[0] * uv[0, 0] + b1 / z[1] * uv[1, 0] + b2 / z[2] * uv[2, 0]) * depth
            v = (b0 / z[0] * uv[0, 1] + b1 / z[1] * uv[1, 1] + b2 / z[2] * uv[2, 1]) * depth
            th, tw = texture.shape[:2]
            # glTF UV origin is the upper-left corner: v grows downward, so the
            # image row is v*height with no flip. Getting this wrong cancels an
            # equal and opposite error on import and hides both.
            tx = np.clip((u % 1.0) * (tw - 1), 0, tw - 1).astype(np.int32)
            ty = np.clip((v % 1.0) * (th - 1), 0, th - 1).astype(np.int32)
            albedo = texture[ty, tx]
        else:
            albedo = np.broadcast_to(base_colour, (*n.shape[:2], 3))

        rgb = albedo * shade[..., None]
        target = colour_buf[y0:y1, x0:x1]
        if alpha >= 1.0:
            target[visible] = rgb[visible]
            window[visible] = depth[visible]
        else:
            # Transparent surfaces composite over what is already there and do
            # not claim the depth slot — otherwise the cornea would hide the
            # iris behind an opaque white cap.
            target[visible] = rgb[visible] * alpha + target[visible] * (1.0 - alpha)


def render(
    path: Path,
    view: str = "face",
    size: int = 720,
    morphs: dict[str, float] | None = None,
    background: tuple[float, float, float] = (0.09, 0.10, 0.12),
    skinned: bool = True,
    only: set[str] | None = None,
    poses: dict[str, float] | None = None,
) -> Image.Image:
    gltf = GLTF2().load(str(path))
    blob = gltf.binary_blob()
    morphs = morphs or {}
    apply_poses(gltf, poses or {})

    eye_pos, target_pos, fov = VIEWS[view]
    eye = np.array(eye_pos, dtype=np.float64)
    target = np.array(target_pos, dtype=np.float64)
    basis = look_at(eye, target)
    focal = (size / 2) / np.tan(np.radians(fov) / 2)

    colour_buf = np.zeros((size, size, 3), dtype=np.float32)
    colour_buf[:] = background
    depth_buf = np.full((size, size), -np.inf, dtype=np.float64)

    # Key, fill and rim, in world space.
    lights = [
        (np.array([0.45, 0.55, 0.70]), 0.85),
        (np.array([-0.65, 0.10, 0.45]), 0.28),
        (np.array([0.0, 0.25, -1.0]), 0.22),
    ]
    lights = [(d / np.linalg.norm(d), i) for d, i in lights]

    # Opaque first so the depth buffer is complete, then transparent surfaces
    # composite over it. Without this ordering the cornea wins the depth test
    # and paints an opaque white cap over the iris.
    def is_blended(mesh) -> bool:
        return gltf.materials[mesh.primitives[0].material].alphaMode == "BLEND"

    matrices = skin_matrices(gltf, blob, 0) if gltf.skins else None

    for mesh in sorted(gltf.meshes, key=is_blended):
        if only is not None and mesh.name not in only:
            continue
        prim = mesh.primitives[0]
        pos = read_accessor(gltf, blob, prim.attributes.POSITION).astype(np.float64).copy()

        names = (mesh.extras or {}).get("targetNames", [])
        nrm = read_accessor(gltf, blob, prim.attributes.NORMAL).astype(np.float64)
        for name, weight in morphs.items():
            if name in names and prim.targets:
                target = prim.targets[names.index(name)]
                pos += read_accessor(gltf, blob, target["POSITION"]).astype(np.float64) * weight
                if "NORMAL" in target:
                    nrm += read_accessor(gltf, blob, target["NORMAL"]).astype(np.float64) * weight
        lengths = np.linalg.norm(nrm, axis=1, keepdims=True)
        lengths[lengths < 1e-9] = 1.0
        nrm = nrm / lengths

        # Deform through the skin exactly as an engine would. Skipping this is
        # what let a file with 73 broken bind matrices render as a clean image.
        if skinned and matrices is not None and prim.attributes.JOINTS_0 is not None:
            j = read_accessor(gltf, blob, prim.attributes.JOINTS_0).astype(np.int64)
            w = read_accessor(gltf, blob, prim.attributes.WEIGHTS_0).astype(np.float64)
            pos, nrm = apply_skin(pos, nrm, j, w, matrices)

        uv = read_accessor(gltf, blob, prim.attributes.TEXCOORD_0).astype(np.float64)
        idx = read_accessor(gltf, blob, prim.indices).astype(np.int64).reshape(-1, 3)

        cam = (pos - eye) @ basis.T
        z = np.minimum(cam[:, 2], -1e-6)
        screen = np.empty((len(pos), 3))
        screen[:, 0] = size / 2 + focal * cam[:, 0] / -z
        screen[:, 1] = size / 2 - focal * cam[:, 1] / -z
        screen[:, 2] = cam[:, 2]

        material = gltf.materials[prim.material]
        texture = load_texture(gltf, blob, prim.material)
        factor = material.pbrMetallicRoughness.baseColorFactor or [1, 1, 1, 1]
        base = np.array(factor[:3], dtype=np.float32)
        alpha = float(factor[3]) if material.alphaMode == "BLEND" else 1.0

        rasterise(
            screen[idx], uv[idx], nrm[idx], texture, base, colour_buf, depth_buf, lights, alpha
        )

    srgb = np.clip(colour_buf, 0.0, 1.0) ** (1 / 2.2)
    return Image.fromarray((srgb * 255).astype(np.uint8), "RGB")


def parse_morphs(values: list[str]) -> dict[str, float]:
    out = {}
    for item in values:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {item!r}")
        name, weight = item.split("=", 1)
        out[name] = float(weight)
    return out


def parse_poses(values: list[str]) -> dict[str, float]:
    out = {}
    for item in values:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"expected JOINT=DEGREES, got {item!r}")
        name, degrees = item.split("=", 1)
        out[name] = float(degrees)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("glb", type=Path)
    p.add_argument("--view", default="face", choices=sorted(VIEWS))
    p.add_argument("--size", type=int, default=720)
    p.add_argument("--morph", action="append", default=[], help="NAME=WEIGHT, repeatable")
    p.add_argument("--pose", action="append", default=[],
                   help="JOINT=DEGREES, repeatable, rotates about local Z away from T-pose rest")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--no-skin", action="store_true",
                   help="render rest-pose vertices without deforming through the skin")
    args = p.parse_args(argv)

    image = render(args.glb, args.view, args.size, parse_morphs(args.morph),
                   skinned=not args.no_skin, poses=parse_poses(args.pose))
    out = args.out or args.glb.with_suffix(f".{args.view}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
