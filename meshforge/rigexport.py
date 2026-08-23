"""General rigged-GLB writer: several primitives (textured and/or
vertex-colored, each with its own material) on ONE skin, an N-joint
hierarchy with measured pivots, and any number of animation clips with
rotation / translation / scale tracks.

Generalizes export.write_multi_skinned_glb (body + part joints, one clip,
rotation-only, vertex color only) to what a full-body character needs.
Conventions shared with meshforge.fk:

* every joint has identity rest rotation;
* a joint node's translation is pivot − parent pivot;
* inverseBindMatrices = translate(−pivot);
* JOINTS_0 indexes skin.joints, which is the order of `joints` as given.

Vertex colors are converted sRGB → linear on write (glTF COLOR_0 is
linear; writing sRGB values there is exactly what made the previous owl
look washed out). Textures are embedded as JPEG (opaque) or PNG.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pygltflib
from PIL import Image

from meshforge.fk import Joint, local_rest_translation, ordered_joints


@dataclass
class MaterialSpec:
    name: str
    base_color_factor: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    base_color_image: Image.Image | None = None
    image_format: str = "JPEG"           # "JPEG" or "PNG"
    jpeg_quality: int = 88
    metallic: float = 0.0
    roughness: float = 0.8
    alpha_mode: str = "OPAQUE"           # OPAQUE | BLEND | MASK
    double_sided: bool = False
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal_image: Image.Image | None = None   # always PNG — lossy JPEG corrupts normal directions
    normal_scale: float = 1.0


@dataclass
class PrimitiveSpec:
    name: str
    vertices: np.ndarray                 # (n, 3)
    faces: np.ndarray                    # (m, 3)
    normals: np.ndarray                  # (n, 3)
    material: MaterialSpec
    uvs: np.ndarray | None = None        # (n, 2)
    colors: np.ndarray | None = None     # (n, 3) or (n, 4), sRGB in [0, 1]
    joints: np.ndarray | None = None     # (n, 4) uint8 skin-joint indices
    weights: np.ndarray | None = None    # (n, 4) float


@dataclass
class Track:
    joint: str
    path: str                            # rotation | translation | scale
    times: np.ndarray                    # (k,)
    values: np.ndarray                   # (k, 4) quaternions xyzw for rotation, (k, 3) otherwise


@dataclass
class AnimationSpec:
    name: str
    tracks: list[Track] = field(default_factory=list)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


class _Buffer:
    def __init__(self):
        self.blob = bytearray()
        self.views: list[pygltflib.BufferView] = []
        self.accessors: list[pygltflib.Accessor] = []

    def _align(self, n: int = 4):
        while len(self.blob) % n:
            self.blob.append(0)

    def add_view(self, data: bytes, target: int | None = None) -> int:
        self._align()
        offset = len(self.blob)
        self.blob.extend(data)
        view = pygltflib.BufferView(buffer=0, byteOffset=offset, byteLength=len(data))
        if target is not None:
            view.target = target
        self.views.append(view)
        return len(self.views) - 1

    def add_accessor(self, array: np.ndarray, component_type: int, type_: str, normalized: bool = False,
                     minmax: bool = False, target: int | None = None) -> int:
        array = np.ascontiguousarray(array)
        view = self.add_view(array.tobytes(), target=target)
        acc = pygltflib.Accessor(bufferView=view, componentType=component_type, count=len(array), type=type_)
        if normalized:
            acc.normalized = True
        if minmax:
            acc.min = np.asarray(array.min(axis=0)).ravel().tolist()
            acc.max = np.asarray(array.max(axis=0)).ravel().tolist()
        self.accessors.append(acc)
        return len(self.accessors) - 1


ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963


def _pack_primitive(buf: _Buffer, prim: PrimitiveSpec, material_index: int) -> pygltflib.Primitive:
    n = len(prim.vertices)
    if prim.faces.size and prim.faces.max() >= n:
        raise ValueError(f"{prim.name}: face index out of range")
    if prim.normals.shape != (n, 3):
        raise ValueError(f"{prim.name}: normals must be ({n}, 3)")
    pos = buf.add_accessor(prim.vertices.astype(np.float32), pygltflib.FLOAT, pygltflib.VEC3, minmax=True, target=ARRAY_BUFFER)
    nrm = buf.add_accessor(prim.normals.astype(np.float32), pygltflib.FLOAT, pygltflib.VEC3, target=ARRAY_BUFFER)
    attrs = pygltflib.Attributes(POSITION=pos, NORMAL=nrm)
    if prim.uvs is not None:
        if prim.uvs.shape != (n, 2):
            raise ValueError(f"{prim.name}: uvs must be ({n}, 2)")
        attrs.TEXCOORD_0 = buf.add_accessor(prim.uvs.astype(np.float32), pygltflib.FLOAT, pygltflib.VEC2, target=ARRAY_BUFFER)
    if prim.colors is not None:
        cols = np.asarray(prim.colors, dtype=np.float64)
        if cols.shape[0] != n or cols.shape[1] not in (3, 4):
            raise ValueError(f"{prim.name}: colors must be ({n}, 3|4)")
        rgb = srgb_to_linear(cols[:, :3])
        if cols.shape[1] == 4:
            rgba = np.concatenate([rgb, cols[:, 3:4]], axis=1).astype(np.float32)
            attrs.COLOR_0 = buf.add_accessor(rgba, pygltflib.FLOAT, pygltflib.VEC4, target=ARRAY_BUFFER)
        else:
            attrs.COLOR_0 = buf.add_accessor(rgb.astype(np.float32), pygltflib.FLOAT, pygltflib.VEC3, target=ARRAY_BUFFER)
    if (prim.joints is None) != (prim.weights is None):
        raise ValueError(f"{prim.name}: joints and weights must be given together")
    if prim.joints is not None:
        if prim.joints.shape != (n, 4) or prim.weights.shape != (n, 4):
            raise ValueError(f"{prim.name}: joints/weights must be ({n}, 4)")
        w = np.asarray(prim.weights, dtype=np.float64)
        s = w.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        w = w / s
        attrs.JOINTS_0 = buf.add_accessor(prim.joints.astype(np.uint8), pygltflib.UNSIGNED_BYTE, pygltflib.VEC4, target=ARRAY_BUFFER)
        attrs.WEIGHTS_0 = buf.add_accessor(w.astype(np.float32), pygltflib.FLOAT, pygltflib.VEC4, target=ARRAY_BUFFER)
    if n > 65535:
        idx = buf.add_accessor(prim.faces.astype(np.uint32).ravel(), pygltflib.UNSIGNED_INT, pygltflib.SCALAR, target=ELEMENT_ARRAY_BUFFER)
    else:
        idx = buf.add_accessor(prim.faces.astype(np.uint16).ravel(), pygltflib.UNSIGNED_SHORT, pygltflib.SCALAR, target=ELEMENT_ARRAY_BUFFER)
    return pygltflib.Primitive(attributes=attrs, indices=idx, material=material_index, mode=4)


def _pack_material(gltf: pygltflib.GLTF2, buf: _Buffer, spec: MaterialSpec) -> int:
    pbr = pygltflib.PbrMetallicRoughness(
        baseColorFactor=list(spec.base_color_factor), metallicFactor=float(spec.metallic), roughnessFactor=float(spec.roughness),
    )
    if spec.base_color_image is not None:
        bio = io.BytesIO()
        img = spec.base_color_image
        if spec.image_format.upper() == "JPEG":
            img.convert("RGB").save(bio, format="JPEG", quality=int(spec.jpeg_quality), subsampling=0, optimize=True)
            mime = "image/jpeg"
        else:
            img.save(bio, format="PNG", optimize=True)
            mime = "image/png"
        view = buf.add_view(bio.getvalue())
        gltf.images.append(pygltflib.Image(bufferView=view, mimeType=mime, name=spec.name))
        gltf.samplers.append(pygltflib.Sampler(magFilter=9729, minFilter=9987, wrapS=33071, wrapT=33071))
        gltf.textures.append(pygltflib.Texture(sampler=len(gltf.samplers) - 1, source=len(gltf.images) - 1))
        pbr.baseColorTexture = pygltflib.TextureInfo(index=len(gltf.textures) - 1, texCoord=0)
    mat = pygltflib.Material(name=spec.name, pbrMetallicRoughness=pbr, alphaMode=spec.alpha_mode,
                             doubleSided=bool(spec.double_sided), emissiveFactor=list(spec.emissive))
    if spec.normal_image is not None:
        bio = io.BytesIO()
        spec.normal_image.save(bio, format="PNG", optimize=True)
        view = buf.add_view(bio.getvalue())
        gltf.images.append(pygltflib.Image(bufferView=view, mimeType="image/png", name=f"{spec.name}_normal"))
        gltf.samplers.append(pygltflib.Sampler(magFilter=9729, minFilter=9987, wrapS=33071, wrapT=33071))
        gltf.textures.append(pygltflib.Texture(sampler=len(gltf.samplers) - 1, source=len(gltf.images) - 1))
        mat.normalTexture = pygltflib.NormalMaterialTexture(
            index=len(gltf.textures) - 1, texCoord=0, scale=float(spec.normal_scale)
        )
    gltf.materials.append(mat)
    return len(gltf.materials) - 1


def write_rigged_glb(
    primitives: list[PrimitiveSpec],
    joints: list[Joint],
    animations: list[AnimationSpec],
    out_path: str,
    mesh_name: str = "owl_mesh",
    asset_extras: dict | None = None,
    node_extras: dict[str, dict] | None = None,
) -> None:
    if not primitives:
        raise ValueError("write_rigged_glb: at least one primitive is required")
    joint_names = [j.name for j in joints]
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("write_rigged_glb: joint names must be unique")
    ordered_joints(joints)  # validates parents / cycles
    by_name = {j.name: j for j in joints}
    j_index = {name: i for i, name in enumerate(joint_names)}
    skinned = any(p.joints is not None for p in primitives)
    for p in primitives:
        if p.joints is not None and p.joints.max() >= len(joints):
            raise ValueError(f"{p.name}: joint index {int(p.joints.max())} out of range for {len(joints)} joints")

    gltf = pygltflib.GLTF2()
    gltf.asset = pygltflib.Asset(generator="meshforge.rigexport", version="2.0")
    buf = _Buffer()

    material_cache: dict[int, int] = {}
    gltf_prims = []
    for prim in primitives:
        key = id(prim.material)
        if key not in material_cache:
            material_cache[key] = _pack_material(gltf, buf, prim.material)
        gltf_prims.append(_pack_primitive(buf, prim, material_cache[key]))
    gltf.meshes = [pygltflib.Mesh(name=mesh_name, primitives=gltf_prims)]

    # nodes: 0 = mesh node, 1.. = joints in the given order
    node_extras = node_extras or {}
    mesh_node = pygltflib.Node(name=mesh_name, mesh=0, skin=0 if skinned else None)
    nodes = [mesh_node]
    node_index = {name: 1 + i for i, name in enumerate(joint_names)}
    for j in joints:
        t = local_rest_translation(j, by_name)
        children = [node_index[c.name] for c in joints if c.parent == j.name]
        node = pygltflib.Node(name=j.name, translation=[float(v) for v in t], children=children or None,
                              extras=node_extras.get(j.name))
        nodes.append(node)
    gltf.nodes = nodes
    roots = [node_index[j.name] for j in joints if j.parent is None]

    if skinned:
        ibms = []
        for j in joints:
            m = np.eye(4, dtype=np.float32)
            m[3, :3] = -np.asarray(j.pivot, dtype=np.float32)  # column-major: translation in the last row
            ibms.append(m)
        ibm_acc = buf.add_accessor(np.stack(ibms), pygltflib.FLOAT, pygltflib.MAT4)
        gltf.skins = [pygltflib.Skin(inverseBindMatrices=ibm_acc, joints=[node_index[n] for n in joint_names],
                                     skeleton=roots[0] if roots else None, name="owl_skin")]

    gltf_anims = []
    for anim in animations:
        samplers = []
        channels = []
        for track in anim.tracks:
            if track.joint not in node_index:
                raise ValueError(f"animation {anim.name!r}: unknown joint {track.joint!r}")
            times = np.asarray(track.times, dtype=np.float32)
            values = np.asarray(track.values, dtype=np.float64)
            if np.any(np.diff(times) < 0):
                raise ValueError(f"animation {anim.name!r}/{track.joint}: times must be non-decreasing")
            if track.path == "rotation":
                if values.shape != (len(times), 4):
                    raise ValueError(f"animation {anim.name!r}/{track.joint}: rotation values must be (k, 4)")
                values = values / np.linalg.norm(values, axis=1, keepdims=True)
                type_ = pygltflib.VEC4
            elif track.path in ("translation", "scale"):
                if values.shape != (len(times), 3):
                    raise ValueError(f"animation {anim.name!r}/{track.joint}: {track.path} values must be (k, 3)")
                if track.path == "translation":
                    values = values + local_rest_translation(by_name[track.joint], by_name)
                type_ = pygltflib.VEC3
            else:
                raise ValueError(f"animation {anim.name!r}: unsupported path {track.path!r}")
            t_acc = buf.add_accessor(times, pygltflib.FLOAT, pygltflib.SCALAR, minmax=True)
            v_acc = buf.add_accessor(values.astype(np.float32), pygltflib.FLOAT, type_)
            samplers.append(pygltflib.AnimationSampler(input=t_acc, output=v_acc, interpolation="LINEAR"))
            channels.append(pygltflib.AnimationChannel(
                sampler=len(samplers) - 1,
                target=pygltflib.AnimationChannelTarget(node=node_index[track.joint], path=track.path),
            ))
        gltf_anims.append(pygltflib.Animation(name=anim.name, samplers=samplers, channels=channels))
    gltf.animations = gltf_anims

    gltf.scenes = [pygltflib.Scene(nodes=[0] + roots)]
    gltf.scene = 0
    gltf.accessors = buf.accessors
    gltf.bufferViews = buf.views
    gltf.buffers = [pygltflib.Buffer(byteLength=len(buf.blob))]
    if asset_extras:
        gltf.extras = asset_extras
    gltf.set_binary_blob(bytes(buf.blob))
    gltf.save(out_path, asset=gltf.asset)
