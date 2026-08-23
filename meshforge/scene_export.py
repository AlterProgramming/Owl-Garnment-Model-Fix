"""Small glTF scene exporter for interaction and cutscene control fixtures."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pygltflib
import trimesh

from meshforge.export import _pack_mesh_buffers, _pack_multi_rotation_animation


@dataclass(frozen=True)
class SceneNodeSpec:
    """One local-space node in a rigid interaction scene."""

    name: str
    mesh: trimesh.Trimesh | None = None
    parent: str | None = None
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    extras: dict | None = None


def _validate_specs(specs: list[SceneNodeSpec]) -> None:
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("write_scene_glb: node names must be unique")
    known = set(names)
    for spec in specs:
        translation = np.asarray(spec.translation, dtype=np.float64)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError(f"write_scene_glb: {spec.name!r} translation must be a finite 3-vector")
        if spec.parent == spec.name:
            raise ValueError(f"write_scene_glb: node {spec.name!r} cannot parent itself")
        if spec.parent is not None and spec.parent not in known:
            raise ValueError(f"write_scene_glb: unknown parent {spec.parent!r} for {spec.name!r}")


def write_scene_glb(
    specs: list[SceneNodeSpec],
    animations: dict[str, list],
    asset_extras: dict | None,
    out_path: str,
) -> None:
    """Write a named rigid scene with optional rotation clips.

    Mesh vertices are already expressed in each node's local space. Empty
    nodes are first-class: sockets, hinge pivots, collider markers, camera
    targets, and cutscene anchors use the same hierarchy as render nodes.
    """
    if not specs:
        raise ValueError("write_scene_glb: at least one node is required")
    _validate_specs(specs)

    gltf = pygltflib.GLTF2()
    buffer_blob = bytearray()
    buffer_views: list[pygltflib.BufferView] = []
    accessors: list[pygltflib.Accessor] = []
    meshes: list[pygltflib.Mesh] = []
    nodes: list[pygltflib.Node] = []

    material = pygltflib.Material(
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorFactor=[0.72, 0.76, 0.82, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.85,
        ),
    )

    index_of = {spec.name: index for index, spec in enumerate(specs)}
    children: dict[str, list[int]] = {spec.name: [] for spec in specs}
    for spec in specs:
        if spec.parent is not None:
            children[spec.parent].append(index_of[spec.name])

    for spec in specs:
        node = pygltflib.Node(
            name=spec.name,
            translation=np.asarray(spec.translation, dtype=np.float64).tolist(),
            children=children[spec.name] or None,
            extras=spec.extras,
        )
        if spec.mesh is not None:
            attributes, indices_accessor = _pack_mesh_buffers(spec.mesh, buffer_blob, buffer_views, accessors)
            mesh_index = len(meshes)
            meshes.append(
                pygltflib.Mesh(
                    name=f"{spec.name}_mesh",
                    primitives=[
                        pygltflib.Primitive(
                            attributes=attributes,
                            indices=indices_accessor,
                            material=0,
                        )
                    ],
                )
            )
            node.mesh = mesh_index
        nodes.append(node)

    node_keyframes = {
        index_of[name]: keyframes
        for name, keyframes in animations.items()
    }
    for name in animations:
        if name not in index_of:
            raise ValueError(f"write_scene_glb: animation target {name!r} is not a scene node")

    roots = [index_of[spec.name] for spec in specs if spec.parent is None]
    gltf.scenes = [pygltflib.Scene(nodes=roots)]
    gltf.scene = 0
    gltf.nodes = nodes
    gltf.meshes = meshes
    gltf.materials = [material]
    gltf.accessors = accessors
    gltf.bufferViews = buffer_views
    gltf.animations = [
        _pack_multi_rotation_animation(node_keyframes, buffer_blob, buffer_views, accessors, name="interaction")
    ] if node_keyframes else []
    gltf.extras = asset_extras
    gltf.buffers = [pygltflib.Buffer(byteLength=len(buffer_blob))]
    gltf.set_binary_blob(bytes(buffer_blob))
    gltf.save(out_path)
