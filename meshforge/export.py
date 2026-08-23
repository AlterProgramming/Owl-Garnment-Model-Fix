"""Assemble the final animated GLB with pygltflib directly. trimesh's own
exporter has no skinning/animation support (confirmed this session), so
the node graph, buffers, accessors, and Animation are built by hand here;
trimesh is used only as the source of each static mesh's raw vertex/face/
color arrays.
"""
from __future__ import annotations

import numpy as np
import pygltflib

from meshforge.animate import Keyframe
from meshforge.rig import RigNode


def _pack_mesh_buffers(mesh, buffer_blob: bytearray, buffer_views, accessors):
    positions = mesh.vertices.astype(np.float32)
    normals = mesh.vertex_normals.astype(np.float32)
    indices = mesh.faces.astype(np.uint32).ravel()
    colors = None
    if mesh.visual.kind == "vertex":
        colors = (mesh.visual.vertex_colors[:, :4].astype(np.float32) / 255.0)

    pos_offset = len(buffer_blob)
    buffer_blob.extend(positions.tobytes())
    buffer_views.append(pygltflib.BufferView(buffer=0, byteOffset=pos_offset, byteLength=positions.nbytes))
    pos_accessor_idx = len(accessors)
    accessors.append(pygltflib.Accessor(
        bufferView=len(buffer_views) - 1, componentType=pygltflib.FLOAT,
        count=len(positions), type=pygltflib.VEC3,
        min=positions.min(axis=0).tolist(), max=positions.max(axis=0).tolist(),
    ))

    while len(buffer_blob) % 4 != 0:
        buffer_blob.append(0)
    normal_offset = len(buffer_blob)
    buffer_blob.extend(normals.tobytes())
    buffer_views.append(pygltflib.BufferView(buffer=0, byteOffset=normal_offset, byteLength=normals.nbytes))
    normal_accessor_idx = len(accessors)
    accessors.append(pygltflib.Accessor(
        bufferView=len(buffer_views) - 1, componentType=pygltflib.FLOAT,
        count=len(normals), type=pygltflib.VEC3,
    ))

    while len(buffer_blob) % 4 != 0:
        buffer_blob.append(0)
    idx_offset = len(buffer_blob)
    buffer_blob.extend(indices.tobytes())
    buffer_views.append(pygltflib.BufferView(buffer=0, byteOffset=idx_offset, byteLength=indices.nbytes))
    idx_accessor_idx = len(accessors)
    accessors.append(pygltflib.Accessor(
        bufferView=len(buffer_views) - 1, componentType=pygltflib.UNSIGNED_INT,
        count=len(indices), type=pygltflib.SCALAR,
    ))

    color_accessor_idx = None
    if colors is not None:
        while len(buffer_blob) % 4 != 0:
            buffer_blob.append(0)
        color_offset = len(buffer_blob)
        buffer_blob.extend(colors.tobytes())
        buffer_views.append(pygltflib.BufferView(buffer=0, byteOffset=color_offset, byteLength=colors.nbytes))
        color_accessor_idx = len(accessors)
        accessors.append(pygltflib.Accessor(
            bufferView=len(buffer_views) - 1, componentType=pygltflib.FLOAT,
            count=len(colors), type=pygltflib.VEC4,
        ))

    attributes = pygltflib.Attributes(POSITION=pos_accessor_idx, NORMAL=normal_accessor_idx)
    if color_accessor_idx is not None:
        attributes.COLOR_0 = color_accessor_idx
    return attributes, idx_accessor_idx


def _flatten_nodes(root: RigNode) -> list[RigNode]:
    flat = [root]
    for child in root.children:
        flat.extend(_flatten_nodes(child))
    return flat


def _pack_rotation_channel(keyframes, target_node_idx, buffer_blob, buffer_views, accessors, sampler_idx=0):
    """Pack one keyframe track into a (sampler, channel) pair. Factored
    out of _pack_rotation_animation so multiple independently-animated
    joints (e.g. two wings) can share one Animation object, each with
    its own sampler index — a single glTF Animation may hold any number
    of channels, but each needs its own sampler."""
    times = np.array([kf.time for kf in keyframes], dtype=np.float32)
    rotations = np.array([kf.rotation for kf in keyframes], dtype=np.float32)

    while len(buffer_blob) % 4 != 0:
        buffer_blob.append(0)
    time_offset = len(buffer_blob)
    buffer_blob.extend(times.tobytes())
    buffer_views.append(pygltflib.BufferView(buffer=0, byteOffset=time_offset, byteLength=times.nbytes))
    time_accessor_idx = len(accessors)
    accessors.append(pygltflib.Accessor(
        bufferView=len(buffer_views) - 1, componentType=pygltflib.FLOAT,
        count=len(times), type=pygltflib.SCALAR,
        min=[float(times.min())], max=[float(times.max())],
    ))

    while len(buffer_blob) % 4 != 0:
        buffer_blob.append(0)
    rot_offset = len(buffer_blob)
    buffer_blob.extend(rotations.tobytes())
    buffer_views.append(pygltflib.BufferView(buffer=0, byteOffset=rot_offset, byteLength=rotations.nbytes))
    rot_accessor_idx = len(accessors)
    accessors.append(pygltflib.Accessor(
        bufferView=len(buffer_views) - 1, componentType=pygltflib.FLOAT,
        count=len(rotations), type=pygltflib.VEC4,
    ))

    sampler = pygltflib.AnimationSampler(input=time_accessor_idx, output=rot_accessor_idx, interpolation="LINEAR")
    channel = pygltflib.AnimationChannel(
        sampler=sampler_idx,
        target=pygltflib.AnimationChannelTarget(node=target_node_idx, path="rotation"),
    )
    return sampler, channel


def _pack_rotation_animation(keyframes, target_node_idx, buffer_blob, buffer_views, accessors):
    sampler, channel = _pack_rotation_channel(keyframes, target_node_idx, buffer_blob, buffer_views, accessors, sampler_idx=0)
    return pygltflib.Animation(samplers=[sampler], channels=[channel], name="wave")


def _pack_multi_rotation_animation(node_keyframes: dict, buffer_blob, buffer_views, accessors, name="wave"):
    """node_keyframes maps target glTF node index -> list[Keyframe].
    Builds ONE Animation with one sampler+channel per entry, in dict
    iteration order (Python dicts preserve insertion order, so callers
    control determinism by how they build this dict)."""
    samplers = []
    channels = []
    for sampler_idx, (target_node_idx, keyframes) in enumerate(node_keyframes.items()):
        sampler, channel = _pack_rotation_channel(
            keyframes, target_node_idx, buffer_blob, buffer_views, accessors, sampler_idx=sampler_idx,
        )
        samplers.append(sampler)
        channels.append(channel)
    return pygltflib.Animation(samplers=samplers, channels=channels, name=name)


def write_glb(
    root: RigNode,
    keyframes: list[Keyframe],
    animated_node_name: str,
    out_path: str,
) -> None:
    flat_nodes = _flatten_nodes(root)
    names = [n.name for n in flat_nodes]
    if animated_node_name not in names:
        raise ValueError(
            f"write_glb: animated_node_name={animated_node_name!r} does not "
            f"match any node in the rig hierarchy {names}"
        )

    gltf = pygltflib.GLTF2()
    buffer_blob = bytearray()
    buffer_views: list[pygltflib.BufferView] = []
    accessors: list[pygltflib.Accessor] = []
    meshes: list[pygltflib.Mesh] = []
    nodes: list[pygltflib.Node] = []

    # A plain, non-metallic, fairly rough material so vertex color reads
    # correctly. Without an assigned material, glTF's default
    # (metallicFactor=1.0, roughnessFactor=1.0) renders near-black in
    # Three.js with no environment map.
    material = pygltflib.Material(
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0], metallicFactor=0.0, roughnessFactor=0.9,
        ),
    )
    materials = [material]
    material_idx = 0

    index_of = {id(n): i for i, n in enumerate(flat_nodes)}

    for node in flat_nodes:
        gltf_node = pygltflib.Node(name=node.name)
        if not np.allclose(node.translation, 0.0):
            gltf_node.translation = node.translation.tolist()
        if node.mesh is not None:
            attributes, idx_accessor_idx = _pack_mesh_buffers(node.mesh, buffer_blob, buffer_views, accessors)
            mesh_idx = len(meshes)
            meshes.append(pygltflib.Mesh(primitives=[
                pygltflib.Primitive(attributes=attributes, indices=idx_accessor_idx, material=material_idx)
            ]))
            gltf_node.mesh = mesh_idx
        if node.children:
            gltf_node.children = [index_of[id(c)] for c in node.children]
        nodes.append(gltf_node)

    # animation: rotation channel on the target node
    target_node_idx = names.index(animated_node_name)
    animation = _pack_rotation_animation(keyframes, target_node_idx, buffer_blob, buffer_views, accessors)

    gltf.scenes = [pygltflib.Scene(nodes=[0])]
    gltf.scene = 0
    gltf.nodes = nodes
    gltf.meshes = meshes
    gltf.materials = materials
    gltf.accessors = accessors
    gltf.bufferViews = buffer_views
    gltf.animations = [animation]
    gltf.buffers = [pygltflib.Buffer(byteLength=len(buffer_blob))]
    gltf.set_binary_blob(bytes(buffer_blob))
    gltf.save(out_path)


def write_skinned_glb(
    mesh,
    wing_weight: np.ndarray,
    pivot_point: np.ndarray,
    keyframes: list[Keyframe],
    out_path: str,
) -> None:
    """Export a single skinned mesh with a two-joint (body/wing) skeleton
    and a baked rotation clip on the wing joint — real glTF skinning
    (JOINTS_0/WEIGHTS_0/Skin/inverseBindMatrices) rather than write_glb's
    rigid two-node hinge. Vertices near the seam are partially driven by
    both joints (see meshforge.skin.compute_skin_weights), so the surface
    deforms continuously and there is no hard cut for an animated pose to
    expose — this is the actual production fix for the rigid hinge's
    "body looks ripped apart" hole, not a cosmetic patch on top of it.

    `mesh` keeps its own (world-space) vertex positions unchanged; the
    body joint's bind pose is identity and the wing joint's bind pose is
    a pure translation to `pivot_point`, so at rest (identity rotation)
    both joints reproduce the original vertex positions exactly.
    """
    pivot_point = np.asarray(pivot_point, dtype=np.float64)
    wing_weight = np.clip(np.asarray(wing_weight, dtype=np.float64), 0.0, 1.0)
    if wing_weight.shape != (len(mesh.vertices),):
        raise ValueError(
            f"write_skinned_glb: wing_weight must have shape "
            f"({len(mesh.vertices)},), got {wing_weight.shape}"
        )

    gltf = pygltflib.GLTF2()
    buffer_blob = bytearray()
    buffer_views: list[pygltflib.BufferView] = []
    accessors: list[pygltflib.Accessor] = []

    material = pygltflib.Material(
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0], metallicFactor=0.0, roughnessFactor=0.9,
        ),
    )

    attributes, idx_accessor_idx = _pack_mesh_buffers(mesh, buffer_blob, buffer_views, accessors)

    n_verts = len(mesh.vertices)
    # JOINTS_0 indices are into skin.joints ([body_joint, wing_joint]),
    # i.e. joint 0 = body, joint 1 = wing — not glTF node indices.
    joints0 = np.zeros((n_verts, 4), dtype=np.uint8)
    joints0[:, 1] = 1
    weights0 = np.zeros((n_verts, 4), dtype=np.float32)
    weights0[:, 0] = (1.0 - wing_weight).astype(np.float32)
    weights0[:, 1] = wing_weight.astype(np.float32)

    while len(buffer_blob) % 4 != 0:
        buffer_blob.append(0)
    joints_offset = len(buffer_blob)
    buffer_blob.extend(joints0.tobytes())
    buffer_views.append(pygltflib.BufferView(buffer=0, byteOffset=joints_offset, byteLength=joints0.nbytes))
    joints_accessor_idx = len(accessors)
    accessors.append(pygltflib.Accessor(
        bufferView=len(buffer_views) - 1, componentType=pygltflib.UNSIGNED_BYTE,
        count=n_verts, type=pygltflib.VEC4,
    ))

    while len(buffer_blob) % 4 != 0:
        buffer_blob.append(0)
    weights_offset = len(buffer_blob)
    buffer_blob.extend(weights0.tobytes())
    buffer_views.append(pygltflib.BufferView(buffer=0, byteOffset=weights_offset, byteLength=weights0.nbytes))
    weights_accessor_idx = len(accessors)
    accessors.append(pygltflib.Accessor(
        bufferView=len(buffer_views) - 1, componentType=pygltflib.FLOAT,
        count=n_verts, type=pygltflib.VEC4,
    ))

    attributes.JOINTS_0 = joints_accessor_idx
    attributes.WEIGHTS_0 = weights_accessor_idx

    gltf_mesh = pygltflib.Mesh(primitives=[
        pygltflib.Primitive(attributes=attributes, indices=idx_accessor_idx, material=0)
    ])

    # node 0 = skinned mesh (identity transform), node 1 = body joint
    # (root joint, identity bind pose), node 2 = wing joint (bind pose is
    # a pure translation to pivot_point, animated with local rotation).
    mesh_node = pygltflib.Node(name="owl_mesh", mesh=0, skin=0)
    body_joint_node = pygltflib.Node(name="body_joint", children=[2])
    wing_joint_node = pygltflib.Node(name="wing_joint", translation=pivot_point.tolist())
    nodes = [mesh_node, body_joint_node, wing_joint_node]

    # inverseBindMatrices: identity for the body joint, translate(-pivot)
    # for the wing joint. glTF matrices are column-major float[16]; a
    # numpy 4x4 with the translation in the last row, flattened row-major
    # (numpy's default), produces exactly that column-major layout.
    inv_body = np.eye(4, dtype=np.float32)
    inv_wing = np.eye(4, dtype=np.float32)
    inv_wing[3, :3] = -pivot_point

    while len(buffer_blob) % 4 != 0:
        buffer_blob.append(0)
    ibm_offset = len(buffer_blob)
    ibm = np.stack([inv_body, inv_wing])
    buffer_blob.extend(ibm.tobytes())
    buffer_views.append(pygltflib.BufferView(buffer=0, byteOffset=ibm_offset, byteLength=ibm.nbytes))
    ibm_accessor_idx = len(accessors)
    accessors.append(pygltflib.Accessor(
        bufferView=len(buffer_views) - 1, componentType=pygltflib.FLOAT,
        count=2, type=pygltflib.MAT4,
    ))

    skin = pygltflib.Skin(inverseBindMatrices=ibm_accessor_idx, joints=[1, 2], skeleton=1)

    animation = _pack_rotation_animation(keyframes, 2, buffer_blob, buffer_views, accessors)

    gltf.scenes = [pygltflib.Scene(nodes=[0, 1])]
    gltf.scene = 0
    gltf.nodes = nodes
    gltf.meshes = [gltf_mesh]
    gltf.materials = [material]
    gltf.accessors = accessors
    gltf.bufferViews = buffer_views
    gltf.skins = [skin]
    gltf.animations = [animation]
    gltf.buffers = [pygltflib.Buffer(byteLength=len(buffer_blob))]
    gltf.set_binary_blob(bytes(buffer_blob))
    gltf.save(out_path)


def write_multi_skinned_glb(
    mesh,
    joint_weights: dict,
    joint_pivots: dict,
    animations: dict,
    out_path: str,
    *,
    joint_parents: dict[str, str] | None = None,
    asset_extras: dict | None = None,
    node_metadata: dict[str, dict] | None = None,
) -> None:
    """N-joint generalization of write_skinned_glb: one skinned mesh, a
    body joint (root, identity bind pose) plus one child joint per named
    part in `joint_weights` (every key except "body"), each with its own
    baked rotation clip if present in `animations`. Same real-skinning
    machinery as write_skinned_glb (JOINTS_0/WEIGHTS_0/Skin/
    inverseBindMatrices), generalized past the hardcoded body+wing pair
    so multiple independently-articulated parts can share one skin.

    `joint_weights` must contain a "body" key (see
    meshforge.skin.compute_multi_part_skin_weights). `joint_pivots` must
    have an entry for every non-body key in `joint_weights`. `animations`
    maps a subset of those same part names to baked keyframe lists —
    parts with no entry are exported unanimated (skinned but static).
    """
    if "body" not in joint_weights:
        raise ValueError("write_multi_skinned_glb: joint_weights must include a 'body' entry")

    part_names = sorted(k for k in joint_weights if k != "body")
    parent_map = {name: "body" for name in part_names}
    if joint_parents is not None:
        unknown = set(joint_parents) - set(part_names)
        if unknown:
            raise ValueError(f"write_multi_skinned_glb: parent specified for unknown parts {sorted(unknown)}")
        parent_map.update(joint_parents)
    valid_parents = {"body", *part_names}
    for name, parent in parent_map.items():
        if parent not in valid_parents or parent == name:
            raise ValueError(f"write_multi_skinned_glb: invalid parent {parent!r} for part {name!r}")
    for name in part_names:
        seen = {name}
        parent = parent_map[name]
        while parent != "body":
            if parent in seen:
                raise ValueError(f"write_multi_skinned_glb: joint parent cycle at {name!r}")
            seen.add(parent)
            parent = parent_map[parent]
    n_verts = len(mesh.vertices)

    if joint_weights["body"].shape != (n_verts,):
        raise ValueError(
            f"write_multi_skinned_glb: joint_weights['body'] must have shape "
            f"({n_verts},), got {joint_weights['body'].shape}"
        )
    for name in part_names:
        if name not in joint_pivots:
            raise ValueError(f"write_multi_skinned_glb: missing pivot for part {name!r}")
        if joint_weights[name].shape != (n_verts,):
            raise ValueError(
                f"write_multi_skinned_glb: joint_weights[{name!r}] must have shape "
                f"({n_verts},), got {joint_weights[name].shape}"
            )

    gltf = pygltflib.GLTF2()
    buffer_blob = bytearray()
    buffer_views: list[pygltflib.BufferView] = []
    accessors: list[pygltflib.Accessor] = []

    material = pygltflib.Material(
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0], metallicFactor=0.0, roughnessFactor=0.9,
        ),
    )

    attributes, idx_accessor_idx = _pack_mesh_buffers(mesh, buffer_blob, buffer_views, accessors)

    # skin.joints order: body first (skin-joint-index 0), then each part
    # in sorted-name order (skin-joint-index 1..N) — must match the
    # column order of weights_matrix below, since JOINTS_0 values index
    # into this same ordering.
    joint_names = ["body"] + part_names
    n_joints = len(joint_names)
    weights_matrix = np.stack([joint_weights[name] for name in joint_names], axis=1)  # (n_verts, n_joints)

    # glTF VEC4 JOINTS_0/WEIGHTS_0 carries at most 4 influences per
    # vertex. In practice every vertex here has body plus at most one
    # part with nonzero weight (parts don't overlap — see
    # compute_multi_part_skin_weights), so 4 is never actually tight;
    # taking the top-4 by magnitude and renormalizing is a no-op for that
    # case but keeps this correct in general rather than assuming it.
    top_k = min(4, n_joints)
    order = np.argsort(-weights_matrix, axis=1)[:, :top_k]
    top_weights = np.take_along_axis(weights_matrix, order, axis=1)
    row_sums = top_weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    top_weights = top_weights / row_sums

    joints0 = np.zeros((n_verts, 4), dtype=np.uint8)
    weights0 = np.zeros((n_verts, 4), dtype=np.float32)
    joints0[:, :top_k] = order.astype(np.uint8)
    weights0[:, :top_k] = top_weights.astype(np.float32)

    while len(buffer_blob) % 4 != 0:
        buffer_blob.append(0)
    joints_offset = len(buffer_blob)
    buffer_blob.extend(joints0.tobytes())
    buffer_views.append(pygltflib.BufferView(buffer=0, byteOffset=joints_offset, byteLength=joints0.nbytes))
    joints_accessor_idx = len(accessors)
    accessors.append(pygltflib.Accessor(
        bufferView=len(buffer_views) - 1, componentType=pygltflib.UNSIGNED_BYTE,
        count=n_verts, type=pygltflib.VEC4,
    ))

    while len(buffer_blob) % 4 != 0:
        buffer_blob.append(0)
    weights_offset = len(buffer_blob)
    buffer_blob.extend(weights0.tobytes())
    buffer_views.append(pygltflib.BufferView(buffer=0, byteOffset=weights_offset, byteLength=weights0.nbytes))
    weights_accessor_idx = len(accessors)
    accessors.append(pygltflib.Accessor(
        bufferView=len(buffer_views) - 1, componentType=pygltflib.FLOAT,
        count=n_verts, type=pygltflib.VEC4,
    ))

    attributes.JOINTS_0 = joints_accessor_idx
    attributes.WEIGHTS_0 = weights_accessor_idx

    gltf_mesh = pygltflib.Mesh(primitives=[
        pygltflib.Primitive(attributes=attributes, indices=idx_accessor_idx, material=0)
    ])

    # node 0 = skinned mesh (identity transform), node 1 = body joint
    # (root joint, identity bind pose), nodes 2..N+1 = one joint per
    # part (in part_names order), each a child of the body joint.
    # Preserve the established mesh-node name used by the MeshForge owl
    # pipeline; semantic interaction metadata lives on the joints/extras.
    mesh_node = pygltflib.Node(name="owl_mesh", mesh=0, skin=0)
    node_metadata = node_metadata or {}
    body_joint_node = pygltflib.Node(
        name="body_joint",
        extras=node_metadata.get("body"),
    )
    node_indices = {name: 2 + index for index, name in enumerate(part_names)}
    children_by_parent = {"body": []}
    children_by_parent.update({name: [] for name in part_names})
    for name in part_names:
        children_by_parent[parent_map[name]].append(node_indices[name])
    body_joint_node.children = children_by_parent["body"] or None
    part_nodes = []
    for name in part_names:
        parent = parent_map[name]
        parent_pivot = np.zeros(3) if parent == "body" else np.asarray(joint_pivots[parent], dtype=np.float64)
        local_translation = np.asarray(joint_pivots[name], dtype=np.float64) - parent_pivot
        part_nodes.append(
            pygltflib.Node(
                name=f"{name}_joint",
                translation=local_translation.tolist(),
                children=children_by_parent[name] or None,
                extras=node_metadata.get(name),
            )
        )
    nodes = [mesh_node, body_joint_node] + part_nodes

    # inverseBindMatrices: identity for the body joint, translate(-pivot)
    # for each part joint — same convention as write_skinned_glb, applied
    # once per part in the same order as skin.joints.
    ibm_list = [np.eye(4, dtype=np.float32)]
    for name in part_names:
        inv = np.eye(4, dtype=np.float32)
        inv[3, :3] = -np.asarray(joint_pivots[name], dtype=np.float32)
        ibm_list.append(inv)

    while len(buffer_blob) % 4 != 0:
        buffer_blob.append(0)
    ibm_offset = len(buffer_blob)
    ibm = np.stack(ibm_list)
    buffer_blob.extend(ibm.tobytes())
    buffer_views.append(pygltflib.BufferView(buffer=0, byteOffset=ibm_offset, byteLength=ibm.nbytes))
    ibm_accessor_idx = len(accessors)
    accessors.append(pygltflib.Accessor(
        bufferView=len(buffer_views) - 1, componentType=pygltflib.FLOAT,
        count=n_joints, type=pygltflib.MAT4,
    ))

    skin_joint_node_indices = [1] + list(range(2, 2 + len(part_names)))
    skin = pygltflib.Skin(inverseBindMatrices=ibm_accessor_idx, joints=skin_joint_node_indices, skeleton=1)

    node_keyframes = {}
    for i, name in enumerate(part_names):
        if name in animations:
            node_keyframes[2 + i] = animations[name]
    animation = _pack_multi_rotation_animation(node_keyframes, buffer_blob, buffer_views, accessors)

    gltf.scenes = [pygltflib.Scene(nodes=[0, 1])]
    gltf.scene = 0
    gltf.nodes = nodes
    gltf.meshes = [gltf_mesh]
    gltf.materials = [material]
    gltf.accessors = accessors
    gltf.bufferViews = buffer_views
    gltf.skins = [skin]
    gltf.animations = [animation]
    gltf.extras = asset_extras
    gltf.buffers = [pygltflib.Buffer(byteLength=len(buffer_blob))]
    gltf.set_binary_blob(bytes(buffer_blob))
    gltf.save(out_path)
