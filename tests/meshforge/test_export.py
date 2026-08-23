import numpy as np
import pygltflib
import trimesh
from meshforge.rig import build_rig
from meshforge.animate import wave_clip
from meshforge.export import write_glb, write_skinned_glb, write_multi_skinned_glb


def _built_rig_and_clip(tmp_path):
    body = trimesh.creation.box(extents=[1, 1, 1])
    part = trimesh.creation.box(extents=[0.5, 0.2, 0.2])
    part.apply_translation([0.75, 0.0, 0.0])
    pivot = np.array([0.5, 0.0, 0.0])
    root = build_rig(body, part, pivot)
    keyframes = wave_clip(np.array([0.0, 0.0, 1.0]), keyframe_count=6)
    out_path = tmp_path / "test_rig.glb"
    write_glb(root, keyframes, animated_node_name="wing_pivot", out_path=str(out_path))
    return out_path


def test_writes_valid_gltf_with_two_mesh_nodes(tmp_path):
    out_path = _built_rig_and_clip(tmp_path)
    gltf = pygltflib.GLTF2().load(str(out_path))

    mesh_bearing_nodes = [n for n in gltf.nodes if n.mesh is not None]
    assert len(mesh_bearing_nodes) == 2


def test_animation_channel_targets_correct_node(tmp_path):
    out_path = _built_rig_and_clip(tmp_path)
    gltf = pygltflib.GLTF2().load(str(out_path))

    assert len(gltf.animations) == 1
    animation = gltf.animations[0]
    assert len(animation.channels) == 1
    channel = animation.channels[0]
    assert channel.target.path == "rotation"

    target_node = gltf.nodes[channel.target.node]
    assert target_node.name == "wing_pivot"


def test_animation_sampler_input_output_lengths_match(tmp_path):
    out_path = _built_rig_and_clip(tmp_path)
    gltf = pygltflib.GLTF2().load(str(out_path))

    sampler = gltf.animations[0].samplers[0]
    input_accessor = gltf.accessors[sampler.input]
    output_accessor = gltf.accessors[sampler.output]
    assert input_accessor.count == output_accessor.count == 6


def test_raises_on_unknown_animated_node_name(tmp_path):
    body = trimesh.creation.box(extents=[1, 1, 1])
    part = trimesh.creation.box(extents=[0.5, 0.2, 0.2])
    pivot = np.array([0.5, 0.0, 0.0])
    root = build_rig(body, part, pivot)
    keyframes = wave_clip(np.array([0.0, 0.0, 1.0]), keyframe_count=6)
    out_path = tmp_path / "should_not_exist.glb"
    try:
        write_glb(root, keyframes, animated_node_name="not_a_real_node", out_path=str(out_path))
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "not_a_real_node" in str(exc)


def test_exports_and_roundtrips_vertex_colors(tmp_path):
    """Regression test: verify COLOR_0 is exported and data round-trips correctly.

    This test catches the bug where mesh.visual.kind == "vertex_colors" (wrong)
    should be mesh.visual.kind == "vertex" (correct). The old condition never
    matched, silently dropping all vertex color data at export.
    """
    # Create colored meshes using ColorVisuals (same pattern as clean.py)
    body = trimesh.creation.box(extents=[1, 1, 1])
    body_colors = np.tile([255, 0, 0, 255], (len(body.vertices), 1)).astype(np.uint8)  # Red
    body.visual = trimesh.visual.ColorVisuals(mesh=body, vertex_colors=body_colors)

    part = trimesh.creation.box(extents=[0.5, 0.2, 0.2])
    part.apply_translation([0.75, 0.0, 0.0])
    part_colors = np.tile([0, 255, 0, 255], (len(part.vertices), 1)).astype(np.uint8)  # Green
    part.visual = trimesh.visual.ColorVisuals(mesh=part, vertex_colors=part_colors)

    # Build rig and export
    pivot = np.array([0.5, 0.0, 0.0])
    root = build_rig(body, part, pivot)
    keyframes = wave_clip(np.array([0.0, 0.0, 1.0]), keyframe_count=6)
    out_path = tmp_path / "test_colored_rig.glb"
    write_glb(root, keyframes, animated_node_name="wing_pivot", out_path=str(out_path))

    # Re-parse and verify COLOR_0 is present
    gltf = pygltflib.GLTF2().load(str(out_path))

    # Check that both mesh primitives have COLOR_0 attribute
    for mesh_idx, mesh in enumerate(gltf.meshes):
        for prim_idx, prim in enumerate(mesh.primitives):
            assert prim.attributes.COLOR_0 is not None, (
                f"Mesh {mesh_idx} primitive {prim_idx} missing COLOR_0 attribute"
            )

    # Verify color data round-trips: extract colors from GLB and compare
    # Body (node 1) should have red colors; wing (node 2) should have green
    for mesh_idx, expected_color in [(0, np.array([1.0, 0.0, 0.0, 1.0])),
                                      (1, np.array([0.0, 1.0, 0.0, 1.0]))]:
        mesh = gltf.meshes[mesh_idx]
        prim = mesh.primitives[0]
        color_accessor_idx = prim.attributes.COLOR_0
        color_accessor = gltf.accessors[color_accessor_idx]

        # Extract color data from buffer
        buffer_view = gltf.bufferViews[color_accessor.bufferView]
        byte_offset = (buffer_view.byteOffset or 0) + (color_accessor.byteOffset or 0)
        byte_count = color_accessor.count * 4 * 4  # 4 floats per color, 4 bytes each

        color_bytes = gltf.binary_blob()[byte_offset:byte_offset + byte_count]
        colors = np.frombuffer(color_bytes, dtype=np.float32).reshape(-1, 4)

        # All vertices should have the expected color (allow small float tolerance)
        assert np.allclose(colors, expected_color, atol=0.01), (
            f"Mesh {mesh_idx} colors did not round-trip correctly. "
            f"Expected {expected_color}, got {colors[0]}"
        )


def test_exports_material_on_every_primitive(tmp_path):
    """Regression test for Critical #1: a primitive with no material renders
    near-black in Three.js (default metallicFactor=1.0, roughnessFactor=1.0).
    Every mesh primitive must have a non-metallic, vertex-color-friendly
    material assigned."""
    out_path = _built_rig_and_clip(tmp_path)
    gltf = pygltflib.GLTF2().load(str(out_path))

    assert len(gltf.materials) >= 1
    for mesh in gltf.meshes:
        for primitive in mesh.primitives:
            assert primitive.material is not None
            mat = gltf.materials[primitive.material]
            assert mat.pbrMetallicRoughness.metallicFactor == 0.0


def test_exports_normal_attribute_on_every_primitive(tmp_path):
    """Regression test for Critical #2: no NORMAL attribute exported means
    the mesh renders faceted/flat-shaded instead of smooth."""
    out_path = _built_rig_and_clip(tmp_path)
    gltf = pygltflib.GLTF2().load(str(out_path))

    for mesh_idx, mesh in enumerate(gltf.meshes):
        for prim_idx, prim in enumerate(mesh.primitives):
            assert prim.attributes.NORMAL is not None, (
                f"Mesh {mesh_idx} primitive {prim_idx} missing NORMAL attribute"
            )

    # spot-check the normal data round-trips as float32 VEC3 and is unit length
    mesh = gltf.meshes[0]
    prim = mesh.primitives[0]
    normal_accessor = gltf.accessors[prim.attributes.NORMAL]
    buffer_view = gltf.bufferViews[normal_accessor.bufferView]
    byte_offset = buffer_view.byteOffset or 0
    byte_count = normal_accessor.count * 3 * 4
    normal_bytes = gltf.binary_blob()[byte_offset:byte_offset + byte_count]
    normals = np.frombuffer(normal_bytes, dtype=np.float32).reshape(-1, 3)
    lengths = np.linalg.norm(normals, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-3)


def _build_skinned_glb(tmp_path):
    body = trimesh.creation.box(extents=[1, 1, 1])
    part = trimesh.creation.box(extents=[0.5, 0.2, 0.2])
    part.apply_translation([0.75, 0.0, 0.0])
    mesh = trimesh.util.concatenate([body, part])
    wing_weight = np.where(mesh.vertices[:, 0] > 0.6, 1.0, 0.0)
    pivot = np.array([0.5, 0.0, 0.0])
    keyframes = wave_clip(np.array([0.0, 0.0, 1.0]), keyframe_count=6)
    out_path = tmp_path / "skinned.glb"
    write_skinned_glb(mesh, wing_weight, pivot, keyframes, out_path=str(out_path))
    return out_path


def test_skinned_glb_has_single_mesh_node_with_skin(tmp_path):
    out_path = _build_skinned_glb(tmp_path)
    gltf = pygltflib.GLTF2().load(str(out_path))

    mesh_nodes = [n for n in gltf.nodes if n.mesh is not None]
    assert len(mesh_nodes) == 1
    assert mesh_nodes[0].skin is not None
    assert len(gltf.skins) == 1
    assert len(gltf.skins[0].joints) == 2


def test_skinned_glb_weights_sum_to_one_per_vertex(tmp_path):
    out_path = _build_skinned_glb(tmp_path)
    gltf = pygltflib.GLTF2().load(str(out_path))

    mesh_node = next(n for n in gltf.nodes if n.mesh is not None)
    prim = gltf.meshes[mesh_node.mesh].primitives[0]
    assert prim.attributes.JOINTS_0 is not None
    assert prim.attributes.WEIGHTS_0 is not None

    w_accessor = gltf.accessors[prim.attributes.WEIGHTS_0]
    bv = gltf.bufferViews[w_accessor.bufferView]
    raw = gltf.binary_blob()[bv.byteOffset: bv.byteOffset + bv.byteLength]
    weights = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-5)


def test_skinned_glb_animation_targets_wing_joint(tmp_path):
    out_path = _build_skinned_glb(tmp_path)
    gltf = pygltflib.GLTF2().load(str(out_path))

    assert len(gltf.animations) == 1
    channel = gltf.animations[0].channels[0]
    assert gltf.nodes[channel.target.node].name == "wing_joint"


def test_skinned_glb_rest_pose_reproduces_original_vertices(tmp_path):
    """At bind pose (identity rotation on the wing joint) the skin
    formula must reproduce the mesh's original vertex positions exactly,
    for both fully-body and fully-wing vertices."""
    body = trimesh.creation.box(extents=[1, 1, 1])
    part = trimesh.creation.box(extents=[0.5, 0.2, 0.2])
    part.apply_translation([0.75, 0.0, 0.0])
    mesh = trimesh.util.concatenate([body, part])
    wing_weight = np.where(mesh.vertices[:, 0] > 0.6, 1.0, 0.0)
    pivot = np.array([0.5, 0.0, 0.0])
    keyframes = wave_clip(np.array([0.0, 0.0, 1.0]), keyframe_count=6)
    out_path = tmp_path / "skinned_rest.glb"
    write_skinned_glb(mesh, wing_weight, pivot, keyframes, out_path=str(out_path))

    from meshforge.skin import pose_vertices_lbs
    posed_at_rest = pose_vertices_lbs(mesh.vertices, wing_weight, pivot, keyframes[0].rotation)
    assert np.allclose(posed_at_rest, mesh.vertices)


def _build_multi_skinned_glb(tmp_path, animate_both=True):
    body = trimesh.creation.box(extents=[3, 1, 1])
    part_a = trimesh.creation.box(extents=[0.5, 0.2, 0.2])
    part_a.apply_translation([2.0, 0.0, 0.0])
    part_b = trimesh.creation.box(extents=[0.5, 0.2, 0.2])
    part_b.apply_translation([-2.0, 0.0, 0.0])
    mesh = trimesh.util.concatenate([body, part_a, part_b])

    joint_weights = {
        "body": np.where(np.abs(mesh.vertices[:, 0]) < 1.6, 1.0, 0.0),
        "part_a": np.where(mesh.vertices[:, 0] > 1.6, 1.0, 0.0),
        "part_b": np.where(mesh.vertices[:, 0] < -1.6, 1.0, 0.0),
    }
    joint_pivots = {"part_a": np.array([1.5, 0.0, 0.0]), "part_b": np.array([-1.5, 0.0, 0.0])}
    keyframes = wave_clip(np.array([0.0, 0.0, 1.0]), keyframe_count=6)
    animations = {"part_a": keyframes}
    if animate_both:
        animations["part_b"] = keyframes

    out_path = tmp_path / "multi_skinned.glb"
    write_multi_skinned_glb(mesh, joint_weights, joint_pivots, animations, out_path=str(out_path))
    return out_path, mesh, joint_weights, joint_pivots, keyframes


def test_multi_skinned_glb_has_three_joints_and_one_mesh_node(tmp_path):
    out_path, *_ = _build_multi_skinned_glb(tmp_path)
    gltf = pygltflib.GLTF2().load(str(out_path))

    mesh_nodes = [n for n in gltf.nodes if n.mesh is not None]
    assert len(mesh_nodes) == 1
    assert mesh_nodes[0].skin is not None
    assert len(gltf.skins) == 1
    # body + part_a + part_b = 3 joints
    assert len(gltf.skins[0].joints) == 3


def test_multi_skinned_glb_weights_sum_to_one_and_joints_in_range(tmp_path):
    out_path, *_ = _build_multi_skinned_glb(tmp_path)
    gltf = pygltflib.GLTF2().load(str(out_path))

    mesh_node = next(n for n in gltf.nodes if n.mesh is not None)
    prim = gltf.meshes[mesh_node.mesh].primitives[0]
    skin = gltf.skins[mesh_node.skin]

    w_accessor = gltf.accessors[prim.attributes.WEIGHTS_0]
    w_bv = gltf.bufferViews[w_accessor.bufferView]
    raw_w = gltf.binary_blob()[w_bv.byteOffset: w_bv.byteOffset + w_bv.byteLength]
    weights = np.frombuffer(raw_w, dtype=np.float32).reshape(-1, 4)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-5)

    j_accessor = gltf.accessors[prim.attributes.JOINTS_0]
    j_bv = gltf.bufferViews[j_accessor.bufferView]
    raw_j = gltf.binary_blob()[j_bv.byteOffset: j_bv.byteOffset + j_bv.byteLength]
    joints = np.frombuffer(raw_j, dtype=np.uint8).reshape(-1, 4)
    assert np.all(joints < len(skin.joints))


def test_multi_skinned_glb_animation_has_one_channel_per_animated_part(tmp_path):
    out_path, *_ = _build_multi_skinned_glb(tmp_path, animate_both=True)
    gltf = pygltflib.GLTF2().load(str(out_path))

    assert len(gltf.animations) == 1
    animation = gltf.animations[0]
    assert len(animation.channels) == 2
    assert len(animation.samplers) == 2

    target_names = {gltf.nodes[ch.target.node].name for ch in animation.channels}
    assert target_names == {"part_a_joint", "part_b_joint"}
    for ch in animation.channels:
        assert ch.target.path == "rotation"


def test_multi_skinned_glb_unanimated_part_gets_no_channel(tmp_path):
    out_path, *_ = _build_multi_skinned_glb(tmp_path, animate_both=False)
    gltf = pygltflib.GLTF2().load(str(out_path))

    animation = gltf.animations[0]
    assert len(animation.channels) == 1
    target_name = gltf.nodes[animation.channels[0].target.node].name
    assert target_name == "part_a_joint"
    # part_b is still a real skinned joint in the skeleton, just static
    joint_node_names = {gltf.nodes[j].name for j in gltf.skins[0].joints}
    assert "part_b_joint" in joint_node_names


def test_multi_skinned_glb_rest_pose_matches_pose_vertices_multi_lbs(tmp_path):
    out_path, mesh, joint_weights, joint_pivots, keyframes = _build_multi_skinned_glb(tmp_path)

    from meshforge.skin import pose_vertices_multi_lbs
    rest_rotations = {"part_a": keyframes[0].rotation, "part_b": keyframes[0].rotation}
    posed_at_rest = pose_vertices_multi_lbs(mesh.vertices, joint_weights, joint_pivots, rest_rotations)
    assert np.allclose(posed_at_rest, mesh.vertices)


def test_multi_skinned_glb_raises_without_body_key(tmp_path):
    body = trimesh.creation.box(extents=[1, 1, 1])
    joint_weights = {"part_a": np.ones(len(body.vertices))}
    joint_pivots = {"part_a": np.array([0.5, 0.0, 0.0])}
    try:
        write_multi_skinned_glb(body, joint_weights, joint_pivots, {}, out_path=str(tmp_path / "bad.glb"))
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "body" in str(exc)


def test_multi_skinned_glb_exports_parent_hierarchy_and_semantic_metadata(tmp_path):
    out_path, mesh, joint_weights, joint_pivots, keyframes = _build_multi_skinned_glb(tmp_path)
    out_path = tmp_path / "hierarchical.glb"
    write_multi_skinned_glb(
        mesh,
        joint_weights,
        joint_pivots,
        {"part_a": keyframes},
        out_path=str(out_path),
        joint_parents={"part_a": "body", "part_b": "part_a"},
        asset_extras={"contract": "interaction-lab", "version": 1},
        node_metadata={"part_a": {"role": "hinge"}, "part_b": {"role": "child"}},
    )
    gltf = pygltflib.GLTF2().load(str(out_path))

    by_name = {node.name: node for node in gltf.nodes}
    indices = {node.name: index for index, node in enumerate(gltf.nodes)}
    assert by_name["body_joint"].children == [indices["part_a_joint"]]
    assert by_name["part_a_joint"].children == [indices["part_b_joint"]]
    assert np.allclose(by_name["part_b_joint"].translation, [
        joint_pivots["part_b"][0] - joint_pivots["part_a"][0],
        joint_pivots["part_b"][1] - joint_pivots["part_a"][1],
        joint_pivots["part_b"][2] - joint_pivots["part_a"][2],
    ])
    assert gltf.extras == {"contract": "interaction-lab", "version": 1}
    assert by_name["part_a_joint"].extras == {"role": "hinge"}
    assert by_name["part_b_joint"].extras == {"role": "child"}
