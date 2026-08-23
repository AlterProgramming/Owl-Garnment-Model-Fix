"""Round-trip tests for meshforge.rigexport.write_rigged_glb through
pygltflib: two primitives (a textured box and a vertex-colored icosphere)
on one skin over a 3-joint chain, with one clip carrying a rotation track
and a translation track.

meshforge.validate.validate is deliberately NOT used here: it expects the
older single-clip layout."""
import numpy as np
import pygltflib
import pytest
import trimesh
from PIL import Image

from meshforge.fk import Joint
from meshforge.rigexport import (
    AnimationSpec,
    MaterialSpec,
    PrimitiveSpec,
    Track,
    srgb_to_linear,
    write_rigged_glb,
)

PIVOTS = {"root": np.array([0.0, 0.0, 0.0]), "a": np.array([0.0, 1.0, 0.0]), "b": np.array([0.0, 2.0, 0.0])}
TIMES = np.array([0.0, 0.5, 1.0])
ROTATION_VALUES = np.array([
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, np.sin(np.pi / 8), np.cos(np.pi / 8)],
    [0.0, 0.0, 0.0, 2.0],   # deliberately non-unit: the writer must normalize it
])
TRANSLATION_OFFSETS = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.0, 0.0, 0.5]])
SPHERE_SRGB = np.array([0.5, 1.0, 0.0])
EXTRAS = {"contract": "owl-rig", "version": 2}


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------

def chain_joints():
    return [
        Joint("root", None, PIVOTS["root"]),
        Joint("a", "root", PIVOTS["a"]),
        Joint("b", "a", PIVOTS["b"]),
    ]


def textured_box_primitive(material=None, joints=None, normals=None):
    box = trimesh.creation.box()
    vertices = np.asarray(box.vertices, dtype=np.float64)
    n = len(vertices)
    # trimesh's box ships ColorVisuals without uv -> a simple planar projection
    uvs = vertices[:, :2] + 0.5
    if joints is None:
        joints = np.zeros((n, 4), dtype=np.uint8)
        joints[:, 0] = 1  # everything on joint "a"
    weights = np.zeros((n, 4))
    weights[:, 0] = 1.0
    if material is None:
        material = MaterialSpec("box_skin", base_color_image=Image.new("RGB", (4, 4), (200, 100, 50)))
    if normals is None:
        normals = np.asarray(box.vertex_normals, dtype=np.float64)
    return PrimitiveSpec("box", vertices, np.asarray(box.faces), normals, material,
                         uvs=uvs, joints=joints, weights=weights)


def colored_sphere_primitive():
    ico = trimesh.creation.icosphere(subdivisions=1)
    n = len(ico.vertices)
    colors = np.tile(SPHERE_SRGB, (n, 1))
    joints = np.zeros((n, 4), dtype=np.uint8)
    joints[:, 0] = 1
    joints[:, 1] = 2
    weights = np.zeros((n, 4))
    weights[:, :2] = 0.5
    material = MaterialSpec("sphere_flat")
    return PrimitiveSpec("sphere", np.asarray(ico.vertices), np.asarray(ico.faces),
                         np.asarray(ico.vertex_normals), material, colors=colors, joints=joints, weights=weights)


def clip():
    return AnimationSpec("clip", tracks=[
        Track("a", "rotation", TIMES, ROTATION_VALUES),
        Track("b", "translation", TIMES, TRANSLATION_OFFSETS),
    ])


def _accessor(gltf, index, dtype, ncomp):
    acc = gltf.accessors[index]
    view = gltf.bufferViews[acc.bufferView]
    start = (view.byteOffset or 0) + (acc.byteOffset or 0)
    nbytes = acc.count * ncomp * np.dtype(dtype).itemsize
    raw = gltf.binary_blob()[start:start + nbytes]
    arr = np.frombuffer(raw, dtype=dtype)
    return arr.reshape(acc.count, ncomp) if ncomp > 1 else arr


@pytest.fixture
def gltf(tmp_path):
    out_path = tmp_path / "rig.glb"
    write_rigged_glb([textured_box_primitive(), colored_sphere_primitive()], chain_joints(), [clip()],
                     str(out_path), asset_extras=dict(EXTRAS))
    assert out_path.exists()
    return pygltflib.GLTF2().load(str(out_path))


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------

def test_one_mesh_two_primitives_two_materials_one_texture(gltf):
    assert len(gltf.meshes) == 1
    assert len(gltf.meshes[0].primitives) == 2
    assert len(gltf.materials) == 2
    assert len(gltf.images) == 1
    assert len(gltf.textures) == 1
    assert len(gltf.samplers) == 1
    assert gltf.images[0].mimeType == "image/jpeg"

    box_prim, sphere_prim = gltf.meshes[0].primitives
    box_mat = gltf.materials[box_prim.material]
    sphere_mat = gltf.materials[sphere_prim.material]
    assert box_mat.pbrMetallicRoughness.baseColorTexture.index == 0
    assert sphere_mat.pbrMetallicRoughness.baseColorTexture is None
    assert box_mat.pbrMetallicRoughness.metallicFactor == 0.0
    assert box_prim.attributes.TEXCOORD_0 is not None
    assert sphere_prim.attributes.COLOR_0 is not None
    assert box_prim.attributes.COLOR_0 is None


def test_mesh_node_carries_the_skin_and_scene_lists_mesh_plus_root(gltf):
    mesh_nodes = [n for n in gltf.nodes if n.mesh is not None]
    assert len(mesh_nodes) == 1
    assert mesh_nodes[0].skin == 0
    assert len(gltf.scenes) == 1
    scene_names = {gltf.nodes[i].name for i in gltf.scenes[0].nodes}
    assert scene_names == {mesh_nodes[0].name, "root"}


def test_skin_joints_follow_the_given_order(gltf):
    assert len(gltf.skins) == 1
    skin = gltf.skins[0]
    assert len(skin.joints) == 3
    assert [gltf.nodes[j].name for j in skin.joints] == ["root", "a", "b"]
    assert gltf.nodes[skin.skeleton].name == "root"


def test_joint_node_translations_are_pivot_minus_parent_pivot(gltf):
    by_name = {n.name: n for n in gltf.nodes}
    index = {n.name: i for i, n in enumerate(gltf.nodes)}
    assert np.allclose(by_name["root"].translation, PIVOTS["root"])
    assert np.allclose(by_name["a"].translation, PIVOTS["a"] - PIVOTS["root"])
    assert np.allclose(by_name["b"].translation, PIVOTS["b"] - PIVOTS["a"])
    assert by_name["root"].children == [index["a"]]
    assert by_name["a"].children == [index["b"]]
    assert not by_name["b"].children
    # rest rotation is identity (omitted)
    for name in ("root", "a", "b"):
        assert by_name[name].rotation is None or np.allclose(by_name[name].rotation, [0, 0, 0, 1])


def test_inverse_bind_matrices_translate_by_minus_pivot(gltf):
    skin = gltf.skins[0]
    acc = gltf.accessors[skin.inverseBindMatrices]
    assert acc.type == "MAT4"
    assert acc.count == 3
    mats = _accessor(gltf, skin.inverseBindMatrices, np.float32, 16)
    for m, name in zip(mats, ["root", "a", "b"]):
        # column-major: translation lives in elements 12, 13, 14
        assert np.allclose(m[[12, 13, 14]], -PIVOTS[name], atol=1e-6)
        # the rest is the identity
        rest = np.eye(4, dtype=np.float32).ravel()
        rest[[12, 13, 14]] = m[[12, 13, 14]]
        assert np.allclose(m, rest, atol=1e-6)


# --------------------------------------------------------------------------
# vertex data
# --------------------------------------------------------------------------

def test_srgb_to_linear_matches_the_standard_curve():
    assert np.isclose(srgb_to_linear(0.5), 0.2140, atol=1e-3)
    assert np.isclose(srgb_to_linear(1.0), 1.0)
    assert np.isclose(srgb_to_linear(0.0), 0.0)
    assert np.isclose(srgb_to_linear(0.04), 0.04 / 12.92)   # linear toe segment
    assert np.isclose(srgb_to_linear(2.0), 1.0)              # clipped


def test_color0_is_written_in_linear_space(gltf):
    sphere_prim = gltf.meshes[0].primitives[1]
    acc = gltf.accessors[sphere_prim.attributes.COLOR_0]
    assert acc.type == "VEC3"
    assert acc.componentType == pygltflib.FLOAT
    colors = _accessor(gltf, sphere_prim.attributes.COLOR_0, np.float32, 3)
    assert colors.shape == (acc.count, 3)
    # sRGB 0.5 -> ~0.214, 1.0 -> 1.0, 0.0 -> 0.0
    assert np.allclose(colors, [0.2140, 1.0, 0.0], atol=1e-3)
    assert not np.allclose(colors[:, 0], 0.5, atol=0.05)


def test_indices_use_unsigned_short_for_small_meshes(gltf):
    for prim in gltf.meshes[0].primitives:
        acc = gltf.accessors[prim.indices]
        assert acc.componentType == pygltflib.UNSIGNED_SHORT
        assert acc.type == "SCALAR"
        assert acc.count % 3 == 0
    box_prim = gltf.meshes[0].primitives[0]
    indices = _accessor(gltf, box_prim.indices, np.uint16, 1)
    assert indices.max() < gltf.accessors[box_prim.attributes.POSITION].count


def test_joints_and_weights_round_trip(gltf):
    box_prim, sphere_prim = gltf.meshes[0].primitives
    for prim in (box_prim, sphere_prim):
        assert gltf.accessors[prim.attributes.JOINTS_0].componentType == pygltflib.UNSIGNED_BYTE
        weights = _accessor(gltf, prim.attributes.WEIGHTS_0, np.float32, 4)
        assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-6)
        joints = _accessor(gltf, prim.attributes.JOINTS_0, np.uint8, 4)
        assert joints.max() < 3
    sphere_joints = _accessor(gltf, sphere_prim.attributes.JOINTS_0, np.uint8, 4)
    sphere_weights = _accessor(gltf, sphere_prim.attributes.WEIGHTS_0, np.float32, 4)
    assert np.all(sphere_joints[:, :2] == [1, 2])
    assert np.allclose(sphere_weights[:, :2], 0.5)


def test_positions_and_normals_round_trip(gltf):
    box = trimesh.creation.box()
    box_prim = gltf.meshes[0].primitives[0]
    pos_acc = gltf.accessors[box_prim.attributes.POSITION]
    assert np.allclose(pos_acc.min, [-0.5, -0.5, -0.5])
    assert np.allclose(pos_acc.max, [0.5, 0.5, 0.5])
    positions = _accessor(gltf, box_prim.attributes.POSITION, np.float32, 3)
    assert np.allclose(positions, box.vertices, atol=1e-6)
    normals = _accessor(gltf, box_prim.attributes.NORMAL, np.float32, 3)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
    uvs = _accessor(gltf, box_prim.attributes.TEXCOORD_0, np.float32, 2)
    assert uvs.min() >= 0.0 and uvs.max() <= 1.0


# --------------------------------------------------------------------------
# animation
# --------------------------------------------------------------------------

def test_translation_track_outputs_rest_translation_plus_offset(gltf):
    assert len(gltf.animations) == 1
    anim = gltf.animations[0]
    assert anim.name == "clip"
    assert len(anim.channels) == 2 and len(anim.samplers) == 2

    channel = next(c for c in anim.channels if c.target.path == "translation")
    assert gltf.nodes[channel.target.node].name == "b"
    sampler = anim.samplers[channel.sampler]
    assert sampler.interpolation == "LINEAR"
    times = _accessor(gltf, sampler.input, np.float32, 1)
    values = _accessor(gltf, sampler.output, np.float32, 3)
    assert np.allclose(times, TIMES)
    rest_local = PIVOTS["b"] - PIVOTS["a"]
    assert np.allclose(values, TRANSLATION_OFFSETS + rest_local, atol=1e-6)
    # the time accessor carries min/max for the viewer's clip-duration lookup
    assert np.isclose(gltf.accessors[sampler.input].min[0], 0.0)
    assert np.isclose(gltf.accessors[sampler.input].max[0], 1.0)


def test_rotation_track_outputs_unit_quaternions(gltf):
    anim = gltf.animations[0]
    channel = next(c for c in anim.channels if c.target.path == "rotation")
    assert gltf.nodes[channel.target.node].name == "a"
    sampler = anim.samplers[channel.sampler]
    assert gltf.accessors[sampler.output].type == "VEC4"
    quats = _accessor(gltf, sampler.output, np.float32, 4)
    assert quats.shape == (3, 4)
    assert np.allclose(np.linalg.norm(quats, axis=1), 1.0, atol=1e-6)
    assert np.allclose(quats[0], [0, 0, 0, 1], atol=1e-6)
    assert np.allclose(quats[2], [0, 0, 0, 1], atol=1e-6)   # (0,0,0,2) normalized
    assert np.allclose(quats[1], ROTATION_VALUES[1], atol=1e-6)


def test_asset_extras_round_trip(gltf):
    assert gltf.extras == EXTRAS
    # note: pygltflib's GLTF2.save(fname, asset=Asset()) replaces gltf.asset with
    # its default, so the "meshforge.rigexport" generator string does not survive
    # the write; only the glTF version is a stable expectation here.
    assert gltf.asset.version == "2.0"


def test_node_extras_round_trip(tmp_path):
    out_path = tmp_path / "extras.glb"
    write_rigged_glb([colored_sphere_primitive()], chain_joints(), [], str(out_path),
                     node_extras={"a": {"role": "hinge"}})
    gltf = pygltflib.GLTF2().load(str(out_path))
    by_name = {n.name: n for n in gltf.nodes}
    assert by_name["a"].extras == {"role": "hinge"}
    assert not by_name["b"].extras
    assert gltf.animations == []


# --------------------------------------------------------------------------
# validation errors
# --------------------------------------------------------------------------

def test_joint_index_out_of_range_raises(tmp_path):
    bad_joints = np.zeros((8, 4), dtype=np.uint8)
    bad_joints[:, 0] = 3  # only joints 0..2 exist
    out_path = tmp_path / "bad.glb"
    with pytest.raises(ValueError, match="out of range"):
        write_rigged_glb([textured_box_primitive(joints=bad_joints)], chain_joints(), [], str(out_path))
    assert not out_path.exists()


def test_unknown_joint_in_track_raises(tmp_path):
    anim = AnimationSpec("clip", tracks=[Track("ghost", "rotation", TIMES, ROTATION_VALUES)])
    out_path = tmp_path / "bad.glb"
    with pytest.raises(ValueError, match="unknown joint"):
        write_rigged_glb([colored_sphere_primitive()], chain_joints(), [anim], str(out_path))
    assert not out_path.exists()


def test_non_monotonic_times_raise(tmp_path):
    anim = AnimationSpec("clip", tracks=[Track("a", "rotation", np.array([0.0, 1.0, 0.5]), ROTATION_VALUES)])
    out_path = tmp_path / "bad.glb"
    with pytest.raises(ValueError, match="non-decreasing"):
        write_rigged_glb([colored_sphere_primitive()], chain_joints(), [anim], str(out_path))
    assert not out_path.exists()


def test_mismatched_normals_shape_raises(tmp_path):
    out_path = tmp_path / "bad.glb"
    with pytest.raises(ValueError, match="normals must be"):
        write_rigged_glb([textured_box_primitive(normals=np.zeros((7, 3)))], chain_joints(), [], str(out_path))
    assert not out_path.exists()


def test_other_contract_violations_raise(tmp_path):
    out_path = tmp_path / "bad.glb"
    with pytest.raises(ValueError, match="at least one primitive"):
        write_rigged_glb([], chain_joints(), [], str(out_path))
    duplicate = chain_joints() + [Joint("a", "root", np.ones(3))]
    with pytest.raises(ValueError, match="unique"):
        write_rigged_glb([colored_sphere_primitive()], duplicate, [], str(out_path))
    bad_shape = AnimationSpec("clip", tracks=[Track("a", "translation", TIMES, ROTATION_VALUES)])
    with pytest.raises(ValueError, match=r"\(k, 3\)"):
        write_rigged_glb([colored_sphere_primitive()], chain_joints(), [bad_shape], str(out_path))
    bad_path = AnimationSpec("clip", tracks=[Track("a", "wobble", TIMES, TRANSLATION_OFFSETS)])
    with pytest.raises(ValueError, match="unsupported path"):
        write_rigged_glb([colored_sphere_primitive()], chain_joints(), [bad_path], str(out_path))
    assert not out_path.exists()
