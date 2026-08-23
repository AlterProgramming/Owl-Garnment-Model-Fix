"""Exercises validate.py's checks directly, including the new checks added
for Important #5 (material presence, NORMAL attribute presence, rotation
accessor NaN check, unit-quaternion check) that would have caught Critical
#1 and #2 before they shipped.
"""
import numpy as np
import pygltflib
import trimesh

from meshforge.animate import wave_clip
from meshforge.export import write_glb
from meshforge.rig import build_rig
from meshforge.validate import validate


def _build_valid_glb(tmp_path):
    body = trimesh.creation.box(extents=[1, 1, 1])
    part = trimesh.creation.box(extents=[0.5, 0.2, 0.2])
    part.apply_translation([0.75, 0.0, 0.0])
    pivot = np.array([0.5, 0.0, 0.0])
    root = build_rig(body, part, pivot)
    keyframes = wave_clip(np.array([0.0, 0.0, 1.0]), keyframe_count=6)
    out_path = tmp_path / "valid_rig.glb"
    write_glb(root, keyframes, animated_node_name="wing_pivot", out_path=str(out_path))
    return out_path


def test_validate_passes_on_a_correctly_exported_glb(tmp_path):
    out_path = _build_valid_glb(tmp_path)
    assert validate(str(out_path)) == 0


def test_validate_fails_when_a_primitive_has_no_material(tmp_path, capsys):
    out_path = _build_valid_glb(tmp_path)
    gltf = pygltflib.GLTF2().load(str(out_path))
    gltf.meshes[0].primitives[0].material = None
    gltf.save(str(out_path))

    result = validate(str(out_path))
    captured = capsys.readouterr()
    assert result != 0
    assert "material assigned" in captured.out
    assert "FAIL" in captured.out


def test_validate_fails_when_a_primitive_has_no_normal_attribute(tmp_path, capsys):
    out_path = _build_valid_glb(tmp_path)
    gltf = pygltflib.GLTF2().load(str(out_path))
    gltf.meshes[0].primitives[0].attributes.NORMAL = None
    gltf.save(str(out_path))

    result = validate(str(out_path))
    captured = capsys.readouterr()
    assert result != 0
    assert "NORMAL" in captured.out


def test_validate_fails_on_non_unit_rotation_quaternion(tmp_path, capsys):
    out_path = _build_valid_glb(tmp_path)
    gltf = pygltflib.GLTF2().load(str(out_path))
    blob = bytearray(gltf.binary_blob())

    animation = gltf.animations[0]
    sampler = animation.samplers[0]
    output_accessor = gltf.accessors[sampler.output]
    buffer_view = gltf.bufferViews[output_accessor.bufferView]
    offset = buffer_view.byteOffset
    # corrupt the first quaternion's x component to break unit length
    rotations = np.frombuffer(bytes(blob[offset:offset + buffer_view.byteLength]), dtype=np.float32).copy()
    rotations = rotations.reshape(-1, 4)
    rotations[0] = [5.0, 0.0, 0.0, 0.0]
    blob[offset:offset + buffer_view.byteLength] = rotations.astype(np.float32).tobytes()

    gltf.set_binary_blob(bytes(blob))
    gltf.save(str(out_path))

    result = validate(str(out_path))
    captured = capsys.readouterr()
    assert result != 0
    assert "unit quaternion" in captured.out


def test_validate_fails_on_nan_in_rotation_accessor(tmp_path, capsys):
    out_path = _build_valid_glb(tmp_path)
    gltf = pygltflib.GLTF2().load(str(out_path))
    blob = bytearray(gltf.binary_blob())

    animation = gltf.animations[0]
    sampler = animation.samplers[0]
    output_accessor = gltf.accessors[sampler.output]
    buffer_view = gltf.bufferViews[output_accessor.bufferView]
    offset = buffer_view.byteOffset
    rotations = np.frombuffer(bytes(blob[offset:offset + buffer_view.byteLength]), dtype=np.float32).copy()
    rotations = rotations.reshape(-1, 4)
    rotations[0, 0] = np.nan
    blob[offset:offset + buffer_view.byteLength] = rotations.astype(np.float32).tobytes()

    gltf.set_binary_blob(bytes(blob))
    gltf.save(str(out_path))

    result = validate(str(out_path))
    captured = capsys.readouterr()
    assert result != 0
    assert "no NaN" in captured.out
