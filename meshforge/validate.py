"""Re-open a meshforge-exported GLB and check its rig/animation contract.
Mirrors the PASS/FAIL-line, nonzero-exit style of avatarforge/tools/validate.py.

    python3 -m meshforge.validate out/owl-rigged.glb
"""
from __future__ import annotations

import sys

import numpy as np
import pygltflib


def _check(label: str, condition: bool, failures: list[str]) -> None:
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}")
        failures.append(label)


def validate(path: str) -> int:
    gltf = pygltflib.GLTF2().load(path)
    failures: list[str] = []

    node_names = [n.name for n in gltf.nodes]
    _check("has at least 1 mesh-bearing node", sum(1 for n in gltf.nodes if n.mesh is not None) >= 1, failures)
    _check("node names are unique", len(node_names) == len(set(node_names)), failures)

    for i, node in enumerate(gltf.nodes):
        if node.mesh is None:
            continue
        mesh = gltf.meshes[node.mesh]
        for p, primitive in enumerate(mesh.primitives):
            _check(
                f"node {i} ({node.name}) mesh primitive {p} has a material assigned",
                primitive.material is not None, failures,
            )
            _check(
                f"node {i} ({node.name}) mesh primitive {p} has NORMAL attribute",
                primitive.attributes.NORMAL is not None, failures,
            )

    for i, node in enumerate(gltf.nodes):
        for child_idx in (node.children or []):
            _check(f"node {i} child index {child_idx} is in range", 0 <= child_idx < len(gltf.nodes), failures)

    blob = gltf.binary_blob()
    for i, node in enumerate(gltf.nodes):
        if node.skin is None:
            continue
        skin = gltf.skins[node.skin]
        _check(
            f"node {i} skin.joints reference valid nodes",
            all(0 <= j < len(gltf.nodes) for j in skin.joints), failures,
        )
        _check(f"node {i} skin has inverseBindMatrices", skin.inverseBindMatrices is not None, failures)

        mesh = gltf.meshes[node.mesh]
        for p, primitive in enumerate(mesh.primitives):
            _check(f"node {i} skin primitive {p} has JOINTS_0", primitive.attributes.JOINTS_0 is not None, failures)
            _check(f"node {i} skin primitive {p} has WEIGHTS_0", primitive.attributes.WEIGHTS_0 is not None, failures)

            if primitive.attributes.WEIGHTS_0 is not None:
                w_accessor = gltf.accessors[primitive.attributes.WEIGHTS_0]
                bv = gltf.bufferViews[w_accessor.bufferView]
                raw = blob[bv.byteOffset: bv.byteOffset + bv.byteLength]
                weights = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)
                sums = weights.sum(axis=1)
                _check(
                    f"node {i} skin primitive {p} weights sum to 1 per vertex",
                    bool(np.allclose(sums, 1.0, atol=1e-4)), failures,
                )

            if primitive.attributes.JOINTS_0 is not None:
                j_accessor = gltf.accessors[primitive.attributes.JOINTS_0]
                bv = gltf.bufferViews[j_accessor.bufferView]
                raw = blob[bv.byteOffset: bv.byteOffset + bv.byteLength]
                joints = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4)
                _check(
                    f"node {i} skin primitive {p} joint indices in range",
                    bool(np.all(joints < len(skin.joints))), failures,
                )

    _check("has exactly one animation", len(gltf.animations) == 1, failures)
    if gltf.animations:
        animation = gltf.animations[0]
        for channel in animation.channels:
            target_valid = 0 <= channel.target.node < len(gltf.nodes)
            _check(f"animation channel target node {channel.target.node} is in range", target_valid, failures)
            sampler_valid = 0 <= channel.sampler < len(animation.samplers)
            _check(f"animation channel sampler {channel.sampler} is in range", sampler_valid, failures)

        for sampler in animation.samplers:
            input_accessor = gltf.accessors[sampler.input]
            output_accessor = gltf.accessors[sampler.output]
            _check("sampler input/output accessor counts match", input_accessor.count == output_accessor.count, failures)

            blob = gltf.binary_blob()
            buffer_view = gltf.bufferViews[input_accessor.bufferView]
            raw = blob[buffer_view.byteOffset: buffer_view.byteOffset + buffer_view.byteLength]
            times = np.frombuffer(raw, dtype=np.float32)
            _check("keyframe times are monotonic", bool(np.all(np.diff(times) >= 0)), failures)
            _check("keyframe times contain no NaN", not bool(np.any(np.isnan(times))), failures)

            out_buffer_view = gltf.bufferViews[output_accessor.bufferView]
            out_raw = blob[out_buffer_view.byteOffset: out_buffer_view.byteOffset + out_buffer_view.byteLength]
            rotations = np.frombuffer(out_raw, dtype=np.float32).reshape(-1, 4)
            _check("rotation keyframes contain no NaN", not bool(np.any(np.isnan(rotations))), failures)

            quat_lengths = np.linalg.norm(rotations, axis=1)
            _check(
                "rotation keyframes are unit quaternions",
                bool(np.all(np.abs(quat_lengths - 1.0) < 1e-4)), failures,
            )

    return 1 if failures else 0


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    print(f"validating {argv[0]}")
    result = validate(argv[0])
    print("0 failures" if result == 0 else "FAILURES ABOVE")
    return result


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
