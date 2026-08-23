"""Integration test against the real owl asset produced earlier this
session. Skips if that file isn't present (e.g. in a fresh checkout),
rather than failing — this test exercises the full pipeline end-to-end,
which unit tests on synthetic geometry (Tasks 1-6) intentionally don't.
"""
import os
import subprocess
import sys

import pygltflib
import pytest

OWL_GLB = "/Users/AI-CCORE/altageris/AICCORE/AI-GATEWAY/3d-generated/owl-mascot.glb"


@pytest.mark.skipif(not os.path.exists(OWL_GLB), reason="real owl asset not present in this checkout")
def test_cli_produces_valid_rigged_glb(tmp_path):
    out_path = tmp_path / "owl-rigged.glb"
    preview_dir = tmp_path / "previews"

    result = subprocess.run(
        [sys.executable, "-m", "meshforge", OWL_GLB, "--out", str(out_path),
         "--rig", "wing", "--clip", "wave", "--preview-dir", str(preview_dir)],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        capture_output=True, text=True, timeout=300,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert out_path.exists()

    gltf = pygltflib.GLTF2().load(str(out_path))
    # The wing and both lids each have a baked clip (3 channels); the
    # eyes are runtime-aimed (skinned but static at bind pose), so they
    # add no animation channels of their own.
    assert len(gltf.animations) == 1
    assert len(gltf.animations[0].channels) == 3
    # Skinned export: one mesh-bearing node, driven by a 6-joint skin
    # (body + wing + eye_right + eye_left + lid_right + lid_left).
    assert sum(1 for n in gltf.nodes if n.mesh is not None) == 1
    assert len(gltf.skins) == 1
    mesh_node = next(n for n in gltf.nodes if n.mesh is not None)
    assert mesh_node.skin is not None
    assert len(gltf.skins[0].joints) == 6
    primitive = gltf.meshes[mesh_node.mesh].primitives[0]
    assert primitive.attributes.JOINTS_0 is not None
    assert primitive.attributes.WEIGHTS_0 is not None

    # preview.py should have written at least one PNG so the run is
    # visually inspectable without opening a viewer
    assert preview_dir.exists()
    assert any(preview_dir.iterdir())
