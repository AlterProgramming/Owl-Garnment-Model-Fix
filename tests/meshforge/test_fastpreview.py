"""Unit tests for meshforge.fastpreview — the PIL painter's-order flat
renderer used for region / skin-weight debugging loops."""
import numpy as np
import pytest
import trimesh
from PIL import Image

from meshforge.fastpreview import contact_sheet, render_flat

BACKGROUND = (38, 38, 38)


def colored_box():
    box = trimesh.creation.box()
    # one distinct color per vertex so the render is visibly non-uniform
    colors = (np.asarray(box.vertices) + 0.5).clip(0, 1)
    return np.asarray(box.vertices), np.asarray(box.faces), colors


def non_background_fraction(img):
    arr = np.asarray(img)
    return (arr != np.array(BACKGROUND, dtype=np.uint8)).any(axis=-1).mean()


def test_render_flat_returns_image_of_requested_resolution_with_content():
    vertices, faces, colors = colored_box()
    img = render_flat(vertices, faces, colors, yaw_deg=30.0, pitch_deg=20.0, resolution=96)

    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (96, 96)
    covered = non_background_fraction(img)
    # a box framed at 92 % of the view covers a big chunk, but not the corners
    assert 0.2 < covered < 0.95
    arr = np.asarray(img)
    assert tuple(arr[0, 0]) == BACKGROUND
    assert tuple(arr[-1, -1]) == BACKGROUND
    # the centre of the image is on the model
    assert tuple(arr[48, 48]) != BACKGROUND


def test_render_flat_accepts_per_face_colors_and_custom_background():
    vertices, faces, _ = colored_box()
    face_colors = np.zeros((len(faces), 3))
    face_colors[:, 0] = 1.0  # all red, no shading so the fill is exact
    img = render_flat(vertices, faces, face_colors, resolution=64, shade=False, background=(0, 0, 0))
    arr = np.asarray(img)
    painted = arr.any(axis=-1)
    assert painted.any()
    assert np.all(arr[painted] == [255, 0, 0])


def test_render_flat_shading_darkens_faces_turned_away_from_the_light():
    vertices, faces, _ = colored_box()
    white = np.ones((len(vertices), 3))
    shaded = np.asarray(render_flat(vertices, faces, white, yaw_deg=35.0, pitch_deg=25.0, resolution=64, shade=True))
    flat = np.asarray(render_flat(vertices, faces, white, yaw_deg=35.0, pitch_deg=25.0, resolution=64, shade=False))
    painted = (flat != np.array(BACKGROUND, dtype=np.uint8)).any(axis=-1)
    assert np.all(flat[painted] == 255)
    # shaded pixels are never brighter than flat ones and more than one level appears
    assert np.all(shaded[painted] <= 255)
    assert len(np.unique(shaded[painted][:, 0])) >= 2


def test_render_flat_return_mapping_exposes_scale_center_and_rotation():
    vertices, faces, colors = colored_box()
    result = render_flat(vertices, faces, colors, yaw_deg=10.0, pitch_deg=5.0, resolution=128, return_mapping=True)

    assert isinstance(result, tuple) and len(result) == 2
    img, (scale, center, R) = result
    assert isinstance(img, Image.Image) and img.size == (128, 128)
    assert np.isscalar(scale) and scale > 0
    assert np.asarray(center).shape == (3,)
    assert np.asarray(R).shape == (3, 3)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)
    # the mapping reproduces the renderer's pixel space: every projected
    # vertex lands inside the image, and the model's centre maps to the middle
    rotated = vertices @ R.T
    px = (rotated[:, 0] - center[0]) * scale + 127 / 2
    py = -(rotated[:, 1] - center[1]) * scale + 127 / 2
    assert px.min() >= 0 and px.max() <= 127 and py.min() >= 0 and py.max() <= 127
    assert np.isclose((px.min() + px.max()) / 2, 127 / 2, atol=1e-9)
    assert np.isclose((py.min() + py.max()) / 2, 127 / 2, atol=1e-9)


def test_render_flat_bounds_fix_the_framing_across_calls():
    vertices, faces, colors = colored_box()
    bounds = np.array([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]])
    _, (scale_a, center_a, _) = render_flat(vertices, faces, colors, resolution=64, bounds=bounds, return_mapping=True)
    _, (scale_b, center_b, _) = render_flat(vertices * 0.5, faces, colors, resolution=64, bounds=bounds, return_mapping=True)
    assert np.isclose(scale_a, scale_b)
    assert np.allclose(center_a, center_b)
    # without bounds the smaller model is framed tighter (larger scale)
    _, (scale_free, _, _) = render_flat(vertices * 0.5, faces, colors, resolution=64, return_mapping=True)
    assert scale_free > scale_b


def test_contact_sheet_of_two_images_has_expected_size():
    vertices, faces, colors = colored_box()
    a = render_flat(vertices, faces, colors, resolution=64)
    b = render_flat(vertices, faces, colors, yaw_deg=90.0, resolution=64)
    sheet = contact_sheet([a, b])

    assert sheet.mode == "RGB"
    assert sheet.size == (2 * 64 + 3 * 6, 64 + 2 * 6)
    # the tiles are pasted verbatim at their slots
    assert np.array_equal(np.asarray(sheet)[6:70, 6:70], np.asarray(a))
    assert np.array_equal(np.asarray(sheet)[6:70, 76:140], np.asarray(b))
    # the gutter keeps the sheet background
    assert tuple(np.asarray(sheet)[0, 0]) == (20, 20, 20)

    stacked = contact_sheet([a, b], cols=1, pad=2)
    assert stacked.size == (64 + 2 * 2, 2 * 64 + 3 * 2)


def test_contact_sheet_rejects_empty_input():
    with pytest.raises(ValueError, match="no images"):
        contact_sheet([])
