import numpy as np
from meshforge.preview import render_orthographic


def test_render_returns_expected_shape_and_dtype():
    vertices = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]])
    faces = np.array([[0, 1, 2]])
    colors = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    image = render_orthographic(vertices, faces, colors, resolution=64)

    assert image.shape == (64, 64, 3)
    assert image.dtype == np.uint8


def test_triangle_paints_red_pixels_near_center():
    vertices = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]])
    faces = np.array([[0, 1, 2]])
    colors = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    image = render_orthographic(vertices, faces, colors, resolution=64)
    center = image[32, 32]

    assert center[0] > 200  # strongly red
    assert center[1] < 50
    assert center[2] < 50


def test_empty_mesh_returns_background_only():
    vertices = np.zeros((0, 3))
    faces = np.zeros((0, 3), dtype=int)
    colors = np.zeros((0, 3))

    image = render_orthographic(vertices, faces, colors, resolution=32)
    assert image.shape == (32, 32, 3)
    # background should be uniform (no triangles drawn)
    assert np.all(image == image[0, 0])
