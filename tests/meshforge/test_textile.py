"""Tests for meshforge.textile — the procedural kente weave.

These check the properties that matter for a *texture* rather than for a
picture: exact tiling, determinism, variation across seeds, an opaque
alpha channel, and that every pixel is one of the colorway's declared
colors (a piecewise-constant generator has no business producing an
unplanned color).
"""
import numpy as np
import pytest

from meshforge.textile import ASANTE_GOLD, EWE_ADANUDO, WeaveParams, generate_weave, weave_height_at

SMALL = WeaveParams(colorway=ASANTE_GOLD, strip_px=8, strip_cycle=2, block_px=12, seed=0)


def test_shape_and_dtype():
    arr = generate_weave(16, 12, SMALL)
    assert arr.shape == (12, 16, 4)
    assert arr.dtype == np.float64


def test_alpha_is_opaque():
    arr = generate_weave(16, 12, SMALL)
    assert np.all(arr[..., 3] == 1.0)


def test_rejects_non_multiple_dimensions():
    with pytest.raises(ValueError):
        generate_weave(15, 12, SMALL)  # width not a multiple of strip_px * strip_cycle
    with pytest.raises(ValueError):
        generate_weave(16, 10, SMALL)  # height not a multiple of block_px


def test_tiles_exactly_in_both_axes():
    period_x = SMALL.strip_px * SMALL.strip_cycle
    period_y = SMALL.block_px
    arr = generate_weave(period_x * 2, period_y * 2, SMALL)

    left, right = arr[:, :period_x], arr[:, period_x:]
    assert np.array_equal(left, right), "not periodic across the x seam"

    top, bottom = arr[:period_y, :], arr[period_y:, :]
    assert np.array_equal(top, bottom), "not periodic across the y seam"


def test_deterministic_for_same_seed():
    a = generate_weave(32, 24, SMALL)
    b = generate_weave(32, 24, SMALL)
    assert np.array_equal(a, b)


def test_varies_across_seeds():
    a = generate_weave(32, 24, SMALL)
    b = generate_weave(32, 24, dataclasses_replace(SMALL, seed=1))
    assert not np.array_equal(a, b)


def test_varies_across_colorways():
    a = generate_weave(32, 24, SMALL)
    b = generate_weave(32, 24, dataclasses_replace(SMALL, colorway=EWE_ADANUDO))
    assert not np.array_equal(a, b)


def test_every_pixel_is_a_declared_colorway_color():
    arr = generate_weave(32, 24, SMALL)
    declared = np.array(
        [*SMALL.colorway.ground, SMALL.colorway.accent, SMALL.colorway.motif],
        dtype=np.float64,
    )
    pixels = arr[..., :3].reshape(-1, 3)
    # every pixel must match one declared color within float rounding
    close = np.all(
        np.isclose(pixels[:, None, :], declared[None, :, :], atol=1e-9), axis=-1
    )
    assert np.all(close.any(axis=-1)), "found a pixel color not in the colorway"


def dataclasses_replace(params: WeaveParams, **kwargs) -> WeaveParams:
    import dataclasses

    return dataclasses.replace(params, **kwargs)


def test_height_is_grayscale_in_declared_levels():
    x, y = np.meshgrid(np.arange(16), np.arange(24))
    h = weave_height_at(x, y, SMALL)
    assert h.shape == (24, 16)
    assert set(np.unique(h)) <= {0.35, 0.55, 0.80}


def test_height_and_color_agree_on_where_the_motif_is():
    # wherever the color equals the colorway's motif color, height must be
    # the raised level — they're two views of the same band structure and
    # must not disagree about where the motif actually is.
    x, y = np.meshgrid(np.arange(32), np.arange(24))
    color = generate_weave(32, 24, SMALL)
    h = weave_height_at(x, y, SMALL)
    is_motif_color = np.all(np.isclose(color[..., :3], np.asarray(SMALL.colorway.motif)), axis=-1)
    assert np.all(h[is_motif_color] == 0.80)


def test_height_tiles_exactly_like_color_does():
    period_x = SMALL.strip_px * SMALL.strip_cycle
    period_y = SMALL.block_px
    x, y = np.meshgrid(np.arange(period_x * 2), np.arange(period_y * 2))
    h = weave_height_at(x, y, SMALL)
    assert np.array_equal(h[:, :period_x], h[:, period_x:])
    assert np.array_equal(h[:period_y, :], h[period_y:, :])
