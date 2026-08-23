"""Kente decal: a garment patch painted with the procedural weave
(`meshforge.textile`).

Mirrors `meshforge.collar`'s decal pattern — offset a thin duplicate of a
mesh region along its own normals, give it dedicated UVs and material, and
let the caller reuse the body's own skin weights at those source vertices
(`source_vertex_index`), so it deforms with the rig exactly like
`collar_text` does. See `owl_pipeline.build_owl` around the
`collarmod.build_collar_decal` call for the wiring pattern this follows:
`decal.primitive.joints/.weights = body_j/body_w[decal.source_vertex_index]`.

Unlike the collar, the "band" here is already a named body region
(`regions.classify` labels `chest`, `body`, `wing_left`, ...) rather than a
hallucinated feature that needs a custom frame fit, so there is no
band-frame-fitting stage. There is also no texture-wrap trick: rather than
rely on the exporter's texture sampler (`meshforge.rigexport` hardcodes
`CLAMP_TO_EDGE` for every material), the weave is simply rendered at
however many repeats look right across the patch and baked into one
non-repeating image — the same choice `build_collar_decal` makes for its
text image.

Two attempts came before this one, both wrong in ways only the real owl
(not the synthetic test torso) exposed:

1. A hand-rolled cylindrical (theta, height) projection, copying the
   collar's coordinate system, used directly as the mesh's UVs. The
   collar's math assumes a true cylinder, which the neck approximately is
   and the belly is not — a doubly-curved, egg-shaped surface stretches
   badly wherever it curves away from a single global projection axis.
   Visible result: a warped, pinched patch of fabric.
2. Fixing the geometry with `retopo.unwrap` (xatlas chart+pack — the same
   low-distortion unwrap the whole body gets) and then reusing its raw
   output UVs to sample directly into a flat, pre-rendered weave image.
   That fixed the pinching but produced something worse: `meshforge.bake`
   says exactly why, in its own module docstring — "TRELLIS-style atlases
   (and xatlas output) are a jigsaw of small charts, so authoring anything
   in 2D UV space directly is hopeless." A curved band-shaped region needs
   many small charts to stay low-distortion, and each one landed on an
   effectively random, non-contiguous patch of the pre-rendered image, so
   neighbouring triangles in 3D showed unrelated fragments of the
   pattern — visible as torn, frayed-looking streaks.

The fix `meshforge.bake` itself prescribes: keep xatlas for the geometry
(it minimizes per-triangle distortion, which is genuinely what fixes the
pinch), but stop treating its UVs as a place to sample a picture from.
Instead, bake the *texture* by 3D position — `bake_position_map` gives
every atlas texel the real surface point it covers, `cylindrical_coords`
turns that into a smooth (theta, height) pair, and `textile.weave_color_at`
(the same weave logic `generate_weave` uses, generalized to take that
scattered, non-grid coordinate set) paints each texel from a single
continuous pattern in 3D. The pattern is smooth and correct on the
surface; only the polygon boundaries between xatlas charts differ from a
literal single unbroken image, which for a small repeating weave with no
long-range readability requirement (unlike the collar's text) is what
tolerates chart seams — not the raw-UV image-sampling version.
"""
from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import distance_transform_edt

from meshforge import retopo
from meshforge.bake import bake_position_map, cylindrical_coords, dilate_into_padding
from meshforge.regions import RegionMap
from meshforge.rigexport import MaterialSpec, PrimitiveSpec
from meshforge.textile import WeaveParams, to_image, weave_color_at, weave_height_at
from meshforge.weights import graph_distance_to_other_label, vertex_adjacency


class KenteDecal(NamedTuple):
    primitive: PrimitiveSpec
    source_vertex_index: np.ndarray     # (n_decal,) index into the source mesh vertices (for weights)
    info: dict


def _nearest_valid_fill(texture: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill every texel outside `mask` with the value of its single
    nearest `mask`-True texel — exact nearest-neighbor propagation, not a
    ring-averaged blend of everything reachable within some padding.

    This is the correct gutter fill for *vector* data (a normal map):
    `bake.dilate_into_padding` was written for color, where blending two
    unrelated neighbouring pattern colors across a chart gap still reads
    as "some plausible fabric color" (and is the standard "extend the
    border pixels outward" technique described for texture padding in,
    e.g., Substance Painter's own docs). A normal map has no such
    tolerance: averaging two unrelated *directions* from different xatlas
    charts produces a physically meaningless direction, not merely an
    imprecise one. Measured on the real owl: this is what caused the
    broad, wrong-colored smeared streaks in the baked normal texture,
    confirmed by pulling the actual baked image out of the GLB and
    inspecting it flat rather than only judging the lit render. Unlike
    `dilate_into_padding`'s fixed ring count, this fills the *entire*
    uncovered area in one pass regardless of gap size — the "infinite
    padding" option texture-baking tools offer for exactly this reason."""
    if mask.all():
        return texture
    _, (iy, ix) = distance_transform_edt(~mask, return_indices=True)
    return texture[iy, ix]


def _face_tangent_basis(vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray):
    """Per-face (T, B, N) in object space, from each triangle's own UVs —
    the exact basis a glTF tangent-space normal map is interpreted
    against. Needed because that basis is NOT the same thing as "the
    surface normal plus some fixed up vector": xatlas is free to rotate
    each chart's UVs independently, so two adjacent triangles across a
    chart seam can have unrelated tangent directions even though they're
    physically next to each other.

    T is the direction of increasing u. B is the direction of *decreasing*
    v — "+Y is up" in the glTF spec's words, up being toward the top of
    the image, and glTF's v grows downward. Verified against the renderer
    this project ships with (Three.js r160, `tools/three_bundle.js`): its
    shader's bitangent follows increasing v, and its GLTFLoader negates
    `normalScale.y` for materials without a TANGENT attribute, which is
    exactly the flip that makes a decreasing-v green channel light
    correctly. An earlier version returned `cross(N, T)` unconditionally,
    which equals this only where the UV chart is mirrored (the robe's
    single chart is; xatlas charts may or may not be), so the sign of the
    relief could have flipped from chart to chart."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    uv0, uv1, uv2 = uvs[faces[:, 0]], uvs[faces[:, 1]], uvs[faces[:, 2]]
    e1, e2 = v1 - v0, v2 - v0
    duv1, duv2 = uv1 - uv0, uv2 - uv0
    denom = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]
    f = np.where(np.abs(denom) > 1e-12, 1.0 / np.where(denom == 0, 1.0, denom), 0.0)
    T = f[:, None] * (duv2[:, 1:2] * e1 - duv1[:, 1:2] * e2)
    B = f[:, None] * (duv1[:, 0:1] * e2 - duv2[:, 0:1] * e1)
    N = np.cross(e1, e2)
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    T = T - N * np.sum(T * N, axis=1, keepdims=True)  # Gram-Schmidt against N
    Tn = np.linalg.norm(T, axis=1, keepdims=True)
    fallback = np.array([1.0, 0.0, 0.0])
    T = np.where(Tn > 1e-9, T / np.maximum(Tn, 1e-9), fallback)
    # cross(N, T) is the increasing-v direction where the UV winding matches
    # the 3D winding (denom > 0) and the decreasing-v direction where the
    # chart is mirrored (denom < 0); B must be decreasing-v either way
    handedness = np.where(denom >= 0.0, -1.0, 1.0)[:, None]
    B = np.cross(N, T) * handedness
    return T, B, N


def _drop_degenerate_faces(
    local_vertices: np.ndarray, used: np.ndarray, local_faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove zero-area faces from a sub-mesh selection and drop whatever
    vertices that leaves unreferenced — same cleanup `retopo.weld`/
    `retopo.decimate` already do with `nondegenerate_faces()` +
    `remove_unreferenced_vertices()`, but done explicitly (not via those
    in-place trimesh methods) so the caller also gets back a correct
    `used` mapping to the *original* mesh's vertex indices, needed
    afterward (e.g. for `source_vertex_index`, body skin weight lookup),
    which `remove_unreferenced_vertices()` re-indexing in place would
    silently break.

    Takes and returns *local* vertex positions explicitly (not `V[used]`
    reconstructed by the caller) so this is safe to call again after a
    step that only moves positions without changing vertex count/order,
    like Laplacian smoothing — measured on the owl's sash: smoothing a
    narrow, sparse selection can itself collapse a thin tip into a new
    degenerate triangle that wasn't there in the original selection, so
    one cleanup pass before smoothing isn't sufficient on its own."""
    keep_faces = trimesh.Trimesh(vertices=local_vertices, faces=local_faces, process=False).nondegenerate_faces()
    local_faces = local_faces[keep_faces]
    referenced = np.zeros(len(used), dtype=bool)
    referenced[np.unique(local_faces)] = True
    new_used = used[referenced]
    new_vertices = local_vertices[referenced]
    remap = -np.ones(len(referenced), dtype=np.int64)
    remap[referenced] = np.arange(referenced.sum())
    new_faces = remap[local_faces]
    return new_vertices, new_used, new_faces


def _repair_zero_normals(normals: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Replace any near-zero-length row with a safe fallback direction:
    outward from the local centroid.

    Measured on the owl's sash: `nondegenerate_faces()` (checked in
    `_drop_degenerate_faces`, both before and after smoothing) never
    flags the actual cause here, because it isn't a degenerate face —
    `trimesh.vertex_normals` area-weight-averages a vertex's *adjacent*
    face normals, and heavy smoothing (the default 8 iterations) on a
    band this narrow can fold the strip's cross-section enough that two
    individually valid, adjacent faces end up with nearly opposite
    normals, cancelling to near-zero in their shared vertex's average.
    Every fix aimed at "the face is degenerate" (there were three
    attempts, in the sash's own history) missed this because neither
    face actually is; the defect only exists in the *averaged* vertex
    value, so this repairs that value directly instead of chasing another
    upstream face-level cause."""
    lengths = np.linalg.norm(normals, axis=1)
    bad = lengths < 0.5
    if not bad.any():
        return normals
    centroid = positions.mean(axis=0)
    fallback = positions[bad] - centroid
    fallback_lengths = np.linalg.norm(fallback, axis=1, keepdims=True)
    fallback = fallback / np.maximum(fallback_lengths, 1e-9)
    repaired = normals.copy()
    repaired[bad] = fallback
    return repaired


def _laplacian_smooth(mesh: trimesh.Trimesh, iterations: int, factor: float) -> trimesh.Trimesh:
    """Relax vertices toward their neighbour average, `factor` of the way,
    `iterations` times — knocks down fine surface detail (measured on the
    real owl: the belly carries a raised circuit-trace emboss, part of
    the brand mark, faintly visible even in the flat reference art) while
    leaving the belly's own much lower-frequency curvature intact, the
    same principle `collar.flatten_collar`'s relaxation pass and
    `collar.despike_band` both use. Cloth drapes over fine detail like
    that; an offset duplicate of the raw surface (which is what this
    decal otherwise is) faithfully traces it instead, and rendered that
    read as the kente fabric having odd raised veins running through it —
    not a shading bug, actual geometry showing through."""
    V = mesh.vertices.copy()
    adjacency = vertex_adjacency(len(V), mesh.faces)
    deg = np.asarray(adjacency.sum(axis=1)).ravel()
    deg[deg == 0] = 1
    for _ in range(iterations):
        avg = (adjacency @ V) / deg[:, None]
        V = V + factor * (avg - V)
    return trimesh.Trimesh(vertices=V, faces=mesh.faces.copy(), process=False)


def bake_weave_maps(
    posmap,
    px: np.ndarray,
    py: np.ndarray,
    t_dir: np.ndarray,
    b_dir: np.ndarray,
    weave_params: WeaveParams,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    uvs: np.ndarray,
    bake_normal: bool = True,
    bump_strength: float = 0.15,
    dilate: int = 8,
    finite_difference_px: float = 4.0,
) -> tuple["Image.Image", "Image.Image | None"]:
    """Paint the weave into a UV atlas by *position*: `px`/`py` are the
    weave's pixel-unit coordinates of every covered texel (in
    `posmap.mask` order), `t_dir`/`b_dir` the world-space directions in
    which `px`/`py` respectively increase at those texels — (N, 3) or a
    single (3,) direction. Returns (base color image, tangent-space normal
    image or None). Shared by the vest, the sash and the robe so the three
    garments can't drift apart on the parts of this that were hard to get
    right:

      1. `weave_height_at` is piecewise-*constant*, so a naive d=1-texel
         central difference spikes at every band edge — the derivative is
         taken over `finite_difference_px` texels instead (a box-filtered
         derivative).
      2. The circumferential tangent of a cylindrical parameterization
         rotates with theta; a single constant "side" vector goes
         degenerate where the belly's curvature swings far enough around —
         hence per-texel `t_dir`.
      3. `bake.dilate_into_padding` (written for color, where blending two
         unrelated neighbouring pattern colors across a chart gap still
         reads as "some plausible fabric color") was blending unrelated
         *normal directions* from different xatlas charts — physically
         meaningless for a vector field, and visible as broad wrong-colored
         streaks when the baked normal texture was pulled out of the GLB
         and inspected flat. `_nearest_valid_fill` (exact nearest-neighbor
         propagation) replaces it for the normal map.
    """
    h, w = posmap.mask.shape
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    color = np.zeros((h, w, 4), dtype=np.float64)
    color[posmap.mask] = weave_color_at(px, py, weave_params)
    color = dilate_into_padding(color, posmap.mask, padding=dilate)
    img = to_image(color)
    if not bake_normal:
        return img, None

    d = float(finite_difference_px)
    dHdx = (weave_height_at(px + d, py, weave_params) - weave_height_at(px - d, py, weave_params)) / (2 * d)
    dHdy = (weave_height_at(px, py + d, weave_params) - weave_height_at(px, py - d, weave_params)) / (2 * d)

    N_local = posmap.normal[posmap.mask].astype(np.float64)
    t_dir = np.broadcast_to(np.asarray(t_dir, dtype=np.float64), N_local.shape)
    b_dir = np.broadcast_to(np.asarray(b_dir, dtype=np.float64), N_local.shape)
    t_local = t_dir - N_local * np.sum(t_dir * N_local, axis=1, keepdims=True)
    t_local /= np.maximum(np.linalg.norm(t_local, axis=1, keepdims=True), 1e-9)
    b_local = b_dir - N_local * np.sum(b_dir * N_local, axis=1, keepdims=True)
    b_local /= np.maximum(np.linalg.norm(b_local, axis=1, keepdims=True), 1e-9)

    world_n = N_local - bump_strength * dHdx[:, None] * t_local - bump_strength * dHdy[:, None] * b_local
    world_n /= np.maximum(np.linalg.norm(world_n, axis=1, keepdims=True), 1e-9)

    face_T, face_B, face_N = _face_tangent_basis(np.asarray(mesh_vertices, dtype=np.float64),
                                                 np.asarray(mesh_faces), np.asarray(uvs, dtype=np.float64))
    fid = posmap.face_id[posmap.mask]
    tangent_space_n = np.stack([
        np.sum(face_T[fid] * world_n, axis=1),
        np.sum(face_B[fid] * world_n, axis=1),
        np.sum(face_N[fid] * world_n, axis=1),
    ], axis=1)
    tangent_space_n /= np.maximum(np.linalg.norm(tangent_space_n, axis=1, keepdims=True), 1e-9)

    normal_map = np.zeros((h, w, 4), dtype=np.float64)
    normal_map[..., 2] = 1.0  # default "flat" tangent-space normal (0, 0, 1) -> encoded (0.5, 0.5, 1.0)
    normal_map[..., 3] = 1.0
    normal_map[posmap.mask, :3] = tangent_space_n
    normal_map = _nearest_valid_fill(normal_map, posmap.mask)
    normal_img = to_image(normal_map * np.array([0.5, 0.5, 0.5, 1.0]) + np.array([0.5, 0.5, 0.5, 0.0]))
    return img, normal_img


def build_kente_decal(
    mesh: trimesh.Trimesh,
    region_map: RegionMap,
    weave_params: WeaveParams,
    regions_included: Sequence[str] = ("chest", "body"),
    offset: float = 0.006,
    strips_around: float = 10.0,
    blocks_tall: float = 4.0,
    texture_size: int = 1024,
    unwrap_resolution: int = 1024,
    unwrap_padding: int = 4,
    dilate: int = 8,
    erode_rings: int = 1,
    hem_percentile: tuple[float, float] = (2.0, 98.0),
    smooth_iterations: int = 8,
    smooth_factor: float = 0.5,
    bake_normal: bool = True,
    bump_strength: float = 0.15,
    normal_scale: float = 1.0,
    material_name: str = "owl_kente",
) -> KenteDecal:
    """Cut the named regions' faces out as their own sub-mesh, run them
    through `retopo.unwrap` (xatlas chart+pack) for a low-distortion UV,
    offset the result outward along its own normals, and paint it by
    baking `weave_params` from real 3D position — `strips_around` sets how
    many kente strips wrap the region's own measured circumference,
    `blocks_tall` how many blocks span its measured height, independent of
    how xatlas happened to pack the UV atlas. Raises `ValueError` if the
    selected regions cover too little of the mesh to be a garment (fewer
    than 10 faces).

    `erode_rings` shrinks the selection inward from its own label boundary
    before any of that: `region_map.labels` draws the line between "chest"
    and "wing_right" (say) at whatever individual triangle happened to get
    classified which way — a legitimate boundary for skin-weight purposes,
    but not a garment hem, and rendered raw it reads as a jagged, torn
    silhouette wherever the decal's edge is actually visible past the
    neighbouring part (measured on the real owl: unmistakable near the
    wing in a three-quarter view). Eroding a couple of rings back from
    every such boundary tucks the visible edge in rather than exposing it.

    `hem_percentile` does the same job on the top/bottom edge that
    `erode_rings` does on the left/right one, for the same reason: the
    classifier's raw vertical extent is noisy triangle-by-triangle, not a
    hem. Rather than trust it, the selection is clipped to a percentile
    band of the *region's own* measured height — a smooth, single cutoff
    line each way, the same move `collar.measure_band_frame` makes when it
    fits rims to the band's own measured facing instead of using the
    region map's raw (and, there, simply wrong) dark-window bounds.
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)

    sel_v = np.isin(region_map.labels, list(regions_included))
    if erode_rings > 0:
        adjacency = vertex_adjacency(len(V), mesh.faces)
        binary_labels = np.where(sel_v, "in", "out")
        ring = graph_distance_to_other_label(adjacency, binary_labels, max_rings=erode_rings)
        sel_v = sel_v & (ring >= erode_rings)
    if hem_percentile is not None:
        region_only = np.isin(region_map.labels, list(regions_included))
        if region_only.any():
            bmin, bmax = np.array(mesh.bounds)
            f_y = (V[:, 1] - bmin[1]) / (bmax[1] - bmin[1])
            h_lo, h_hi = np.percentile(f_y[region_only], hem_percentile)
            sel_v = sel_v & (f_y >= h_lo) & (f_y <= h_hi)
    face_mask = sel_v[mesh.faces].all(axis=1)
    if face_mask.sum() < 10:
        raise ValueError(f"build_kente_decal: {list(regions_included)} has too few faces")
    faces = mesh.faces[face_mask]
    used = np.unique(faces)
    remap = -np.ones(len(V), dtype=np.int64)
    remap[used] = np.arange(len(used))
    sub_faces = remap[faces]
    sub_mesh = trimesh.Trimesh(vertices=V[used], faces=sub_faces, process=False)

    if smooth_iterations > 0:
        sub_mesh = _laplacian_smooth(sub_mesh, iterations=smooth_iterations, factor=smooth_factor)

    unwrapped = retopo.unwrap(sub_mesh, resolution=unwrap_resolution, padding=unwrap_padding)
    # the same folded-cross-section defect the sash documents on
    # `_repair_zero_normals` shows up here too on the real owl (validate_rig:
    # "normals unit length — range 0.000..1.000" on the vest primitive)
    normals = _repair_zero_normals(unwrapped.normals, unwrapped.mesh.vertices)
    verts = unwrapped.mesh.vertices + normals * offset
    uvs = unwrapped.uvs
    # source_index is into sub_mesh's own vertices; compose with `used` to
    # land back in the *original* mesh's vertex space, same double
    # indirection owl_pipeline.py already uses for the body primitive
    # itself (`body_j[unwrapped.source_index]`).
    source_vertex_index = used[unwrapped.source_index]

    # bake by position, not by the raw atlas UV (see module docstring):
    # every texel of xatlas's own packed layout gets its real 3D point,
    # a smooth (theta, height) around a vertical axis through the region's
    # own centre, and a weave color evaluated continuously from that —
    # so the pattern is unbroken across chart boundaries even though the
    # UV layout backing it is a jigsaw of many small charts.
    posmap = bake_position_map(unwrapped.mesh.vertices, unwrapped.mesh.faces, uvs,
                               size=texture_size, vertex_normals=normals)
    centre = unwrapped.mesh.vertices.mean(axis=0)
    covered = posmap.position[posmap.mask]
    theta, height, _radius = cylindrical_coords(
        covered, axis_point=np.array([centre[0], 0.0, centre[2]]),
        axis_dir=np.array([0.0, 1.0, 0.0]), forward=np.array([0.0, 0.0, 1.0]),
    )
    period_x = weave_params.strip_px * weave_params.strip_cycle
    period_y = weave_params.block_px
    px_per_radian = (period_x * strips_around) / (2 * np.pi * weave_params.strip_cycle)
    h_span = max(float(height.max() - height.min()), 1e-6)
    px_per_height = (period_y * blocks_tall) / h_span

    # the circumferential tangent rotates with theta (see bake_weave_maps);
    # "up" is world +Y, both projected onto the local tangent plane there
    cos_t, sin_t = np.cos(theta)[:, None], np.sin(theta)[:, None]
    t_dir = -sin_t * np.array([0.0, 0.0, 1.0])[None, :] + cos_t * np.array([1.0, 0.0, 0.0])[None, :]
    img, normal_img = bake_weave_maps(
        posmap, theta * px_per_radian, height * px_per_height, t_dir, np.array([0.0, 1.0, 0.0]),
        weave_params, unwrapped.mesh.vertices, unwrapped.mesh.faces, uvs,
        bake_normal=bake_normal, bump_strength=bump_strength, dilate=dilate,
    )

    mat = MaterialSpec(name=material_name, base_color_image=img, image_format="PNG",
                       roughness=0.85, metallic=0.0, double_sided=False,
                       normal_image=normal_img, normal_scale=normal_scale)
    prim = PrimitiveSpec(name="kente", vertices=verts, faces=unwrapped.mesh.faces, normals=normals,
                         uvs=uvs, material=mat)
    info = {
        "faces": int(len(unwrapped.mesh.faces)), "vertices": int(len(source_vertex_index)),
        "regions": list(regions_included), "colorway": weave_params.colorway.name,
        "image": img.size, "coverage": round(float(posmap.mask.mean()), 3),
        "theta_range_deg": [round(float(np.degrees(theta.min())), 1), round(float(np.degrees(theta.max())), 1)],
    }
    return KenteDecal(primitive=prim, source_vertex_index=source_vertex_index, info=info)


def _point_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perpendicular distance and progress-along-[0, 1] of each row of
    `points` to the segment a->b."""
    ab = b - a
    ab_len2 = max(float(ab @ ab), 1e-12)
    t = np.clip(((points - a) @ ab) / ab_len2, 0.0, 1.0)
    proj = a + t[:, None] * ab
    dist = np.linalg.norm(points - proj, axis=1)
    return dist, t


def _sash_anchors(
    mesh: trimesh.Trimesh,
    region_map: RegionMap,
    shoulder_regions: Sequence[str],
    hip_regions: Sequence[str],
    shoulder_side: str,
    hip_side: str,
    shoulder_height_percentile: float,
    hip_height_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Two anchor points measured from the mesh's own geometry, not
    guessed coordinates: the shoulder end is the centroid of
    `shoulder_regions` vertices on `shoulder_side` of the collar's own
    measured centre, above `shoulder_height_percentile` of that set's own
    height range; the hip end is the mirror idea on `hip_side`, below
    `hip_height_percentile`.

    Also biased to the front hemisphere (`collar["centre_xz"]`'s own z,
    the same reference `collar.py` measures the neck band's front facing
    against): the first version of this only filtered by left/right and
    high/low, and since `chest`/`body` wrap the full 360 degrees, that
    left front/back unconstrained — measured on the real owl, both
    anchors landed with negative z (the back), so the whole sash rendered
    across the back, mostly hidden behind the wing and feather ridges,
    and invisible from the front the owl is actually viewed from."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    bmin, bmax = np.array(mesh.bounds)
    f = (V - bmin) / (bmax - bmin)
    # region_map.collar["centre_xz"] is computed (regions.measure_collar)
    # from *bbox-fraction* coordinates, not world ones — comparing it
    # directly against raw V would silently compare the wrong units (this
    # was in fact a real bug in the first version of this function, not
    # just the missing front/back filter). Compare against `f`, the
    # per-vertex fraction array already in the same space, instead.
    cx, cz = region_map.collar["centre_xz"]

    def _anchor(regions_: Sequence[str], side: str, height_percentile: float, above: bool) -> np.ndarray:
        sel = np.isin(region_map.labels, list(regions_))
        sided = sel & ((f[:, 0] < cx) if side == "left" else (f[:, 0] > cx))
        if sided.any():
            sel = sided
        fronted = sel & (f[:, 2] > cz)
        if fronted.any():
            sel = fronted
        if not sel.any():
            raise ValueError(f"_sash_anchors: no vertices in {list(regions_)}")
        cutoff = np.percentile(f[sel, 1], height_percentile)
        band = sel & ((f[:, 1] >= cutoff) if above else (f[:, 1] <= cutoff))
        if not band.any():
            band = sel
        return V[band].mean(axis=0)

    shoulder = _anchor(shoulder_regions, shoulder_side, shoulder_height_percentile, above=True)
    hip = _anchor(hip_regions, hip_side, hip_height_percentile, above=False)
    return shoulder, hip


def build_kente_sash_decal(
    mesh: trimesh.Trimesh,
    region_map: RegionMap,
    weave_params: WeaveParams,
    shoulder_regions: Sequence[str] = ("chest",),
    hip_regions: Sequence[str] = ("body",),
    shoulder_side: str = "left",
    hip_side: str = "right",
    shoulder_height_percentile: float = 55.0,
    hip_height_percentile: float = 15.0,
    band_half_width: float | None = None,
    band_half_width_frac: float = 0.16,
    base_regions: Sequence[str] = ("chest", "body", "wing_left", "wing_right"),
    offset: float = 0.006,
    strips_across: float = 3.0,
    blocks_along: float = 6.0,
    texture_size: int = 1024,
    unwrap_resolution: int = 1024,
    unwrap_padding: int = 4,
    dilate: int = 8,
    erode_rings: int = 2,
    smooth_iterations: int = 8,
    smooth_factor: float = 0.5,
    bake_normal: bool = True,
    bump_strength: float = 0.15,
    normal_scale: float = 1.0,
    material_name: str = "owl_kente_sash",
) -> KenteDecal:
    """The diagonal-stole alternative to `build_kente_decal`'s vest: a
    straight band from a measured shoulder point to the opposite hip,
    instead of a whole already-labelled region. Everything downstream of
    selection — smoothing, `retopo.unwrap`, offset, position-baked color
    and (optional) tangent-space normal, `_nearest_valid_fill` — mirrors
    `build_kente_decal` exactly and carries the same fixes forward. The
    one structural difference: a straight segment's own direction is
    constant everywhere, unlike the vest's cylindrical wrap where the
    circumferential tangent rotates with angle (`build_kente_decal`'s
    bug (2) in its own docstring) — so the sash's tangent-space normal
    math doesn't need that per-point rotation at all, only the same
    tangent-plane projection already used there for the "up" axis.

    `band_half_width` is in world units; if not given, it's
    `band_half_width_frac` of the shoulder-to-hip distance, so it scales
    with the body instead of being an arbitrary constant. Raises
    `ValueError` if the band selects too little of the mesh to be a
    garment (fewer than 10 faces)."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    shoulder, hip = _sash_anchors(mesh, region_map, shoulder_regions, hip_regions,
                                  shoulder_side, hip_side, shoulder_height_percentile, hip_height_percentile)
    segment_len = float(np.linalg.norm(hip - shoulder))
    if segment_len < 1e-9:
        raise ValueError("build_kente_sash_decal: shoulder and hip anchors coincide")
    half_width = band_half_width if band_half_width is not None else segment_len * band_half_width_frac

    base_sel = np.isin(region_map.labels, list(base_regions))
    dist, t = _point_segment_distance(V, shoulder, hip)
    sel_v = base_sel & (dist <= half_width)
    if erode_rings > 0:
        adjacency = vertex_adjacency(len(V), mesh.faces)
        binary_labels = np.where(sel_v, "in", "out")
        ring = graph_distance_to_other_label(adjacency, binary_labels, max_rings=erode_rings)
        sel_v = sel_v & (ring >= erode_rings)
    face_mask = sel_v[mesh.faces].all(axis=1)
    if face_mask.sum() < 10:
        raise ValueError("build_kente_sash_decal: the band selects too few faces")
    faces = mesh.faces[face_mask]
    used = np.unique(faces)
    remap = -np.ones(len(V), dtype=np.int64)
    remap[used] = np.arange(len(used))
    sub_faces = remap[faces]

    # a small, narrow selection like this band is far more likely than the
    # vest's whole-region one to catch a genuinely degenerate, near-zero-area
    # sliver triangle right at its distance-threshold edge — measured on the
    # real owl: three vertices landing within 1e-6 of each other, each
    # belonging to exactly one face, so trimesh's vertex_normals for them
    # came out as the zero vector (caught by validate_rig's "normals unit
    # length" check, range 0.000..1.000 instead of all 1.0).
    sub_verts, used, sub_faces = _drop_degenerate_faces(V[used], used, sub_faces)
    sub_mesh = trimesh.Trimesh(vertices=sub_verts, faces=sub_faces, process=False)

    if smooth_iterations > 0:
        sub_mesh = _laplacian_smooth(sub_mesh, iterations=smooth_iterations, factor=smooth_factor)
        # smoothing can itself collapse a thin tip of a narrow band into a
        # new degenerate triangle that wasn't there before it — measured:
        # the pre-smoothing cleanup above alone left the owl's sash with
        # 0 bad normals when smooth_iterations=0, but 3 reappeared at the
        # default smooth_iterations=8, so this isn't optional here. Vertex
        # count/order is unchanged by smoothing (only positions move), so
        # `used` is still a valid mapping to re-clean against.
        sub_verts, used, sub_faces = _drop_degenerate_faces(sub_mesh.vertices, used, sub_mesh.faces)
        sub_mesh = trimesh.Trimesh(vertices=sub_verts, faces=sub_faces, process=False)

    unwrapped = retopo.unwrap(sub_mesh, resolution=unwrap_resolution, padding=unwrap_padding)
    # see _repair_zero_normals's own docstring: this is not another
    # degenerate-face case (nondegenerate_faces() genuinely can't see the
    # actual defect, which only exists in the post-smoothing *averaged*
    # vertex normal, not any single face) — a real, if unusual, geometric
    # fold from smoothing a narrow band, repaired directly here.
    normals = _repair_zero_normals(unwrapped.normals, unwrapped.mesh.vertices)
    verts = unwrapped.mesh.vertices + normals * offset
    uvs = unwrapped.uvs
    source_vertex_index = used[unwrapped.source_index]

    along_dir = (hip - shoulder) / segment_len
    across_dir = np.cross(along_dir, np.array([0.0, 1.0, 0.0]))
    across_norm = np.linalg.norm(across_dir)
    if across_norm < 1e-9:
        across_dir = np.array([1.0, 0.0, 0.0])
    else:
        across_dir = across_dir / across_norm

    posmap = bake_position_map(unwrapped.mesh.vertices, unwrapped.mesh.faces, uvs,
                               size=texture_size, vertex_normals=normals)
    covered = posmap.position[posmap.mask]
    along = (covered - shoulder) @ along_dir
    across = (covered - shoulder) @ across_dir

    period_x = weave_params.strip_px * weave_params.strip_cycle
    period_y = weave_params.block_px
    across_span = max(float(across.max() - across.min()), 1e-6)
    along_span = max(float(along.max() - along.min()), 1e-6)
    px_per_across = (period_x * strips_across) / across_span
    px_per_along = (period_y * blocks_along) / along_span

    # a straight segment's direction is constant everywhere (unlike the
    # vest's rotating circumferential tangent) — bake_weave_maps projects
    # both directions onto each texel's own tangent plane
    img, normal_img = bake_weave_maps(
        posmap, across * px_per_across, along * px_per_along, across_dir, along_dir,
        weave_params, unwrapped.mesh.vertices, unwrapped.mesh.faces, uvs,
        bake_normal=bake_normal, bump_strength=bump_strength, dilate=dilate,
    )

    mat = MaterialSpec(name=material_name, base_color_image=img, image_format="PNG",
                       roughness=0.85, metallic=0.0, double_sided=False,
                       normal_image=normal_img, normal_scale=normal_scale)
    prim = PrimitiveSpec(name="kente_sash", vertices=verts, faces=unwrapped.mesh.faces, normals=normals,
                         uvs=uvs, material=mat)
    info = {
        "faces": int(len(unwrapped.mesh.faces)), "vertices": int(len(source_vertex_index)),
        "colorway": weave_params.colorway.name, "image": img.size,
        "coverage": round(float(posmap.mask.mean()), 3), "half_width": round(half_width, 4),
        "segment_len": round(segment_len, 4), "shoulder": [round(float(c), 4) for c in shoulder],
        "hip": [round(float(c), 4) for c in hip],
    }
    return KenteDecal(primitive=prim, source_vertex_index=source_vertex_index, info=info)
