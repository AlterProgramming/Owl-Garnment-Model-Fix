"""The AI-CCORE owl recipe: hi-res TRELLIS export -> rigged, textured,
animated GLB (OwlV1 skeleton, six baked clips, eyes v2, AI-CCORE collar).

    python3 -m meshforge.owl_pipeline --hires "~/Downloads/AI-CCORE Owl.glb" \
        --out viewer/assets/owl.glb --cache .owl_cache --previews out/owl_previews

Every slow stage caches its result under --cache keyed by its inputs, so
re-running after changing only the paint or the clips takes seconds.
See docs/superpowers/specs/2026-08-21-owl-full-body-rig-design.md.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

from meshforge import clips as clipmod
from meshforge import collar as collarmod
from meshforge import paint, regions, retopo, weights as weightmod
from meshforge.bake import PositionMap, dilate_into_padding
from meshforge.eyes2 import EyeParams, build_eye_set
from meshforge.fk import Joint
from meshforge.rigexport import AnimationSpec, MaterialSpec, PrimitiveSpec, write_rigged_glb
from meshforge.validate_rig import validate

EYE_SEEDS = [np.array([-0.097, 0.354, 0.334]), np.array([0.246, 0.352, 0.331])]  # left (−X), right (+X)
REF_FACES = 120_000   # the master decimation; small builds borrow its collar measurement for the tunic's neckline


def _log(msg: str, t0: float):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def _cache_key(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode())
    return h.hexdigest()[:12]


class Stage:
    def __init__(self, cache_dir: Path | None):
        self.dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, name: str, key: str, compute):
        if self.dir is None:
            return compute()
        path = self.dir / f"{name}_{key}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        value = compute()
        with open(path, "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        return value


def build_owl(
    hires_path: str,
    out_path: str,
    cache_dir: str | None = None,
    target_faces: int = 120_000,
    texture_size: int = 2048,
    n_samples: int = 3_000_000,
    previews_dir: str | None = None,
    text: str = "AI-CCORE",
    eye_params: EyeParams = EyeParams(),
    jpeg_quality: int = 88,
    decal_height: int = 512,
    kente: bool = False,
    kente_colorway: str = "asante_gold",
    kente_seed: int = 0,
    kente_regions: tuple[str, ...] = ("chest", "body"),
    kente_style: str = "tunic",
    kente_normal: bool = False,
    robe_params: "RobeParams | None" = None,
    robe_explicit: "set[str] | None" = None,
    cloth_params: "ClothParams | None" = None,
    cloth_explicit: "set[str] | None" = None,
    sleeve: bool = True,
    sleeve_params: "SleeveParams | None" = None,
    sleeve_explicit: "set[str] | None" = None,
    wrap_params: "WrapParams | None" = None,
    beads: bool = False,
    bead_count: int = 14,
) -> dict:
    t0 = time.time()
    stage = Stage(Path(cache_dir) if cache_dir else None)
    hires_path = os.path.expanduser(hires_path)
    src_stat = os.stat(hires_path)
    src_key = _cache_key(hires_path, src_stat.st_size, int(src_stat.st_mtime))

    # 1. geometry ---------------------------------------------------------
    def _geometry():
        src = retopo.load_mesh(hires_path)
        welded = retopo.weld(src)
        decimated = retopo.decimate(welded, target_faces=target_faces)
        return decimated
    base = stage.get("decimated", _cache_key(src_key, target_faces), _geometry)
    _log(f"base mesh: {len(base.vertices)} verts / {len(base.faces)} faces", t0)

    # 2. color samples of the hi-res surface -------------------------------
    def _sampler():
        src = retopo.load_mesh(hires_path)
        s = retopo.SurfaceColorSampler(src, n_samples=n_samples)
        return {"points": s.points.astype(np.float32), "colors": s.colors}
    samp = stage.get("samples", _cache_key(src_key, n_samples), _sampler)
    sampler = retopo.SurfaceColorSampler.__new__(retopo.SurfaceColorSampler)
    sampler.points = samp["points"].astype(np.float64)
    sampler.colors = samp["colors"]
    sampler.tree = cKDTree(sampler.points)
    _log(f"color sampler: {len(sampler.points)} samples", t0)

    # 3. regions on the base mesh -----------------------------------------
    base_colors = sampler.query(base.vertices, k=3)
    region_map = regions.classify(base, base_colors)
    _log(f"regions: {region_map.stats}", t0)

    # 4. measure the band, then flatten the embossed lettering inside it ----
    # The frame is measured *before* flattening and everything downstream
    # rides it: the flatten's movable set, the despike, the texture fill and
    # the decal. Flattening against the coarse dark-rim bounds instead
    # dragged chest vertices onto the band's radius and tore the seam.
    band_frame = collarmod.measure_band_frame(base, region_map, base_colors)
    _log(f"band frame: {band_frame.diagnostics}", t0)
    flat, moved = collarmod.flatten_collar(base, region_map, vertex_colors=base_colors,
                                           frame=band_frame)
    _log(f"collar flattened: {int(moved.sum())} vertices relaxed", t0)
    flat, despiked = collarmod.despike_band(flat, region_map, band_frame)
    _log(f"band despiked: {despiked} ridge vertices clamped", t0)

    # 5. unwrap + bake ----------------------------------------------------
    def _unwrap():
        return retopo.unwrap(flat, resolution=texture_size, padding=6)
    unwrapped = stage.get("unwrapped", _cache_key(src_key, target_faces, texture_size, "flat6"), _unwrap)
    _log(f"unwrapped: {len(unwrapped.mesh.vertices)} split verts", t0)

    def _bake():
        # no gutter dilation here: paint first, dilate after, otherwise the
        # gutters keep pre-paint colors and bleed them back in at chart seams
        baked = retopo.bake_texture(unwrapped, sampler, size=texture_size, padding=0, k=3)
        return {"color": baked.color, "posmap": baked.position_map}
    bk = stage.get("baked", _cache_key(src_key, target_faces, texture_size, n_samples, "flat6"), _bake)
    color = bk["color"]
    posmap: PositionMap = bk["posmap"]
    _log(f"baked texture: coverage {posmap.mask.mean():.2f}", t0)

    # 6. paint ------------------------------------------------------------
    texel_lab = paint.texel_labels(posmap, flat.vertices, region_map.labels)
    color, collar_info = paint.paint_collar_text(color, posmap, region_map, np.array(flat.bounds), flat.vertices,
                                                 text=None, texel_label_map=texel_lab,
                                                 frame=band_frame, piping_frac=0.10)
    _log(f"collar filled: {collar_info}", t0)

    bmin, bmax = flat.bounds
    size = bmax - bmin
    fP = (sampler.points - bmin) / size
    win = (fP[:, 1] > 0.60) & (fP[:, 1] < 0.78) & (fP[:, 2] > 0.70)
    vtree = cKDTree(flat.vertices)
    VN = np.asarray(flat.vertex_normals)

    def normal_at(p):
        _, i = vtree.query(p, k=12)
        n = VN[i].mean(axis=0)
        return n / np.linalg.norm(n)

    discs = paint.find_eye_discs(sampler.points[win], sampler.colors[win], seeds=EYE_SEEDS, surface_normals_at=normal_at)
    disc_by_side = {"left": discs[0], "right": discs[1]}
    for side, dsc in disc_by_side.items():
        _log(f"eye {side}: centre {np.round(dsc.center, 3)} iris {dsc.iris_radius:.3f} disc {dsc.disc_radius:.3f}", t0)
    color = paint.blank_eye_discs(color, posmap, discs)
    color = paint.grade(color, saturation=1.06, contrast=1.02)
    color = dilate_into_padding(color, posmap.mask, padding=16)
    texture = paint.to_image(color)
    _log("eyes blanked + graded + gutters dilated", t0)

    # face cream per side for the lids: bright unsaturated samples in a ring outside the disc
    face_colors = {}
    for side, dsc in disc_by_side.items():
        ring = sampler.tree.query_ball_point(dsc.center, dsc.disc_radius * 1.6)
        ring_pts = sampler.points[ring]
        d = np.linalg.norm(ring_pts - dsc.center, axis=1)
        cols = sampler.colors[ring][(d > dsc.disc_radius * 1.15)]
        sat, val = paint._hsv_sat_val(cols)
        cream = cols[(sat < 0.35) & (val > 0.6)]
        face_colors[side] = np.median(cream, axis=0) if len(cream) > 20 else np.array([0.95, 0.92, 0.86])

    # 7. skeleton + weights ------------------------------------------------
    eye_set = build_eye_set(disc_by_side, face_colors, eye_params)
    joints = regions.with_pivots(region_map.joints, eye_set.pivots)
    joint_names = [j.name for j in joints]
    wr = weightmod.harmonic_weights(flat.faces, region_map.labels, regions.BODY_REGION_JOINTS,
                                    rings=regions.DEFAULT_BLEND_RINGS)
    _log(f"weights: {wr.diagnostics['influence_histogram']} free={wr.diagnostics['free_vertices']}", t0)

    # body weights (base topology) -> full joint index space, top-4
    full = np.zeros((len(flat.vertices), len(joint_names)), dtype=np.float32)
    for k, name in enumerate(wr.joint_names):
        full[:, joint_names.index(name)] = wr.weights[:, k]

    def top4(W):
        order = np.argsort(-W, axis=1)[:, :4]
        w = np.take_along_axis(W, order, axis=1)
        s = w.sum(axis=1, keepdims=True)
        s[s == 0] = 1
        return order.astype(np.uint8), (w / s).astype(np.float32)

    body_j, body_w = top4(full)
    # the clips are needed before the garment: cloth is draped clear of where
    # the legs *go*, not only where they rest
    anims: list[AnimationSpec] = clipmod.default_clips(eye_set.blink_axes, eye_set.blink_close_deg)
    split_idx = unwrapped.source_index
    # the split mesh came from the *flattened* base: rebuild split positions/normals from `flat`
    split_vertices = flat.vertices[split_idx]
    split_normals = np.asarray(flat.vertex_normals)[split_idx]

    body_mat = MaterialSpec(name="owl_body", base_color_image=texture, roughness=0.72, metallic=0.0, jpeg_quality=jpeg_quality)
    primitives = [PrimitiveSpec(
        name="body", vertices=split_vertices, faces=unwrapped.mesh.faces, normals=split_normals, uvs=unwrapped.uvs,
        material=body_mat, joints=body_j[split_idx], weights=body_w[split_idx],
    )]

    # collar decal shares the body weights of its source vertices
    decal = collarmod.build_collar_decal(flat, region_map, text=text, frame=band_frame,
                                         plate_color=collar_info["fill_color"], offset=0.005,
                                         image_height=decal_height)
    dp = decal.primitive
    dp.joints = body_j[decal.source_vertex_index]
    dp.weights = body_w[decal.source_vertex_index]
    primitives.append(dp)
    _log(f"collar decal: {decal.info}", t0)

    # kente: a robe (a real garment mesh, skinned from the nearest torso
    # vertices) or one of the legacy decals (which reuse the body's own
    # weights at their source vertices, same as the collar)
    kente_info = None
    robe_check = None
    sleeve_info = None
    sleeve_check = None
    wrap_gate_args = None
    report_gates = None
    if kente:
        from meshforge import kente as kentemod
        from meshforge.textile import COLORWAYS, WeaveParams

        weave_params = WeaveParams(colorway=COLORWAYS[kente_colorway], seed=kente_seed)
        if kente_style in ("tunic", "robe", "wrap"):
            from meshforge import robe as robemod

            # scale the grid and the chart with the build: the guide build
            # (14k faces, 512 texture) must not carry an 18k-triangle robe
            scale = float(np.clip(np.sqrt(target_faces / 120_000.0), 0.3, 1.0))
            scaled = dict(n_theta=max(48, int(round(144 * scale / 4)) * 4), n_rows=max(24, int(round(64 * scale))),
                          texture_size=(max(1024, min(2048, texture_size)), max(256, min(512, texture_size // 4))))
            # fields the caller set explicitly (`robe_explicit`, the CLI's
            # flags) are never auto-scaled, even when set to the default value
            if robe_params is None:
                robe_params = dataclasses.replace(robemod.RobeParams(), **scaled)
            else:
                keep = {k: v for k, v in scaled.items() if k not in (robe_explicit or set())}
                robe_params = dataclasses.replace(robe_params, **keep)
            rp = dataclasses.replace(robe_params, bake_normal=kente_normal)
            if kente_style in ("tunic", "wrap"):
                from meshforge.drape import ClothParams

                # the floor is higher than the robe's was: folds and a cut
                # armhole need cells, and at 60 x 36 the sleeve's root came
                # out from under a tunic that was fine at full size
                c_scaled = dict(n_u=max(80, int(round(152 * scale / 4)) * 4), n_v=max(44, int(round(84 * scale))),
                                texture_size=(max(1024, min(2048, texture_size)),
                                              max(512, min(1024, texture_size // 2))))
                if cloth_params is None:
                    cloth_params = dataclasses.replace(ClothParams(), **c_scaled)
                else:
                    keep = {k: v for k, v in c_scaled.items() if k not in (cloth_explicit or set())}
                    cloth_params = dataclasses.replace(cloth_params, **keep)
                cloth_params = dataclasses.replace(cloth_params, bake_normal=kente_normal)
            from meshforge import sleeve as sleevemod

            # the sleeve first: the tunic is cut around where the sleeve
            # leaves it (its root ring stays under the tunic's cloth)
            slv = None
            obstacles = None
            if sleeve and kente_style != "wrap":
                sscaled = dict(n_phi=max(24, int(round(48 * scale / 4)) * 4),
                               station_spacing=0.01 / max(scale, 0.3),
                               texture_size=(max(512, min(1024, texture_size // 2)), max(128, min(256, texture_size // 8))))
                if sleeve_params is None:
                    sleeve_params = dataclasses.replace(sleevemod.SleeveParams(), **sscaled)
                else:
                    keep = {k: v for k, v in sscaled.items() if k not in (sleeve_explicit or set())}
                    sleeve_params = dataclasses.replace(sleeve_params, **keep)
                sp = dataclasses.replace(sleeve_params, bake_normal=kente_normal)
                wing_pivot = next(j.pivot for j in joints if j.name == "wing_left")
                slv = sleevemod.build_sleeve(flat, region_map.labels, wing_pivot, weave_params, sp)
                s_along = (slv.primitive.vertices - slv.frame.pivot) @ slv.frame.u
                obstacles = [slv.primitive.vertices[s_along >= slv.info["s_separation"]]]
                _log(f"kente sleeve: {slv.info}", t0)

            # the tunic hangs from the collar's lower rim. A small build (the
            # guide, 14k faces) cannot measure that rim — `measure_band_frame`
            # falls back and the coarse rim reads 0.40 of the height all
            # round, 0.1 under the real rim at the back, which left the
            # sleeve's root bare above the neckline — so builds under half
            # the reference size take the rim measured on the 120k mesh
            # (cached once; it is the same body)
            neck_map, neck_frame = region_map, band_frame
            if target_faces < REF_FACES // 2:
                def _collar_ref():
                    ref = stage.get("decimated", _cache_key(src_key, REF_FACES),
                                    lambda: retopo.decimate(retopo.weld(retopo.load_mesh(hires_path)), target_faces=REF_FACES))
                    ref_colors = sampler.query(ref.vertices, k=3)
                    ref_map = regions.classify(ref, ref_colors)
                    return {"collar": ref_map.collar, "frame": collarmod.measure_band_frame(ref, ref_map, ref_colors)}
                ref = stage.get("collar_ref", _cache_key(src_key, REF_FACES, n_samples), _collar_ref)
                neck_map, neck_frame = region_map._replace(collar=ref["collar"]), ref["frame"]
                _log(f"tunic neckline from the {REF_FACES}-face reference collar: {ref['frame'].diagnostics}", t0)

            # both wings' fused roots are draped over, the hand gets its
            # slit from the rest pose: no posed sweep is needed because the
            # cloth over each root rides that root's joint (weights below)
            if kente_style in ("tunic", "wrap"):
                from meshforge import drape as drapemod
                from meshforge import fk as fkmod

                sweeps = []
                for anim in [*anims, _head_follow_probe()]:
                    if anim.name not in ("hop", "idle", "wave", "tablet_show", "head_follow_probe"):
                        continue
                    dur = max(float(np.max(t.times)) for t in anim.tracks) if anim.tracks else 0.0
                    # the wrap's loop stands off where the limbs *go*: three
                    # samples per clip missed the wave's peaks (measured: 20
                    # pinned vertices inside at t=0.95, none at 0.53/1.07)
                    n_sweep = 13 if kente_style == "wrap" else 4
                    for tm in np.linspace(0.0, dur, n_sweep)[1:]:
                        rot, tr = robemod.clip_pose_at(anim, tm)
                        skin = fkmod.skin_matrices(joints, fkmod.world_matrices(joints, rotations=rot, translations=tr))
                        sweeps.append(fkmod.pose_vertices(flat.vertices, full, joint_names, skin))
                sleeve_mesh = None
                if slv is not None:
                    sleeve_mesh = trimesh.Trimesh(vertices=slv.primitive.vertices, faces=slv.primitive.faces,
                                                  process=False)
                if kente_style == "wrap":
                    from meshforge import wrap as wrapmod

                    wp = wrap_params or wrapmod.WrapParams()
                    if target_faces < REF_FACES // 2:
                        # the guide build: a coarser tail and knot, JPEG cloth textures
                        wp = dataclasses.replace(wp, tail_n_u=max(12, int(round(30 * scale))),
                                                 tail_n_v=max(24, int(round(80 * scale))),
                                                 tail_texture_size=(512, 256), knot_strands=5, image_format="JPEG")
                    wrap = wrapmod.build_kente_wrap(flat, neck_map, weave_params, cloth_params, wp,
                                                    band_frame=neck_frame, sweep_vertices=sweeps)
                    robe = None
                else:
                    robe = drapemod.build_kente_tunic(flat, neck_map, weave_params, cloth_params,
                                                      band_frame=neck_frame, obstacle_points=obstacles,
                                                      obstacle_mesh=sleeve_mesh, sweep_vertices=sweeps)
            else:
                robe = robemod.build_kente_robe(flat, neck_map, weave_params, rp, band_frame=neck_frame,
                                                obstacle_points=obstacles)
            if kente_style == "wrap":
                from meshforge import wrap as wrapmod

                # skinning from the support graph, not from the nearest skin
                y_hip = float(next(j.pivot[1] for j in joints if j.name == "body"))
                rows, cols = wrap.sheet_grid
                W_sheet = wrapmod.support_weights(wrap.sheet.vertices, np.arange(rows * cols), wrap.sheet_grid,
                                                  wrap.pins["sheet"], flat.vertices, full, joint_names, y_hip)
                rows_t, cols_t = wrap.tail_grid
                chest_one = np.zeros((cols_t, len(joint_names)))
                chest_one[:, joint_names.index("chest")] = 1.0
                W_tail = wrapmod.support_weights(wrap.tail.vertices, np.arange(rows_t * cols_t), wrap.tail_grid,
                                                 wrap.pins["tail"], flat.vertices, full, joint_names, y_hip,
                                                 pin_weights=chest_one, periodic=False)
                W_knot = np.zeros((len(wrap.knot.vertices), len(joint_names)))
                W_knot[:, joint_names.index("chest")] = 1.0
                dense = {}
                for prim, Wd in ((wrap.sheet, W_sheet), (wrap.tail, W_tail), (wrap.knot, W_knot)):
                    prim.joints, prim.weights = top4(Wd.astype(np.float32))
                    E = np.zeros_like(Wd)
                    np.put_along_axis(E, prim.joints.astype(np.int64), prim.weights, axis=1)
                    dense[prim.name] = E
                    primitives.append(prim)
                kente_info = wrap.info
                _log(f"kente wrap: support={wrap.info['support']} sheet={wrap.info['sheet']['grid']} "
                     f"tail={wrap.info['tail']['grid']} knot={wrap.info['knot']}", t0)
                cloth_V = np.vstack([wrap.sheet.vertices, wrap.tail.vertices])
                cloth_W = np.vstack([dense["kente_wrap"], dense["kente_wrap_tail"]])
                robe_check = {"rest": robemod.clearance(cloth_V, flat, surface_tree=sampler.tree)}
                _log(f"wrap clearance at rest: {robe_check['rest']}", t0)
                robe_pose_args = (cloth_V, cloth_W)
                wrap_gate_args = {"wrap": wrap, "dense": dense, "params": wp,
                                  "W_all": np.vstack([W_sheet, W_tail, W_knot])}
            else:
                # weights: body + chest, plus each wing for the cloth that lies
                # over its fused root — that cloth rests on the root and lifts
                # with it (the folded wing on `tablet_show`, the raised wing's
                # shoulder on `wave`), so the hand slit stays put relative to
                # the hand and the shoulder never pushes through the cloth.
                # Not the neck (runtime-driven by cursor follow; a neckline does
                # not turn with the head), and not the folded wing's mislabelled
                # tip below 0.28 of the height (it does not move with the wing).
                fy = (flat.vertices[:, 1] - flat.bounds[0][1]) / (flat.bounds[1][1] - flat.bounds[0][1])
                wing_labels = ["wing_right", "wing_left", "wing_left_tip"]
                pool_body = np.isin(region_map.labels, ["chest", "body"])
                pool_wing = pool_body | (np.isin(region_map.labels, wing_labels) & (fy > 0.28))
                W_body, _ = robemod.transfer_body_weights(robe.primitive.vertices, flat.vertices, pool_body, full,
                                                          allowed_joints=np.isin(joint_names, ["body", "chest"]))
                W_wing, _ = robemod.transfer_body_weights(robe.primitive.vertices, flat.vertices, pool_wing, full,
                                                          allowed_joints=np.isin(joint_names, ["body", "chest"] + wing_labels))
                # only cloth at the wings' own height may ride them; the hem
                # below hangs beside the folded wing's static, mislabelled tip
                # and must stay with the body (measured: 5 hem vertices swung
                # into that tip)
                ry = (robe.primitive.vertices[:, 1] - flat.bounds[0][1]) / (flat.bounds[1][1] - flat.bounds[0][1])
                a = np.clip((ry - 0.28) / 0.06, 0.0, 1.0)
                a = (a * a * (3 - 2 * a))[:, None]
                robe_W = a * W_wing + (1.0 - a) * W_body
                robe_W /= np.maximum(robe_W.sum(axis=1, keepdims=True), 1e-12)
                robe_W = robemod.smooth_grid_weights(robe_W, robe.grid_index, robe.grid_shape, sigma=1.0)
                # (tried and removed 2026-08-22: handing the shoulder cloth's
                # wing_right weight back to the chest wherever tablet_show drove
                # it into the collar oscillated — 139, 42, 62, 97 inside per
                # round — because the wing's shoulder then pierces the cloth
                # that stopped following it. The cloth rides the wing; what it
                # enters at the peak of the lift is the collar band's own volume.)
                robe.primitive.joints, robe.primitive.weights = top4(robe_W.astype(np.float32))
                primitives.append(robe.primitive)
                kente_info = robe.info
                _log(f"kente robe: {robe.info}", t0)
                robe_check = {"rest": robemod.clearance(robe.primitive.vertices, flat, surface_tree=sampler.tree)}
                _log(f"robe clearance at rest: {robe_check['rest']}", t0)
                # the posed check must deform the robe with the weights that ship,
                # not the dense ones they were pruned from
                exported_W = np.zeros_like(robe_W)
                np.put_along_axis(exported_W, robe.primitive.joints.astype(np.int64), robe.primitive.weights, axis=1)
                robe_pose_args = (robe.primitive.vertices, exported_W)

            if slv is not None:
                # the sleeve moves like the *wing's* skin under it (the two
                # wing joints along its length, the chest share the root's
                # own skin already carries). Chest vertices in the pool pulled
                # the root ring's weights toward the static collar 0.03 away,
                # and the lifting root skin moved into the lagging cloth on
                # the wave (measured: 1 vertex inside at t=0.96)
                pool_sleeve = np.isin(region_map.labels, ["wing_left", "wing_left_tip"])
                S_W, _ = robemod.transfer_body_weights(slv.primitive.vertices, flat.vertices, pool_sleeve, full,
                                                       allowed_joints=np.isin(joint_names, ["wing_left", "wing_left_tip", "chest"]))
                S_W = robemod.smooth_grid_weights(S_W, slv.grid_index, slv.grid_shape, sigma=1.0)
                slv.primitive.joints, slv.primitive.weights = top4(S_W.astype(np.float32))
                primitives.append(slv.primitive)
                sleeve_info = slv.info
                # the root ring must hide under the tunic's cloth over the
                # root, and the tunic must not pass through the sleeve
                root_mask = s_along <= slv.info["s_separation"]
                root_zone = slv.primitive.vertices[root_mask]
                near = cKDTree(slv.primitive.vertices).query(robe.primitive.vertices, distance_upper_bound=0.06)[0] < 0.06
                tunic_near = robemod.garment_overlap(robe.primitive.vertices[near], slv.primitive.vertices, slv.primitive.faces)
                sleeve_check = {
                    "rest": robemod.clearance(slv.primitive.vertices, flat, surface_tree=sampler.tree),
                    "under_tunic": robemod.garment_overlap(root_zone, robe.primitive.vertices, robe.primitive.faces),
                    "tunic_in_sleeve": {"near_count": tunic_near["count"],
                                        "inside_count": tunic_near["count"] - tunic_near["outside_count"],
                                        "max_depth": tunic_near["max_depth"]},
                }
                _log(f"sleeve clearance at rest: {sleeve_check['rest']}", t0)
                _log(f"sleeve root under the tunic: {sleeve_check['under_tunic']}", t0)
                _log(f"tunic vertices inside the sleeve: {sleeve_check['tunic_in_sleeve']}", t0)
                exported_SW = np.zeros_like(S_W)
                np.put_along_axis(exported_SW, slv.primitive.joints.astype(np.int64), slv.primitive.weights, axis=1)
                sleeve_pose_args = (slv.primitive.vertices, exported_SW)
                sleeve_root_args = (root_zone, exported_SW[root_mask])
        else:
            if kente_style == "sash":
                kdecal = kentemod.build_kente_sash_decal(flat, region_map, weave_params, bake_normal=kente_normal)
            else:
                kdecal = kentemod.build_kente_decal(flat, region_map, weave_params, regions_included=kente_regions,
                                                    bake_normal=kente_normal)
            kp = kdecal.primitive
            kp.joints = body_j[kdecal.source_vertex_index]
            kp.weights = body_w[kdecal.source_vertex_index]
            primitives.append(kp)
            kente_info = kdecal.info
            _log(f"kente decal: {kdecal.info}", t0)

    # eye parts: rigid to their joint
    for prim, bound in zip(eye_set.primitives, eye_set.bindings):
        n = len(prim.vertices)
        prim.joints = np.zeros((n, 4), dtype=np.uint8)
        prim.joints[:, 0] = joint_names.index(bound)
        prim.weights = np.zeros((n, 4), dtype=np.float32)
        prim.weights[:, 0] = 1.0
        primitives.append(prim)

    # beads: rigid to their leg joint, same pattern as the eye parts —
    # individual beads on a string don't deform with the skin the way a
    # decal does, so blended body weights would be wrong here.
    bead_info = None
    if beads:
        from meshforge import beads as beadsmod

        # build_symmetric_bead_rings, not two independent build_bead_ring
        # calls: regions.classify labels noticeably more geometry
        # leg_right than leg_left on this asset (the tablet-holding wing
        # folds down near that side, the same asymmetry collar.py already
        # documents for wing_right vs wing_left), so measuring each leg's
        # radius independently gave a visibly larger, looser anklet on
        # the right — confirmed by rendering, not just the numbers.
        # 28 beads at 320 faces each outweighed the whole garment in the guide GLB (253 KB); 80 faces read the same at 200 px
        for ring in beadsmod.build_symmetric_bead_rings(flat, region_map, bead_count=bead_count,
                                                        subdivisions=1 if target_faces < REF_FACES // 2 else 2):
            n = len(ring.primitive.vertices)
            ring.primitive.joints = np.zeros((n, 4), dtype=np.uint8)
            ring.primitive.joints[:, 0] = joint_names.index(ring.joint_name)
            ring.primitive.weights = np.zeros((n, 4), dtype=np.float32)
            ring.primitive.weights[:, 0] = 1.0
            primitives.append(ring.primitive)
            bead_info = (bead_info or []) + [ring.info]
            _log(f"beads ({ring.info['leg']}): {ring.info}", t0)

    # 8. clips -------------------------------------------------------------
    if robe_check is not None:
        from meshforge import robe as robemod

        # the clips plus the runtime's own head motion (cursor follow turns
        # the neck and pitches the head; the raised wing stands beside the
        # head, so the sleeve is checked against it too)
        check_clips = anims + [_head_follow_probe()]
        robe_check["posed"] = robemod.posed_clearance(robe_pose_args[0], robe_pose_args[1], flat, full,
                                                      joints, joint_names, check_clips)
        _log(f"robe clearance through the clips: {robe_check['posed']['worst']}", t0)
        if wrap_gate_args is not None:
            from meshforge import gates as gatesmod

            wrap, dense, wp = wrap_gate_args["wrap"], wrap_gate_args["dense"], wrap_gate_args["params"]
            H = float(flat.bounds[1][1] - flat.bounds[0][1])
            y0 = float(flat.bounds[0][1])
            bead_top = max((r["world_y"] + r["bead_radius"] for r in (bead_info or [])), default=y0 + 0.10 * H)
            gates = gatesmod.run_gates(
                pins=wrap.pins["sheet"], pin_W=dense["kente_wrap"][:wrap.sheet_grid[1]],
                body_V=flat.vertices, body_W=full, torso_mask=np.isin(region_map.labels, ["chest", "body"]),
                joints=joints, joint_names=joint_names, clips=check_clips, H=H,
                W_all=wrap_gate_args["W_all"], posed=robe_check["posed"],
                hem_points=wrap.sheet.vertices[-wrap.sheet_grid[1]:], y0=y0, bead_top_y=bead_top,
                # the garment answers for the root below the tie; the shoulder above it is bare in the reference
                root_ring=wrap.support.root_ring[wrap.support.root_ring[:, 1] <= wrap.support.y_k + 0.5 * wp.knot_size * H],
                axis_xz=wrap.body.axis_xz,
                garments=[(p.vertices, p.faces) for p in (wrap.sheet, wrap.tail, wrap.knot)],
                tail_V=wrap.tail.vertices, tail_W=dense["kente_wrap_tail"], body_mesh=flat,
                wave_clips=[a for a in anims if a.name == "wave"],
                sheet_P=wrap.sheet.vertices, sheet_sets=wrap.sheet_sets, stretch_k=cloth_params.stretch,
                sheet_cols=wrap.sheet_grid[1],
                cloth_V=np.vstack([wrap.sheet.vertices, wrap.tail.vertices]),
                theta_grid=wrap.body.theta, rim_y=wrap.body.neck_y)
            _log("gates:\n" + gatesmod.format_table(gates), t0)
            report_gates = [dict(g._asdict()) for g in gates]
        if sleeve_check is not None:
            sleeve_check["posed"] = robemod.posed_clearance(sleeve_pose_args[0], sleeve_pose_args[1], flat, full,
                                                            joints, joint_names, check_clips)
            _log(f"sleeve clearance through the clips: {sleeve_check['posed']['worst']}", t0)
            sleeve_check["under_tunic_posed"] = robemod.posed_overlap(
                sleeve_root_args[0], sleeve_root_args[1], robe_pose_args[0], robe.primitive.faces, robe_pose_args[1],
                joints, joint_names, check_clips)
            _log(f"sleeve root under the tunic through the clips: {sleeve_check['under_tunic_posed']['worst']}", t0)

    node_extras = {
        name: {"gaze_forward": [float(v) for v in fwd]} for name, fwd in eye_set.gaze_forward.items()
    }
    for name, axis in eye_set.blink_axes.items():
        node_extras[name] = {"blink_axis": [float(v) for v in axis], "blink_close_deg": float(eye_set.blink_close_deg)}
    asset_extras = {
        "meshforge": {
            "skeleton": "OwlV1",
            "clips": [a.name for a in anims],
            "joints": joint_names,
            "runtime_joints": {"gaze": ["eye_left", "eye_right"], "head_follow": "neck"},
            "eye_radius": {k: float(v) for k, v in eye_set.radius.items()},
            "collar_text": text,
        }
    }

    # 9. export + validate -------------------------------------------------
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_rigged_glb(primitives, joints, anims, str(out), asset_extras=asset_extras, node_extras=node_extras)
    _log(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)", t0)
    failures = validate(str(out), expect_clips=[a.name for a in anims], expect_joints=joint_names, verbose=False)
    if failures:
        for f in failures:
            print("FAIL ", f)
        raise SystemExit(f"validation failed with {len(failures)} failure(s)")
    _log("validation passed", t0)

    report = {
        "out": str(out), "bytes": out.stat().st_size, "vertices": int(len(split_vertices)), "faces": int(len(unwrapped.mesh.faces)),
        "joints": joint_names, "clips": [a.name for a in anims], "regions": region_map.stats,
        "weights": wr.diagnostics, "eyes": eye_set.info, "collar": decal.info, "kente": kente_info,
        "kente_style": kente_style if kente else None, "robe_clearance": robe_check,
        "sleeve": sleeve_info, "sleeve_clearance": sleeve_check,
        "beads": bead_info,
        "gates": report_gates,
    }
    if previews_dir:
        _write_previews(Path(previews_dir), flat, base_colors, region_map, joints, full, joint_names, eye_set, t0)
    with open(out.with_suffix(".report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def _head_follow_probe(yaw_deg: float = 25.0, pitch_deg: float = 15.0, duration: float = 1.0) -> AnimationSpec:
    """Not a clip that ships: the head-follow range the viewer drives at
    runtime (neck yaw, head pitch), as an animation so the garments can be
    checked through it like through the clips."""
    from scipy.spatial.transform import Rotation

    from meshforge.rigexport import Track

    times = np.linspace(0.0, duration, 5)
    f = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    neck = Rotation.from_euler("y", (yaw_deg * f)[:, None], degrees=True).as_quat()
    head = Rotation.from_euler("x", (pitch_deg * f)[:, None], degrees=True).as_quat()
    return AnimationSpec("head_follow_probe", [Track("neck", "rotation", times, neck), Track("head", "rotation", times, head)])


def _write_previews(previews_dir: Path, mesh, colors, region_map, joints, W, joint_names, eye_set, t0):
    from PIL import ImageDraw
    from scipy.spatial.transform import Rotation

    from meshforge import fk
    from meshforge.fastpreview import contact_sheet, render_flat

    previews_dir.mkdir(parents=True, exist_ok=True)
    bounds = np.array(mesh.bounds)
    lc = regions.label_colors(region_map.labels)
    sheet = [render_flat(mesh.vertices, mesh.faces, lc, yaw_deg=a, resolution=600, bounds=bounds) for a in (0, 90, 180, 270)]
    contact_sheet(sheet, cols=4).save(previews_dir / "regions.png")

    def q(axis, deg):
        return Rotation.from_euler(axis, deg, degrees=True).as_quat()
    poses = {
        "rest": {}, "head_pitch+15": {"head": q("x", 15)}, "neck_yaw-25": {"neck": q("y", -25)},
        "wing_left_up": {"wing_left": q("z", -35)}, "wing_left_down": {"wing_left": q("z", 20)},
        "chest_lean": {"chest": q("x", 8)}, "legs": {"leg_left": q("x", 14), "leg_right": q("x", 14), "foot_left": q("x", -11), "foot_right": q("x", -11)},
        "tablet_show": {"wing_right": (Rotation.from_euler("y", -10, degrees=True) * Rotation.from_euler("x", -15, degrees=True)).as_quat()},
        "tassel+tail": {"tassel": q("z", 25), "tail": q("z", 12)},
    }
    imgs = []
    for name, rots in poses.items():
        world = fk.world_matrices(joints, rotations=rots)
        skin = fk.skin_matrices(joints, world)
        posed = fk.pose_vertices(mesh.vertices, W, joint_names, skin)
        img = render_flat(posed, mesh.faces, colors, yaw_deg=35 if "tablet" in name else 0, resolution=500, bounds=bounds)
        ImageDraw.Draw(img).text((10, 10), name, fill=(255, 255, 255))
        imgs.append(img)
    contact_sheet(imgs, cols=3).save(previews_dir / "poses.png")
    _log(f"previews written to {previews_dir}", t0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hires", required=True, help="hi-res textured TRELLIS GLB")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--faces", type=int, default=120_000)
    ap.add_argument("--tex", type=int, default=2048)
    ap.add_argument("--samples", type=int, default=3_000_000)
    ap.add_argument("--previews", default=None)
    ap.add_argument("--text", default="AI-CCORE")
    ap.add_argument("--jpeg-quality", type=int, default=88)
    ap.add_argument("--eye-subdiv", type=int, default=None, help="eyeball tessellation (4 default; 2 for a small on-page guide)")
    ap.add_argument("--decal-height", type=int, default=512, help="collar decal image height in px")
    ap.add_argument("--kente", action="store_true", help="dress the owl in kente: a robe by default (see --kente-style)")
    ap.add_argument("--kente-colorway", default="asante_gold", choices=("asante_gold", "ewe_adanudo"))
    ap.add_argument("--kente-seed", type=int, default=0)
    ap.add_argument("--kente-normal", action="store_true", help="bake a woven-relief normal map for the kente garment")
    ap.add_argument("--kente-style", default="tunic", choices=("tunic", "robe", "vest", "sash", "wrap"),
                    help="tunic: a drafted pattern draped onto the body by a cloth solve (default). "
                         "robe: the older radial-hull garment. vest / sash: the oldest skin-tight decals")
    ap.add_argument("--cloth-hem", type=float, default=None, help="tunic hem height as a fraction of the bbox height (default 0.135)")
    ap.add_argument("--cloth-ease", type=float, default=None, help="cloth circumference over the measured chest girth (default 0.20)")
    ap.add_argument("--cloth-hem-ease", type=float, default=None, help="... at the hem: the taper (default 0.06)")
    ap.add_argument("--cloth-res", default=None, help="tunic pattern grid as columns,rows (default 152,84)")
    ap.add_argument("--cloth-tex", default=None, help="tunic texture as WxH (default 2048x1024)")
    ap.add_argument("--cloth-band", type=float, default=None, help="width of the gold selvedge along every cut edge (default 0.030; 0 = none)")
    ap.add_argument("--cloth-iterations", type=int, default=None, help="drape solver steps (default 300)")
    ap.add_argument("--cloth-strips", type=int, default=None, help="kente strips around the piece; a multiple of 3 (default 24)")
    ap.add_argument("--wrap-hem", type=float, default=None, help="wrap: hem height, fraction of the height (default 0.125)")
    ap.add_argument("--wrap-gather", type=float, default=None, help="wrap: cloth per loop length inside the knot arc (default 1.3)")
    ap.add_argument("--wrap-knot-drop", type=float, default=None, help="wrap: knot top under the collar rim, fraction of the height (default 0.015)")
    ap.add_argument("--wrap-knot-size", type=float, default=None, help="wrap: knot bulge height, fraction of the height (default 0.07)")
    ap.add_argument("--wrap-tail-width", type=float, default=None, help="wrap: tail width, fraction of the height (default 0.22)")
    ap.add_argument("--wrap-tail-end", type=float, default=None, help="wrap: where the tail ends, fraction of the height (default 0.15)")
    ap.add_argument("--wrap-tail-offset", type=float, default=None, help="wrap: degrees behind the knot arc the tail is pinned (default 30)")
    ap.add_argument("--robe-hem", type=float, default=None, help="hem height as a fraction of the bbox height (default 0.12)")
    ap.add_argument("--robe-clearance", type=float, default=None, help="air between cloth and body at the hem, mesh units (default 0.10)")
    ap.add_argument("--robe-pleats", type=int, default=None, help="number of vertical pleats (default 10)")
    ap.add_argument("--robe-res", default=None, help="robe grid as columns,rows (default 144,64; the guide build uses ~72,32)")
    ap.add_argument("--robe-tex", default=None, help="robe texture as WxH (default 2048x512)")
    ap.add_argument("--robe-band", type=float, default=None, help="width of the plain gold binding along every edge (default 0.035; 0 = none)")
    ap.add_argument("--no-sleeve", action="store_true", help="kente robe without the sleeve on the wing that waves")
    ap.add_argument("--sleeve-length", type=float, default=None,
                    help="how much of the free wing the sleeve covers, 0-1 (default 0.72: the feathered tip shows past the cuff)")
    ap.add_argument("--sleeve-clearance", type=float, default=None,
                    help="air between sleeve and wing at its widest, just past the shoulder (default 0.06; the cuff keeps 0.02)")
    ap.add_argument("--beads", action="store_true", help="add beaded anklets")
    ap.add_argument("--bead-count", type=int, default=14)
    args = ap.parse_args(argv)
    eye_params = EyeParams()
    if args.eye_subdiv is not None:
        eye_params = eye_params._replace(subdivisions=args.eye_subdiv) if hasattr(eye_params, "_replace") \
            else dataclasses.replace(eye_params, subdivisions=args.eye_subdiv)
    robe_params = None
    overrides = {}
    if args.kente and args.kente_style == "robe":
        from meshforge.robe import RobeParams

        overrides = {}
        if args.robe_hem is not None:
            overrides["hem_frac"] = args.robe_hem
        if args.robe_clearance is not None:
            overrides["clearance_hem"] = args.robe_clearance
        if args.robe_pleats is not None:
            overrides["pleats"] = args.robe_pleats
        if args.robe_res is not None:
            cols, rows = (int(v) for v in args.robe_res.split(","))
            overrides["n_theta"], overrides["n_rows"] = cols, rows
        if args.robe_tex is not None:
            w, h = (int(v) for v in args.robe_tex.lower().split("x"))
            overrides["texture_size"] = (w, h)
        if args.robe_band is not None:
            overrides["edge_band"] = args.robe_band
        # no overrides -> None, so build_owl scales the grid/chart to the build
        robe_params = RobeParams(**overrides) if overrides else None
    robe_explicit = set(overrides) if args.kente and args.kente_style == "robe" else set()
    cloth_params, cloth_explicit = None, set()
    if args.kente and args.kente_style in ("tunic", "wrap"):
        from meshforge.drape import ClothParams

        c_over = {}
        if args.cloth_hem is not None:
            c_over["hem_frac"] = args.cloth_hem
        if args.cloth_ease is not None:
            c_over["chest_ease"] = args.cloth_ease
        if args.cloth_hem_ease is not None:
            c_over["hem_ease"] = args.cloth_hem_ease
        if args.cloth_res is not None:
            cu, cv = (int(v) for v in args.cloth_res.split(","))
            c_over["n_u"], c_over["n_v"] = cu, cv
        if args.cloth_tex is not None:
            w, h = (int(v) for v in args.cloth_tex.lower().split("x"))
            c_over["texture_size"] = (w, h)
        if args.cloth_band is not None:
            c_over["edge_band"] = args.cloth_band
        if args.cloth_iterations is not None:
            c_over["iterations"] = args.cloth_iterations
        if args.cloth_strips is not None:
            c_over["strips_around"] = args.cloth_strips
        cloth_params = ClothParams(**c_over) if c_over else None
        cloth_explicit = set(c_over)
    sleeve_params = None
    sleeve_explicit = set()
    if args.kente and args.kente_style in ("tunic", "robe") and not args.no_sleeve:
        from meshforge.sleeve import SleeveParams

        s_over = {}
        if args.sleeve_length is not None:
            s_over["length_frac"] = args.sleeve_length
        if args.sleeve_clearance is not None:
            s_over["clearance_wide"] = args.sleeve_clearance
        sleeve_params = SleeveParams(**s_over) if s_over else None
        sleeve_explicit = set(s_over)
    wrap_params = None
    if args.kente and args.kente_style == "wrap":
        from meshforge.wrap import WrapParams

        w_over = {k: v for k, v in (("hem_frac", args.wrap_hem), ("gather_ratio", args.wrap_gather),
                                    ("knot_drop", args.wrap_knot_drop), ("knot_size", args.wrap_knot_size),
                                    ("tail_width_frac", args.wrap_tail_width), ("tail_end_frac", args.wrap_tail_end),
                                    ("tail_offset_deg", args.wrap_tail_offset)) if v is not None}
        wrap_params = WrapParams(**w_over)
    report = build_owl(args.hires, args.out, cache_dir=args.cache, target_faces=args.faces, texture_size=args.tex,
                       n_samples=args.samples, previews_dir=args.previews, text=args.text,
                       jpeg_quality=args.jpeg_quality, eye_params=eye_params, decal_height=args.decal_height,
                       kente=args.kente, kente_colorway=args.kente_colorway, kente_seed=args.kente_seed,
                       kente_normal=args.kente_normal, kente_style=args.kente_style, robe_params=robe_params,
                       robe_explicit=robe_explicit, cloth_params=cloth_params, cloth_explicit=cloth_explicit, sleeve=not args.no_sleeve, sleeve_params=sleeve_params,
                       sleeve_explicit=sleeve_explicit, wrap_params=wrap_params,
                       beads=args.beads, bead_count=args.bead_count)
    print(json.dumps({k: report[k] for k in ("out", "bytes", "vertices", "faces", "clips")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
