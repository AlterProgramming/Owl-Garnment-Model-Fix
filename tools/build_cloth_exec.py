#!/usr/bin/env python3
"""Build the owl with the contact-driven shoulder-supported cloth path."""
from __future__ import annotations

import dataclasses
import sys


def _concept_args(argv):
    out = list(argv)
    for i, arg in enumerate(out):
        if arg == "--kente-style" and i + 1 < len(out) and out[i + 1] == "wrap":
            out[i + 1] = "tunic"
        elif arg == "--kente-style=wrap":
            out[i] = "--kente-style=tunic"
    return out


def _replace_unset(instance, overrides, explicit):
    return dataclasses.replace(instance, **{k: v for k, v in overrides.items() if k not in explicit})


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    keep_shell = "--exec-keep-swept-shell" in argv
    argv = [a for a in argv if a != "--exec-keep-swept-shell"]
    argv = _concept_args(argv)

    from meshforge import cloth_exec as E
    from meshforge import drape as D
    from meshforge import gates as G
    from meshforge import shoulder_exec as S
    from meshforge import sleeve as SL
    from meshforge import wrap as W

    D.initial_positions = S.initial_positions
    D.drape = S.drape
    W.support_weights = E.support_weights

    OriginalClothParams = D.ClothParams

    class ExecClothParams:
        def __new__(cls, *args, **kwargs):
            p = OriginalClothParams(*args, **kwargs)
            p = _replace_unset(p, {
                "chest_ease": 0.055,
                "hem_ease": 0.018,
                "hem_min_frac": 0.68,
                "yoke_frac": 0.11,
                "yoke_keep": 0.16,
                "shoulder_keep": 0.32,
                "shoulder_margin": 0.085,
                "length_slack": 0.04,
                "bend": 0.12,
                "bend_wide": 0.055,
                "damping": 0.93,
                "friction": 0.58,
            }, set(kwargs))
            return p

    D.ClothParams = ExecClothParams

    OriginalSleeveParams = SL.SleeveParams

    class ExecSleeveParams:
        def __new__(cls, *args, **kwargs):
            p = OriginalSleeveParams(*args, **kwargs)
            p = _replace_unset(p, {
                "length_frac": 0.80,
                "root_overlap": 0.065,
                "clearance_root": 0.025,
                "clearance_wide": 0.075,
                "wide_at": 0.075,
                "clearance_cuff": 0.045,
                "body_margin": 0.025,
                "smooth_phi_deg": 22.0,
                "smooth_stations": 2.0,
                "cuff_band": 0.025,
            }, set(kwargs))
            return p

    SL.SleeveParams = ExecSleeveParams

    legal_dynamic_support = {"wing_left", "wing_left_tip", "wing_right"}
    G.FORBIDDEN = tuple(n for n in G.FORBIDDEN if n not in legal_dynamic_support)

    if not keep_shell:
        OriginalWrapParams = W.WrapParams

        class ExecWrapParams:
            def __new__(cls, *args, **kwargs):
                p = OriginalWrapParams(*args, **kwargs)
                return dataclasses.replace(
                    p,
                    root_bound=False,
                    sweep_band=0.0,
                    gather_half_deg=max(float(p.gather_half_deg), 34.0),
                )

        W.WrapParams = ExecWrapParams

    from meshforge import owl_pipeline

    print("[cloth-exec] shoulder-supported concept garment", flush=True)
    return owl_pipeline.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
