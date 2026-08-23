#!/usr/bin/env python3
"""Build the owl with the contact-driven shoulder-supported cloth path."""
from __future__ import annotations

import dataclasses
import sys


def _concept_args(argv):
    """The visual reference is a shoulder garment, not the old tied wrap.

    Preserve the public CLI while routing legacy `--kente-style wrap` calls to
    the tunic+sleeve construction so existing build commands produce the
    intended silhouette instead of the diagonal under-arm loop.
    """
    out = list(argv)
    for i, arg in enumerate(out):
        if arg == "--kente-style" and i + 1 < len(out) and out[i + 1] == "wrap":
            out[i + 1] = "tunic"
        elif arg == "--kente-style=wrap":
            out[i] = "--kente-style=tunic"
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    keep_shell = "--exec-keep-swept-shell" in argv
    argv = [a for a in argv if a != "--exec-keep-swept-shell"]
    argv = _concept_args(argv)

    from meshforge import cloth_exec as E
    from meshforge import drape as D
    from meshforge import gates as G
    from meshforge import shoulder_exec as S
    from meshforge import wrap as W

    D.initial_positions = S.initial_positions
    D.drape = S.drape

    W.support_weights = E.support_weights

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
