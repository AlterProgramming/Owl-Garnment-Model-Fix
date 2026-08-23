#!/usr/bin/env python3
"""Build the owl with the contact-driven shoulder-supported cloth path."""
from __future__ import annotations

import dataclasses
import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    keep_shell = "--exec-keep-swept-shell" in argv
    argv = [a for a in argv if a != "--exec-keep-swept-shell"]

    from meshforge import cloth_exec as E
    from meshforge import drape as D
    from meshforge import gates as G
    from meshforge import shoulder_exec as S
    from meshforge import wrap as W

    # Seed the upper garment directly on the measured support surface and
    # keep that yoke registered while the lower cloth remains free.
    D.initial_positions = S.initial_positions
    D.drape = S.drape

    # Cloth touching a moving wing root inherits a local amount of that
    # support's motion rather than watching the wing move under a static shell.
    W.support_weights = E.support_weights

    legal_dynamic_support = {"wing_left", "wing_left_tip", "wing_right"}
    G.FORBIDDEN = tuple(n for n in G.FORBIDDEN if n not in legal_dynamic_support)

    if not keep_shell:
        OriginalWrapParams = W.WrapParams

        class ExecWrapParams:
            def __new__(cls, *args, **kwargs):
                p = OriginalWrapParams(*args, **kwargs)
                # The execution path follows actual support and contact instead
                # of pre-clearing a future swept shell around the character.
                return dataclasses.replace(
                    p,
                    root_bound=False,
                    sweep_band=0.0,
                    gather_half_deg=max(float(p.gather_half_deg), 34.0),
                )

        W.WrapParams = ExecWrapParams

    from meshforge import owl_pipeline

    mode = "shoulder-supported contact cloth" + (" + baseline swept shell" if keep_shell else " (no swept shell)")
    print(f"[cloth-exec] {mode}", flush=True)
    return owl_pipeline.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
