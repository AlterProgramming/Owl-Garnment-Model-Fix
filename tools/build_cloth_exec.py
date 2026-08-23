#!/usr/bin/env python3
"""Build the owl with the contact-driven cloth execution path.

Usage is the normal owl pipeline command with this script in front::

    PYTHONPATH=. python3 tools/build_cloth_exec.py \
        --hires "assets/AI-CCORE Owl.glb" \
        --out build/owl-kente-wrap-exec.glb --cache .owl_cache \
        --kente --kente-style wrap --beads --cloth-res 76,42

The baseline pipeline is not modified on disk.  This process installs the
experimental solver and support-weight transfer, disables the wrap's static
swept-root shell, and then calls ``meshforge.owl_pipeline.main``.

Pass ``--exec-keep-swept-shell`` to keep the old swept-root bounds while still
using the new solver.  That is useful as an A/B isolation run.
"""
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
    from meshforge import wrap as W

    # Physics: reconcile constraints with body contact every inner iteration.
    D.drape = E.drape

    # Animation approximation: cloth touching a moving wing root can share a
    # local amount of that root's skinning, instead of the root moving inside a
    # body/chest-only garment shell.
    W.support_weights = E.support_weights

    # G2's old definition treated every wing weight as forbidden because the
    # baseline intentionally prohibited local moving support.  In this path,
    # only the three wing joints are newly legal; head/neck/legs/tail remain
    # forbidden supports.
    legal_dynamic_support = {"wing_left", "wing_left_tip", "wing_right"}
    G.FORBIDDEN = tuple(n for n in G.FORBIDDEN if n not in legal_dynamic_support)

    if not keep_shell:
        OriginalWrapParams = W.WrapParams

        class ExecWrapParams:
            def __new__(cls, *args, **kwargs):
                p = OriginalWrapParams(*args, **kwargs)
                # These two mechanisms pre-clear where the raised wing *will*
                # go.  That is the shell behaviour this execution path replaces
                # with local support on the moving root.
                return dataclasses.replace(p, root_bound=False, sweep_band=0.0)

        W.WrapParams = ExecWrapParams

    from meshforge import owl_pipeline

    mode = "solver + local support" + (" + baseline swept shell" if keep_shell else " (no swept shell)")
    print(f"[cloth-exec] {mode}", flush=True)
    return owl_pipeline.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
