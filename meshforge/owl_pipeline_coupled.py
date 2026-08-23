"""Build the owl with the animation-coupled garment experiment enabled.

This entry point is intentionally separate from ``meshforge.owl_pipeline`` so
main remains an A/B control while the user judges the motion result.

Example (same arguments as the production pipeline)::

    python -m meshforge.owl_pipeline_coupled \
      --hires "assets/AI-CCORE Owl.glb" \
      --out build/owl-kente-coupled.glb \
      --cache .owl_cache \
      --kente --kente-style wrap --beads

The coupling layer changes only wrap/cloth defaults, wrap skinning, and the two
acceptance gates whose old definitions explicitly prohibited legitimate wing
root coupling.
"""
from __future__ import annotations

import sys

from meshforge import coupling


def main(argv=None) -> int:
    coupling.install()
    # Import after installation: owl_pipeline's CLI imports WrapParams and
    # ClothParams lazily, so the coupled variants become the experiment's
    # defaults without editing the control implementation.
    from meshforge.owl_pipeline import main as production_main
    return production_main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
