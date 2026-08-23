# AI-CCORE Owl — 2D Puppet Rig

This directory is an intentionally independent 2D replacement for the 3D garment experiment.

The implementation is a layered SVG character with a small deterministic skeletal runtime. There is no cloth solver, no 3D collision, and no garment shell moving separately from the mascot.

## Run

From the repository root:

```bash
python3 -m http.server 8000
```

Open:

```text
http://127.0.0.1:8000/puppet2d/
```

Do not open `index.html` directly from `file://`; the page fetches `owl.svg` as a module asset.

## Rig structure

The rig is declared in `rig-data.mjs`.

```text
root
└─ body
   ├─ garment_back
   ├─ garment_front
   ├─ wing_l
   │  └─ cuff_l
   ├─ wing_r
   │  └─ cuff_r
   ├─ head
   │  ├─ eye_l
   │  ├─ eye_r
   │  ├─ brow_l
   │  ├─ brow_r
   │  └─ beak
   └─ collar
root
├─ tail
├─ leg_l
└─ leg_r
```

The important modeling rule is encoded directly in topology:

- the kente body panels inherit `body`;
- the left patterned cuff inherits `wing_l`;
- the right patterned cuff inherits `wing_r`.

There is therefore no independent garment transform capable of remaining in place while the owl moves beneath it.

## Animation clips

- `idle` — breathing, subtle head/wing/tail motion, blink
- `wave` — raised left wing with several controlled wave beats and eye/head follow
- `talk` — looping beak shapes with small head/wing motion
- `celebrate` — two-wing raise plus body bounce

Keys `1`–`4` trigger those clips. `Space` toggles playback. `D` shows the live rig hierarchy/pivots.

All clip transforms are rest-pose deltas. Rotation/scale bounds are checked by `validateRig()`.

## Validation

```bash
node --test tests/puppet2d/rig.test.mjs
node tools/validate_puppet2d.mjs
```

The tests cover:

- acyclic hierarchy and known parents;
- garment-to-body attachment contract;
- cuff-to-wing attachment contract;
- clip loop closure;
- action return-to-rest;
- bounded facial channels;
- dense sampling against joint limits.

The diagnostic script additionally samples each clip at 60 Hz and reports attachment-distance errors and actual wing-tip motion.

## Runtime evidence

With Playwright installed:

```bash
python3 tools/render_puppet2d.py --out renders/puppet2d
```

This captures the real browser runtime at deterministic points in each animation and writes:

```text
renders/puppet2d/idle.png
renders/puppet2d/wave.png
renders/puppet2d/talk.png
renders/puppet2d/celebrate.png
renders/puppet2d/wave_rig_debug.png
renders/puppet2d/contact_sheet.png
```

These are runtime screenshots, not generated concept renders.
