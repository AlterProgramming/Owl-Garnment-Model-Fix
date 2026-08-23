// Declarative 2D puppet rig. All transform channels are DELTAS from the rest pose.
// The garment is deliberately part of the hierarchy instead of an independent shell:
// front/back panels inherit BODY; cuffs inherit their WING pivots.

export const RIG = {
  root:          { parent: null,     element: null,                 rest: { x: 450, y: 505, r: 0, sx: 1, sy: 1 } },
  tail:          { parent: 'root',   element: 'part-tail',          rest: { x: 0, y: 145, r: 0, sx: 1, sy: 1 }, limits: { r: [-12, 12] } },
  leg_l:         { parent: 'root',   element: 'part-leg-l',         rest: { x: -78, y: 165, r: 0, sx: 1, sy: 1 }, limits: { r: [-12, 12] } },
  leg_r:         { parent: 'root',   element: 'part-leg-r',         rest: { x:  78, y: 165, r: 0, sx: 1, sy: 1 }, limits: { r: [-12, 12] } },
  body:          { parent: 'root',   element: 'part-body',          rest: { x: 0, y: 0, r: 0, sx: 1, sy: 1 }, limits: { r: [-8, 8], sx: [0.94, 1.06], sy: [0.94, 1.06] } },
  garment_back:  { parent: 'body',   element: 'part-garment-back',  rest: { x: 0, y: -8, r: 0, sx: 1, sy: 1 } },
  garment_front: { parent: 'body',   element: 'part-garment-front', rest: { x: 0, y: -8, r: 0, sx: 1, sy: 1 } },
  wing_l:        { parent: 'body',   element: 'part-wing-l',        rest: { x: -134, y: -55, r: -7, sx: 1, sy: 1 }, limits: { r: [-35, 125] } },
  wing_r:        { parent: 'body',   element: 'part-wing-r',        rest: { x:  134, y: -52, r:  7, sx: 1, sy: 1 }, limits: { r: [-125, 35] } },
  cuff_l:        { parent: 'wing_l', element: 'part-cuff-l',        rest: { x: -9, y: 18, r: 0, sx: 1, sy: 1 } },
  cuff_r:        { parent: 'wing_r', element: 'part-cuff-r',        rest: { x:  9, y: 18, r: 0, sx: 1, sy: 1 } },
  head:          { parent: 'body',   element: 'part-head',          rest: { x: 0, y: -224, r: 0, sx: 1, sy: 1 }, limits: { r: [-16, 16], sx: [0.96, 1.04], sy: [0.96, 1.04] } },
  eye_l:         { parent: 'head',   element: 'part-eye-l',         rest: { x: -58, y: 7, r: 0, sx: 1, sy: 1 }, limits: { sy: [0.12, 1.05] } },
  eye_r:         { parent: 'head',   element: 'part-eye-r',         rest: { x:  58, y: 7, r: 0, sx: 1, sy: 1 }, limits: { sy: [0.12, 1.05] } },
  brow_l:        { parent: 'head',   element: 'part-brow-l',        rest: { x: -58, y: -56, r: 0, sx: 1, sy: 1 }, limits: { r: [-18, 18] } },
  brow_r:        { parent: 'head',   element: 'part-brow-r',        rest: { x:  58, y: -56, r: 0, sx: 1, sy: 1 }, limits: { r: [-18, 18] } },
  beak:          { parent: 'head',   element: 'part-beak',          rest: { x: 0, y: 43, r: 0, sx: 1, sy: 1 }, limits: { r: [-8, 8] } },
  collar:        { parent: 'body',   element: 'part-collar',        rest: { x: 0, y: -117, r: 0, sx: 1, sy: 1 }, limits: { r: [-5, 5] } },
};

const K = (t, value) => ({ t, ...value });

export const CLIPS = {
  idle: {
    duration: 4.0,
    loop: true,
    tracks: {
      root: [K(0,{y:0}),K(1,{y:-4}),K(2,{y:0}),K(3,{y:3}),K(4,{y:0})],
      body: [K(0,{sy:1,sx:1}),K(1,{sy:1.012,sx:0.994}),K(2,{sy:1,sx:1}),K(3,{sy:0.993,sx:1.004}),K(4,{sy:1,sx:1})],
      head: [K(0,{r:-1.2,y:0}),K(1,{r:0.8,y:-2}),K(2,{r:1.4,y:0}),K(3,{r:-0.5,y:1}),K(4,{r:-1.2,y:0})],
      wing_l: [K(0,{r:0}),K(2,{r:-3}),K(4,{r:0})],
      wing_r: [K(0,{r:0}),K(2,{r:3}),K(4,{r:0})],
      tail: [K(0,{r:0}),K(2,{r:2}),K(4,{r:0})],
    },
    face: {
      blink: [K(0,{v:0}),K(2.62,{v:0}),K(2.68,{v:1}),K(2.74,{v:0}),K(4,{v:0})],
      mouth: [K(0,{v:0}),K(4,{v:0})],
      lookX: [K(0,{v:0}),K(4,{v:0})],
      lookY: [K(0,{v:0}),K(4,{v:0})],
    },
  },

  wave: {
    duration: 2.8,
    loop: false,
    tracks: {
      root: [K(0,{y:0}),K(.35,{y:-5}),K(1.4,{y:-2}),K(2.8,{y:0})],
      body: [K(0,{r:0}),K(.45,{r:2.2}),K(2.25,{r:1}),K(2.8,{r:0})],
      head: [K(0,{r:0}),K(.45,{r:-5}),K(1.4,{r:-3}),K(2.8,{r:0})],
      wing_l: [K(0,{r:0}),K(.38,{r:104}),K(.82,{r:91}),K(1.16,{r:112}),K(1.50,{r:91}),K(1.84,{r:111}),K(2.30,{r:96}),K(2.8,{r:0})],
      wing_r: [K(0,{r:0}),K(.45,{r:5}),K(2.3,{r:3}),K(2.8,{r:0})],
      tail: [K(0,{r:0}),K(.45,{r:-3}),K(2.2,{r:-2}),K(2.8,{r:0})],
    },
    face: {
      blink: [K(0,{v:0}),K(.55,{v:0}),K(.61,{v:1}),K(.67,{v:0}),K(2.8,{v:0})],
      mouth: [K(0,{v:.15}),K(.45,{v:.55}),K(1.8,{v:.35}),K(2.8,{v:.15})],
      lookX: [K(0,{v:0}),K(.38,{v:-5}),K(2.2,{v:-4}),K(2.8,{v:0})],
      lookY: [K(0,{v:0}),K(.38,{v:-2}),K(2.2,{v:-1}),K(2.8,{v:0})],
    },
  },

  talk: {
    duration: 1.6,
    loop: true,
    tracks: {
      head: [K(0,{r:0,y:0}),K(.4,{r:1.5,y:1}),K(.8,{r:-1,y:-1}),K(1.2,{r:1,y:0}),K(1.6,{r:0,y:0})],
      wing_l: [K(0,{r:0}),K(.8,{r:5}),K(1.6,{r:0})],
      wing_r: [K(0,{r:0}),K(.8,{r:-4}),K(1.6,{r:0})],
    },
    face: {
      mouth: [K(0,{v:.1}),K(.16,{v:.85}),K(.34,{v:.25}),K(.53,{v:.65}),K(.72,{v:.12}),K(.93,{v:.95}),K(1.12,{v:.35}),K(1.34,{v:.72}),K(1.6,{v:.1})],
      blink: [K(0,{v:0}),K(1.05,{v:0}),K(1.11,{v:1}),K(1.17,{v:0}),K(1.6,{v:0})],
      lookX: [K(0,{v:0}),K(1.6,{v:0})],
      lookY: [K(0,{v:0}),K(1.6,{v:0})],
    },
  },

  celebrate: {
    duration: 2.4,
    loop: false,
    tracks: {
      root: [K(0,{y:0,sy:1}),K(.35,{y:-20,sy:.98}),K(.62,{y:3,sy:1.02}),K(1.2,{y:-7}),K(2.4,{y:0,sy:1})],
      body: [K(0,{r:0}),K(.35,{r:-2}),K(.8,{r:2}),K(1.4,{r:-1}),K(2.4,{r:0})],
      head: [K(0,{r:0}),K(.35,{r:3}),K(.8,{r:-4}),K(1.4,{r:3}),K(2.4,{r:0})],
      wing_l: [K(0,{r:0}),K(.38,{r:103}),K(1.55,{r:112}),K(2.4,{r:0})],
      wing_r: [K(0,{r:0}),K(.38,{r:-103}),K(1.55,{r:-112}),K(2.4,{r:0})],
      tail: [K(0,{r:0}),K(.4,{r:6}),K(.9,{r:-6}),K(1.4,{r:5}),K(2.4,{r:0})],
    },
    face: {
      mouth: [K(0,{v:.2}),K(.35,{v:1}),K(1.8,{v:.85}),K(2.4,{v:.2})],
      blink: [K(0,{v:0}),K(.72,{v:0}),K(.78,{v:1}),K(.86,{v:0}),K(2.4,{v:0})],
      lookX: [K(0,{v:0}),K(2.4,{v:0})],
      lookY: [K(0,{v:0}),K(.4,{v:-3}),K(1.7,{v:-2}),K(2.4,{v:0})],
    },
  },
};

export const DEFAULT_CLIP = 'idle';
