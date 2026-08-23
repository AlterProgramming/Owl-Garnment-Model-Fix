import { RIG, CLIPS } from './rig-data.mjs';

const EPS = 1e-9;

export function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
export function lerp(a, b, u) { return a + (b - a) * u; }

export function sampleKeys(keys = [], time = 0) {
  if (!keys.length) return {};
  if (time <= keys[0].t) return { ...keys[0] };
  if (time >= keys[keys.length - 1].t) return { ...keys[keys.length - 1] };
  let hi = 1;
  while (hi < keys.length && keys[hi].t < time) hi++;
  const a = keys[hi - 1], b = keys[hi];
  const u = (time - a.t) / Math.max(EPS, b.t - a.t);
  const out = { t: time };
  const channels = new Set([...Object.keys(a), ...Object.keys(b)]);
  channels.delete('t');
  for (const ch of channels) {
    const av = Number.isFinite(a[ch]) ? a[ch] : b[ch];
    const bv = Number.isFinite(b[ch]) ? b[ch] : a[ch];
    out[ch] = lerp(av ?? 0, bv ?? 0, u);
  }
  return out;
}

export function normalizedTime(clip, time) {
  if (clip.loop) {
    const d = Math.max(EPS, clip.duration);
    return ((time % d) + d) % d;
  }
  return clamp(time, 0, clip.duration);
}

export function sampleClip(name, time) {
  const clip = CLIPS[name];
  if (!clip) throw new Error(`Unknown clip: ${name}`);
  const t = normalizedTime(clip, time);
  const nodes = {};
  for (const id of Object.keys(RIG)) {
    const k = sampleKeys(clip.tracks?.[id] ?? [], t);
    nodes[id] = {
      x: k.x ?? 0,
      y: k.y ?? 0,
      r: k.r ?? 0,
      sx: k.sx ?? 1,
      sy: k.sy ?? 1,
    };
  }
  const face = {};
  for (const channel of ['blink', 'mouth', 'lookX', 'lookY']) {
    const k = sampleKeys(clip.face?.[channel] ?? [], t);
    face[channel] = k.v ?? 0;
  }
  return { clip: name, time: t, nodes, face };
}

export function matIdentity() { return [1, 0, 0, 1, 0, 0]; }

export function matMul(a, b) {
  return [
    a[0] * b[0] + a[2] * b[1],
    a[1] * b[0] + a[3] * b[1],
    a[0] * b[2] + a[2] * b[3],
    a[1] * b[2] + a[3] * b[3],
    a[0] * b[4] + a[2] * b[5] + a[4],
    a[1] * b[4] + a[3] * b[5] + a[5],
  ];
}

export function localMatrix(spec, delta) {
  const rest = spec.rest ?? {};
  const x = (rest.x ?? 0) + (delta.x ?? 0);
  const y = (rest.y ?? 0) + (delta.y ?? 0);
  const r = ((rest.r ?? 0) + (delta.r ?? 0)) * Math.PI / 180;
  const sx = (rest.sx ?? 1) * (delta.sx ?? 1);
  const sy = (rest.sy ?? 1) * (delta.sy ?? 1);
  const c = Math.cos(r), s = Math.sin(r);
  return [c * sx, s * sx, -s * sy, c * sy, x, y];
}

export function worldMatrices(pose) {
  const cache = {};
  const visiting = new Set();
  function solve(id) {
    if (cache[id]) return cache[id];
    if (visiting.has(id)) throw new Error(`Rig cycle at ${id}`);
    visiting.add(id);
    const spec = RIG[id];
    if (!spec) throw new Error(`Unknown rig node: ${id}`);
    const local = localMatrix(spec, pose.nodes?.[id] ?? {});
    const world = spec.parent ? matMul(solve(spec.parent), local) : local;
    visiting.delete(id);
    cache[id] = world;
    return world;
  }
  for (const id of Object.keys(RIG)) solve(id);
  return cache;
}

export function svgMatrix(m) {
  return `matrix(${m.map(v => Number(v.toFixed(6))).join(' ')})`;
}

export function transformPoint(m, p = [0, 0]) {
  return [m[0] * p[0] + m[2] * p[1] + m[4], m[1] * p[0] + m[3] * p[1] + m[5]];
}

export function angleDeg(m) {
  return Math.atan2(m[1], m[0]) * 180 / Math.PI;
}

function trackValueAtEnds(keys, channel, fallback) {
  if (!keys?.length) return [fallback, fallback];
  const first = Number.isFinite(keys[0][channel]) ? keys[0][channel] : fallback;
  const last = Number.isFinite(keys.at(-1)[channel]) ? keys.at(-1)[channel] : first;
  return [first, last];
}

export function validateRig() {
  const errors = [];
  const warnings = [];

  // topology
  for (const [id, spec] of Object.entries(RIG)) {
    if (spec.parent && !RIG[spec.parent]) errors.push(`${id}: missing parent ${spec.parent}`);
    if (!spec.rest) errors.push(`${id}: missing rest transform`);
  }
  try { worldMatrices(sampleClip('idle', 0)); } catch (err) { errors.push(String(err.message ?? err)); }

  // Semantic attachment contract: garment cannot become an independent root.
  if (RIG.garment_front?.parent !== 'body') errors.push('garment_front must inherit body');
  if (RIG.garment_back?.parent !== 'body') errors.push('garment_back must inherit body');
  if (RIG.cuff_l?.parent !== 'wing_l') errors.push('left cuff must inherit left wing');
  if (RIG.cuff_r?.parent !== 'wing_r') errors.push('right cuff must inherit right wing');

  // animation data
  for (const [name, clip] of Object.entries(CLIPS)) {
    if (!(clip.duration > 0)) errors.push(`${name}: non-positive duration`);
    for (const [id, keys] of Object.entries(clip.tracks ?? {})) {
      if (!RIG[id]) errors.push(`${name}: unknown track node ${id}`);
      for (let i = 1; i < keys.length; i++) if (keys[i].t <= keys[i - 1].t) errors.push(`${name}/${id}: key times not increasing`);
      if (keys.length && (keys[0].t < -EPS || keys.at(-1).t > clip.duration + EPS)) errors.push(`${name}/${id}: key outside clip duration`);
    }
    for (const [ch, keys] of Object.entries(clip.face ?? {})) {
      for (let i = 1; i < keys.length; i++) if (keys[i].t <= keys[i - 1].t) errors.push(`${name}/face.${ch}: key times not increasing`);
    }
    if (clip.loop) {
      for (const [id, keys] of Object.entries(clip.tracks ?? {})) {
        for (const ch of ['x','y','r','sx','sy']) {
          const fallback = (ch === 'sx' || ch === 'sy') ? 1 : 0;
          const [a,b] = trackValueAtEnds(keys, ch, fallback);
          if (Math.abs(a-b) > 1e-5) errors.push(`${name}/${id}.${ch}: loop does not close (${a} != ${b})`);
        }
      }
      for (const [ch, keys] of Object.entries(clip.face ?? {})) {
        const [a,b] = trackValueAtEnds(keys, 'v', 0);
        if (Math.abs(a-b) > 1e-5) errors.push(`${name}/face.${ch}: loop does not close`);
      }
    }
  }

  // Dense sampling against node limits.
  for (const [name, clip] of Object.entries(CLIPS)) {
    const steps = Math.max(20, Math.ceil(clip.duration * 60));
    for (let i = 0; i <= steps; i++) {
      const pose = sampleClip(name, clip.duration * i / steps);
      for (const [id, spec] of Object.entries(RIG)) {
        const d = pose.nodes[id];
        const finals = {
          r: (spec.rest.r ?? 0) + d.r,
          sx: (spec.rest.sx ?? 1) * d.sx,
          sy: (spec.rest.sy ?? 1) * d.sy,
        };
        for (const [ch, range] of Object.entries(spec.limits ?? {})) {
          if (finals[ch] < range[0] - 1e-5 || finals[ch] > range[1] + 1e-5) errors.push(`${name}@${pose.time.toFixed(3)} ${id}.${ch}=${finals[ch].toFixed(3)} outside [${range}]`);
        }
      }
      if (pose.face.blink < -EPS || pose.face.blink > 1 + EPS) errors.push(`${name}: blink outside [0,1]`);
      if (pose.face.mouth < -EPS || pose.face.mouth > 1 + EPS) errors.push(`${name}: mouth outside [0,1]`);
    }
  }

  return { ok: errors.length === 0, errors, warnings };
}
