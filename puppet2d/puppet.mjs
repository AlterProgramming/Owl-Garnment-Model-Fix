import { RIG, CLIPS, DEFAULT_CLIP } from './rig-data.mjs';
import { sampleClip, worldMatrices, svgMatrix, transformPoint } from './rig-core.mjs';

class OwlPuppet {
  constructor(stage) {
    this.stage = stage;
    this.svg = null;
    this.clip = DEFAULT_CLIP;
    this.time = 0;
    this.playing = true;
    this.debug = false;
    this._lastFrame = null;
    this._raf = null;
  }

  async init() {
    const markup = await fetch('./owl.svg').then(r => {
      if (!r.ok) throw new Error(`Failed to load owl.svg: ${r.status}`);
      return r.text();
    });
    this.stage.innerHTML = markup;
    this.svg = this.stage.querySelector('svg');
    this.svg.classList.add('owl-svg');
    this._bindUI();
    this.apply(sampleClip(this.clip, 0));
    this._raf = requestAnimationFrame(ts => this._tick(ts));
    this.stage.dataset.ready = 'true';
    return this;
  }

  _bindUI() {
    document.querySelectorAll('[data-clip]').forEach(btn => {
      btn.addEventListener('click', () => this.setClip(btn.dataset.clip));
    });
    document.querySelector('[data-action="pause"]')?.addEventListener('click', ev => {
      this.playing ? this.pause() : this.play();
      ev.currentTarget.textContent = this.playing ? 'Pause' : 'Play';
    });
    document.querySelector('[data-action="debug"]')?.addEventListener('click', ev => {
      this.setDebug(!this.debug);
      ev.currentTarget.setAttribute('aria-pressed', String(this.debug));
    });
    window.addEventListener('keydown', ev => {
      const keymap = { '1':'idle', '2':'wave', '3':'talk', '4':'celebrate' };
      if (keymap[ev.key]) this.setClip(keymap[ev.key]);
      if (ev.key === ' ') { ev.preventDefault(); this.playing ? this.pause() : this.play(); }
      if (ev.key.toLowerCase() === 'd') this.setDebug(!this.debug);
    });
  }

  setClip(name, restart = true) {
    if (!CLIPS[name]) throw new Error(`Unknown clip ${name}`);
    this.clip = name;
    if (restart) this.time = 0;
    this._lastFrame = performance.now();
    this.apply(sampleClip(this.clip, this.time));
    document.querySelectorAll('[data-clip]').forEach(btn => btn.classList.toggle('active', btn.dataset.clip === name));
    this._updateReadout();
  }

  seek(seconds) {
    this.time = Math.max(0, Number(seconds) || 0);
    const clip = CLIPS[this.clip];
    if (!clip.loop) this.time = Math.min(this.time, clip.duration);
    this.apply(sampleClip(this.clip, this.time));
    this._updateReadout();
  }

  pause() { this.playing = false; this._lastFrame = null; }
  play() { this.playing = true; this._lastFrame = performance.now(); }

  setDebug(enabled) {
    this.debug = Boolean(enabled);
    this._renderDebug();
  }

  apply(pose) {
    if (!this.svg) return;

    // Facial expression becomes controlled deformation of rig nodes. The eye
    // squash occurs in HEAD space so it cannot detach from the face.
    const p = {
      ...pose,
      nodes: Object.fromEntries(Object.entries(pose.nodes).map(([id,v]) => [id, {...v}])),
      face: {...pose.face},
    };
    const blinkScale = 1 - 0.84 * Math.max(0, Math.min(1, p.face.blink));
    p.nodes.eye_l.sy *= blinkScale;
    p.nodes.eye_r.sy *= blinkScale;
    p.nodes.brow_l.y -= 3 * p.face.blink;
    p.nodes.brow_r.y -= 3 * p.face.blink;

    const world = worldMatrices(p);
    for (const [id, spec] of Object.entries(RIG)) {
      if (!spec.element) continue;
      const el = this.svg.getElementById(spec.element);
      if (el) el.setAttribute('transform', svgMatrix(world[id]));
    }

    const lookX = Math.max(-9, Math.min(9, p.face.lookX));
    const lookY = Math.max(-7, Math.min(7, p.face.lookY));
    this.svg.getElementById('pupil-l')?.setAttribute('transform', `translate(${lookX} ${lookY})`);
    this.svg.getElementById('pupil-r')?.setAttribute('transform', `translate(${lookX} ${lookY})`);

    const mouth = Math.max(0, Math.min(1, p.face.mouth));
    const lower = this.svg.getElementById('beak-lower');
    if (lower) lower.setAttribute('transform', `translate(0 ${(12 * mouth).toFixed(2)}) scale(1 ${(1 + 0.10 * mouth).toFixed(3)})`);
    const cavity = this.svg.getElementById('mouth-cavity');
    if (cavity) cavity.setAttribute('opacity', String((0.88 * mouth).toFixed(3)));

    this._pose = p;
    this._world = world;
    if (this.debug) this._renderDebug();
  }

  _tick(ts) {
    if (this.playing) {
      if (this._lastFrame == null) this._lastFrame = ts;
      const dt = Math.min(0.05, Math.max(0, (ts - this._lastFrame) / 1000));
      this._lastFrame = ts;
      this.time += dt;
      const clip = CLIPS[this.clip];
      if (!clip.loop && this.time >= clip.duration) {
        this.time = clip.duration;
        this.apply(sampleClip(this.clip, this.time));
        this.setClip('idle');
      } else {
        this.apply(sampleClip(this.clip, this.time));
      }
      this._updateReadout();
    } else {
      this._lastFrame = ts;
    }
    this._raf = requestAnimationFrame(t => this._tick(t));
  }

  _updateReadout() {
    const clip = CLIPS[this.clip];
    const el = document.querySelector('[data-readout]');
    if (el) el.textContent = `${this.clip}  ${this.time.toFixed(2)} / ${clip.duration.toFixed(2)}s`;
  }

  _renderDebug() {
    if (!this.svg) return;
    const layer = this.svg.getElementById('debug-layer');
    if (!layer) return;
    if (!this.debug || !this._world) {
      layer.setAttribute('opacity', '0');
      layer.replaceChildren();
      return;
    }
    layer.setAttribute('opacity', '1');
    layer.replaceChildren();
    const ns = 'http://www.w3.org/2000/svg';
    for (const [id, spec] of Object.entries(RIG)) {
      const p = transformPoint(this._world[id], [0,0]);
      if (spec.parent) {
        const q = transformPoint(this._world[spec.parent], [0,0]);
        const line = document.createElementNS(ns, 'line');
        line.setAttribute('x1', q[0]); line.setAttribute('y1', q[1]);
        line.setAttribute('x2', p[0]); line.setAttribute('y2', p[1]);
        line.setAttribute('stroke', '#2f7ef8'); line.setAttribute('stroke-width', '2');
        line.setAttribute('stroke-opacity', '.55');
        layer.append(line);
      }
      const dot = document.createElementNS(ns, 'circle');
      dot.setAttribute('cx', p[0]); dot.setAttribute('cy', p[1]); dot.setAttribute('r', id.includes('garment') ? '6' : '4.5');
      dot.setAttribute('fill', id.includes('garment') || id.includes('cuff') ? '#34b56f' : '#ff4d79');
      dot.setAttribute('stroke', '#fff'); dot.setAttribute('stroke-width', '2');
      layer.append(dot);
    }
  }

  snapshot() {
    return { clip: this.clip, time: this.time, pose: this._pose, world: this._world };
  }
}

const stage = document.getElementById('stage');
const puppet = new OwlPuppet(stage);
window.owlPuppet = puppet;
puppet.init().catch(err => {
  console.error(err);
  stage.innerHTML = `<div class="load-error">${err.message}</div>`;
});

export { OwlPuppet };
