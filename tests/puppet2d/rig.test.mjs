import test from 'node:test';
import assert from 'node:assert/strict';
import { RIG, CLIPS } from '../../puppet2d/rig-data.mjs';
import { sampleClip, worldMatrices, transformPoint, validateRig } from '../../puppet2d/rig-core.mjs';

const dist = (a,b) => Math.hypot(a[0]-b[0], a[1]-b[1]);
const origin = (m) => transformPoint(m,[0,0]);

function snapshot(clip,t){
  const pose = sampleClip(clip,t);
  return { pose, world: worldMatrices(pose) };
}

test('rig topology and all dense clip samples validate', () => {
  const report = validateRig();
  assert.deepEqual(report.errors, []);
  assert.equal(report.ok, true);
});

test('garment panels are structurally attached to body', () => {
  assert.equal(RIG.garment_front.parent, 'body');
  assert.equal(RIG.garment_back.parent, 'body');
  const a = snapshot('wave',0);
  const b = snapshot('wave',0.45);
  const bodyMove = dist(origin(a.world.body), origin(b.world.body));
  const clothMove = dist(origin(a.world.garment_front), origin(b.world.garment_front));
  assert.ok(bodyMove > 1, `body should actually move in the test, got ${bodyMove}`);
  assert.ok(clothMove >= bodyMove * 0.85, `garment must inherit body transport: body=${bodyMove}, cloth=${clothMove}`);
});

test('patterned cuffs inherit wing pivots instead of floating in body space', () => {
  assert.equal(RIG.cuff_l.parent, 'wing_l');
  assert.equal(RIG.cuff_r.parent, 'wing_r');
  const a = snapshot('wave',0);
  const b = snapshot('wave',1.16);
  const wingTravel = dist(origin(a.world.wing_l), transformPoint(b.world.wing_l,[0,120]));
  const cuffTravel = dist(origin(a.world.cuff_l), origin(b.world.cuff_l));
  assert.ok(cuffTravel > 80, `left cuff should travel with waving wing, got ${cuffTravel}`);
  assert.ok(wingTravel > 80, `wave must produce meaningful wing motion, got ${wingTravel}`);
});

test('non-loop actions settle back into their authored rest pose', () => {
  for (const name of ['wave','celebrate']) {
    const start = sampleClip(name,0);
    const end = sampleClip(name,CLIPS[name].duration);
    for (const id of Object.keys(RIG)) assert.deepEqual(end.nodes[id], start.nodes[id], `${name}/${id} did not close`);
  }
});

test('idle and talk loops are mathematically closed', () => {
  for (const name of ['idle','talk']) {
    const a = sampleClip(name,0);
    const b = sampleClip(name,CLIPS[name].duration);
    assert.deepEqual(b.nodes,a.nodes,`${name} node loop mismatch`);
    assert.deepEqual(b.face,a.face,`${name} face loop mismatch`);
  }
});

test('blink and mouth channels stay bounded during dense sampling', () => {
  for (const [name,clip] of Object.entries(CLIPS)) {
    for (let i=0;i<=240;i++) {
      const p=sampleClip(name,clip.duration*i/240);
      assert.ok(p.face.blink>=0 && p.face.blink<=1,`${name} blink ${p.face.blink}`);
      assert.ok(p.face.mouth>=0 && p.face.mouth<=1,`${name} mouth ${p.face.mouth}`);
    }
  }
});
