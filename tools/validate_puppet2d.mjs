import { CLIPS } from '../puppet2d/rig-data.mjs';
import { sampleClip, worldMatrices, transformPoint, validateRig } from '../puppet2d/rig-core.mjs';

const d=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
const p=(m,xy=[0,0])=>transformPoint(m,xy);

const report=validateRig();
if(!report.ok){
  console.error(JSON.stringify(report,null,2));
  process.exit(1);
}

const metrics={};
for(const [name,clip] of Object.entries(CLIPS)){
  let maxGarmentBodyError=0;
  let maxLeftCuffWingError=0;
  let maxRightCuffWingError=0;
  let maxWingTipTravel=0;
  const rest=worldMatrices(sampleClip(name,0));
  const restLeftTip=p(rest.wing_l,[0,190]);
  for(let i=0;i<=Math.ceil(clip.duration*60);i++){
    const t=clip.duration*i/Math.ceil(clip.duration*60);
    const w=worldMatrices(sampleClip(name,t));
    // These are relative-distance invariants in a hierarchical 2D rig. If
    // they change unexpectedly, a garment/cuff has become an independent layer.
    const body=p(w.body), cloth=p(w.garment_front);
    const body0=p(rest.body), cloth0=p(rest.garment_front);
    maxGarmentBodyError=Math.max(maxGarmentBodyError,Math.abs(d(body,cloth)-d(body0,cloth0)));

    const wl=p(w.wing_l), cl=p(w.cuff_l), wl0=p(rest.wing_l), cl0=p(rest.cuff_l);
    const wr=p(w.wing_r), cr=p(w.cuff_r), wr0=p(rest.wing_r), cr0=p(rest.cuff_r);
    maxLeftCuffWingError=Math.max(maxLeftCuffWingError,Math.abs(d(wl,cl)-d(wl0,cl0)));
    maxRightCuffWingError=Math.max(maxRightCuffWingError,Math.abs(d(wr,cr)-d(wr0,cr0)));
    maxWingTipTravel=Math.max(maxWingTipTravel,d(restLeftTip,p(w.wing_l,[0,190])));
  }
  metrics[name]={
    garment_body_distance_error_px:+maxGarmentBodyError.toFixed(6),
    left_cuff_wing_distance_error_px:+maxLeftCuffWingError.toFixed(6),
    right_cuff_wing_distance_error_px:+maxRightCuffWingError.toFixed(6),
    left_wing_tip_max_travel_px:+maxWingTipTravel.toFixed(3),
  };
}

console.log(JSON.stringify({ok:true,metrics},null,2));
