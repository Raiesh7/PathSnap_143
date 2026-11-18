// Entirely synthetic fixtures. These are not model predictions or evaluator GPS.
export const DEMO_TRIPS = Object.freeze([
  ['demo-01','City centre connector','Parallel streets · steady sampling'],
  ['demo-02','Garden district loop','Local turns · dense observations'],
  ['demo-03','Eastbound commute','Longer journey · steady movement'],
  ['demo-04','Neighbourhood crossing','Junctions · close road candidates'],
  ['demo-05','Sparse afternoon track','Fewer observations · wider spacing'],
  ['demo-06','South district link','Mixed direction · short segments'],
  ['demo-07','Evening connection','Local streets · positional noise'],
  ['demo-08','North quarter journey','Several turns · regular sampling'],
  ['demo-09','Market street passage','Dense candidates · short journey'],
  ['demo-10','Outer district connector','Long segments · sparse sampling'],
].map(([id,name,description],index)=>Object.freeze({id,name,description,index})));

const ANCHORS = [[77.5947,12.9756],[77.5955,12.9750],[77.5968,12.9745],[77.5985,12.9741],[77.6003,12.9738],[77.6021,12.9737],[77.6041,12.9735],[77.6062,12.9732],[77.6080,12.9727],[77.6098,12.9723],[77.6118,12.9721],[77.6130,12.9725],[77.6142,12.9734],[77.6152,12.9745],[77.6162,12.9756],[77.6173,12.9766],[77.6183,12.9774]];

export function createDemoTrip(id) {
  const trip = DEMO_TRIPS.find(t=>t.id===id);
  if (!trip) throw new Error('That demo journey is not available. Please choose another.');
  const shift = (trip.index-4)*0.00022;
  const anchors = ANCHORS.map(([lon,lat])=>[lon+shift,lat+shift*.5]);
  const steps = trip.index===4 ? 4 : 10;
  const route=[];
  for(let i=0;i<anchors.length-1;i++) for(let j=0;j<steps;j++) {
    const f=j/steps;
    route.push(anchors[i].map((v,k)=>v+(anchors[i+1][k]-v)*f));
  }
  route.push(anchors.at(-1));
  const observations=route.map(([lon,lat],i)=>({
    id:i,coordinates:[lon+Math.sin(i*1.7+trip.index)*.00015,lat+Math.cos(i*1.1)*.00014],timestamp_s:i*3,
  }));
  const classifier=route.map(([lon,lat],i)=>[lon,lat+(i>steps*6&&i<steps*9?.00035:0)]);
  return { trip:{id:trip.id,name:trip.name,description:trip.description,observation_count:observations.length,
    duration_s:(observations.length-1)*3,distance_m:3260+trip.index*47},observations,
    matched_route:{type:'MultiLineString',coordinates:[route]},
    classifier_route:{type:'MultiLineString',coordinates:[classifier]},
    metrics:{top1_accuracy:.942-(trip.index%3)*.009,route_continuity:.981-(trip.index%4)*.004,processing_ms:842+trip.index*73},
    demo:true,warnings:[] };
}
