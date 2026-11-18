// Opt-in integration check. Refuses to predict unless the server lists STUDENT-TEST trips.
// Run from ui: node tests/live-api-smoke.js http://127.0.0.1:8000
import assert from 'node:assert/strict';
import {readdir} from 'node:fs/promises';
import {createApiService,normalizeBackendUrl} from '../services/trajectory-service.js';

const url=normalizeBackendUrl(process.argv[2]||'http://127.0.0.1:8000');
const service=createApiService(url);
const folders=await readdir(new URL('../../data/Student%20folder/student_test/',import.meta.url),{withFileTypes:true});
const expected=folders.filter(entry=>entry.isDirectory()).map(entry=>entry.name).sort();
const trips=await service.listTrips();
assert.deepEqual(trips.map(t=>t.id).sort(),expected,'Refusing to run: backend must expose the student-testing partition only.');
assert.equal(expected.length,10);
await service.health();
const preflight=await fetch(url+'/predict',{method:'OPTIONS',headers:{Origin:'http://127.0.0.1:5173','Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'content-type'}});
assert.equal(preflight.status,200);assert.ok(['*','http://127.0.0.1:5173'].includes(preflight.headers.get('access-control-allow-origin')));
const results=[];
for(const {id} of trips){
  const result=await service.reconstruct(id);
  assert.equal(result.demo,false);assert.equal(result.trip.id,id);
  assert.equal(result.observations.length,result.counts.valid_points);
  assert.equal(result.matched_route.coordinates.length,result.counts.route_segments);
  assert.ok(result.matched_route.coordinates.length>0);
  assert.ok(!('reference_route' in result)&&!('labels' in result));
  results.push({trip_id:id,gps_points:result.observations.length,route_sections:result.matched_route.coordinates.length,
    top1_accuracy:result.metrics.top1_accuracy,route_f1:result.metrics.route_f1});
  console.log('PASS',id,`(${result.observations.length} GPS points, ${result.counts.route_segments} sections)`);
}
console.log(JSON.stringify({passed:true,trips:results.length,cors_verified:true,evaluator_data_used:false,results},null,2));
