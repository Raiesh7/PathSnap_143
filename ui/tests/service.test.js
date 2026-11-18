import test from 'node:test';
import assert from 'node:assert/strict';
import {createDemoTrip,DEMO_TRIPS} from '../services/demo-data.js';
import {createApiService,validateResponse,normalizePrediction,normalizeBackendUrl,demoService} from '../services/trajectory-service.js';

function apiResult(){return {
  trip_id:'00000052',gps_points:[[48.135,11.582,1],[48.136,11.584,4]],
  predicted_route:[[[48.135,11.582],[48.136,11.584]],[[48.14,11.59],[48.141,11.591]]],
  counts:{gps_points:2,valid_points:2,points_with_candidates:2,candidate_rows:8,hmm_breaks:1,route_segments:2,route_arcs:4},
  metrics:{top1_accuracy:.95,route_continuity:.98,route_f1:.97,classifier_only_top1_accuracy:.89,scored_points:2,unscored_points:0},
  timings_s:{total_prediction:1.25,scoring:.4},model:{file:'lightgbm_tuned.joblib',hmm_beta_m:3},
};}

test('all ten fixtures are synthetic and use valid GeoJSON coordinates',()=>{
  assert.equal(DEMO_TRIPS.length,10);
  assert.equal(new Set(DEMO_TRIPS.map(t=>t.id)).size,10);
  for(const trip of DEMO_TRIPS){const result=validateResponse(createDemoTrip(trip.id),trip.id);assert.equal(result.demo,true);assert.ok(result.observations.length);assert.ok(result.matched_route.coordinates.length);}
});
test('reference fields never pass through the display adapter',()=>{
  const input=createDemoTrip('demo-01');input.reference_route={secret:true};input.ground_truth=[1,2];input.observations[0].label=1;
  const result=validateResponse(input,'demo-01');assert.ok(!('ground_truth' in result));assert.ok(!('reference_route' in result));assert.ok(!('label' in result.observations[0]));
});
test('mismatched trips, invalid coordinates and invalid metrics are rejected',()=>{
  assert.throws(()=>validateResponse(createDemoTrip('demo-01'),'demo-02'),/mismatched/);
  const input=createDemoTrip('demo-01');input.observations[0].coordinates=[200,12];assert.throws(()=>validateResponse(input,'demo-01'),/incomplete/);
  const other=createDemoTrip('demo-01');other.metrics.top1_accuracy=97;assert.throws(()=>validateResponse(other,'demo-01'),/metrics/);
});
test('reconstruction sends only the selected trip ID to the API',async()=>{
  let captured;
  const api=createApiService('http://localhost:8000/',async(url,options)=>{captured={url,...options};return {ok:true,json:async()=>apiResult()};});
  const result=await api.reconstruct('00000052');assert.equal(captured.url,'http://localhost:8000/predict');assert.equal(captured.method,'POST');assert.deepEqual(JSON.parse(captured.body),{trip_id:'00000052'});assert.equal(result.demo,false);
});
test('HTTP and network failures produce useful errors',async()=>{
  const http=createApiService('http://localhost',async()=>({ok:false,status:503}));await assert.rejects(()=>http.listTrips(),/503/);
  const network=createApiService('http://localhost',async()=>{throw new TypeError('Failed to fetch');});await assert.rejects(()=>network.listTrips(),/Unable to reach/);
});
test('a cancelled demo reconstruction stops without returning stale results',async()=>{
  const controller=new AbortController();const pending=demoService.reconstruct('demo-01',{signal:controller.signal});controller.abort();await assert.rejects(pending,{name:'AbortError'});
});
test('API coordinates are converted once and disconnected route sections stay separate',()=>{
  const input=apiResult();input.reference_route=['secret'];input.metrics.labels=[1,2];
  const result=normalizePrediction(input,'00000052');
  assert.deepEqual(result.observations[0].coordinates,[11.582,48.135]);
  assert.deepEqual(result.matched_route.coordinates[0][0],[11.582,48.135]);
  assert.equal(result.matched_route.coordinates.length,2);assert.equal(result.metrics.processing_ms,1250);
  assert.equal(result.trip.duration_s,3);assert.ok(result.trip.distance_m>0&&result.trip.distance_m<600);
  assert.equal(result.classifier_route,null);assert.equal(result.metrics.route_f1,.97);
  assert.ok(!('reference_route' in result));assert.ok(!('labels' in result.metrics));
});
test('no-reference predictions display unavailable metrics without inventing accuracy',()=>{
  const input=apiResult();input.metrics=null;
  const result=normalizePrediction(input,'00000052');assert.equal(result.metrics.top1_accuracy,null);assert.equal(result.metrics.route_f1,null);assert.ok(result.warnings.length);
});
test('trip IDs keep leading zeroes and selecting a trip makes no preview request',async()=>{
  let requests=0;
  const api=createApiService('http://localhost',async()=>{requests++;return {ok:true,json:async()=>({count:1,trips:[{trip_id:'00000052'}]})};});
  const trips=await api.listTrips();assert.equal(trips[0].id,'00000052');
  const preview=await api.preview(trips[0].id);assert.equal(preview.observations.length,0);assert.equal(requests,1);
});
test('empty backend trip lists stay empty instead of returning demo data',async()=>{
  const api=createApiService('http://localhost',async()=>({ok:true,json:async()=>({count:0,trips:[]})}));assert.deepEqual(await api.listTrips(),[]);
});
test('invalid API geometry, identity and aggregate metrics are rejected',()=>{
  assert.throws(()=>normalizePrediction(apiResult(),'00000000'),/mismatched/);
  const point=apiResult();point.predicted_route[0][0]=[200,11];assert.throws(()=>normalizePrediction(point,'00000052'),/geometry/);
  const metric=apiResult();metric.metrics.route_f1=1.5;assert.throws(()=>normalizePrediction(metric,'00000052'),/aggregate/);
});
test('backend addresses accept LAN hosts but reject credentials and unsupported schemes',()=>{
  assert.equal(normalizeBackendUrl('http://192.168.1.20:8000/'),'http://192.168.1.20:8000');
  for(const value of ['file:///C:/secret','http://user:pass@localhost:8000','not a URL','http://localhost:8000/?token=secret'])assert.throws(()=>normalizeBackendUrl(value));
});
