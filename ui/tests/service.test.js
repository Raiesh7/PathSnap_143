import test from 'node:test';
import assert from 'node:assert/strict';
import {createDemoTrip,DEMO_TRIPS} from '../services/demo-data.js';
import {createApiService,validateResponse,demoService} from '../services/trajectory-service.js';

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
  const api=createApiService('http://localhost:8000/',async(url,options)=>{captured={url,...options};return {ok:true,json:async()=>createDemoTrip('demo-03')};});
  await api.reconstruct('demo-03');assert.equal(captured.url,'http://localhost:8000/match');assert.equal(captured.method,'POST');assert.deepEqual(JSON.parse(captured.body),{trip_id:'demo-03'});
});
test('HTTP and network failures produce useful errors',async()=>{
  const http=createApiService('http://localhost',async()=>({ok:false,status:503}));await assert.rejects(()=>http.listTrips(),/503/);
  const network=createApiService('http://localhost',async()=>{throw new TypeError('Failed to fetch');});await assert.rejects(()=>network.listTrips(),/Unable to reach/);
});
test('a cancelled demo reconstruction stops without returning stale results',async()=>{
  const controller=new AbortController();const pending=demoService.reconstruct('demo-01',{signal:controller.signal});controller.abort();await assert.rejects(pending,{name:'AbortError'});
});
