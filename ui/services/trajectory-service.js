import { CONFIG } from '../config.js';
import { DEMO_TRIPS, createDemoTrip } from './demo-data.js';

const coordinate = p=>Array.isArray(p)&&p.length===2&&p.every(Number.isFinite)&&Math.abs(p[0])<=180&&Math.abs(p[1])<=90;
function geometry(value,name){
  if(value==null)return null;
  if(value.type!=='MultiLineString'||!Array.isArray(value.coordinates)||!value.coordinates.every(line=>Array.isArray(line)&&line.length>=2&&line.every(coordinate)))throw new Error(`The service returned invalid ${name} geometry.`);
  return {type:'MultiLineString',coordinates:value.coordinates};
}
export function validateResponse(data, expectedId) {
  if(!data||data.trip?.id!==expectedId||!Array.isArray(data.observations)||!data.observations.length||!data.observations.every(o=>coordinate(o.coordinates)))throw new Error('The service returned an incomplete or mismatched trip.');
  for(const key of ['top1_accuracy','route_continuity'])if(data.metrics?.[key]!=null&&(!Number.isFinite(data.metrics[key])||data.metrics[key]<0||data.metrics[key]>1))throw new Error('The service returned invalid metrics.');
  if(data.metrics?.processing_ms!=null&&(!Number.isFinite(data.metrics.processing_ms)||data.metrics.processing_ms<0))throw new Error('The service returned an invalid processing time.');
  // Whitelist display fields: reference routes and ground-truth labels are never forwarded.
  return { trip:{id:data.trip.id,name:String(data.trip.name||data.trip.id),description:String(data.trip.description||''),
    observation_count:data.observations.length,duration_s:data.trip.duration_s,distance_m:data.trip.distance_m},
    observations:data.observations.map(o=>({id:o.id,coordinates:o.coordinates,timestamp_s:o.timestamp_s})),
    matched_route:geometry(data.matched_route,'reconstructed route'),classifier_route:geometry(data.classifier_route,'classifier'),
    metrics:{top1_accuracy:data.metrics?.top1_accuracy??null,route_continuity:data.metrics?.route_continuity??null,processing_ms:data.metrics?.processing_ms??null},
    warnings:Array.isArray(data.warnings)?data.warnings.map(String):[],demo:data.demo===true };
}
function wait(ms,signal){return new Promise((resolve,reject)=>{
  const abort=()=>{clearTimeout(timer);reject(new DOMException('Cancelled','AbortError'));};
  if(signal?.aborted){reject(new DOMException('Cancelled','AbortError'));return;}
  const timer=setTimeout(()=>{signal?.removeEventListener('abort',abort);resolve();},ms);
  signal?.addEventListener('abort',abort,{once:true});
});}
export const demoService={
  async listTrips(){return DEMO_TRIPS.map(({id,name,description})=>({id,name,description}));},
  async preview(id){const data=createDemoTrip(id);return {trip:data.trip,observations:data.observations,demo:true};},
  async reconstruct(id,{signal,onProgress=()=>{}}={}){
    for(const phase of ['Preparing observations','Scoring road candidates','Connecting the route']){onProgress(phase);await wait(360,signal);}
    return validateResponse(createDemoTrip(id),id);
  },
};
export function createApiService(baseUrl,fetcher=fetch){
  const base=baseUrl.replace(/\/$/,'');
  async function request(path,options={}){
    const controller=new AbortController();
    const external=options.signal;
    const abort=()=>controller.abort(external?.reason);
    if(external?.aborted)abort();else external?.addEventListener('abort',abort,{once:true});
    const timer=setTimeout(()=>controller.abort(new DOMException('Request timed out','TimeoutError')),CONFIG.timeoutMs);
    try{
      const response=await fetcher(base+path,{...options,signal:controller.signal,headers:{'Content-Type':'application/json',...options.headers}});
      if(!response.ok)throw new Error(`The route service returned HTTP ${response.status}. Please try again.`);
      return await response.json();
    }catch(error){if(external?.aborted)throw new DOMException('Cancelled','AbortError');if(error.name==='TimeoutError'||controller.signal.aborted)throw new Error('The route service took too long to respond. Please try again.');if(error instanceof TypeError)throw new Error('Unable to reach the route service. Check the backend connection.');throw error;}
    finally{clearTimeout(timer);external?.removeEventListener('abort',abort);}
  }
  return {
    async listTrips(){const data=await request('/trips');if(!Array.isArray(data.trips)||!data.trips.length)throw new Error('No trips are available from the route service.');return data.trips.map(t=>({id:String(t.id),name:String(t.name||t.id),description:String(t.description||'')}));},
    async preview(id){const data=await request(`/trips/${encodeURIComponent(id)}`);return validateResponse(data,id);},
    async reconstruct(id,{signal,onProgress=()=>{}}={}){onProgress('Reconstructing the journey');const data=await request('/match',{method:'POST',body:JSON.stringify({trip_id:id}),signal});return validateResponse(data,id);},
  };
}
export const service=CONFIG.mode==='demo'?demoService:createApiService(CONFIG.apiBaseUrl);
