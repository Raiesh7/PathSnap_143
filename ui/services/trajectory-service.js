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

function routeDistance(lines){
  const radians=value=>value*Math.PI/180;
  let distance=0;
  for(const line of lines)for(let i=1;i<line.length;i++){
    const [lon1,lat1]=line[i-1], [lon2,lat2]=line[i];
    const a=Math.sin(radians(lat2-lat1)/2)**2+Math.cos(radians(lat1))*Math.cos(radians(lat2))*Math.sin(radians(lon2-lon1)/2)**2;
    distance+=6371000*2*Math.asin(Math.sqrt(Math.min(1,a)));
  }
  return distance;
}
const apiPoint=point=>Array.isArray(point)&&point.length>=2&&coordinate([point[1],point[0]]);
export function normalizePrediction(data,expectedId){
  if(data?.trip_id!==expectedId||!Array.isArray(data.gps_points)||!data.gps_points.length||
      !data.gps_points.every(point=>apiPoint(point)&&Number.isFinite(point[2]))||
      !Array.isArray(data.predicted_route))throw new Error('The service returned an incomplete or mismatched prediction.');
  const observations=data.gps_points.map(([lat,lon,t],id)=>({id,coordinates:[lon,lat],timestamp_s:t}));
  const lines=[];
  for(const line of data.predicted_route){
    if(!Array.isArray(line)||!line.every(apiPoint))throw new Error('The service returned invalid reconstructed route geometry.');
    if(line.length>=2)lines.push(line.map(([lat,lon])=>[lon,lat]));
  }
  const timestamps=observations.map(o=>o.timestamp_s);
  let minimum=Infinity,maximum=-Infinity;
  for(const value of timestamps){minimum=Math.min(minimum,value);maximum=Math.max(maximum,value);}
  const totalSeconds=data.timings_s?.total_prediction;
  if(totalSeconds!=null&&(!Number.isFinite(totalSeconds)||totalSeconds<0))throw new Error('The service returned invalid prediction timing.');
  const result=validateResponse({trip:{id:expectedId,name:`Trip ${expectedId}`,description:'Tuned LightGBM + HMM / Viterbi',
    duration_s:maximum-minimum,distance_m:routeDistance(lines)},observations,
    matched_route:{type:'MultiLineString',coordinates:lines},classifier_route:null,
    metrics:{top1_accuracy:data.metrics?.top1_accuracy??null,route_continuity:data.metrics?.route_continuity??null,
      processing_ms:totalSeconds==null?null:totalSeconds*1000},demo:false},expectedId);
  // Accept aggregate numbers only. Never forward reference geometry or per-point labels.
  const extraMetrics={};
  for(const key of ['route_f1','classifier_only_top1_accuracy']){
    const value=data.metrics?.[key];
    if(value!=null&&(!Number.isFinite(value)||value<0||value>1))throw new Error('The service returned invalid aggregate metrics.');
    extraMetrics[key]=value??null;
  }
  for(const key of ['scored_points','unscored_points']){
    const value=data.metrics?.[key];
    if(value!=null&&(!Number.isInteger(value)||value<0))throw new Error('The service returned invalid scoring counts.');
    extraMetrics[key]=value??null;
  }
  const counts={};
  for(const key of ['gps_points','valid_points','points_with_candidates','candidate_rows','hmm_breaks','route_segments','route_arcs']){
    const value=data.counts?.[key];
    if(value!=null&&(!Number.isInteger(value)||value<0))throw new Error('The service returned invalid route counts.');
    counts[key]=value??null;
  }
  result.metrics={...result.metrics,...extraMetrics};result.counts=counts;
  if(!data.metrics)result.warnings.push('Scoring is unavailable because this trip has no server-side reference.');
  if(counts.valid_points!=null&&counts.points_with_candidates!=null&&counts.points_with_candidates<counts.valid_points)
    result.warnings.push(`${counts.valid_points-counts.points_with_candidates} GPS observations had no road candidates.`);
  return result;
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
      const response=await fetcher(base+path,{...options,signal:controller.signal,headers:{...(options.body?{'Content-Type':'application/json'}:{}),...options.headers}});
      if(!response.ok)throw new Error(`The route service returned HTTP ${response.status}. ${response.status===404?'This trip may no longer be available. Refresh the trip list.':'Please try again.'}`);
      return await response.json();
    }catch(error){if(external?.aborted)throw new DOMException('Cancelled','AbortError');if(error.name==='TimeoutError'||controller.signal.aborted)throw new Error('The route service took too long to respond. Please try again.');if(error instanceof TypeError)throw new Error('Unable to reach the route service. Check the backend connection.');throw error;}
    finally{clearTimeout(timer);external?.removeEventListener('abort',abort);}
  }
  return {
    async health(){const data=await request('/health');if(data.status!=='ok')throw new Error('The backend is not ready.');return {status:data.status,trips_available:data.trips_available};},
    async listTrips(){
      const data=await request('/trips');
      if(!Array.isArray(data.trips)||!data.trips.every(t=>typeof t.trip_id==='string'&&/^[0-9A-Za-z_-]{1,64}$/.test(t.trip_id)))throw new Error('The service returned an invalid trip list.');
      if(new Set(data.trips.map(t=>t.trip_id)).size!==data.trips.length)throw new Error('The service returned duplicate trip IDs.');
      return data.trips.map(t=>({id:t.trip_id,name:`Trip ${t.trip_id}`,description:'Ready for reconstruction'}));
    },
    // The API intentionally has no preview endpoint. Do not run prediction on selection.
    async preview(id){return {trip:{id,name:`Trip ${id}`,description:'Reconstruct to load observations and the matched route'},observations:[],demo:false};},
    async reconstruct(id,{signal,onProgress=()=>{}}={}){
      onProgress('Running the prediction pipeline');
      const data=await request('/predict',{method:'POST',body:JSON.stringify({trip_id:id}),signal});
      return normalizePrediction(data,id);
    },
  };
}

export function normalizeBackendUrl(value){
  let url;try{url=new URL(value.trim());}catch{throw new Error('Enter a complete backend address, such as http://192.168.1.20:8000.');}
  if(!['http:','https:'].includes(url.protocol)||url.username||url.password||url.search||url.hash)throw new Error('Use an HTTP or HTTPS backend address without credentials, query parameters or fragments.');
  return url.href.replace(/\/$/,'');
}
export function getBackendUrl(){
  try{return normalizeBackendUrl(globalThis.localStorage?.getItem('pathsnap.backendUrl')||CONFIG.apiBaseUrl);}catch{return CONFIG.apiBaseUrl;}
}
export let service=CONFIG.mode==='demo'?demoService:createApiService(getBackendUrl());
export function configureBackend(value){
  const url=normalizeBackendUrl(value);
  try{globalThis.localStorage?.setItem('pathsnap.backendUrl',url);}catch{/* Connection still works when storage is disabled. */}
  service=createApiService(url);return url;
}
