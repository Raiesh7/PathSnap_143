import {CONFIG} from './config.js';
import {service} from './services/trajectory-service.js';
import {JourneyMap} from './map-view.js';

const $=id=>document.getElementById(id);
const state={trip:null,result:null,controller:null,token:0,busy:false};
let map;
try{map=new JourneyMap();}catch(error){$('map-empty').hidden=false;$('fit-button').disabled=true;console.error(error);}
const say=message=>{$('announcer').textContent=message;};
const formatNumber=value=>Number.isFinite(value)?new Intl.NumberFormat('en-IN').format(value):'—';
function metric(id,value,unit){const element=$(id);element.replaceChildren(document.createTextNode(Number.isFinite(value)?value.toFixed(unit==='s'?2:1):'—'));const small=document.createElement('small');small.textContent=unit;element.append(small);}
function setBusy(busy){
  state.busy=busy;$('map-loading').hidden=!busy;$('map').setAttribute('aria-busy',String(busy));
  $('reconstruct-button').disabled=busy||!state.trip;
  $('reconstruct-button').querySelector('span').textContent=busy?'Reconstructing…':state.result?'Reconstruct again':'Reconstruct route';
  $('export-button').disabled=busy||!state.result;
}
function clearResult(){
  map?.clearResult();
  state.result=null;metric('metric-accuracy',null,'%');metric('metric-continuity',null,'%');metric('metric-time',null,'s');
  $('results-badge').textContent='AWAITING A RUN';$('result-title').textContent='Your next journey starts here';
  $('result-description').textContent='Select a trip and reconstruct its route to explore the results.';
  $('export-button').disabled=true;$('error-panel').hidden=true;
  $('layer-classifier').disabled=true;$('layer-matched').disabled=true;
}
function showError(error){
  $('error-message').textContent=error.message||'An unexpected error occurred. Please try again.';
  $('error-panel').hidden=false;$('results-badge').textContent='COULD NOT COMPLETE';say($('error-message').textContent);
}
function setEnvironment(demo){
  document.querySelector('.demo-notice').hidden=!demo;
  document.querySelector('.demo-badge').lastChild.textContent=demo?'Demo mode':'API mode';
  document.querySelector('.connection-state strong').textContent=demo?'Frontend preview':'Connected workspace';
  document.querySelector('.connection-state span:last-child').textContent=demo?'Backend not connected':'Route service connected';
  document.querySelector('.field-label').textContent=demo?'Demo trip':'Trip';
  $('map-subtitle').textContent=demo?'Synthetic GPS track · illustrative street network':'GPS observations · road network';
  document.querySelector('.map-label').lastChild.textContent=demo?'ILLUSTRATIVE MAP':'JOURNEY MAP';
  document.querySelector('.page-footer>span:last-child').textContent=demo?'Demo data only · No backend requests':'Connected to the route service';
  document.querySelector('.button-hint').textContent=demo?'Preview a simulated reconstruction':'Process this journey through the route service';
  document.querySelector('.map-loading>span').textContent=demo?'Simulated demo run':'Waiting for the route service';
  document.querySelector('.dialog-note').textContent=demo?'This interface currently simulates that flow with synthetic fixtures. No model is executed, no evaluator files are loaded, and displayed metrics are illustrative.':'This workspace requests predictions from the configured route service. Reference routes and labels remain on the backend.';
}
async function selectTrip(id){
  const token=++state.token;state.controller?.abort();state.trip=null;clearResult();map?.clearPreview();setBusy(false);
  $('trip-name').textContent='Loading journey';$('trip-description').textContent='Preparing the preview';
  $('trip-id-label').textContent=id.toUpperCase();
  for(const field of ['point-count','duration','distance'])$(field).textContent='—';
  if(!id)return;
  try{
    const preview=await service.preview(id);
    if(token!==state.token)return;
    state.trip=preview.trip;$('trip-name').textContent=preview.trip.name;
    $('trip-description').textContent=preview.trip.description;
    $('trip-id-label').textContent=preview.trip.id.toUpperCase();
    $('point-count').textContent=formatNumber(preview.observations.length);
    const duration=preview.trip.duration_s;
    $('duration').textContent=Number.isFinite(duration)?`${Math.floor(duration/60)}m ${Math.floor(duration%60)}s`:'—';
    $('distance').textContent=Number.isFinite(preview.trip.distance_m)?`${(preview.trip.distance_m/1000).toFixed(2)} km`:'—';
    setEnvironment(preview.demo===true);map?.preview(preview);
    setBusy(false);say(`${preview.trip.name} loaded. ${preview.observations.length} GPS observations.`);
  }catch(error){if(token===state.token){showError(error);$('trip-name').textContent='Journey unavailable';$('trip-description').textContent='Choose another trip or retry.';}}
}
async function reconstruct(){
  if(!state.trip||state.busy)return;
  const id=state.trip.id,token=++state.token;state.controller?.abort();state.controller=new AbortController();
  clearResult();setBusy(true);$('results-badge').textContent='PROCESSING';
  metric('metric-accuracy',null,'%');metric('metric-continuity',null,'%');metric('metric-time',null,'s');
  try{
    const result=await service.reconstruct(id,{signal:state.controller.signal,onProgress:phase=>{
      if(token===state.token){$('loading-label').textContent=phase;say(phase);}
    }});
    if(token!==state.token)return;
    if(!result.matched_route?.coordinates.length)throw new Error(result.warnings[0]||'No connected route was returned for this journey.');
    state.result=result;map?.result(result);
    metric('metric-accuracy',result.metrics.top1_accuracy==null?null:result.metrics.top1_accuracy*100,'%');
    metric('metric-continuity',result.metrics.route_continuity==null?null:result.metrics.route_continuity*100,'%');
    metric('metric-time',result.metrics.processing_ms==null?null:result.metrics.processing_ms/1000,'s');
    $('results-badge').textContent=result.demo?'DEMO RESULTS':'COMPLETE';
    $('result-title').textContent=result.demo?'A connected journey, illustrated.':'Journey reconstructed';
    $('result-description').textContent=result.demo?'Synthetic route and metrics · no model was executed.':result.warnings.join(' ')||'Explore the reconstructed path or export its geometry.';
    $('layer-matched').disabled=false;$('layer-classifier').disabled=!result.classifier_route?.coordinates.length;
    say(result.demo?'Demo reconstruction complete. Displayed metrics are synthetic.':'Journey reconstruction complete.');
    if(matchMedia('(max-width:760px)').matches)document.querySelector('.map-panel').scrollIntoView({
      behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?'instant':'smooth',block:'start',
    });
  }catch(error){if(error.name!=='AbortError'&&token===state.token)showError(error);}
  finally{if(token===state.token)setBusy(false);}
}
$('trip-select').addEventListener('change',event=>selectTrip(event.target.value));
$('reconstruct-button').addEventListener('click',reconstruct);
$('retry-button').addEventListener('click',()=>state.trip?reconstruct():selectTrip($('trip-select').value));
$('fit-button').addEventListener('click',()=>map?.fit());
for(const name of ['gps','classifier','matched'])$(`layer-${name}`).addEventListener('change',event=>map?.setVisible(name,event.target.checked));
$('export-button').addEventListener('click',()=>{
  const result=state.result;if(!result)return;
  const geojson={type:'FeatureCollection',features:[{type:'Feature',properties:{trip_id:result.trip.id,demo:result.demo,
    source:result.demo?'Synthetic UI fixture; not a model prediction':'Route service'},geometry:result.matched_route}]};
  const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([JSON.stringify(geojson,null,2)],{type:'application/geo+json'}));
  link.download=`pathsnap-${result.trip.id}${result.demo?'-synthetic':''}.geojson`;link.click();
  setTimeout(()=>URL.revokeObjectURL(link.href),1000);say('Route GeoJSON exported.');
});
$('help-button').addEventListener('click',()=>$('help-dialog').showModal());
for(const id of ['close-help','help-done'])$(id).addEventListener('click',()=>$('help-dialog').close());
async function init(){
  try{
    const trips=await service.listTrips();$('trip-select').replaceChildren();
    for(const trip of trips){const option=document.createElement('option');option.value=trip.id;option.textContent=`${trip.id.toUpperCase()} · ${trip.name}`;$('trip-select').append(option);}
    $('trip-help').textContent=CONFIG.mode==='demo'?`${trips.length} sample journeys · no evaluator data`:`${trips.length} journeys available`;
    await selectTrip(trips[0]?.id);
  }catch(error){showError(error);$('trip-select').disabled=true;}
}
init();
