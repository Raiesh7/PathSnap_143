import {CONFIG} from './config.js';
import {service,configureBackend,getBackendUrl} from './services/trajectory-service.js';
import {JourneyMap} from './map-view.js';

const $=id=>document.getElementById(id);
const state={trip:null,result:null,controller:null,token:0,busy:false,trips:[],connected:false};
let map;
try{map=new JourneyMap();}catch(error){$('map-empty').hidden=false;$('fit-button').disabled=true;console.error(error);}
const say=message=>{$('announcer').textContent=message;};
const formatNumber=value=>Number.isFinite(value)?new Intl.NumberFormat('en-IN').format(value):'—';
const percent=value=>Number.isFinite(value)?`${(value*100).toFixed(1)}%`:'—';
function metric(id,value,unit){
  const element=$(id);element.replaceChildren(document.createTextNode(Number.isFinite(value)?value.toFixed(unit==='s'?2:1):'—'));
  const small=document.createElement('small');small.textContent=unit;element.append(small);
}
function setBusy(busy){
  state.busy=busy;$('map-loading').hidden=!busy;$('map').setAttribute('aria-busy',String(busy));
  $('reconstruct-button').disabled=busy||!state.trip;
  $('reconstruct-button').querySelector('span').textContent=busy?'Reconstructing…':state.result?'Reconstruct again':'Reconstruct route';
  $('export-button').disabled=busy||!state.result;
}
function clearResult(){
  map?.clearResult();state.result=null;
  metric('metric-accuracy',null,'%');metric('metric-continuity',null,'%');metric('metric-time',null,'s');
  $('results-badge').textContent='AWAITING A RUN';$('result-title').textContent='Your next journey starts here';
  $('result-description').textContent='Select a trip and reconstruct its route to explore the results.';
  $('export-button').disabled=true;$('error-panel').hidden=true;$('run-details').hidden=true;
  $('layer-classifier').disabled=true;$('layer-matched').disabled=true;
}
function showError(error){
  document.querySelector('#error-panel strong').textContent='The request couldn’t be completed.';
  $('error-message').textContent=error.message||'An unexpected error occurred. Please try again.';
  $('error-panel').hidden=false;$('results-badge').textContent='COULD NOT COMPLETE';say($('error-message').textContent);
}
function setConnection(status){
  state.connected=status==='connected';
  document.querySelector('.connection-state').dataset.state=status;
  const labels=CONFIG.mode==='demo'?['Frontend preview','Backend not connected']:
    status==='connected'?['Backend connected',getBackendUrl()]:
    status==='connecting'?['Connecting to backend','Checking the route service']:['Backend unavailable','Open Connection to check the address'];
  document.querySelector('.connection-state strong').textContent=labels[0];
  document.querySelector('.connection-state span:last-child').textContent=labels[1];
  document.querySelector('.page-footer>span:last-child').textContent=CONFIG.mode==='demo'?'Demo data only · No backend requests':
    status==='connected'?'Predictions and scoring from the API':'Connect to the route service to begin';
}
function setEnvironment(demo){
  document.querySelector('.demo-notice').hidden=!demo;
  document.querySelector('.demo-badge').classList.toggle('api-mode',!demo);
  document.querySelector('.demo-badge').lastChild.textContent=demo?'Demo mode':'API mode';
  document.querySelector('.field-label').textContent=demo?'Demo trip':'Trip';
  $('map-subtitle').textContent=demo?'Synthetic GPS track · illustrative street network':'GPS observations and predicted road geometry';
  document.querySelector('.map-label').lastChild.textContent=demo?'ILLUSTRATIVE MAP':'JOURNEY MAP';
  document.querySelector('.button-hint').textContent=demo?'Preview a simulated reconstruction':'Run the model and decoder on the backend';
  document.querySelector('.map-loading>span').textContent=demo?'Simulated demo run':'Waiting for the API response';
  document.querySelector('.dialog-note').textContent=demo?'This interface simulates the flow with synthetic fixtures. No model is executed and metrics are illustrative.':
    'The backend runs tuned LightGBM and HMM / Viterbi, then scores its predictions separately. Reference routes and labels stay on the server.';
  $('layer-classifier').closest('label').hidden=!demo;
}
function durationLabel(seconds){
  if(!Number.isFinite(seconds))return '—';
  if(seconds>=3600)return `${Math.floor(seconds/3600)}h ${Math.floor(seconds%3600/60)}m`;
  return `${Math.floor(seconds/60)}m ${Math.floor(seconds%60)}s`;
}
function displayTrip(data,hasObservations){
  state.trip=data.trip;$('trip-name').textContent=data.trip.name;$('trip-description').textContent=data.trip.description;
  $('trip-id-label').textContent=data.trip.id.toUpperCase();
  $('point-count').textContent=hasObservations?formatNumber(data.observations.length):'—';
  $('duration').textContent=durationLabel(data.trip.duration_s);
  $('distance').textContent=Number.isFinite(data.trip.distance_m)?`${(data.trip.distance_m/1000).toFixed(2)} km`:'—';
  $('map-placeholder').hidden=hasObservations;document.querySelector('.map-label').hidden=!hasObservations;
  $('fit-button').disabled=!hasObservations||!map;
}
async function selectTrip(id){
  const token=++state.token;state.controller?.abort();state.trip=null;clearResult();map?.clearPreview();setBusy(false);
  $('trip-name').textContent='Loading journey';$('trip-description').textContent='Preparing the workspace';
  $('trip-id-label').textContent=(id||'').toUpperCase();
  for(const field of ['point-count','duration','distance'])$(field).textContent='—';
  $('map-placeholder').hidden=false;document.querySelector('.map-label').hidden=true;$('fit-button').disabled=true;
  if(!id)return;
  try{
    const preview=await service.preview(id);
    if(token!==state.token)return;
    setEnvironment(preview.demo===true);displayTrip(preview,preview.observations.length>0);map?.preview(preview);
    setBusy(false);say(`${preview.trip.name} selected. ${preview.observations.length?'GPS observations loaded.':'Reconstruct to load its GPS and predicted route.'}`);
  }catch(error){if(token===state.token){showError(error);$('trip-name').textContent='Journey unavailable';$('trip-description').textContent='Choose another trip or retry.';}}
}
async function reconstruct(){
  if(!state.trip||state.busy)return;
  const id=state.trip.id,token=++state.token;state.controller?.abort();state.controller=new AbortController();
  clearResult();setBusy(true);$('results-badge').textContent='PROCESSING';
  try{
    const result=await service.reconstruct(id,{signal:state.controller.signal,onProgress:phase=>{
      if(token===state.token){$('loading-label').textContent=phase;say(phase);}
    }});
    if(token!==state.token)return;
    displayTrip(result,true);map?.preview(result);
    if(!result.matched_route?.coordinates.length)throw new Error(result.warnings[0]||'No connected route was returned. GPS observations remain visible.');
    state.result=result;map?.result(result);setConnection('connected');
    metric('metric-accuracy',result.metrics.top1_accuracy==null?null:result.metrics.top1_accuracy*100,'%');
    metric('metric-continuity',result.metrics.route_continuity==null?null:result.metrics.route_continuity*100,'%');
    metric('metric-time',result.metrics.processing_ms==null?null:result.metrics.processing_ms/1000,'s');
    $('results-badge').textContent=result.demo?'DEMO RESULTS':'COMPLETE';
    $('result-title').textContent=result.demo?'A connected journey, illustrated.':`${result.trip.name} reconstructed`;
    const description=result.demo?'Synthetic route and metrics · no model was executed.':
      `${formatNumber(result.observations.length)} GPS observations · ${formatNumber(result.counts?.route_segments)} route ${result.counts?.route_segments===1?'section':'sections'}. ${result.warnings.join(' ')}`;
    $('result-description').textContent=description;
    $('run-details').hidden=result.demo;
    $('metric-route-f1').textContent=Number.isFinite(result.metrics.route_f1)?result.metrics.route_f1.toFixed(4):'—';
    $('metric-classifier').textContent=percent(result.metrics.classifier_only_top1_accuracy);
    $('metric-scored').textContent=Number.isFinite(result.metrics.scored_points)?`${formatNumber(result.metrics.scored_points)} / ${formatNumber(result.observations.length)}`:'Unavailable';
    $('metric-sections').textContent=formatNumber(result.counts?.route_segments);
    $('layer-matched').disabled=false;$('layer-classifier').disabled=!result.classifier_route?.coordinates.length;
    const note=document.querySelector('.layer-note');
    note.lastChild.textContent=result.demo?'Switch layers to compare the demo paths.':'Showing GPS fixes and the HMM / Viterbi route.';
    say(result.demo?'Demo reconstruction complete. Displayed metrics are synthetic.':'Reconstruction complete. Displayed metrics were calculated on the server.');
    if(matchMedia('(max-width:760px)').matches)document.querySelector('.map-panel').scrollIntoView({
      behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?'instant':'smooth',block:'start',
    });
  }catch(error){
    if(error.name!=='AbortError'&&token===state.token){
      showError(error);
      if(/Unable to reach|too long/.test(error.message))setConnection('error');
    }
  }finally{if(token===state.token)setBusy(false);}
}
$('trip-select').addEventListener('change',event=>selectTrip(event.target.value));
$('reconstruct-button').addEventListener('click',reconstruct);
$('retry-button').addEventListener('click',()=>state.trip?reconstruct():init());
$('fit-button').addEventListener('click',()=>map?.fit());
for(const name of ['gps','classifier','matched'])$(`layer-${name}`).addEventListener('change',event=>map?.setVisible(name,event.target.checked));
$('export-button').addEventListener('click',()=>{
  const result=state.result;if(!result)return;
  const geojson={type:'FeatureCollection',features:[{type:'Feature',properties:{trip_id:result.trip.id,demo:result.demo,
    source:result.demo?'Synthetic UI fixture; not a model prediction':'PathSnap /predict'},geometry:result.matched_route}]};
  const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([JSON.stringify(geojson,null,2)],{type:'application/geo+json'}));
  link.download=`pathsnap-${result.trip.id}${result.demo?'-synthetic':''}.geojson`;link.click();
  setTimeout(()=>URL.revokeObjectURL(link.href),1000);say('Route GeoJSON exported.');
});
$('help-button').addEventListener('click',()=>$('help-dialog').showModal());
for(const id of ['close-help','help-done'])$(id).addEventListener('click',()=>$('help-dialog').close());
$('connection-button').addEventListener('click',()=>{$('backend-url').value=getBackendUrl();$('connection-error').hidden=true;$('connection-dialog').showModal();});
$('close-connection').addEventListener('click',()=>$('connection-dialog').close());
$('connection-form').addEventListener('submit',async event=>{
  event.preventDefault();$('save-connection').disabled=true;$('connection-error').hidden=true;
  try{
    if(CONFIG.mode==='demo')throw new Error('Set mode to api in config.js before connecting a backend.');
    configureBackend($('backend-url').value);
    if(await init())$('connection-dialog').close();
    else{$('connection-error').textContent=$('error-message').textContent;$('connection-error').hidden=false;}
  }catch(error){$('connection-error').textContent=error.message;$('connection-error').hidden=false;}
  finally{$('save-connection').disabled=false;}
});
async function init(){
  const token=++state.token;state.controller?.abort();state.trip=null;clearResult();map?.clearPreview();setBusy(false);
  $('trip-select').disabled=true;$('trip-select').replaceChildren(new Option('Connecting to backend…',''));
  $('map-placeholder').hidden=false;document.querySelector('.map-label').hidden=true;$('fit-button').disabled=true;
  $('trip-name').textContent='Connecting to the backend';$('trip-description').textContent='Loading available trips';$('trip-id-label').textContent='—';
  for(const field of ['point-count','duration','distance'])$(field).textContent='—';
  setEnvironment(CONFIG.mode==='demo');setConnection('connecting');
  try{
    if(service.health)await service.health();
    const trips=await service.listTrips();
    if(token!==state.token)return false;
    state.trips=trips;setConnection('connected');$('trip-select').replaceChildren();
    $('trip-help').textContent=CONFIG.mode==='demo'?`${trips.length} sample journeys · no evaluator data`:`${trips.length} trips available from the backend`;
    if(!trips.length){
      $('trip-select').append(new Option('No trips available',''));$('trip-name').textContent='No trip data available';
      $('trip-description').textContent='Add trip files on the backend, then refresh.';
      showError(new Error('The backend is connected but has no trip data. Add trips on Laptop 1, then try again.'));
      return true;
    }
    for(const trip of trips){const option=new Option(CONFIG.mode==='demo'?`${trip.id.toUpperCase()} · ${trip.name}`:trip.name,trip.id);$('trip-select').append(option);}
    $('trip-select').disabled=false;await selectTrip(trips[0].id);return true;
  }catch(error){
    if(token!==state.token)return false;
    setConnection('error');showError(error);$('trip-select').replaceChildren(new Option('Backend unavailable',''));
    $('trip-name').textContent='Connect to the backend';$('trip-description').textContent='Use Connection in the header to set its address.';
    $('trip-help').textContent='No demo fallback is used in API mode';return false;
  }
}
init();
