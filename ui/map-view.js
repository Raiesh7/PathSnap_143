const asLatLng = coordinates => coordinates.map(([lon,lat])=>[lat,lon]);

export class JourneyMap {
  constructor(){
    if(!window.L)throw new Error('The local map library could not be loaded.');
    const L=window.L;
    this.map=L.map('map',{zoomControl:false,attributionControl:true,scrollWheelZoom:false,minZoom:10,maxZoom:19});
    L.control.zoom({position:'topright'}).addTo(this.map);
    this.map.attributionControl.setPrefix('<a href="https://leafletjs.com" target="_blank" rel="noopener noreferrer">Leaflet</a>');
    this.map.attributionControl.addAttribution('PathSnap · synthetic geometry');
    this.background=L.layerGroup();
    this.gps=L.layerGroup();this.classifier=L.layerGroup();this.matched=L.layerGroup();this.endpoints=L.layerGroup();
    this.map.createPane('gps-points');this.map.getPane('gps-points').style.zIndex=440;
    this.map.createPane('route-line');this.map.getPane('route-line').style.zIndex=420;
    this.buildBackground();
    this.visibility={gps:true,classifier:false,matched:true};
    this.bounds=null;
    this.map.setView([12.9747,77.606],15);
    this.resizeObserver=new ResizeObserver(()=>this.map.invalidateSize({pan:false}));
    this.resizeObserver.observe(document.getElementById('map'));
  }
  buildBackground(){
    const L=window.L;
    const addRoad=(points,major=false)=>{
      L.polyline(points,{color:'#d1d9cc',weight:major?13:7,opacity:1,interactive:false}).addTo(this.background);
      L.polyline(points,{color:'#ffffff',weight:major?10:5,opacity:1,interactive:false}).addTo(this.background);
    };
    for(let row=0;row<31;row++)for(let col=0;col<38;col++){
      const south=12.945+row*.00195,west=77.572+col*.00205;
      const green=(row*7+col*11)%17===0;
      L.rectangle([[south+.00018,west+.00018],[south+.00177,west+.00185]],{color:'#e0e4dc',weight:.7,
        fillColor:green?'#d3dfcb':(row+col)%3?'#e6e9e1':'#e0e5da',fillOpacity:1,interactive:false}).addTo(this.background);
    }
    for(let i=0;i<39;i++){const lon=77.572+i*.00205;addRoad([[12.94,lon],[13.01,lon+.0006]],i%5===0);}
    for(let i=0;i<32;i++){const lat=12.945+i*.00195;addRoad([[lat,77.565],[lat+.0009,77.66]],i%4===0);}
    addRoad([[12.978,77.588],[12.9741,77.5985],[12.9721,77.6118],[12.9793,77.6205]],true);
    addRoad([[12.983,77.595],[12.9734,77.6045],[12.965,77.614]],true);
    for(const [lat,lon,label] of [[12.978,77.597,'CENTRAL DISTRICT'],[12.9693,77.6085,'GARDEN QUARTER'],[12.9803,77.613,'EASTSIDE']]){
      L.marker([lat,lon],{icon:L.divIcon({className:'map-district-label',html:label,iconSize:[180,16]}),interactive:false,keyboard:false}).addTo(this.background);
    }
  }
  preview(data){
    const L=window.L;
    for(const group of [this.gps,this.classifier,this.matched,this.endpoints])group.clearLayers();
    if(data.demo){
      if(this.streetTiles){this.map.removeLayer(this.streetTiles);this.streetTiles=null;}
      this.background.addTo(this.map);
      this.map.setMaxBounds([[12.948,77.575],[13.0,77.645]]);
    }else{
      this.map.removeLayer(this.background);this.map.setMaxBounds([[-85,-180],[85,180]]);
      if(!this.streetTiles)this.streetTiles=L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{
        maxZoom:19,attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      }).addTo(this.map);
    }
    for(const observation of data.observations){
      const marker=L.circleMarker([observation.coordinates[1],observation.coordinates[0]],{
        radius:3,weight:.9,color:'#fff',fillColor:'#6688be',fillOpacity:.9,pane:'gps-points',
      });
      const content=document.createElement('span');
      content.textContent=`GPS observation ${observation.id} · ${observation.timestamp_s??'—'} s`;
      marker.bindTooltip(content).addTo(this.gps);
    }
    const points=asLatLng(data.observations.map(o=>o.coordinates));
    this.bounds=L.latLngBounds(points);
    this.syncLayers();this.fit();
  }
  result(data){
    const L=window.L;
    this.classifier.clearLayers();this.matched.clearLayers();this.endpoints.clearLayers();
    for(const line of data.classifier_route?.coordinates||[])L.polyline(asLatLng(line),{color:'#b58b4c',weight:3,dashArray:'7 7',opacity:.85,pane:'route-line'}).addTo(this.classifier);
    for(const line of data.matched_route?.coordinates||[]){
      const points=asLatLng(line);
      L.polyline(points,{color:'#fff',weight:8,opacity:.95,pane:'route-line',interactive:false}).addTo(this.matched);
      L.polyline(points,{color:'#218767',weight:4.5,opacity:1,pane:'route-line'}).addTo(this.matched);
      for(const [point,end] of [[points[0],false],[points.at(-1),true]]){
        L.marker(point,{icon:L.divIcon({className:'',html:`<div class="endpoint-marker${end?' end':''}"></div>`,iconSize:[13,13],iconAnchor:[6,6]}),
          title:end?'Route end':'Route start'}).addTo(this.endpoints);
      }
    }
    this.syncLayers();
  }
  clearResult(){this.classifier.clearLayers();this.matched.clearLayers();this.endpoints.clearLayers();}
  clearPreview(){this.clearResult();this.gps.clearLayers();}
  setVisible(name,visible){this.visibility[name]=visible;this.syncLayers();}
  syncLayers(){
    for(const name of ['gps','classifier','matched']){
      if(this.visibility[name])this[name].addTo(this.map);else this.map.removeLayer(this[name]);
    }
    if(this.visibility.matched)this.endpoints.addTo(this.map);else this.map.removeLayer(this.endpoints);
  }
  fit(){if(this.bounds?.isValid())this.map.fitBounds(this.bounds,{padding:[22,35],maxZoom:16,animate:false});}
}
