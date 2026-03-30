import os
import json
import asyncio
import logging
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from environment import DeliveryCityEnvironment
from models import Assignment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="OptiBatch Swarm Command")
env = DeliveryCityEnvironment()

MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

def get_ai_decision(state: dict) -> list:
    pending_orders = [{"id": o["id"], "pickup": o["pickup_loc"]} for o in state["orders"] if o["status"] == "pending"]
    available_riders = [{"id": r["id"], "loc": r["loc"], "load": r["load"]} for r in state["riders"] if r["status"] in ["idle", "relocating"] or (r["status"] in ["heading_to_pickup", "waiting_at_hub"] and r["load"] < 4)][:30]
    
    if not pending_orders or not available_riders: return []
    
    prompt = f"Context: Shift {state.get('shift')} | Weather {state.get('weather')}. Available Riders (Max Load 4): {available_riders}. Pending Orders: {pending_orders}. Match riders to orders efficiently. Reply ONLY with JSON array: [{{\"rider_id\": <id>, \"order_id\": <id>, \"action\": \"pickup\"}}]"
    
    valid_assignments = []
    if HF_TOKEN:
        headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
        try:
            res = requests.post(API_URL, headers=headers, json={"inputs": prompt, "parameters": {"max_new_tokens": 150, "temperature": 0.01}}, timeout=1.5)
            res.raise_for_status()
            
            data = res.json()
            text = data[0].get("generated_text", "").replace(prompt, "").strip()
            start, end = text.find('['), text.rfind(']') + 1
            
            if start != -1 and end != 0: 
                parsed = json.loads(text[start:end])
                pending_ids = [o["id"] for o in pending_orders]
                avail_ids = [r["id"] for r in available_riders]
                for d in parsed:
                    if d.get("rider_id") in avail_ids and d.get("order_id") in pending_ids:
                        valid_assignments.append(Assignment(**d))
                        pending_ids.remove(d.get("order_id")) 
                        
        except requests.exceptions.Timeout:
            pass
        except Exception as e:
            pass
            
    if valid_assignments: return valid_assignments

    assignments = []
    for o in pending_orders:
        if not available_riders: break
        best_r = min(available_riders, key=lambda r: ((r["loc"][0]-o["pickup"][0])**2 + (r["loc"][1]-o["pickup"][1])**2))
        assignments.append(Assignment(rider_id=best_r["id"], order_id=o["id"], action="pickup"))
        best_r["load"] += 1
        if best_r["load"] >= 4: available_riders.remove(best_r) 
    return assignments

     @app.post("/reset")
async def reset_env():
    obs = env.reset()
    return obs

@app.post("/step")
async def step_env(assignments: List[Assignment]):
    obs = env.step(assignments)
    return obs
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>OptiBatch Swarm Command</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

    <style>
        body { background-color: #020617; color: #f8fafc; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; overflow: hidden; height: 100vh;}
        .main-container { display: flex; flex-direction: row; height: calc(100vh - 80px); gap: 24px; position: relative; z-index: 10;}
        #city-map { flex-grow: 1; height: 100%; border-radius: 12px; border: 1px solid #334155; position: relative; z-index: 5;}
        
        .leaflet-layer, .leaflet-control-zoom-in, .leaflet-control-zoom-out, .leaflet-control-attribution { filter: invert(100%) hue-rotate(180deg) brightness(85%) contrast(120%); }
        .glass { background: rgba(15, 23, 42, 0.8); backdrop-filter: blur(10px); border: 1px solid #334155; }
        
        .hub-icon { background: rgba(16, 185, 129, 0.2); border-radius: 50%; border: 1px solid #10b981; box-shadow: 0 0 15px rgba(16, 185, 129, 0.5); }
        .rider-idle { background: #10b981; border-radius: 50%; opacity: 0.8; transition: all 0.4s linear; }
        .rider-pickup { background: #f59e0b; border-radius: 50%; box-shadow: 0 0 5px #f59e0b; transition: all 0.4s linear; }
        .rider-waiting { background: #06b6d4; border-radius: 50%; box-shadow: 0 0 8px #06b6d4; transition: all 0.4s linear; animation: pulse-fast 1s infinite; }
        .rider-deliver { background: #3b82f6; border-radius: 50%; box-shadow: 0 0 5px #3b82f6; transition: all 0.4s linear; }
        .rider-relocate { background: #8b5cf6; border-radius: 50%; opacity: 0.6; transition: all 0.4s linear; } 
        
        .order-pending { border: 2px solid #ef4444; border-radius: 50%; animation: pulse 0.5s infinite; }
        .order-assigned { background: #d946ef; border-radius: 50%; }
        @keyframes pulse { 0% { transform: scale(0.5); opacity: 1; } 100% { transform: scale(2); opacity: 0; } }
        @keyframes pulse-fast { 0% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(1.2); } 100% { opacity: 1; transform: scale(1); } }
    </style>
</head>
<body class="p-6">
    
    <div class="flex justify-between items-end mb-4 h-[60px]">
        <div>
            <h1 class="text-3xl font-bold text-emerald-400">OptiBatch Swarm Engine (15 Hubs)</h1>
            <div class="flex gap-3 items-center mt-1">
                <p class="text-slate-400 text-sm font-mono" id="live-clock">--:--:--</p>
                <span class="px-2 py-0.5 rounded bg-slate-800 text-emerald-300 text-xs border border-slate-700" id="weather-badge">🌤️ Loading...</span>
            </div>
        </div>
        <span class="text-emerald-500 text-sm animate-pulse font-bold tracking-widest">● LIVE SCALE</span>
    </div>
    
    <div class="main-container">
        <div id="city-map"></div>
        
        <div class="glass p-6 rounded-xl w-[400px] flex flex-col z-10">
            <h2 class="text-xl text-emerald-400 border-b border-slate-700 pb-2 mb-4 uppercase tracking-widest">Swarm Telemetry</h2>
            <div class="text-center mb-6 bg-slate-800/50 p-4 rounded-lg border border-slate-700 relative overflow-hidden">
                <div class="absolute inset-0 bg-emerald-500/5 blur-xl"></div>
                <p class="text-slate-400 text-sm mb-1 relative">Context Shift: <span id="shift-name" class="text-emerald-300 font-bold">...</span></p>
                <p class="text-7xl font-bold relative" id="score-text">1.00</p>
                <p class="text-xs text-slate-500 mt-2 relative">Fleet Balance & SLA Score (0.0 to 1.0)</p>
            </div>
            <div class="flex justify-between mb-3 border-b border-slate-700/50 pb-2">
                <span class="text-slate-400">Total Delivered:</span> <span id="delivered" class="text-emerald-300 font-bold text-2xl">0</span>
            </div>
            <div class="flex justify-between mb-2">
                <span class="text-slate-400">Active Agent Queue:</span> <span id="pending" class="text-rose-400 font-bold text-xl">0</span>
            </div>
            <h2 class="text-xs text-slate-400 mt-3 mb-2 border-b border-slate-700 pb-1 uppercase tracking-wider">Active Swarm (150 Units | Max 4 Load)</h2>
            <div class="text-[9px] text-slate-500 mb-2 flex justify-between"><span>🟢 Idle</span><span>🟠 To Hub</span><span>🟡 Waiting</span><span>🔵 Deliver</span><span>🟣 Reloc</span></div>
            <div id="fleet-status" class="flex-1 overflow-y-auto text-xs space-y-2 text-slate-300 pr-1"></div>
        </div>
    </div>
    
    <button onclick="shutdownEngine()" class="fixed bottom-6 left-6 w-[700px] bg-rose-600 hover:bg-rose-500 text-white font-bold py-3 px-4 rounded shadow-[0_0_20px_rgba(225,29,72,0.4)] transition-all z-20">
        🛑 SHUTDOWN HUB & GENERATE DAILY REPORT
    </button>

    <div id="report-modal" class="hidden fixed inset-0 bg-black/90 z-50 flex items-center justify-center backdrop-blur-md">
        <div class="glass p-8 rounded-2xl w-[600px] text-center border-emerald-500 border-2">
            <h1 class="text-4xl text-emerald-400 font-bold mb-2 uppercase tracking-widest">DAILY HUB REPORT</h1>
            <p class="text-slate-400 mb-6">Operations Suspended (Day Complete).</p>
            <div class="bg-slate-900 p-6 rounded-xl mb-6 border border-slate-800">
                <p class="text-sm text-slate-400 uppercase tracking-widest">Final Swarm Score</p>
                <p class="text-8xl font-bold text-emerald-400 mt-3" id="final-score">0.00</p>
            </div>
            <div class="text-left bg-slate-800/80 p-5 rounded-lg border border-slate-700 mb-6 font-mono text-sm space-y-2" id="shift-breakdown"></div>
            <button onclick="location.reload()" class="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-3 px-6 rounded text-lg uppercase">RESTART NEXT DAY</button>
        </div>
    </div>

    <script>
        setInterval(() => {
            const now = new Date();
            document.getElementById('live-clock').innerText = now.toLocaleTimeString('en-US', { timeZone: 'Asia/Kolkata', hour12: true }) + " IST";
        }, 1000);

        const BASE_LAT = 12.9716; const BASE_LNG = 77.5946; const SCALE = 0.015; 
        function toLatLng(loc) { return [BASE_LAT + ((loc[1]-5) * SCALE), BASE_LNG + ((loc[0]-5) * SCALE)]; }

        const map = L.map('city-map', { zoomControl: false }).setView(toLatLng([5, 5]), 12);
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', { attribution: '&copy; OSM' }).addTo(map);

        let hubMarkers = {}, riderMarkers = {}, orderMarkers = {}, riderLines = {};
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(wsProtocol + '//' + window.location.host + '/ws');

        ws.onmessage = function(event) {
            const state = JSON.parse(event.data);
            if(!state.is_running) return;

            document.getElementById('weather-badge').innerText = state.weather;
            let scoreEl = document.getElementById('score-text');
            scoreEl.innerText = state.current_score.toFixed(2);
            scoreEl.className = state.current_score >= 0.8 ? "text-7xl font-bold relative text-emerald-400" : (state.current_score >= 0.5 ? "text-7xl font-bold relative text-yellow-400" : "text-7xl font-bold relative text-rose-500");
            document.getElementById('shift-name').innerText = state.shift;
            document.getElementById('delivered').innerText = state.delivered_total;
            document.getElementById('pending').innerText = state.orders.filter(o => o.status === "pending").length;

            if(Object.keys(hubMarkers).length === 0) {
                for (const [name, loc] of Object.entries(state.hubs)) {
                    hubMarkers[name] = L.marker(toLatLng(loc), {icon: L.divIcon({className: 'hub-icon', iconSize: [12, 12]})}).addTo(map);
                }
            }

            let currentOrderIds = state.orders.map(o => o.id);
            for(let id in orderMarkers) {
                if(!currentOrderIds.includes(parseInt(id))) { map.removeLayer(orderMarkers[id]); delete orderMarkers[id]; }
            }
            state.orders.forEach(o => {
                let isAssigned = o.status === "assigned";
                let iconClass = isAssigned ? 'order-assigned' : 'order-pending';
                let iconSize = isAssigned ? [6, 6] : [10, 10];
                if(!orderMarkers[o.id]) orderMarkers[o.id] = L.marker(toLatLng(o.dropoff_loc), {icon: L.divIcon({className: iconClass, iconSize: iconSize})}).addTo(map);
                else orderMarkers[o.id].setIcon(L.divIcon({className: iconClass, iconSize: iconSize}));
            });

            let fleetHtml = '';
            let visibleRidersCount = 0; 
            
            state.riders.forEach(r => {
                let colorHex = '#10b981'; let cssClass = 'rider-idle'; let statusText = 'IDLE';
                if(r.status === 'heading_to_pickup') { colorHex = '#f59e0b'; cssClass = 'rider-pickup'; statusText = 'TO HUB'; }
                else if(r.status === 'waiting_at_hub') { colorHex = '#06b6d4'; cssClass = 'rider-waiting'; statusText = 'WAITING'; }
                else if(r.status === 'delivering') { colorHex = '#3b82f6'; cssClass = 'rider-deliver'; statusText = 'DELIVER'; }
                else if(r.status === 'relocating') { colorHex = '#8b5cf6'; cssClass = 'rider-relocate'; statusText = 'RELOCAT'; } 

                let latLng = toLatLng(r.loc);

                if(!riderMarkers[r.id]) riderMarkers[r.id] = L.marker(latLng, {icon: L.divIcon({className: cssClass, iconSize: [8, 8]})}).addTo(map);
                else { riderMarkers[r.id].setLatLng(latLng); riderMarkers[r.id].setIcon(L.divIcon({className: cssClass, iconSize: [8, 8]})); }

                if(r.target_loc && r.status !== 'relocating' && r.status !== 'waiting_at_hub') {
                    let targetLatLng = toLatLng(r.target_loc);
                    if(!riderLines[r.id]) riderLines[r.id] = L.polyline([latLng, targetLatLng], {color: colorHex, dashArray: '3, 3', weight: 1, opacity: 0.5}).addTo(map);
                    else { riderLines[r.id].setLatLngs([latLng, targetLatLng]); riderLines[r.id].setStyle({color: colorHex}); }
                } else { if(riderLines[r.id]) { map.removeLayer(riderLines[r.id]); delete riderLines[r.id]; } }

                if(visibleRidersCount < 20 || r.load > 0) {
                    fleetHtml += `<div class="bg-slate-800/40 p-1.5 rounded border border-slate-700/50 flex justify-between items-center"><span style="color:${colorHex}; font-weight:bold;">U-${r.id} <span class="text-slate-500 text-[10px] ml-1">[Load: ${r.load}/4]</span></span> <span class="text-slate-400 text-[10px] uppercase">${statusText}</span></div>`;
                    visibleRidersCount++;
                }
            });
            document.getElementById('fleet-status').innerHTML = fleetHtml;
        };

        async function shutdownEngine() {
            const res = await fetch('/shutdown', {method: 'POST'});
            const data = await res.json();
            document.getElementById('report-modal').classList.remove('hidden');
            document.getElementById('final-score').innerText = data.avg_score.toFixed(2);
            let finalScoreEl = document.getElementById('final-score');
            finalScoreEl.className = data.avg_score >= 0.8 ? "text-8xl font-bold text-emerald-400 mt-3" : (data.avg_score >= 0.5 ? "text-8xl font-bold text-yellow-400 mt-3" : "text-8xl font-bold text-rose-500 mt-3");

            let html = '';
            for (const [shift, stats] of Object.entries(data.shifts)) {
                if(stats.total > 0) html += `<div class="flex justify-between border-b border-slate-700/50 pb-1 mb-1 font-mono text-sm space-y-2 text-slate-300"><span class="text-slate-400 w-24">${shift}:</span> <span>${stats.delivered}/${stats.total} Delivered</span> <span class="text-rose-400">Breach: ${stats.breach} | Imbalance: ${stats.imbalance}</span></div>`;
            }
            document.getElementById('shift-breakdown').innerHTML = html;
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    with open("index.html", "w") as f:
        f.write(html_content)
    with open("index.html", "r") as f:
        return HTMLResponse(f.read())

@app.post("/shutdown")
def shutdown():
    global env
    return env.stop_engine()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global env
    env.reset()
    try:
        while env.is_running:
            state = env._get_observation()
            await websocket.send_json(state)
            env.step(get_ai_decision(state))
            await asyncio.sleep(0.4) 
    except WebSocketDisconnect: 
        logging.info("Client Disconnected.")
