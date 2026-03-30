import os
import json
import asyncio
import logging
import requests
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from environment import DeliveryCityEnvironment
from models import Assignment

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="OptiBatch Swarm Command")
env = DeliveryCityEnvironment()

# AI Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

def get_ai_decision(state: dict) -> list:
    pending_orders = [{"id": o["id"], "pickup": o["pickup_loc"]} for o in state["orders"] if o["status"] == "pending"]
    available_riders = [{"id": r["id"], "loc": r["loc"], "load": r["load"]} for r in state["riders"] if r["status"] in ["idle", "relocating"] or (r["status"] in ["heading_to_pickup", "waiting_at_hub"] and r["load"] < 4)][:30]
    
    if not pending_orders or not available_riders: 
        return []
    
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
                        
        except Exception:
            pass
            
    if valid_assignments: 
        return valid_assignments

    # MATH FALLBACK
    assignments = []
    for o in pending_orders:
        if not available_riders: break
        best_r = min(available_riders, key=lambda r: ((r["loc"][0]-o["pickup"][0])**2 + (r["loc"][1]-o["pickup"][1])**2))
        assignments.append(Assignment(rider_id=best_r["id"], order_id=o["id"], action="pickup"))
        best_r["load"] += 1
        if best_r["load"] >= 4: 
            available_riders.remove(best_r) 
    return assignments


@app.post("/reset")
async def reset_env():
    obs = env.reset()
    return obs

@app.post("/step")
async def step_env(assignments: List[Assignment]):
    obs = env.step(assignments)
    return obs

@app.post("/shutdown")
async def shutdown():
    return env.stop_engine()

# --- DASHBOARD LOGIC ---

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
        body { background-color: #020617; color: #f8fafc; font-family: sans-serif; overflow: hidden; height: 100vh;}
        .main-container { display: flex; height: calc(100vh - 80px); gap: 24px; }
        #city-map { flex-grow: 1; border-radius: 12px; border: 1px solid #334155; }
        .leaflet-layer { filter: invert(100%) hue-rotate(180deg) brightness(85%); }
        .glass { background: rgba(15, 23, 42, 0.8); backdrop-filter: blur(10px); border: 1px solid #334155; }
        .hub-icon { background: rgba(16, 185, 129, 0.2); border-radius: 50%; border: 1px solid #10b981; }
        .rider-idle { background: #10b981; border-radius: 50%; }
        .rider-pickup { background: #f59e0b; border-radius: 50%; }
        .rider-waiting { background: #06b6d4; border-radius: 50%; animation: pulse 1s infinite; }
        .rider-deliver { background: #3b82f6; border-radius: 50%; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    </style>
</head>
<body class="p-6">
    <div class="flex justify-between items-center mb-4">
        <h1 class="text-3xl font-bold text-emerald-400">OptiBatch Swarm Engine</h1>
        <div id="weather-badge" class="px-3 py-1 bg-slate-800 rounded border border-slate-700 text-emerald-300">🌤️ Loading...</div>
    </div>
    
    <div class="main-container">
        <div id="city-map"></div>
        <div class="glass p-6 rounded-xl w-[350px] flex flex-col">
            <h2 class="text-emerald-400 border-b border-slate-700 pb-2 mb-4">TELEMETRY</h2>
            <div class="text-center mb-6">
                <p class="text-slate-400 text-sm">SLA SCORE</p>
                <p id="score-text" class="text-6xl font-bold text-emerald-400">1.00</p>
            </div>
            <div class="flex justify-between mb-2"><span>Delivered:</span> <span id="delivered" class="text-emerald-400 font-bold">0</span></div>
            <div class="flex justify-between mb-4"><span>Pending:</span> <span id="pending" class="text-rose-400 font-bold">0</span></div>
            <div id="fleet-status" class="flex-1 overflow-y-auto space-y-1 text-xs"></div>
        </div>
    </div>

    <div id="report-modal" class="hidden fixed inset-0 bg-black/90 z-50 flex items-center justify-center">
        <div class="glass p-8 rounded-2xl w-[500px] text-center border-emerald-500 border-2">
            <h1 class="text-3xl text-emerald-400 font-bold mb-4">DAILY HUB REPORT</h1>
            <div class="bg-slate-900 p-6 rounded-xl mb-6">
                <p class="text-8xl font-bold text-emerald-400" id="final-score">0.84</p>
            </div>
            <button onclick="location.reload()" class="bg-emerald-600 px-6 py-2 rounded">RESTART NEXT DAY</button>
        </div>
    </div>

    <script>
        const map = L.map('city-map', {zoomControl: false}).setView([12.9716, 77.5946], 12);
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png').addTo(map);
        let riderMarkers = {};

        const ws = new WebSocket((window.location.protocol==='https:'?'wss:':'ws:') + '//' + window.location.host + '/ws');
        ws.onmessage = function(e) {
            const state = JSON.parse(e.data);
            document.getElementById('score-text').innerText = state.current_score.toFixed(2);
            document.getElementById('delivered').innerText = state.delivered_total;
            document.getElementById('pending').innerText = state.orders.filter(o=>o.status==='pending').length;
            
            state.riders.forEach(r => {
                const pos = [12.9716 + (r.loc[1]-5)*0.015, 77.5946 + (r.loc[0]-5)*0.015];
                if(!riderMarkers[r.id]) riderMarkers[r.id] = L.marker(pos, {icon: L.divIcon({className: 'rider-idle', iconSize: [6,6]})}).addTo(map);
                else riderMarkers[r.id].setLatLng(pos);
            });
        };

        async function shutdownEngine() {
            const res = await fetch('/shutdown', {method: 'POST'});
            const data = await res.json();
            document.getElementById('final-score').innerText = data.avg_score.toFixed(2);
            document.getElementById('report-modal').classList.remove('hidden');
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    env.reset()
    try:
        while env.is_running:
            state = env._get_observation()
            await websocket.send_json(state)
            # AI/Math Step
            decision = get_ai_decision(state)
            env.step(decision)
            await asyncio.sleep(0.4)
    except WebSocketDisconnect:
        logging.info("Client Disconnected")
