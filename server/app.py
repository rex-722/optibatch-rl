import os
import sys
import json
import asyncio
import logging
import requests
import time
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# --- Fast Logic for Decision Making ---
def get_ai_decision(state: dict, force_manual: bool = False) -> list:
    pending_orders = [{"id": o["id"], "pickup": o["pickup_loc"]} for o in state["orders"] if o["status"] == "pending"]
    pending_orders = pending_orders[:15] # Task size reduce
    
    available_riders = [{"id": r["id"], "loc": r["loc"], "load": r["load"]} for r in state["riders"] if r["status"] in ["idle", "relocating"] or (r["status"] in ["heading_to_pickup", "waiting_at_hub"] and r["load"] < 4)][:20]
    
    if not pending_orders or not available_riders: 
        return []

    # Agar frequency skip hai ya AI off hai toh seedha manual logic (Micro-seconds)
    if force_manual or not HF_TOKEN:
        return manual_fallback(pending_orders, available_riders)

    # AI Call with Strict Timeout
    prompt = f"Riders:{available_riders}. Orders:{pending_orders}. JSON format: [{{'rider_id':id,'order_id':id,'action':'pickup'}}]. No talk."
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    try:
        # POINT 1: TIMEOUT - Sirf 0.5 second wait
        res = requests.post(API_URL, headers=headers, json={"inputs": prompt, "parameters": {"max_new_tokens": 80, "temperature": 0.01}}, timeout=0.5)
        res.raise_for_status()
        data = res.json()
        text = data[0].get("generated_text", "").replace(prompt, "").strip()
        start, end = text.find('['), text.rfind(']') + 1
        if start != -1 and end != 0: 
            parsed = json.loads(text[start:end])
            valid_assignments = []
            avail_ids = [r["id"] for r in available_riders]
            pending_ids = [o["id"] for o in pending_orders]
            for d in parsed:
                if d.get("rider_id") in avail_ids and d.get("order_id") in pending_ids:
                    valid_assignments.append(Assignment(**d))
                    pending_ids.remove(d.get("order_id"))
            return valid_assignments if valid_assignments else manual_fallback(pending_orders, available_riders)
    except:
        pass
    return manual_fallback(pending_orders, available_riders)

def manual_fallback(pending_orders, available_riders):
    assignments = []
    for o in pending_orders:
        if not available_riders: break
        best_r = min(available_riders, key=lambda r: ((r["loc"][0]-o["pickup"][0])**2 + (r["loc"][1]-o["pickup"][1])**2))
        assignments.append(Assignment(rider_id=best_r["id"], order_id=o["id"], action="pickup"))
        best_r["load"] += 1
        if best_r["load"] >= 4: available_riders.remove(best_r)
    return assignments

# --- Dashboard & Demo Logic ---
@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    # ... html_content same rahega jo pehle tha ...
    return HTMLResponse(content="Dashboard is loading...") # Replace with your original HTML string

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global env
    env.reset()
    
    start_time = time.time()
    demo_duration = 5 * 60 # 🚀 5 Minute Auto-Shutdown
    step_cnt = 0
    
    try:
        while env.is_running:
            # POINT 2: Auto-Stop logic
            elapsed = time.time() - start_time
            if elapsed >= demo_duration:
                break
            
            state = env._get_observation()
            state["timer"] = round(demo_duration - elapsed)
            await websocket.send_json(state)
            
            # POINT 3: Frequency Skip (Har 10th step par AI)
            if step_cnt % 10 == 0:
                env.step(get_ai_decision(state))
            else:
                env.step(get_ai_decision(state, force_manual=True))
            
            step_cnt += 1
            await asyncio.sleep(0.05) # POINT 4: Non-blocking fast simulation
            
    except WebSocketDisconnect:
        pass
    finally:
        stats = env.stop_engine()
        await websocket.send_json({"type": "REPORT", "stats": stats})

@app.post("/reset")
async def reset_env():
    return env.reset()

@app.post("/step")
async def step_env(assignments: List[Assignment]):
    return env.step(assignments)

@app.post("/shutdown")
def shutdown():
    return env.stop_engine()

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
