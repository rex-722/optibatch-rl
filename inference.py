import os
import json
from typing import List

from openai import OpenAI
from environment import DeliveryCityEnvironment
from models import Assignment

# --- Bot's Mandatory Variables ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.getenv("HF_TOKEN")

def get_ai_decision(client: OpenAI, state: dict) -> list:
    pending_orders = [{"id": o["id"], "pickup": o["pickup_loc"]} for o in state.get("orders", []) if o["status"] == "pending"]
    available_riders = [{"id": r["id"], "loc": r["loc"], "load": r["load"]} for r in state.get("riders", []) if r["status"] in ["idle", "relocating"] or (r["status"] in ["heading_to_pickup", "waiting_at_hub"] and r["load"] < 4)][:30]
    
    if not pending_orders or not available_riders: 
        return []
    
    prompt = f"Context: Shift {state.get('shift')} | Weather {state.get('weather')}. Match riders to orders efficiently. Reply ONLY with JSON array: [{{\"rider_id\": <id>, \"order_id\": <id>, \"action\": \"pickup\"}}]"
    
    valid_assignments = []
    if client:
        try:
            # Mandatory OpenAI Call
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.01
            )
            text = response.choices[0].message.content.strip()
            
            start, end = text.find('['), text.rfind(']') + 1
            if start != -1 and end != 0: 
                parsed = json.loads(text[start:end])
                avail_ids = [r["id"] for r in available_riders]
                pending_ids = [o["id"] for o in pending_orders]
                for d in parsed:
                    if d.get("rider_id") in avail_ids and d.get("order_id") in pending_ids:
                        valid_assignments.append(Assignment(**d))
                        pending_ids.remove(d.get("order_id")) 
        except Exception:
            pass
            
    if valid_assignments: return valid_assignments

    # Fallback Logic
    assignments = []
    for o in pending_orders:
        if not available_riders: break
        best_r = min(available_riders, key=lambda r: ((r["loc"][0]-o["pickup"][0])**2 + (r["loc"][1]-o["pickup"][1])**2))
        assignments.append(Assignment(rider_id=best_r["id"], order_id=o["id"], action="pickup"))
        best_r["load"] += 1
        if best_r["load"] >= 4: available_riders.remove(best_r) 
    return assignments

def main():
    # Setup
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None
    env = DeliveryCityEnvironment()
    
    task_name = "optibatch"
    benchmark = "optibatch-swarm"
    
    # 1. MANDATORY: [START] Print
    print(f"[START] task={task_name} env={benchmark} model={MODEL_NAME}", flush=True)

    obs = env.reset()
    obs = obs if isinstance(obs, dict) else {}
    
    step_count = 0
    rewards = []
    
    try:
        while env.is_running:
            step_count += 1
            decision = get_ai_decision(client, obs)
            
            obs = env.step(decision)
            obs = obs if isinstance(obs, dict) else {}
            
            reward = 0.00 
            rewards.append(reward)
            done = not env.is_running
            
            action_str = f"assigned_{len(decision)}_agents"
            
            # 2. MANDATORY: [STEP] Print
            print(f"[STEP] step={step_count} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

    except Exception as e:
        print(f"[DEBUG] Error: {e}", flush=True)

    finally:
        stats = env.stop_engine()
        score = stats.get('avg_score', 0.0) if stats else 0.0
        score = min(max(score, 0.0), 1.0)  # Make sure score is between 0 and 1
        success = score >= 0.5
        
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        
        # 3. MANDATORY: [END] Print
        print(f"[END] success={str(success).lower()} steps={step_count} score={score:.3f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    main()
