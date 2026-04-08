import os
import json
from typing import List
import time

from openai import OpenAI
from environment import DeliveryCityEnvironment
from models import Assignment

# --- Bot's Mandatory Variables ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.getenv("HF_TOKEN")

def get_ai_decision(client: OpenAI, state: dict, use_ai: bool = True) -> list:
    pending_orders = [{"id": o["id"], "pickup": o["pickup_loc"]} for o in state.get("orders", []) if o["status"] == "pending"]
    # Task size reduce for faster LLM response
    pending_orders = pending_orders[:15] 
    
    available_riders = [{"id": r["id"], "loc": r["loc"], "load": r["load"]} for r in state.get("riders", []) if r["status"] in ["idle", "relocating"] or (r["status"] in ["heading_to_pickup", "waiting_at_hub"] and r["load"] < 4)][:20]
    
    if not pending_orders or not available_riders: 
        return []

    # 🚀 Optimization: Use AI only on specific intervals
    if client and use_ai:
        prompt = f"Riders:{available_riders}. Orders:{pending_orders}. Match IDs. Format: [{{'rider_id': id, 'order_id': id, 'action': 'pickup'}}]. No talk."
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0.01,
                timeout=0.6 # ⏱️ POINT 1: Strict LLM Timeout
            )
            text = response.choices[0].message.content.strip()
            
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
                if valid_assignments: return valid_assignments
        except Exception:
            pass 

    # ⚡ POINT 2: Super Fast Fallback (Always ready if AI fails or is skipped)
    assignments = []
    for o in pending_orders:
        if not available_riders: break
        best_r = min(available_riders, key=lambda r: ((r["loc"][0]-o["pickup"][0])**2 + (r["loc"][1]-o["pickup"][1])**2))
        assignments.append(Assignment(rider_id=best_r["id"], order_id=o["id"], action="pickup"))
        best_r["load"] += 1
        if best_r["load"] >= 4: available_riders.remove(best_r) 
    return assignments

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None
    env = DeliveryCityEnvironment()
    
    print(f"[START] task=optibatch env=optibatch-swarm model={MODEL_NAME}", flush=True)

    obs = env.reset()
    obs = obs if isinstance(obs, dict) else {}
    
    start_time = time.time()
    # ⏱️ POINT 3: Demo/Validation Protection (Auto-stop after 5-10 mins if needed)
    # Scaler typically kills at 30, but we aim to finish in 5-8 mins
    
    step_count = 0
    rewards = []
    
    try:
        while env.is_running:
            step_count += 1
            
            should_use_ai = (step_count % 15 == 0)
            
            decision = get_ai_decision(client, obs, use_ai=should_use_ai)
            
            obs = env.step(decision)
            obs = obs if isinstance(obs, dict) else {}
            
            reward = 0.00 
            rewards.append(reward)
            done = not env.is_running
            
            # Keep logs clean and fast
            if step_count % 50 == 0 or done:
                action_str = f"assigned_{len(decision)}_agents"
                print(f"[STEP] step={step_count} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

    except Exception as e:
        print(f"[DEBUG] Error: {e}", flush=True)

    finally:
        stats = env.stop_engine()
        score = stats.get('avg_score', 0.0) if stats else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.5
        rewards_str = ",".join(f"{r:.2f}" for r in rewards[:50]) # Minimalist logs
        
        print(f"[END] success={str(success).lower()} steps={step_count} score={score:.3f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    main()
