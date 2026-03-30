import os
import requests
import time
from datetime import datetime, timedelta, timezone

# HF Direct API configuration
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

BASE_URL = "https://rex-722ra-optibatch-rl.hf.space"

def query_huggingface(prompt):
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 5, "temperature": 0.01}
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=5)
        result = response.json()
        if isinstance(result, list):
            # Extract just the generated number
            text = result[0].get("generated_text", "")
            ans = text.replace(prompt, "").strip()
            return int(ans) if ans in ["0", "1"] else None
    except Exception as e:
        return None
    return None

def run_continuous_simulation():
    requests.post(f"{BASE_URL}/reset")
    done = False
    prev_riders = None
    
    print("\n" + "="*80)
    print("🚀 INITIATING CONTINUOUS LIVE SIMULATION (300 TICKS)")
    print("="*80)
    
    while not done:
        state = requests.get(f"{BASE_URL}/state").json()
        
        tick = state.get("time", 0)
        pending = state.get("pending_orders_count", 0)
        riders = state.get("available_riders", 0)
        oldest = state.get("oldest_order_time", 0)
        sla = state.get("sla_limit", 30)
        cap = state.get("max_capacity", 4)
        phase = state.get("live_phase", "NORMAL")
        weather = state.get("weather", "UNKNOWN")
        total_in = state.get("total_orders", 0)
        total_out = state.get("total_delivered", 0)
        
        time_waiting = (tick - oldest) if pending > 0 else 0
        ist_time = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%I:%M %p")

        # --- Dashboard Updates ---
        if tick % 15 == 0 or tick == 1:
            print(f"\n[🕒 {ist_time}] TICK: {tick:03d}/300 | 🌍 {weather} | 📊 PHASE: {phase}")
            print(f"📦 Total Received: {total_in} | ✅ Total Delivered: {total_out} | 🛵 Riders in Hub: {riders}")
            print("-" * 80)
            
        if prev_riders is not None:
            if riders < prev_riders:
                print(f"   🚚 [Tick {tick:03d}] AI DISPATCHED! Rider out for delivery.")
            elif riders > prev_riders:
                returned = riders - prev_riders
                print(f"   ✅ [Tick {tick:03d}] DELIVERY COMPLETE! {returned} Rider(s) returned.")
        prev_riders = riders
        
        # --- Fallback Math ---
        def get_fallback_action():
            if pending == 0 or riders == 0: return 0 
            if time_waiting >= (sla - 5): return 1 
            if pending >= cap: return 1
            if pending >= 2: return 1
            return 0

        # --- AI Inference ---
        decision = None
        if HF_TOKEN:
            prompt = f"System: {phase}, {weather}. Pending orders:{pending}, Available riders:{riders}. Wait time:{time_waiting}m. Max capacity:{cap}. Reply ONLY with 1 to dispatch or 0 to wait."
            decision = query_huggingface(prompt)
            
        used_ai = True
        if decision is None:
            decision = get_fallback_action()
            used_ai = False

        if pending > 0 and tick % 5 == 0:
            agent_str = "🧠 HuggingFace AI" if used_ai else "⚙️ Local Fallback"
            action_str = "DISPATCHING" if decision == 1 else "HOLDING"
            print(f"   > {agent_str} decision: {action_str} (Pending Queue: {pending})")

        # Execute
        requests.post(f"{BASE_URL}/step", json={"action_type": decision})
        
        state_after = requests.get(f"{BASE_URL}/state").json()
        if state_after.get("time", 0) >= 300:
            done = True

    grader = requests.get(f"{BASE_URL}/grader").json()
    print("\n" + "="*80)
    print(f"🏁 SIMULATION COMPLETE | Final Score: {grader['score']} / 1.0")
    print("="*80)

if __name__ == "__main__":
    run_continuous_simulation()
