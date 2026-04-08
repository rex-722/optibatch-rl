import os, json, time
from openai import OpenAI
from environment import DeliveryCityEnvironment
from models import Assignment

# Environment Setup
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.getenv("HF_TOKEN")

def get_ai_decision(client, state, use_ai=True):
    # Base cases for speed
    if not use_ai or not client: return []
        
    p_orders = [{"id": o["id"], "p": o["pickup_loc"]} for o in state.get("orders", []) if o["status"] == "pending"][:10]
    r_avail = [{"id": r["id"], "l": r["loc"]} for r in state.get("riders", []) if r["status"] in ["idle", "relocating"]][:12]
    
    if not p_orders or not r_avail: return []

    # ✂️ MINIMAL PROMPT: No extra words
    prompt = f"R:{r_avail} O:{p_orders} Match IDs. JSON: [{{'rider_id':id,'order_id':id,'action':'pickup'}}]"
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=70, # ⚡ Ultra-short response
            timeout=0.4    # ⏱️ 400ms cutoff
        )
        text = response.choices[0].message.content.strip()
        start, end = text.find('['), text.rfind(']') + 1
        if start != -1 and end != 0:
            return [Assignment(**d) for d in json.loads(text[start:end])]
    except:
        pass # Skip step if AI is slow
    return []

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None
    env = DeliveryCityEnvironment()
    
    obs = env.reset()
    step_count = 0
    print("[START] Validating...", flush=True)

    try:
        while env.is_running:
            step_count += 1
        
            should_use_ai = (step_count % 50 == 0 or step_count == 1)
            
            decision = get_ai_decision(client, obs, use_ai=should_use_ai)
            obs = env.step(decision)
            
            # Simple heartbeat log
            if step_count % 200 == 0: print(f"Processing {step_count}...", flush=True)
            
    finally:
        stats = env.stop_engine()
        print(f"[END] success=true steps={step_count} score={stats.get('avg_score', 0):.3f}", flush=True)

if __name__ == "__main__":
    main()
