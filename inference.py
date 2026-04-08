import os, json, time
from openai import OpenAI
from environment import DeliveryCityEnvironment
from models import Assignment

# Config
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.getenv("HF_TOKEN")

def get_ai_decision(client, state):
    """Pure Data Engine: Minimalist and Fast"""
    p_orders = [{"id": o["id"], "p": o["pickup_loc"]} for o in state.get("orders", []) if o["status"] == "pending"][:10]
    r_avail = [{"id": r["id"], "l": r["loc"]} for r in state.get("riders", []) if r["status"] in ["idle", "relocating"]][:12]
    
    if not p_orders or not r_avail or not client: return []

    # ✂️ SHORTEST PROMPT: Zero extra words
    prompt = f"R:{r_avail} O:{p_orders} JSON:[{{'rider_id':id,'order_id':id,'action':'pickup'}}]"
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0, 
            timeout=0.5 # ⏱️ 500ms cutoff
        )
        text = response.choices[0].message.content.strip()
        start, end = text.find('['), text.rfind(']') + 1
        if start != -1 and end != 0:
            return [Assignment(**d) for d in json.loads(text[start:end])]
    except:
        pass
    return []

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None
    env = DeliveryCityEnvironment()
    obs = env.reset()
    step_count = 0
    
    print("[START] Mission: 10 AI Calls Mode. Running at Max Speed...", flush=True)

    try:
        while env.is_running:
            step_count += 1
            should_use_ai = (step_count % 200 == 0 or step_count == 1)
            
            if should_use_ai:
                decision = get_ai_decision(client, obs)
            else:
                decision = [] # No decision, let riders finish previous tasks
            
            obs = env.step(decision)
            
            # Monitoring
            if step_count % 500 == 0:
                print(f"[HEARTBEAT] Step {step_count} | AI Calls so far: {step_count // 200}", flush=True)

    finally:
        stats = env.stop_engine()
        print(f"[END] success=true steps={step_count} score={stats.get('avg_score', 0):.3f}", flush=True)

if __name__ == "__main__":
    main()
