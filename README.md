# 🚀 OptiBatch-RL: Real-World Last-Mile Dispatch Environment

An OpenEnv-compliant reinforcement learning environment that simulates the intense trade-offs of quick-commerce logistics. It forces AI agents to balance **Fuel Efficiency (Batching)** against **Customer Satisfaction (SLA Breaches)**.

## 🎯 Motivation
Static, rule-based algorithms completely choke during peak hours (like Diwali or heavy rain). OptiBatch-RL acts as a smart hub manager, calculating real-time mathematical trade-offs to decide whether to wait for more orders or dispatch immediately.

## 🌟 Key Features & Industry Realism
* **Physical Constraints (Reward Hacking Prevention):** Agents cannot infinitely stack orders to farm fuel efficiency. A strict physical constraint of `MAX_CAPACITY = 4` (or 5 in hard mode) is enforced per rider.
* **Dynamic ETAs:** During 'Hard Mode', the system mimics real-world platforms by automatically shifting the SLA promise from 30 mins to 45 mins.
* **Graceful Degradation:** The baseline script features a programmatic fallback mechanism. If the OpenAI API key is missing or rate-limited, it smoothly transitions to a Heuristic Agent without crashing the CI/CD pipeline.

---

## 📊 Action & Observation Spaces

### Observation Space (Pydantic Typed)
The environment returns a strict JSON observation of the current hub state:
* `time` (int): Current tick/minute (0 to 100).
* `pending_orders_count` (int): Number of orders waiting in the hub.
* `available_riders` (int): Idle riders ready for dispatch.
* `oldest_order_time` (int): Timestamp of the most delayed order.
* `sla_limit` (int): Current SLA promise based on difficulty.
* `max_capacity` (int): Maximum physical bag capacity for a rider.

### Action Space (Discrete)
The agent outputs a single integer:
* `0` (WAIT): Hold orders to build a larger batch. Risks SLA breaches (-50 penalty).
* `1` (DISPATCH): Send a rider with up to `max_capacity` orders. Rewards partial efficiency (+15 per order) but consumes a rider for 15 ticks.

---

## 🛠️ Tasks & Difficulties
1.  **Easy:** Slump hours. Excess riders (5), few orders. SLA: 25 mins.
2.  **Medium:** Standard operations. Balanced fleet (4). SLA: 30 mins.
3.  **Hard:** Peak rush. Severe rider shortage vs. massive order influx. SLA: 45 mins.

---

## 🤖 Baseline Scores (0.0 to 1.0 Scale)
Using `gpt-4o-mini` via OpenAI API (Score formula = SLA Fill Rate %):
* **Easy:** ~0.95
* **Medium:** ~0.82
* **Hard:** ~0.45 

---

## 💻 Setup & Usage Instructions

### 1. Build and Run the Server (Docker Recommended)
```bash
docker build -t optibatch-env .
docker run -p 7860:7860 optibatch-env
```

### 2. Run the Baseline Agent
```bash
# Set your API Key (If omitted, script falls back to Heuristic Mode)
export OPENAI_API_KEY="your-api-key"
python baseline.py
```

