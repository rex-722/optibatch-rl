import random
import math
import requests
from typing import List, Dict, Any
from datetime import datetime, timedelta, timezone

class DeliveryCityEnvironment:
    def __init__(self):
        self.num_hubs = 15 
        self.num_riders = 150
        self.max_capacity = 4 
        self.max_wait_ticks = 5 
        
        self.hubs = {f"Store_{i}": (round(random.uniform(1, 9), 2), round(random.uniform(1, 9), 2)) for i in range(1, self.num_hubs + 1)}
        self.reset()

    def fetch_live_weather(self) -> None:
        try:
            url = "https://api.open-meteo.com/v1/forecast?latitude=12.9716&longitude=77.5946&current_weather=true"
            data = requests.get(url, timeout=3).json()
            weather_code = data.get("current_weather", {}).get("weathercode", 0)
            self.temperature = data.get("current_weather", {}).get("temperature", 25)
            self.is_raining = weather_code >= 50
        except Exception:
            self.is_raining, self.temperature = False, 25
        self.weather_desc = f"🌧️ Rain ({self.temperature}°C)" if self.is_raining else f"🌤️ Clear ({self.temperature}°C)"

    def reset(self) -> Dict[str, Any]:
        self.is_running = True 
        self.delivered_count = 0
        self.fetch_live_weather()
        
        self.riders = []
        hubs_list = list(self.hubs.values())
        
        # 🔥 FIX 1: PERFECT EVEN DISTRIBUTION (10 Riders per Hub exactly) 🔥
        for i in range(self.num_riders):
            start_hub = hubs_list[i % self.num_hubs]
            self.riders.append({
                "id": i + 1, 
                "loc": [start_hub[0] + random.uniform(-0.1, 0.1), start_hub[1] + random.uniform(-0.1, 0.1)], 
                "status": "idle", 
                "target_loc": None, 
                "active_orders": [], 
                "load": 0,
                "wait_timer": 0 
            })
            
        self.orders = []
        self.order_counter = 0
        self.shift_data = {
            "Morning": {"total": 0, "delivered": 0, "breach": 0, "imbalance": 0},
            "Afternoon": {"total": 0, "delivered": 0, "breach": 0, "imbalance": 0},
            "Evening": {"total": 0, "delivered": 0, "breach": 0, "imbalance": 0},
            "Night": {"total": 0, "delivered": 0, "breach": 0, "imbalance": 0}
        }
        return self._get_observation()

    def get_current_shift(self) -> str:
        h = datetime.now(timezone(timedelta(hours=5, minutes=30))).hour
        if 6 <= h < 12: return "Morning"
        elif 12 <= h < 17: return "Afternoon"
        elif 17 <= h < 22: return "Evening"
        else: return "Night"
    def calculate_0_to_1_reward(self, shift: str) -> float:
        data = self.shift_data[shift]
        if data["total"] == 0: return 1.0 
        
        # 🔥 FIX: Scale the live penalty by the volume of orders (matching the final report) 🔥
        raw_penalty = (data["breach"] * 0.15) + (data["imbalance"] * 0.005)
        scaled_penalty = raw_penalty / max(1, (data["total"] / 10))
        
        return max(0.0, min(1.0, 1.0 - scaled_penalty))

    def stop_engine(self) -> Dict[str, Any]:
        self.is_running = False
        return self._get_daily_summary()

    def _get_daily_summary(self) -> Dict[str, Any]:
        total_orders = sum(s["total"] for s in self.shift_data.values())
        if total_orders == 0: return {"avg_score": 1.0, "shifts": self.shift_data, "total_delivered": self.delivered_count}
        total_breach = sum(s["breach"] for s in self.shift_data.values())
        total_imbalance = sum(s["imbalance"] for s in self.shift_data.values())
        
        # Matches the live reward calculation
        final_score = max(0.0, min(1.0, 1.0 - ((total_breach * 0.15 + total_imbalance * 0.005) / max(1, (total_orders/10)))))
        return {"avg_score": round(final_score, 2), "shifts": self.shift_data, "total_delivered": self.delivered_count}

    def _get_hub_rider_count(self, hub_loc: tuple) -> int:
        return sum(1 for r in self.riders if r["status"] in ["idle", "relocating"] and math.hypot(r["loc"][0]-hub_loc[0], r["loc"][1]-hub_loc[1]) < 1.0)

    def step(self, assignments: List[Any]) -> Dict[str, Any]:
        if not self.is_running: return self._get_observation()
        shift = self.get_current_shift()
        
        # 🔥 FIX 3: MORE FORGIVING IMBALANCE RULE 🔥
        imbalance_flag = False
        for h_loc in self.hubs.values():
            count = self._get_hub_rider_count(h_loc)
            # Only trigger severe imbalance if hub is almost empty (<2) or over-crowded (>20)
            if count < 2 or count > 20: imbalance_flag = True
        if imbalance_flag: self.shift_data[shift]["imbalance"] += 1

        num_orders = 1 if (random.random() < 0.6 if shift == "Evening" else random.random() < 0.3) else 0
        if self.is_raining: num_orders += 1 
        
        for _ in range(num_orders):
            hub = random.choice(list(self.hubs.keys()))
            hub_loc = self.hubs[hub]
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0.5, 3.0) 
            drop_x = max(0.0, min(10.0, hub_loc[0] + distance * math.cos(angle)))
            drop_y = max(0.0, min(10.0, hub_loc[1] + distance * math.sin(angle)))
            
            self.order_counter += 1
            self.orders.append({"id": self.order_counter, "hub": hub, "pickup_loc": hub_loc, "dropoff_loc": [round(drop_x, 2), round(drop_y, 2)], "status": "pending", "wait_time": 0})
            self.shift_data[shift]["total"] += 1

        for assign in assignments:
            rider = next((r for r in self.riders if r["id"] == assign.rider_id), None)
            order = next((o for o in self.orders if o["id"] == assign.order_id and o["status"] == "pending"), None)
            
            if rider and order and (rider["status"] in ["idle", "relocating", "waiting_at_hub"] or (rider["status"] == "heading_to_pickup" and rider["load"] < self.max_capacity)):
                if rider["status"] in ["idle", "relocating"]:
                    rider["status"] = "heading_to_pickup"
                    rider["target_loc"] = order["pickup_loc"]
                
                rider["active_orders"].append(order)
                rider["load"] += 1
                order["status"] = "assigned"

        speed = 1.0 if self.is_raining else 2.0 
        for r in self.riders:
            if r["status"] == "waiting_at_hub":
                r["wait_timer"] += 1
                if r["load"] >= self.max_capacity or r["wait_timer"] >= self.max_wait_ticks:
                    r["status"] = "delivering"
                    if r["active_orders"]: r["target_loc"] = r["active_orders"][0]["dropoff_loc"]
                continue 

            if r["target_loc"]:
                rx, ry = r["loc"]
                tx, ty = r["target_loc"]
                dist = math.hypot(tx - rx, ty - ry)
                
                if dist <= speed:
                    r["loc"] = [tx, ty] 
                    if r["status"] == "heading_to_pickup":
                        r["status"] = "waiting_at_hub"
                        r["wait_timer"] = 0

                    elif r["status"] == "delivering":
                        if r["active_orders"]:
                            delivered_order = r["active_orders"].pop(0)
                            self.orders = [o for o in self.orders if o["id"] != delivered_order["id"]]
                            self.delivered_count += 1
                            self.shift_data[shift]["delivered"] += 1
                            r["load"] -= 1

                        if r["active_orders"]:
                            r["target_loc"] = r["active_orders"][0]["dropoff_loc"] 
                        else:
                            best_hub_loc = None
                            min_score = float('inf')
                            for h_loc in self.hubs.values():
                                r_count = self._get_hub_rider_count(h_loc)
                                h_dist = math.hypot(r["loc"][0]-h_loc[0], r["loc"][1]-h_loc[1])
                                score = h_dist + (100 if r_count >= 20 else 0) - (20 if r_count < 2 else 0)
                                if score < min_score:
                                    min_score = score
                                    best_hub_loc = h_loc
                            r["status"], r["target_loc"] = "relocating", best_hub_loc
                else:
                    r["loc"][0] += (tx - rx) / dist * speed
                    r["loc"][1] += (ty - ry) / dist * speed
                    r["loc"][0], r["loc"][1] = round(r["loc"][0], 2), round(r["loc"][1], 2)

        for o in self.orders:
            if o["status"] == "pending":
                o["wait_time"] += 1
                if o["wait_time"] == 30: self.shift_data[shift]["breach"] += 1

        return self._get_observation()

    def _get_observation(self) -> Dict[str, Any]:
        shift = self.get_current_shift()
        return {
            "is_running": self.is_running, "shift": shift, "weather": self.weather_desc,
            "current_score": round(self.calculate_0_to_1_reward(shift), 2),
            "hubs": self.hubs, "riders": self.riders, "orders": self.orders,
            "delivered_total": self.delivered_count
        }
