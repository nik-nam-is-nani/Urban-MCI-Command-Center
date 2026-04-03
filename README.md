---

title: Urban MCI Command Center
emoji: 🚑
colorFrom: red
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
tags:
* openenv
* reinforcement-learning
* emergency-response
* simulation

---

# Urban Mass Casualty Incident (MCI) Command Environment

## Problem #15 — Live Incident Command Agent

A building collapse with 200+ casualties. The agent plays incident commander in real time:

* Triages victims via START protocol
* Dispatches ambulances to capacity-aware hospitals
* Coordinates fire / search-and-rescue teams
* Manages a secondary collapse risk clock
* All under exponential time-decay rewards

**One env step = 60 seconds of simulated time.** The "golden hour" is 60 steps. After that, survival probability decays exponentially.

---

## 🎯 Why This Problem is Hard

1. **Multi-dimensional resource allocation**: Ambulances, SAR teams, fire teams, hospitals with different capabilities
2. **Time-critical decision making**: RED victims have ~60 minutes before mortality spikes
3. **Partial observability**: True triage tags are hidden until properly assessed
4. **Dynamic events**: Secondary collapse, road blockages, media pressure emerge during episode
5. **Competing priorities**: Must balance immediate lifesaving vs. overall throughput

---

## 📋 Task Configurations

| Task | Difficulty | Victims | Ambulances | Hospitals | Events                                              |
| ---- | ---------- | ------- | ---------- | --------- | --------------------------------------------------- |
| 1    | Easy       | 40      | 8          | 2         | None                                                |
| 2    | Medium     | 120     | 5          | 3         | Secondary collapse @ t=30                           |
| 3    | Hard       | 240     | 4          | 4         | Road blockage @ t=10, Media @ t=15, Collapse @ t=20 |

---

## 🔧 Installation

```bash
# Install dependencies
pip install pydantic==1.10.13 pyyaml

# Test the environment
python urban_mci_env.py
```

---

## 🎮 Usage

### Basic Environment Usage

```python
from urban_mci_env import UrbanMCIEnv, IncidentAction, grade

# Create environment
env = UrbanMCIEnv(task=1)

# Reset to initial state
state = env.reset()

# Execute action
action = IncidentAction(directives=[
    {"type": "triage", "victim_id": 0, "tag": 1},  # RED
    {"type": "dispatch", "team_id": 0, "victim_id": 0, "hospital_id": 0},
])

state, reward, done, info = env.step(action)

# Get normalized score (0-1)
score = grade(env)
```

---

### Running Inference

```bash
# Set environment variables
export HF_TOKEN=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

# Run the baseline agent
python inference.py
```

---

## 📊 Observation Space

| Component                 | Type  | Description                             |
| ------------------------- | ----- | --------------------------------------- |
| `step`                    | int   | Current simulation step (1-120)         |
| `golden_hour_remaining`   | int   | Minutes left in golden hour             |
| `secondary_collapse_risk` | float | Probability of secondary collapse (0-1) |
| `road_blocked`            | bool  | Whether main road is blocked            |
| `victims`                 | list  | List of victim observations             |
| `hospitals`               | list  | Hospital states with available beds     |
| `teams`                   | list  | Resource team availability              |

---

## 🎯 Action Space

The agent submits a list of directives each step:

### 1. Triage

```python
{"type": "triage", "victim_id": int, "tag": TriageTag}
# tag: 0=BLACK, 1=RED, 2=YELLOW, 3=GREEN
```

### 2. Dispatch Ambulance

```python
{"type": "dispatch", "team_id": int, "victim_id": int, "hospital_id": int}
```

### 3. Assign SAR Team

```python
{"type": "assign_sar", "team_id": int, "victim_id": int}
```

### 4. Assign Fire Team

```python
{"type": "assign_fire", "team_id": int, "victim_id": int}
```

---

## 🏆 Reward System

### Triage Rewards

| Action                      | Reward |
| --------------------------- | ------ |
| Correct RED triage          | +1.0   |
| Correct non-RED triage      | +0.5   |
| Missed RED (tagged non-RED) | -2.0   |
| Minor mistag                | -0.3   |

### Dispatch Rewards

| Action                    | Reward |
| ------------------------- | ------ |
| Ambulance dispatched      | +0.3   |
| RED → Level 1 trauma      | +0.5   |
| RED → lower tier hospital | -0.3   |
| Sent to full hospital     | -0.5   |
| Dispatch without triage   | -0.3   |

### Delivery Rewards

| Victim Type | Base Reward | Decay                           |
| ----------- | ----------- | ------------------------------- |
| RED         | +10.0       | After 60 min: exp(-(t-60)/30)   |
| YELLOW      | +4.0        | After 120 min: exp(-(t-120)/60) |
| GREEN       | +1.0        | No decay                        |
| BLACK       | -1.0        | Wasted resource                 |

### Time Shaping

After golden hour (step 60), all rewards are multiplied by:
`exp(-overtime/30)`

---

## 📈 Grading

**Score = lives_saved / saveable_victims**

Where:

* `lives_saved`: Victims delivered to hospital alive
* `saveable_victims`: Total victims minus BLACK

Score is normalized to **0.0 - 1.0**

---

## 🧪 Testing

```bash
# Run smoke tests
python urban_mci_env.py

# Run inference
python inference.py
```

---

## 📁 File Structure

```
.
├── urban_mci_env.py
├── openenv.yaml
├── inference.py
├── app.py
├── dashboard/
├── README.md
└── Dockerfile
```

---

## 🔬 Design Decisions

### Why START Protocol?

The START (Simple Triage and Rapid Treatment) protocol is the industry standard for mass casualty incidents.

---

### Why Exponential Decay?

After 60 minutes, survival drops sharply — this creates real-world urgency.

---

### Why Dense Rewards?

Helps the agent learn faster compared to sparse reward systems.

---

## 🚀 Deployment

### HuggingFace Spaces

```bash
docker build -t urban-mci-env .
docker run -p 7860:7860 urban-mci-env
```

---

## 📜 License

MIT License
