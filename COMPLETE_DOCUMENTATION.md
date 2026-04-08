# Urban Mass Casualty Incident (MCI) Command System

## Complete Documentation - From Scratch

---

## 📋 What Did I Build?

I built a **full-stack AI simulation system** for emergency response training. It's like a "disaster management video game" but with real AI agents making decisions.

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    URBAN MCI COMMAND SYSTEM                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐          ┌──────────────────┐           │
│  │   BACKEND        │          │   FRONTEND        │           │
│  │   (Python/Flask) │◄─────────►│   (React/JS)     │           │
│  │                  │   HTTP   │                  │           │
│  │  - RL Environment│   API    │  - Dashboard UI   │           │
│  │  - Auto Agent    │          │  - Real-time Map │           │
│  │  - API Server    │          │  - Metrics Panel │           │
│  │  - Grader        │          │  - Controls      │           │
│  └──────────────────┘          └──────────────────┘           │
│           │                            │                        │
│           ▼                            ▼                        │
│  ┌─────────────────────────────────────────────┐              │
│  │            DOCKER CONTAINER                  │              │
│  │         (For HuggingFace Deployment)         │              │
│  └─────────────────────────────────────────────┘              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 What Problem Does It Solve?

A mass casualty incident (MCI) happens when many people are injured at once (like a building collapse). Emergency responders must:

1. **Triage** - Quickly identify who needs help most (RED = critical, YELLOW = serious, GREEN = minor)
2. **Rescue** - Get trapped victims out of debris (SAR teams)
3. **Transport** - Send ambulances to hospitals
4. **Allocate Resources** - Balance limited ambulances, hospitals, and time

**This system simulates that chaos and lets AI learn to make better decisions.**

---

## 📁 Files Created

### Backend (Root Directory)

| File | What It Does |
|------|---------------|
| `urban_mci_env.py` | The core RL environment - simulates the disaster |
| `app.py` | Flask API server - handles HTTP requests |
| `inference.py` | LLM-based agent - can use OpenAI for decisions |
| `openenv.yaml` | OpenEnv specification - defines tasks |
| `Dockerfile` | Docker configuration for HuggingFace |
| `requirements.txt` | Python dependencies |
| `test_api.py` | API testing script |

### Frontend (`dashboard/` directory)

| File | What It Does |
|------|---------------|
| `src/App.js` | Main React component |
| `src/components/MapView.js` | 2D canvas map showing patients/hospitals |
| `src/components/MetricsPanel.js` | Real-time statistics |
| `src/components/ControlPanel.js` | Task selection and controls |
| `src/services/api.js` | API communication layer |
| `package.json` | React dependencies |

---

## 🔧 How It Works

### 1. The RL Environment (`urban_mci_env.py`)

This is the core simulation:

```python
# Create environment
env = UrbanMCIEnv(task=1)  # Easy task with 40 victims

# Reset to initial state
state = env.reset()

# Execute one step (1 minute of simulation time)
action = IncidentAction(directives=[
    {"type": "triage", "victim_id": 0, "tag": "RED"},
    {"type": "dispatch", "team_id": 0, "victim_id": 0, "hospital_id": 0}
])
state, reward, done, info = env.step(action)
```

**Features:**
- 3 difficulty levels (40, 120, 240 victims)
- Time pressure (golden hour = 60 minutes)
- Dynamic events (road blocks, secondary collapse, media)
- Realistic resource constraints

### 2. The API Server (`app.py`)

Exposes the environment via HTTP:

| Endpoint | Method | What It Does |
|----------|--------|--------------|
| `/reset` | POST | Reset environment for task 1, 2, or 3 |
| `/step` | POST | Run one simulation step |
| `/state` | GET | Get current patient/hospital/ambulance status |
| `/grade` | GET | Get performance score (0-1) |
| `/health` | GET | Health check |
| `/tasks` | GET | List available tasks |

**Auto-Agent Feature:** When `/step` is called without explicit directives, the built-in heuristic agent automatically makes triage/dispatch decisions.

### 3. The Frontend Dashboard

A React-based real-time dashboard:

- **Map View** - Canvas showing patients color-coded by triage status
- **Metrics Panel** - Live statistics (patients saved/lost, hospital capacity)
- **Control Panel** - Task selection, reset, step, and auto-run buttons
- **Alerts** - Shows collapse risk, road blocks, etc.

---

## 🚀 How to Run From Scratch

### Step 1: Install Python Dependencies

```bash
# Navigate to project directory
cd d:/hack

# Install Python packages
pip install -r requirements.txt
```

Requirements include:
- flask - Web server
- flask-cors - Cross-origin support
- pydantic - Data validation
- openai - LLM integration
- numpy - Numerical computing

### Step 2: Start the Backend

```bash
python app.py
```

This starts the Flask server on port 7860.
You should see:
```
* Running on http://0.0.0.0:7860
```

### Step 3: Install Frontend Dependencies

Open a new terminal:

```bash
cd dashboard
npm install
```

This installs React and other frontend packages.

### Step 4: Start the Frontend

```bash
npm start
```

This opens the dashboard at `http://localhost:3000`.

---

## 🎮 How to Use the Dashboard

### 1. Select a Task
- **Easy** (Task 1): 40 victims, 8 ambulances, 2 hospitals
- **Medium** (Task 2): 120 victims, 5 ambulances, secondary collapse at step 30
- **Hard** (Task 3): 240 victims, 4 ambulances, multiple events

### 2. Controls
- **Reset** - Restart the simulation
- **Step** - Run one simulation step (1 minute)
- **Auto Run** - Continuously run steps automatically

### 3. Watch the Simulation
- **Map** - Shows patients as colored dots (RED/YELLOW/GREEN/BLACK/UNTRIAGED)
- **Metrics** - Real-time stats update
- **Alerts** - Warning signs for events

---

## 📊 Understanding the Metrics

### Patient Status
- **Trapped** - Under debris, needs SAR rescue
- **Triaged** - Assessed, waiting for ambulance
- **In Transit** - In ambulance going to hospital
- **At Hospital** - Successfully delivered
- **Deceased** - Did not survive

### Score Calculation
```
Score = Lives Saved / Saveable Victims
```

- Lives saved = patients delivered to hospital alive
- Saveable = total victims minus BLACK (expectant/deceased)

---

## 🐳 Docker Deployment (For HuggingFace)

To deploy to HuggingFace Spaces:

```bash
# Build the Docker image
docker build -t urban-mci-env .

# Run locally
docker run -p 7860:7860 urban-mci-env
```

The Dockerfile is configured to run `python app.py` (the API server, not inference).

---

## 🔬 Testing the API

Run the test script:

```bash
python test_api.py
```

Expected output:
```
POST /reset: 200
POST /step: 200
GET /grade: 200
GET /health: 200
GET /tasks: 200
All API tests passed!
```

---

## 🎯 What Makes This Special

1. **Real-World Complexity** - Not a toy problem, actual disaster management
2. **Time Pressure** - Golden hour creates urgency
3. **Multi-Agent** - Ambulances, SAR, Fire, Hospitals all interact
4. **Auto-Decisions** - Built-in agent makes decisions without LLM
5. **Real-Time Dashboard** - Visual feedback of simulation

---

## 📝 Summary

This is a complete **AI emergency response simulation system** with:

✅ Python backend (Flask + RL environment)
✅ React frontend (real-time dashboard)
✅ Auto-agent for automatic decisions
✅ OpenEnv compliance for competitions
✅ Docker deployment ready
✅ Color-coded patient visualization
✅ Live metrics and alerts

The system simulates a building collapse with multiple victims and challenges the AI to make optimal triage, rescue, and transport decisions under time pressure.
