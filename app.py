"""
================================================
 Flask API Server for Urban MCI Environment
================================================

This provides HTTP endpoints for HuggingFace Spaces deployment.
Required for validation: POST /reset must return 200

Features auto-simulation: when /step is called without directives,
the heuristic agent automatically makes decisions.

Endpoints:
- POST /reset - Reset environment for given task
- POST /step  - Execute action (auto-agent if no directives provided)
- GET  /state - Get current state
- GET  /grade - Get current grade (0-1)
"""

import os
import json
from flask import Flask, request, jsonify, make_response
from flask import send_from_directory
from flask_cors import CORS

from urban_mci_env import UrbanMCIEnv, IncidentAction, TriageTag, TeamType, grade

app = Flask(__name__)

# Enable CORS for all routes - needed for React dashboard
CORS(app, resources={r"/*": {"origins": "*"}})

# Optional static dashboard (served from this backend to avoid file:// CORS issues)
_DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "dashboard", "build")

# Global environment state
env = None
current_task = 1


def get_json_body():
    """
    Safely parse request JSON without triggering Flask 415 errors when
    Content-Type is missing or incorrect.
    """
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        return data

    raw = request.get_data(as_text=True) or ""
    if raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    if request.form:
        return request.form.to_dict(flat=True)

    return {}


def create_env():
    """Create and reset the environment."""
    global env, current_task
    env = UrbanMCIEnv(task=current_task)
    return env.reset()


class AutoAgent:
    """
    Heuristic agent that automatically makes decisions.
    Used when /step is called without explicit directives.
    """
    
    def __init__(self, env):
        self.env = env
        
    def act(self, state):
        """Generate action based on current state."""
        directives = []
        
        # 1. First, triage any untriaged trapped victims
        directives.extend(self._triage_victims(state))
        
        # 2. Dispatch ambulances to triaged victims
        directives.extend(self._dispatch_ambulances(state))
        
        # 3. Assign SAR teams to trapped victims
        directives.extend(self._assign_sar_teams(state))
        
        return IncidentAction(directives=directives)
    
    def _triage_victims(self, state):
        """Triage untriaged victims using heuristic."""
        directives = []
        
        untriaged = [
            v for v in state.get("victims", [])
            if v["status"] in ("TRAPPED", "TRIAGED") and v["assigned_tag"] is None
        ]
        
        for victim in untriaged:
            minutes = victim.get("minutes_since_injury", 0)
            
            if minutes > 10:
                tag = TriageTag.RED
            elif minutes > 5:
                tag = TriageTag.YELLOW
            else:
                tag = TriageTag.GREEN
                
            directives.append({
                "type": "triage",
                "victim_id": victim["id"],
                "tag": tag,
            })
            
        return directives
    
    def _dispatch_ambulances(self, state):
        """Dispatch free ambulances to triaged victims."""
        directives = []
        
        free_ambs = [
            t for t in state.get("teams", [])
            if t["type"] == "AMBULANCE" 
            and t["is_free"] 
            and t.get("transport_victim") is None
        ]
        
        if not free_ambs:
            return directives
            
        # SMART AGENT FIX: Dispatches TRAPPED directly and accounts for wait time!
        triaged = []
        for v in state.get("victims", []):
            if v["status"] in ("TRAPPED", "TRIAGED") and v["assigned_tag"] is not None:
                tag_str = v["assigned_tag"]
                priority = 0 if tag_str == "RED" else (1 if tag_str == "YELLOW" else 2)
                triaged.append((priority, -v.get("minutes_since_injury", 0), v))
        
        triaged.sort(key=lambda x: (x[0], x[1]))
        
        hospitals = state.get("hospitals", [])
        
        for amb in free_ambs:
            if not triaged:
                break
                
            _, _, victim = triaged.pop(0)
            tag_str = victim.get("assigned_tag", "GREEN")
            
            # Select hospital based on victim priority
            if tag_str == "RED":
                accepting = [h for h in hospitals if h["is_accepting"]]
                if accepting:
                    accepting.sort(key=lambda h: h.get("trauma_level", 1))
                    hosp = accepting[0]
                else:
                    continue
            else:
                accepting = [h for h in hospitals if h["is_accepting"]]
                if accepting:
                    accepting.sort(key=lambda h: h.get("travel_time_minutes", 999))
                    hosp = accepting[0]
                else:
                    continue
            
            directives.append({
                "type": "dispatch",
                "team_id": amb["id"],
                "victim_id": victim["id"],
                "hospital_id": hosp["id"],
            })
            
        return directives
    
    def _assign_sar_teams(self, state):
        """Assign SAR teams to trapped victims."""
        directives = []
        
        free_sar = [
            t for t in state.get("teams", [])
            if t["type"] == "SEARCH_RESCUE" and t["is_free"]
        ]
        
        if not free_sar:
            return directives
            
        trapped = [
            v for v in state.get("victims", [])
            if v["status"] == "TRAPPED"
        ]
        
        # Focus on those waiting the longest
        trapped.sort(key=lambda x: x.get("minutes_since_injury", 0), reverse=True)
        
        for sar in free_sar:
            if not trapped:
                break
            victim = trapped.pop(0)
            
            directives.append({
                "type": "assign_sar",
                "team_id": sar["id"],
                "victim_id": victim["id"],
            })
            
        return directives


# Global agent
agent = None


@app.route('/reset', methods=['POST'])
def reset():
    """Reset the environment."""
    global current_task, agent

    body = get_json_body()
    task = body.get('task')
    if task is None:
        task = request.args.get('task', default=1, type=int)
    try:
        task = int(task)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid task. Use 1, 2, or 3."}), 400

    if task not in (1, 2, 3):
        return jsonify({"error": "Invalid task. Use 1, 2, or 3."}), 400
    
    current_task = task
    state = create_env()
    
    # Create new agent - MUST use global keyword!
    global agent
    agent = AutoAgent(env)
    
    return jsonify({
        "state": state,
        "task": current_task
    }), 200


@app.route('/step', methods=['POST'])
def step():
    """Execute one step in the environment."""
    global env, agent
    
    if env is None:
        return jsonify({"error": "Environment not initialized. Call /reset first."}), 400
    
    body = get_json_body()
    directives = body.get('directives', []) if body else []
    if not isinstance(directives, list):
        directives = []
    
    # If no directives provided, use auto-agent
    if not directives and agent:
        state = env.state()
        action = agent.act(state)
        directives = action.directives
        print(f"[AUTO-AGENT] Step {state.get('step', 0)}: {len(directives)} directives")
        for d in directives:
            print(f"  -> {d}")
    
    # If episode already ended, return a stable terminal payload instead of 500.
    if getattr(env, "_done", False):
        state = env.state()
        summary = state.get("summary", {})
        info = {
            "step": state.get("step", 0),
            "lives_saved": env._lives_saved(),
            "lives_lost": env._lives_lost(),
            "still_trapped": summary.get("trapped", 0),
            "still_triaged_waiting": summary.get("triaged", 0),
            "in_transit": summary.get("in_transit", 0),
            "ambulances_free": env._count_free_ambulances(),
            "log": [],
            "total_reward_so_far": sum(getattr(env, "_reward_log", [])),
            "grade": grade(env),
            "episode_done": True,
            "message": "Episode is complete. Call /reset to start a new simulation.",
        }
        return jsonify({
            "state": state,
            "reward": 0.0,
            "done": True,
            "info": info
        }), 200

    action = IncidentAction(directives=directives)
    try:
        state, reward, done, info = env.step(action)
    except RuntimeError as e:
        if "Episode is done" not in str(e):
            raise
        state = env.state()
        summary = state.get("summary", {})
        info = {
            "step": state.get("step", 0),
            "lives_saved": env._lives_saved(),
            "lives_lost": env._lives_lost(),
            "still_trapped": summary.get("trapped", 0),
            "still_triaged_waiting": summary.get("triaged", 0),
            "in_transit": summary.get("in_transit", 0),
            "ambulances_free": env._count_free_ambulances(),
            "log": [],
            "total_reward_so_far": sum(getattr(env, "_reward_log", [])),
            "grade": grade(env),
            "episode_done": True,
            "message": "Episode is complete. Call /reset to start a new simulation.",
        }
        return jsonify({
            "state": state,
            "reward": 0.0,
            "done": True,
            "info": info
        }), 200
    
    return jsonify({
        "state": state,
        "reward": reward,
        "done": done,
        "info": info
    })


@app.route('/state', methods=['GET'])
def get_state():
    """Get current environment state."""
    if env is None:
        return jsonify({"error": "Environment not initialized. Call /reset first."}), 400
    
    return jsonify(env.state())


@app.route('/grade', methods=['GET'])
def get_grade():
    """Get current grade (0-1 normalized score)."""
    if env is None:
        return jsonify({"error": "Environment not initialized. Call /reset first."}), 400
    
    return jsonify({
        "grade": grade(env),
        "lives_saved": env._lives_saved(),
        "lives_lost": env._lives_lost()
    })


@app.route('/tasks', methods=['GET'])
def list_tasks():
    """List available tasks."""
    return jsonify({
        "tasks": [
            {"id": 1, "name": "Task 1 - Easy", "victims": 40},
            {"id": 2, "name": "Task 2 - Medium", "victims": 120},
            {"id": 3, "name": "Task 3 - Hard", "victims": 240}
        ]
    })



@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint used by dashboard and API clients."""
    return jsonify({
        "status": "healthy",
        "agent_connected": agent is not None,
        "env_initialized": env is not None,
    })


def _serve_dashboard_page(filename: str = "index.html"):
    """Serve dashboard HTML with no-cache headers for live updates."""
    response = make_response(send_from_directory(_DASHBOARD_DIR, filename))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route('/static/<path:path>', methods=['GET'])
def serve_static(path):
    """Serve static assets (CSS, JS, etc.)."""
    static_dir = os.path.join(_DASHBOARD_DIR, 'static')
    full_path = os.path.join(static_dir, path)
    print(f"[STATIC] Request: {path}")
    print(f"[STATIC] Full path: {full_path}")
    print(f"[STATIC] Exists: {os.path.exists(full_path)}")
    try:
        return send_from_directory(static_dir, path)
    except Exception as e:
        print(f"[STATIC] Error: {e}")
        return jsonify({"error": "Asset not found", "path": path}), 404


@app.route('/manifest.json', methods=['GET'])
def serve_manifest():
    """Serve manifest.json for PWA."""
    try:
        return send_from_directory(_DASHBOARD_DIR, 'manifest.json')
    except Exception:
        return jsonify({"error": "Manifest not found"}), 404


@app.route('/favicon.ico', methods=['GET'])
def serve_favicon():
    """Serve favicon."""
    try:
        return send_from_directory(_DASHBOARD_DIR, 'favicon.ico')
    except Exception:
        return "", 204


@app.route('/', methods=['GET'])
def root_dashboard():
    """Serve dashboard at root for HuggingFace Spaces."""
    return _serve_dashboard_page("index.html")


@app.route('/dashboard', methods=['GET'])
@app.route('/dashboard/', methods=['GET'])
def dashboard():
    """Serve the static HTML dashboard (dashboard/build/index.html)."""
    return _serve_dashboard_page("index.html")


@app.route('/<path:path>', methods=['GET'])
def catch_all(path):
    """Catch-all for SPA routing - serve index.html for all unknown routes."""
    # Serve static files if they exist
    if os.path.exists(os.path.join(_DASHBOARD_DIR, path)):
        return send_from_directory(_DASHBOARD_DIR, path)
    # Otherwise serve index.html for SPA routing
    return _serve_dashboard_page("index.html")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)
