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

import math
import os
import threading
from flask import Flask, request, jsonify
from flask import send_from_directory
from flask_cors import CORS
from urban_mci_env import UrbanMCIEnv, IncidentAction, TriageTag, TeamType, grade

app = Flask(__name__)

# Enable CORS for all routes - needed for React dashboard
CORS(app, resources={r"/*": {"origins": "*"}})

# Optional static dashboard (served from this backend to avoid file:// CORS issues)
_DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "dashboard")

# Global environment state
env = None
current_task = 1
_env_lock = threading.Lock()


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
        # Tuned to reduce catastrophic RED misses while keeping directive volume bounded.
        self.max_triage_per_step = 10
        self.red_threshold_minutes = 6
        self.yellow_threshold_minutes = 1

    @staticmethod
    def _distance(point_a, point_b):
        return math.sqrt(
            (point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2
        )

    def _estimate_trip_time(self, amb, victim, hospital, road_blocked=False):
        team_loc = amb.get("location", [100.0, 100.0])
        victim_loc = victim.get("location", team_loc)
        hospital_loc = hospital.get("location", victim_loc)
        trip = self._distance(team_loc, victim_loc) + self._distance(victim_loc, hospital_loc)
        if road_blocked:
            trip *= 1.5
        return trip

    def act(self, state):
        """Generate action based on current state."""
        directives = []
        directives.extend(self._triage_victims(state))
        directives.extend(self._dispatch_ambulances(state))
        directives.extend(self._assign_sar_teams(state))
        return IncidentAction(directives=directives)

    def _triage_victims(self, state):
        """Triage untriaged victims using heuristic."""
        directives = []
        untriaged = [
            v for v in state.get("victims", [])
            if v["status"] in ("TRAPPED", "TRIAGED") and v["assigned_tag"] is None
        ]
        untriaged.sort(key=lambda v: v.get("minutes_since_injury", 0), reverse=True)
        for victim in untriaged[: self.max_triage_per_step]:
            minutes = victim.get("minutes_since_injury", 0)
            if minutes > self.red_threshold_minutes:
                tag = TriageTag.RED
            elif minutes > self.yellow_threshold_minutes:
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

        triaged = []
        for v in state.get("victims", []):
            if v["status"] in ("TRAPPED", "TRIAGED") and v["assigned_tag"] is not None:
                triaged.append(v)

        victims_by_tag = {
            "RED": [v for v in triaged if v.get("assigned_tag") == "RED"],
            "YELLOW": [v for v in triaged if v.get("assigned_tag") == "YELLOW"],
            "GREEN": [v for v in triaged if v.get("assigned_tag") == "GREEN"],
        }
        hospitals = [h for h in state.get("hospitals", []) if h.get("is_accepting")]
        if not hospitals:
            return directives

        used_victims = set()
        for amb in free_ambs:
            bucket = victims_by_tag["RED"] or victims_by_tag["YELLOW"] or victims_by_tag["GREEN"]
            bucket = [v for v in bucket if v.get("id") not in used_victims]
            if not bucket:
                continue

            best_choice = None
            for victim in bucket:
                tag_str = victim.get("assigned_tag", "GREEN")
                best_hosp = None
                best_hosp_key = None
                for hosp in hospitals:
                    trip = self._estimate_trip_time(
                        amb,
                        victim,
                        hosp,
                        road_blocked=bool(state.get("road_blocked", False)),
                    )
                    hosp_key = (trip, hosp.get("trauma_level", 99)) if tag_str == "RED" else (trip,)
                    if best_hosp_key is None or hosp_key < best_hosp_key:
                        best_hosp_key = hosp_key
                        best_hosp = hosp

                if best_hosp is None or best_hosp_key is None:
                    continue

                victim_score = (best_hosp_key[0], -victim.get("minutes_since_injury", 0))
                if best_choice is None or victim_score < best_choice[0]:
                    best_choice = (victim_score, victim, best_hosp)

            if best_choice is None:
                continue

            _, victim, hosp = best_choice
            used_victims.add(victim["id"])

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


def _get_json_body():
    """
    Safely parse JSON body regardless of Content-Type header.
    This fixes the 415 Unsupported Media Type error when the
    OpenEnv validator sends POST /reset without Content-Type: application/json.
    """
    # Try standard JSON parsing first
    if request.is_json:
        return request.get_json(silent=True) or {}
    # Fallback: force-parse even if Content-Type is missing/wrong
    try:
        return request.get_json(force=True, silent=True) or {}
    except Exception:
        return {}


@app.route('/reset', methods=['POST'])
def reset():
    """Reset the environment."""
    global current_task, agent, env

    # Use safe JSON parsing (handles missing Content-Type header)
    body = _get_json_body()

    task = body.get('task') if body else None
    if task is None:
        task = request.args.get('task', default=1, type=int)

    if task not in (1, 2, 3):
        return jsonify({"error": "Invalid task. Use 1, 2, or 3."}), 400

    with _env_lock:
        current_task = task
        state = create_env()
        agent = AutoAgent(env)

    return jsonify({
        "state": state,
        "task": current_task
    }), 200


@app.route('/step', methods=['POST'])
def step():
    """Execute one step in the environment."""
    global env, agent

    body = _get_json_body()
    directives = body.get('directives', []) if body else []

    with _env_lock:
        if env is None:
            return jsonify({"error": "Environment not initialized. Call /reset first."}), 400

        # If no directives provided, use auto-agent
        if not directives and agent:
            state = env.state()
            action = agent.act(state)
            directives = action.directives
            print(f"[AUTO-AGENT] Step {state.get('step', 0)}: {len(directives)} directives")
            for d in directives:
                print(f"  -> {d}")

        # If episode already ended, return stable terminal payload
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
    with _env_lock:
        if env is None:
            return jsonify({"error": "Environment not initialized. Call /reset first."}), 400
        state = env.state()
    return jsonify(state)


@app.route('/grade', methods=['GET'])
def get_grade():
    """Get current grade (0-1 normalized score)."""
    with _env_lock:
        if env is None:
            return jsonify({"error": "Environment not initialized. Call /reset first."}), 400
        payload = {
            "grade": grade(env),
            "lives_saved": env._lives_saved(),
            "lives_lost": env._lives_lost()
        }
    return jsonify(payload)


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

@app.route('/', methods=['GET'])
def index():
    """Serve the dashboard at root path for HuggingFace Spaces."""
    return send_from_directory(_DASHBOARD_DIR, "index.html")

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Serve the static HTML dashboard (dashboard/index.html)."""
    return send_from_directory(_DASHBOARD_DIR, "index.html")

def main():
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    main()
