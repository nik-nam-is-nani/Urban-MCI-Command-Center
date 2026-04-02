"""
================================================
 Inference Script for Urban MCI Environment
================================================

This script supports two modes:
1. Heuristic: Simple START triage protocol (default)
2. LLM: Uses OpenAI API for decision making (requires API_BASE_URL, MODEL_NAME, HF_TOKEN)

Environment variables:
  SPACE_URL     - If set, use HTTP client against deployed Space
  API_BASE_URL  - LLM API endpoint (enables LLM mode)
  HF_TOKEN      - API key for LLM
  MODEL_NAME    - Model identifier (default: heuristic-agent)

Grading:
- Scores normalized to 0.0-1.0
- Score = lives_saved / saveable_victims
"""

import os
import sys
import random
import json
from typing import List, Dict, Any, Optional

# Try to import OpenAI client
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

SUCCESS_SCORE_THRESHOLD = 0.5

# ─────────────────────────────────────────────
# Structured Logging (required by evaluator)
# ─────────────────────────────────────────────

def log_start(task: str, env: str, model: str):
    print(json.dumps({
        "event": "START",
        "task": task,
        "env": env,
        "model": model
    }), flush=True)


def log_step(step: int, action: any, reward: float, done: bool, error=None):
    print(json.dumps({
        "event": "STEP",
        "step": step,
        "action": str(action),
        "reward": round(reward, 4),
        "done": done,
        "error": str(error) if error else None
    }), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list):
    print(json.dumps({
        "event": "END",
        "success": success,
        "steps": steps,
        "score": round(score, 4),
        "rewards": [round(r, 4) for r in rewards]
    }), flush=True)


# ─────────────────────────────────────────────
# HTTP Client (for deployed HuggingFace Space)
# ─────────────────────────────────────────────

class HTTPEnvClient:
    def __init__(self, base_url: str):
        import requests
        self.base_url = base_url.rstrip("/")
        self.requests = requests

    def reset(self, task: int = 1):
        r = self.requests.post(f"{self.base_url}/reset", json={"task": task})
        r.raise_for_status()
        return r.json()["state"]

    def step(self, directives: list):
        r = self.requests.post(f"{self.base_url}/step", json={"directives": directives})
        r.raise_for_status()
        data = r.json()
        return data["state"], data["reward"], data["done"], data["info"]

    def state(self):
        r = self.requests.get(f"{self.base_url}/state")
        r.raise_for_status()
        return r.json()

    def grade(self):
        r = self.requests.get(f"{self.base_url}/grade")
        r.raise_for_status()
        return r.json()["grade"]


# ─────────────────────────────────────────────
# Local env wrapper (for testing without HTTP)
# ─────────────────────────────────────────────

class LocalEnvClient:
    """Thin wrapper around UrbanMCIEnv matching HTTPEnvClient interface."""

    def __init__(self, task: int = 1):
        from urban_mci_env import UrbanMCIEnv, IncidentAction, grade as _grade
        self._env_cls = UrbanMCIEnv
        self._action_cls = IncidentAction
        self._grade_fn = _grade
        self._env = UrbanMCIEnv(task=task)
        self._task = task

    def reset(self, task: int = 1):
        self._task = task
        self._env = self._env_cls(task=task)
        return self._env.reset()

    def step(self, directives: list):
        action = self._action_cls(directives=directives)
        return self._env.step(action)

    def state(self):
        return self._env.state()

    def grade(self):
        return self._grade_fn(self._env)


def _make_env_client(task: int = 1):
    """Return HTTPEnvClient if SPACE_URL is set, otherwise LocalEnvClient."""
    space_url = os.environ.get("SPACE_URL", "").strip()
    if space_url:
        client = HTTPEnvClient(space_url)
        client.reset(task=task)
        return client
    else:
        return LocalEnvClient(task=task)


# ─────────────────────────────────────────────
# Heuristic Agent
# ─────────────────────────────────────────────

class HeuristicAgent:
    """
    A heuristic agent that follows START triage protocol.

    Priority order:
    1. RED victims (immediate) - highest priority, fastest transport
    2. YELLOW victims (delayed) - second priority
    3. GREEN victims (minor) - transport when capacity available
    4. BLACK victims (expectant) - do not transport
    """

    def __init__(self, client):
        self.client = client

    def act(self, state: Dict) -> List[Dict]:
        """Generate directives based on current state."""
        directives = []
        directives.extend(self._triage_victims(state))
        directives.extend(self._dispatch_ambulances(state))
        directives.extend(self._assign_sar_teams(state))
        directives.extend(self._assign_fire_teams(state))
        return directives

    def _triage_victims(self, state: Dict) -> List[Dict]:
        directives = []
        untriaged = [
            v for v in state["victims"]
            if v["status"] in ("TRAPPED", "TRIAGED") and v["assigned_tag"] is None
        ]
        for victim in untriaged:
            minutes = victim["minutes_since_injury"]
            if minutes > 10:
                tag = 1   # RED
            elif minutes > 5:
                tag = 2   # YELLOW
            else:
                tag = 3   # GREEN
            directives.append({
                "type": "triage",
                "victim_id": victim["id"],
                "tag": tag,
            })
        return directives

    def _dispatch_ambulances(self, state: Dict) -> List[Dict]:
        directives = []
        free_ambs = [
            t for t in state["teams"]
            if t["type"] == "AMBULANCE"
            and t["is_free"]
            and t.get("transport_victim") is None
        ]
        if not free_ambs:
            return directives

        triaged = []
        for v in state["victims"]:
            if v["status"] in ("TRAPPED", "TRIAGED") and v["assigned_tag"] is not None:
                tag_str = v["assigned_tag"]
                priority = 0 if tag_str == "RED" else (1 if tag_str == "YELLOW" else 2)
                triaged.append((priority, -v.get("minutes_since_injury", 0), v))
        triaged.sort(key=lambda x: (x[0], x[1]))

        hospitals = state["hospitals"]
        for amb in free_ambs:
            if not triaged:
                break
            _, _, victim = triaged.pop(0)
            tag_str = victim.get("assigned_tag", "GREEN")

            if tag_str == "RED":
                accepting = [h for h in hospitals if h["is_accepting"]]
                if accepting:
                    accepting.sort(key=lambda h: h["trauma_level"])
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

    def _assign_sar_teams(self, state: Dict) -> List[Dict]:
        directives = []
        free_sar = [
            t for t in state["teams"]
            if t["type"] == "SEARCH_RESCUE" and t["is_free"]
        ]
        if not free_sar:
            return directives

        trapped = [v for v in state["victims"] if v["status"] == "TRAPPED"]
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

    def _assign_fire_teams(self, state: Dict) -> List[Dict]:
        directives = []
        free_fire = [
            t for t in state.get("teams", [])
            if t["type"] == "FIRE" and t["is_free"]
        ]
        if not free_fire:
            return directives

        collapse_risk = state.get("secondary_collapse_risk", 0.0)
        if collapse_risk < 0.3:
            return directives  # not needed at low risk

        trapped = [
            v for v in state.get("victims", [])
            if v["status"] == "TRAPPED"
        ]
        trapped.sort(key=lambda x: x.get("minutes_since_injury", 0), reverse=True)

        for fire_team in free_fire:
            if not trapped:
                break
            victim = trapped.pop(0)
            directives.append({
                "type": "assign_fire",
                "team_id": fire_team["id"],
                "victim_id": victim["id"],
            })
        return directives


# ─────────────────────────────────────────────
# LLM Agent
# ─────────────────────────────────────────────

class LLMAgent:
    """
    LLM-based agent using OpenAI client.

    Required env vars:
      API_BASE_URL, MODEL_NAME, HF_TOKEN
    """

    def __init__(self, client):
        self.client = client
        self.api_base_url = os.environ.get("API_BASE_URL", "")
        self.model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        self.hf_token = os.environ.get("HF_TOKEN", "")

        if not self.api_base_url:
            raise ValueError("API_BASE_URL environment variable not set")
        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable not set")

        self.llm = OpenAI(
            api_key=self.hf_token,
            base_url=self.api_base_url,
        )
        self._heuristic = HeuristicAgent(client)

    def act(self, state: Dict) -> List[Dict]:
        prompt = self._build_prompt(state)
        try:
            response = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
            )
            llm_response = response.choices[0].message.content
            return self._parse_response(llm_response, state)
        except Exception as e:
            print(f"LLM call failed: {e}, falling back to heuristic", flush=True)
            return self._heuristic.act(state)

    def _system_prompt(self) -> str:
        return """You are an incident commander for a mass casualty incident.

TRIAGE CATEGORIES (START Protocol):
- RED (1): Immediate - life-threatening, requires immediate transport
- YELLOW (2): Delayed - serious but stable, can wait
- GREEN (3): Minor - walking wounded
- BLACK (0): Expectant - deceased or unsalvageable, do not transport

RESPONSE FORMAT:
Return a JSON array of directives. Each directive must have:
- type: "triage" | "dispatch" | "assign_sar" | "assign_fire"
- For triage: victim_id, tag (0-3)
- For dispatch: team_id, victim_id, hospital_id
- For assign_sar: team_id, victim_id
- For assign_fire: team_id, victim_id

Respond ONLY with the JSON array, no other text."""

    def _build_prompt(self, state: Dict) -> str:
        prompt = f"""Step: {state['step']} | Golden hour remaining: {state['golden_hour_remaining']} min
Secondary collapse risk: {state['secondary_collapse_risk']:.2f} | Road blocked: {state['road_blocked']}

Summary: trapped={state['summary']['trapped']} triaged={state['summary']['triaged']} transit={state['summary']['in_transit']} hospital={state['summary']['at_hospital']} deceased={state['summary']['deceased']}

VICTIMS (id, status, tag, minutes):
"""
        for v in state['victims'][:20]:
            tag_str = v['assigned_tag'] if v['assigned_tag'] else 'NONE'
            prompt += f"  V{v['id']}: {v['status']}, tag={tag_str}, {v['minutes_since_injury']:.1f}min\n"

        prompt += "\nHOSPITALS:\n"
        for h in state['hospitals']:
            prompt += f"  H{h['id']}: {h['name']}, beds={h['available_beds']}, level={h['trauma_level']}\n"

        prompt += "\nTEAMS:\n"
        for t in state['teams']:
            prompt += f"  T{t['id']}: {t['type']}, free={t['is_free']}\n"

        prompt += "\nWhat directives? (JSON array)"
        return prompt

    def _parse_response(self, response: str, state: Dict) -> List[Dict]:
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            elif response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            directives = json.loads(response.strip())

            valid = []
            for d in directives:
                if not isinstance(d, dict) or 'type' not in d:
                    continue
                if d['type'] == 'triage' and 'victim_id' in d and 'tag' in d:
                    valid.append(d)
                elif d['type'] == 'dispatch' and all(k in d for k in ['team_id', 'victim_id', 'hospital_id']):
                    valid.append(d)
                elif d['type'] in ('assign_sar', 'assign_fire') and 'team_id' in d and 'victim_id' in d:
                    valid.append(d)
            return valid
        except (json.JSONDecodeError, TypeError):
            return self._heuristic.act(state)


# ─────────────────────────────────────────────
# Core inference loop
# ─────────────────────────────────────────────

def run_inference(
    task: int = 1,
    max_steps: int = 120,
    verbose: bool = True,
    seed: int = 42,
    use_llm: bool = False,
) -> float:
    """
    Run the agent on the given task.
    Calls log_step() every step and log_end() when done.

    Returns:
        float: Normalized score (0.0-1.0)
    """
    random.seed(seed)

    client = _make_env_client(task=task)
    state = client.reset(task=task)

    if use_llm and OPENAI_AVAILABLE:
        agent = LLMAgent(client)
    else:
        agent = HeuristicAgent(client)

    rewards = []
    step_num = 0
    done = False

    for step_num in range(max_steps):
        try:
            directives = agent.act(state)
            state, reward, done, info = client.step(directives)
            rewards.append(reward)
            log_step(step=step_num + 1, action=directives, reward=reward, done=done)
            if done:
                break
        except Exception as e:
            log_step(step=step_num + 1, action=[], reward=0.0, done=True, error=e)
            done = True
            break

    # Get final score
    try:
        final_score = client.grade()
    except Exception:
        final_score = 0.0

    success = final_score >= SUCCESS_SCORE_THRESHOLD
    log_end(success=success, steps=step_num + 1, score=final_score, rewards=rewards)

    return final_score


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    use_llm = bool(os.environ.get("API_BASE_URL") and os.environ.get("HF_TOKEN"))
    model_name = os.environ.get("MODEL_NAME", "heuristic-agent")
    results = {}

    for task_num in [1, 2, 3]:
        task_name = f"urban_mci_task_{task_num}"
        log_start(task=task_name, env="urban-mci-command-center", model=model_name)
        score = run_inference(task=task_num, verbose=False, seed=42, use_llm=use_llm)
        results[task_num] = score

    return results


if __name__ == "__main__":
    main()