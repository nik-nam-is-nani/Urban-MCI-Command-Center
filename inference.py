"""
================================================
Inference Script for Urban MCI Environment
================================================
"""

import os
import json
import random
from typing import List, Dict, Any, Optional

from urban_mci_env import (
    UrbanMCIEnv, IncidentAction, TriageTag, TeamType, VictimStatus, grade,
)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

SUCCESS_SCORE_THRESHOLD = 0.5


def log_start(task: str, env: str, model: str):
    print(json.dumps({"event": "START", "task": task, "env": env, "model": model}), flush=True)

def log_step(step: int, action: Any, reward: float, done: bool, error=None):
    print(json.dumps({"event": "STEP", "step": step, "action": str(action), "reward": round(float(reward), 4), "done": bool(done), "error": str(error) if error else None}), flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(json.dumps({"event": "END", "success": bool(success), "steps": steps, "score": round(float(score), 4), "rewards": [round(float(r), 4) for r in rewards]}), flush=True)


class HeuristicAgent:
    def __init__(self, env):
        self.env = env

    def act(self, state):
        directives = []
        directives.extend(self._triage_victims(state))
        directives.extend(self._dispatch_ambulances(state))
        directives.extend(self._assign_sar_teams(state))
        directives.extend(self._assign_fire_teams(state))
        return IncidentAction(directives=directives)

    def _triage_victims(self, state):
        directives = []
        for v in state["victims"]:
            if v["status"] in ("TRAPPED", "TRIAGED") and v["assigned_tag"] is None:
                m = v["minutes_since_injury"]
                tag = TriageTag.RED if m > 10 else (TriageTag.YELLOW if m > 5 else TriageTag.GREEN)
                directives.append({"type": "triage", "victim_id": v["id"], "tag": tag})
        return directives

    def _dispatch_ambulances(self, state):
        directives = []
        free_ambs = [t for t in state["teams"] if t["type"] == "AMBULANCE" and t["is_free"] and t["transport_victim"] is None]
        if not free_ambs:
            return directives
        triaged = []
        for v in state["victims"]:
            if v["status"] in ("TRAPPED", "TRIAGED") and v["assigned_tag"] is not None:
                p = 0 if v["assigned_tag"] == "RED" else (1 if v["assigned_tag"] == "YELLOW" else 2)
                triaged.append((p, -v.get("minutes_since_injury", 0), v))
        triaged.sort(key=lambda x: (x[0], x[1]))
        hospitals = state["hospitals"]
        for amb in free_ambs:
            if not triaged:
                break
            _, _, victim = triaged.pop(0)
            if victim["assigned_tag"] == "RED":
                accepting = sorted([h for h in hospitals if h["is_accepting"]], key=lambda h: h["trauma_level"])
            else:
                accepting = sorted([h for h in hospitals if h["is_accepting"]], key=lambda h: h["travel_time_minutes"])
            if not accepting:
                continue
            directives.append({"type": "dispatch", "team_id": amb["id"], "victim_id": victim["id"], "hospital_id": accepting[0]["id"]})
        return directives

    def _assign_sar_teams(self, state):
        directives = []
        free_sar = [t for t in state["teams"] if t["type"] == "SEARCH_RESCUE" and t["is_free"]]
        trapped = sorted([v for v in state["victims"] if v["status"] == "TRAPPED"], key=lambda x: x.get("minutes_since_injury", 0), reverse=True)
        for sar in free_sar:
            if not trapped:
                break
            directives.append({"type": "assign_sar", "team_id": sar["id"], "victim_id": trapped.pop(0)["id"]})
        return directives

    def _assign_fire_teams(self, state):
        directives = []
        if state.get("secondary_collapse_risk", 0.0) < 0.3:
            return directives
        free_fire = [t for t in state["teams"] if t["type"] == "FIRE" and t["is_free"]]
        trapped = sorted([v for v in state["victims"] if v["status"] == "TRAPPED"], key=lambda x: x.get("minutes_since_injury", 0), reverse=True)
        for ft in free_fire:
            if not trapped:
                break
            directives.append({"type": "assign_fire", "team_id": ft["id"], "victim_id": trapped.pop(0)["id"]})
        return directives


class LLMAgent:
    def __init__(self, env):
        self.env = env
        self.api_base_url = os.environ.get("API_BASE_URL", "")
        self.model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        self.hf_token = os.environ.get("HF_TOKEN", "")
        if not self.api_base_url:
            raise ValueError("API_BASE_URL not set")
        if not self.hf_token:
            raise ValueError("HF_TOKEN not set")
        self.client = OpenAI(api_key=self.hf_token, base_url=self.api_base_url)

    def act(self, state):
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": "You are an incident commander. Return ONLY a JSON array of directives with types: triage(victim_id,tag 0-3), dispatch(team_id,victim_id,hospital_id), assign_sar(team_id,victim_id), assign_fire(team_id,victim_id). Prioritize RED victims to Level-1 hospitals."}, {"role": "user", "content": str(state)}],
                temperature=0.1, max_tokens=2000,
            )
            raw = resp.choices[0].message.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            directives = json.loads(raw)
            valid = []
            for d in directives:
                if d.get("type") == "triage" and "victim_id" in d and "tag" in d:
                    valid.append(d)
                elif d.get("type") == "dispatch" and all(k in d for k in ["team_id","victim_id","hospital_id"]):
                    valid.append(d)
                elif d.get("type") in ("assign_sar","assign_fire") and "team_id" in d and "victim_id" in d:
                    valid.append(d)
            return IncidentAction(directives=valid)
        except Exception as e:
            print(json.dumps({"event": "DEBUG", "msg": f"LLM failed: {e}"}), flush=True)
            return HeuristicAgent(self.env).act(state)


def run_inference(task=1, max_steps=120, seed=42, use_llm=False):
    random.seed(seed)
    model_name = os.environ.get("MODEL_NAME", "heuristic-agent")
    task_name = f"urban_mci_task_{task}"

    log_start(task=task_name, env="urban-mci-command-center", model=model_name)

    env = UrbanMCIEnv(task=task)
    state = env.reset()
    agent = LLMAgent(env) if use_llm else HeuristicAgent(env)

    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        for step in range(1, max_steps + 1):
            action = agent.act(state)
            error = None
            try:
                state, reward, done, info = env.step(action)
            except Exception as e:
                error = str(e)
                reward = 0.0
                done = True
                info = {}
            rewards.append(float(reward))
            steps_taken = step
            log_step(step=step, action=action.directives, reward=reward, done=done, error=error)
            if done:
                break
        score = min(max(grade(env), 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main():
    use_llm = bool(os.environ.get("API_BASE_URL") and os.environ.get("HF_TOKEN"))
    for task_num in [1, 2, 3]:
        run_inference(task=task_num, seed=42, use_llm=use_llm)


if __name__ == "__main__":
    main()
