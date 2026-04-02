"""
================================================
Inference Script for Urban MCI Environment
================================================
Supports heuristic and LLM modes.
Emits structured [START], [STEP], [END] JSON logs for evaluator.
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


# ─────────────────────────────────────────────
# REQUIRED STRUCTURED LOGGING FUNCTIONS
# ─────────────────────────────────────────────

def log_start(task: str, env: str, model: str):
    print(json.dumps({
        "event": "START",
        "task": task,
        "env": env,
        "model": model
    }), flush=True)


def log_step(step: int, action: Any, reward: float, done: bool, error=None):
    print(json.dumps({
        "event": "STEP",
        "step": step,
        "action": str(action),
        "reward": round(float(reward), 4),
        "done": bool(done),
        "error": str(error) if error else None
    }), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(json.dumps({
        "event": "END",
        "success": bool(success),
        "steps": steps,
        "score": round(float(score), 4),
        "rewards": [round(float(r), 4) for r in rewards]
    }), flush=True)


# ─────────────────────────────────────────────
# HEURISTIC AGENT
# ─────────────────────────────────────────────

class HeuristicAgent:
    def __init__(self, env: UrbanMCIEnv):
        self.env = env

    def act(self, state: Dict) -> IncidentAction:
        directives = []
        directives.extend(self._triage_victims(state))
        directives.extend(self._dispatch_ambulances(state))
        directives.extend(self._assign_sar_teams(state))
        directives.extend(self._assign_fire_teams(state))
        return IncidentAction(directives=directives)

    def _triage_victims(self, state: Dict) -> List[Dict]:
        directives = []
        untriaged = [
            v for v in state["victims"]
            if v["status"] in ("TRAPPED", "TRIAGED") and v["assigned_tag"] is None
        ]
        for victim in untriaged:
            minutes = victim["minutes_since_injury"]
            if minutes > 10:
                tag = TriageTag.RED
            elif minutes > 5:
                tag = TriageTag.YELLOW
            else:
                tag = TriageTag.GREEN
            directives.append({"type": "triage", "victim_id": victim["id"], "tag": tag})
        return directives

    def _dispatch_ambulances(self, state: Dict) -> List[Dict]:
        directives = []
        free_ambs = [
            t for t in state["teams"]
            if t["type"] == "AMBULANCE" and t["is_free"] and t["transport_victim"] is None
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
            tag = TriageTag[victim["assigned_tag"]]
            if tag == TriageTag.RED:
                accepting = sorted(
                    [h for h in hospitals if h["is_accepting"]],
                    key=lambda h: h["trauma_level"]
                )
            else:
                accepting = sorted(
                    [h for h in hospitals if h["is_accepting"]],
                    key=lambda h: h["travel_time_minutes"]
                )
            if not accepting:
                continue
            directives.append({
                "type": "dispatch",
                "team_id": amb["id"],
                "victim_id": victim["id"],
                "hospital_id": accepting[0]["id"],
            })
        return directives

    def _assign_sar_teams(self, state: Dict) -> List[Dict]:
        directives = []
        free_sar = [t for t in state["teams"] if t["type"] == "SEARCH_RESCUE" and t["is_free"]]
        trapped = [v for v in state["victims"] if v["status"] == "TRAPPED"]
        trapped.sort(key=lambda x: x.get("minutes_since_injury", 0), reverse=True)
        for sar in free_sar:
            if not trapped:
                break
            victim = trapped.pop(0)
            directives.append({"type": "assign_sar", "team_id": sar["id"], "victim_id": victim["id"]})
        return directives

    def _assign_fire_teams(self, state: Dict) -> List[Dict]:
        directives = []
        collapse_risk = state.get("secondary_collapse_risk", 0.0)
        if collapse_risk < 0.3:
            return directives
        free_fire = [t for t in state["teams"] if t["type"] == "FIRE" and t["is_free"]]
        trapped = [v for v in state["victims"] if v["status"] == "TRAPPED"]
        trapped.sort(key=lambda x: x.get("minutes_since_injury", 0), reverse=True)
        for fire_team in free_fire:
            if not trapped:
                break
            victim = trapped.pop(0)
            directives.append({"type": "assign_fire", "team_id": fire_team["id"], "victim_id": victim["id"]})
        return directives


# ─────────────────────────────────────────────
# LLM AGENT
# ─────────────────────────────────────────────

class LLMAgent:
    def __init__(self, env: UrbanMCIEnv):
        self.env = env
        self.api_base_url = os.environ.get("API_BASE_URL", "")
        self.model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        self.hf_token = os.environ.get("HF_TOKEN", "")
        if not self.api_base_url:
            raise ValueError("API_BASE_URL not set")
        if not self.hf_token:
            raise ValueError("HF_TOKEN not set")
        self.client = OpenAI(api_key=self.hf_token, base_url=self.api_base_url)

    def act(self, state: Dict) -> IncidentAction:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": self._build_prompt(state)}
                ],
                temperature=0.1,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content or ""
            directives = self._parse(raw, state)
        except Exception as e:
            print(json.dumps({"event": "DEBUG", "msg": f"LLM failed: {e}"}), flush=True)
            directives = HeuristicAgent(self.env).act(state).directives
        return IncidentAction(directives=directives)

    def _system_prompt(self) -> str:
        return (
            "You are an incident commander for a mass casualty incident. "
            "Respond ONLY with a JSON array of directives. "
            "Types: triage(victim_id,tag 0-3), dispatch(team_id,victim_id,hospital_id), "
            "assign_sar(team_id,victim_id), assign_fire(team_id,victim_id). "
            "Prioritize RED victims to Level-1 trauma centers."
        )

    def _build_prompt(self, state: Dict) -> str:
        lines = [
            f"Step: {state['step']}, Golden hour left: {state['golden_hour_remaining']}min",
            f"Collapse risk: {state['secondary_collapse_risk']:.2f}, Road blocked: {state['road_blocked']}",
            f"Summary: {state['summary']}",
            "Victims (first 20):"
        ]
        for v in state["victims"][:20]:
            lines.append(f"  {v['id']}: {v['status']} tag={v['assigned_tag']} {v['minutes_since_injury']:.1f}min")
        lines.append("Hospitals:")
        for h in state["hospitals"]:
            lines.append(f"  {h['id']}: {h['name']} beds={h['available_beds']} level={h['trauma_level']}")
        lines.append("Teams:")
        for t in state["teams"]:
            lines.append(f"  {t['id']}: {t['type']} free={t['is_free']}")
        lines.append("Return JSON array of directives only.")
        return "\n".join(lines)

    def _parse(self, response: str, state: Dict) -> List[Dict]:
        try:
            response = response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            directives = json.loads(response)
            valid = []
            for d in directives:
                if not isinstance(d, dict) or "type" not in d:
                    continue
                if d["type"] == "triage" and "victim_id" in d and "tag" in d:
                    valid.append(d)
                elif d["type"] == "dispatch" and all(k in d for k in ["team_id", "victim_id", "hospital_id"]):
                    valid.append(d)
                elif d["type"] in ("assign_sar", "assign_fire") and "team_id" in d and "victim_id" in d:
                    valid.append(d)
            return valid
        except Exception:
            return HeuristicAgent(self.env).act(state).directives


# ─────────────────────────────────────────────
# CORE RUN FUNCTION
# ─────────────────────────────────────────────

def run_inference(task: int = 1, max_steps: int = 120, seed: int = 42, use_llm: bool = False) -> float:
    random.seed(seed)
    model_name = os.environ.get("MODEL_NAME", "heuristic-agent")
    task_name = f"urban_mci_task_{task}"

    log_start(task=task_name, env="urban-mci-command-center", model=model_name)

    env = UrbanMCIEnv(task=task)
    state = env.reset()
    agent = LLMAgent(env) if use_llm else HeuristicAgent(env)

    rewards: List[float] = []
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

        final_grade = grade(env)
        score = min(max(final_grade, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    use_llm = bool(os.environ.get("API_BASE_URL") and os.environ.get("HF_TOKEN"))
    results = {}
    for task_num in [1, 2, 3]:
        score = run_inference(task=task_num, seed=42, use_llm=use_llm)
        results[task_num] = score
    return results


if __name__ == "__main__":
    main()