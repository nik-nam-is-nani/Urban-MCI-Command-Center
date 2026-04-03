"""
================================================
Inference Script for Urban MCI Environment
================================================

MANDATORY ENV VARIABLES:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT (strictly required by judges):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
import random
from typing import List, Dict, Any, Optional

from urban_mci_env import (
    UrbanMCIEnv,
    IncidentAction,
    TriageTag,
    TeamType,
    VictimStatus,
    grade,
)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ─── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

ENV_NAME              = "urban-mci-command-center"
SUCCESS_SCORE_THRESHOLD = 0.5
MAX_STEPS             = 120
SEED                  = 42

# ─── Mandatory stdout loggers ──────────────────────────────────────────────────

def log_start(task: str, env: str, model: str):
    """Print the mandatory [START] line."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action_str: str, reward: float, done: bool, error=None):
    """Print the mandatory [STEP] line."""
    done_str  = "true" if done else "false"
    error_str = str(error) if error else "null"
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]):
    """Print the mandatory [END] line."""
    success_str  = "true" if success else "false"
    rewards_str  = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)


# ─── Heuristic Agent ──────────────────────────────────────────────────────────

class HeuristicAgent:
    """
    Simple START-protocol heuristic agent.
    Priority: RED > YELLOW > GREEN. Never transports BLACK.
    """

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
        for v in state["victims"]:
            if v["status"] in ("TRAPPED", "TRIAGED") and v["assigned_tag"] is None:
                m   = v["minutes_since_injury"]
                tag = TriageTag.RED if m > 10 else (TriageTag.YELLOW if m > 5 else TriageTag.GREEN)
                directives.append({"type": "triage", "victim_id": v["id"], "tag": tag})
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
                p = 0 if v["assigned_tag"] == "RED" else (1 if v["assigned_tag"] == "YELLOW" else 2)
                triaged.append((p, -v.get("minutes_since_injury", 0), v))
        triaged.sort(key=lambda x: (x[0], x[1]))

        hospitals = state["hospitals"]
        for amb in free_ambs:
            if not triaged:
                break
            _, _, victim = triaged.pop(0)
            if victim["assigned_tag"] == "RED":
                accepting = sorted(
                    [h for h in hospitals if h["is_accepting"]],
                    key=lambda h: h["trauma_level"],
                )
            else:
                accepting = sorted(
                    [h for h in hospitals if h["is_accepting"]],
                    key=lambda h: h["travel_time_minutes"],
                )
            if not accepting:
                continue
            directives.append({
                "type":        "dispatch",
                "team_id":     amb["id"],
                "victim_id":   victim["id"],
                "hospital_id": accepting[0]["id"],
            })
        return directives

    def _assign_sar_teams(self, state: Dict) -> List[Dict]:
        directives = []
        free_sar = [t for t in state["teams"] if t["type"] == "SEARCH_RESCUE" and t["is_free"]]
        trapped  = sorted(
            [v for v in state["victims"] if v["status"] == "TRAPPED"],
            key=lambda x: x.get("minutes_since_injury", 0),
            reverse=True,
        )
        for sar in free_sar:
            if not trapped:
                break
            directives.append({"type": "assign_sar", "team_id": sar["id"], "victim_id": trapped.pop(0)["id"]})
        return directives

    def _assign_fire_teams(self, state: Dict) -> List[Dict]:
        directives = []
        if state.get("secondary_collapse_risk", 0.0) < 0.3:
            return directives
        free_fire = [t for t in state["teams"] if t["type"] == "FIRE" and t["is_free"]]
        trapped   = sorted(
            [v for v in state["victims"] if v["status"] == "TRAPPED"],
            key=lambda x: x.get("minutes_since_injury", 0),
            reverse=True,
        )
        for ft in free_fire:
            if not trapped:
                break
            directives.append({"type": "assign_fire", "team_id": ft["id"], "victim_id": trapped.pop(0)["id"]})
        return directives


# ─── LLM Agent ────────────────────────────────────────────────────────────────

class LLMAgent:
    """
    LLM-based agent using the OpenAI client pointed at any compatible API.
    Falls back to HeuristicAgent on any failure.
    """

    SYSTEM_PROMPT = (
        "You are an incident commander for a mass casualty incident.\n"
        "TRIAGE TAGS: 0=BLACK(expectant), 1=RED(immediate), 2=YELLOW(delayed), 3=GREEN(minor).\n"
        "DIRECTIVE TYPES: triage(victim_id,tag), dispatch(team_id,victim_id,hospital_id), "
        "assign_sar(team_id,victim_id), assign_fire(team_id,victim_id).\n"
        "RULES: Prioritize RED victims. Send RED only to trauma_level=1 hospitals. "
        "Never dispatch to a full hospital. Do not dispatch untriaged victims.\n"
        "Respond ONLY with a JSON array of directive objects — no other text."
    )

    def __init__(self, env: UrbanMCIEnv):
        self.env        = env
        self.model_name = MODEL_NAME
        if not API_BASE_URL:
            raise ValueError("API_BASE_URL not set")
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN not set")
        self.client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    def act(self, state: Dict) -> IncidentAction:
        try:
            user_content = self._build_prompt(state)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                temperature=0.1,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            directives = json.loads(raw.strip())
            return IncidentAction(directives=self._validate(directives))
        except Exception as e:
            # Silently fall back — error will show in [STEP] error field if step fails
            return HeuristicAgent(self.env).act(state)

    def _build_prompt(self, state: Dict) -> str:
        summary = state.get("summary", {})
        lines = [
            f"Step: {state['step']}  |  Golden hour remaining: {state['golden_hour_remaining']} min",
            f"Collapse risk: {state['secondary_collapse_risk']:.2f}  |  Road blocked: {state['road_blocked']}",
            f"Victims — total:{summary.get('total_victims',0)}  trapped:{summary.get('trapped',0)}  "
            f"triaged:{summary.get('triaged',0)}  in_transit:{summary.get('in_transit',0)}  "
            f"at_hospital:{summary.get('at_hospital',0)}  deceased:{summary.get('deceased',0)}",
            "",
            "VICTIMS (id | status | tag | minutes_injured):",
        ]
        for v in state["victims"][:30]:   # cap at 30 to stay within token budget
            tag = v["assigned_tag"] or "UNTRIAGED"
            lines.append(f"  {v['id']:3d} | {v['status']:10s} | {tag:10s} | {v['minutes_since_injury']:.1f}")

        lines.append("\nHOSPITALS (id | name | beds | trauma_level | travel_min):")
        for h in state["hospitals"]:
            lines.append(f"  {h['id']} | {h['name']} | {h['available_beds']} beds | L{h['trauma_level']} | {h['travel_time_minutes']}min")

        lines.append("\nTEAMS (id | type | free):")
        for t in state["teams"]:
            lines.append(f"  {t['id']} | {t['type']} | {'YES' if t['is_free'] else 'NO'}")

        lines.append("\nReturn a JSON array of directives.")
        return "\n".join(lines)

    @staticmethod
    def _validate(directives: List[Dict]) -> List[Dict]:
        valid = []
        for d in directives:
            if not isinstance(d, dict) or "type" not in d:
                continue
            t = d["type"]
            if t == "triage" and "victim_id" in d and "tag" in d:
                valid.append(d)
            elif t == "dispatch" and all(k in d for k in ("team_id", "victim_id", "hospital_id")):
                valid.append(d)
            elif t in ("assign_sar", "assign_fire") and "team_id" in d and "victim_id" in d:
                valid.append(d)
        return valid


# ─── Episode runner ───────────────────────────────────────────────────────────

def run_inference(task: int = 1, seed: int = SEED, use_llm: bool = False) -> float:
    """
    Run one full episode for the given task.

    Emits [START], one [STEP] per env.step(), and [END] to stdout.
    Returns the final grade (0.0 – 1.0).
    """
    random.seed(seed)

    task_name  = f"urban_mci_task_{task}"
    model_name = MODEL_NAME if use_llm else "heuristic-agent"

    # ── [START] ──────────────────────────────────────────────────────────────
    log_start(task=task_name, env=ENV_NAME, model=model_name)

    env   = UrbanMCIEnv(task=task)
    state = env.reset()
    agent = LLMAgent(env) if use_llm else HeuristicAgent(env)

    rewards: List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False

    try:
        for step_num in range(1, MAX_STEPS + 1):
            action     = agent.act(state)
            action_str = f"directives({len(action.directives)})"
            error      = None

            try:
                state, reward, done, info = env.step(action)
            except Exception as exc:
                error  = str(exc)
                reward = 0.0
                done   = True
                info   = {}

            rewards.append(float(reward))
            steps_taken = step_num

            # ── [STEP] ───────────────────────────────────────────────────────
            log_step(
                step       = step_num,
                action_str = action_str,
                reward     = reward,
                done       = done,
                error      = error,
            )

            if done:
                break

        score   = min(max(grade(env), 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        # ── [END] ────────────────────────────────────────────────────────────
        # Always emitted, even on exception
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return score


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    use_llm = bool(API_BASE_URL and HF_TOKEN)

    if use_llm:
        print(f"# LLM mode  |  base_url={API_BASE_URL}  |  model={MODEL_NAME}", flush=True)
    else:
        print("# Heuristic mode  |  set API_BASE_URL + HF_TOKEN for LLM mode", flush=True)

    results = {}
    for task_num in [1, 2, 3]:
        score = run_inference(task=task_num, seed=SEED, use_llm=use_llm)
        results[task_num] = score

    print("\n# ── Final Scores ──────────────────────", flush=True)
    for t, s in results.items():
        print(f"#  Task {t}: {s:.4f}", flush=True)


if __name__ == "__main__":
    main()
