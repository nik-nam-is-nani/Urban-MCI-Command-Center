"""
Inference Script for Urban MCI Command Center
================================================
MANDATORY stdout format:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import sys
import random
import json
from typing import List, Dict, Any, Optional

from openai import OpenAI
from urban_mci_env import (
    UrbanMCIEnv,
    IncidentAction,
    TriageTag,
    TeamType,
    VictimStatus,
    grade,
)

API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "urban-mci-command-center"
MAX_STEPS = 120

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)


class HeuristicAgent:
    """
    A heuristic agent that follows START triage protocol.
    Priority order:
    1. RED victims (immediate) - highest priority, fastest transport
    2. YELLOW victims (delayed) - second priority
    3. GREEN victims (minor) - transport when capacity available
    4. BLACK victims (expectant) - do not transport
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
                accepting = [h for h in hospitals if h["is_accepting"]]
                if accepting:
                    accepting.sort(key=lambda h: h["trauma_level"])
                    hosp = accepting[0]
                else:
                    continue
            else:
                accepting = [h for h in hospitals if h["is_accepting"]]
                if accepting:
                    accepting.sort(key=lambda h: h["travel_time_minutes"])
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
        free_sar = [t for t in state["teams"] if t["type"] == "SEARCH_RESCUE" and t["is_free"]]
        trapped = [v for v in state["victims"] if v["status"] == "TRAPPED"]
        for sar in free_sar[:len(trapped)]:
            if not trapped:
                break
            victim = trapped.pop(0)
            directives.append({"type": "assign_sar", "team_id": sar["id"], "victim_id": victim["id"]})
        return directives

    def _assign_fire_teams(self, state: Dict) -> List[Dict]:
        return []


def run_task(task: int) -> None:
    task_name = f"task-{task}"
    random.seed(42)
    env = UrbanMCIEnv(task=task)
    agent = HeuristicAgent(env)

    state = env.reset()
    rewards_log = []
    last_info = {}
    last_error = None
    step_count = 0
    done = False

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        for step in range(1, MAX_STEPS + 1):
            action = agent.act(state)
            action_str = f"directives({len(action.directives)})"

            try:
                state, reward, done, info = env.step(action)
                last_info = info
                last_error = None
            except Exception as e:
                reward = 0.0
                done = True
                last_error = str(e).replace("\n", " ")
                info = last_info

            rewards_log.append(reward)
            step_count = step
            error_str = last_error if last_error else "null"
            done_str = "true" if done else "false"

            print(
                f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_str} error={error_str}",
                flush=True,
            )

            if done:
                break

    except Exception as e:
        last_error = str(e).replace("\n", " ")

    finally:
        final_score = grade(env)
        success = final_score > 0.0
        rewards_str = ",".join(f"{r:.2f}" for r in rewards_log)
        success_str = "true" if success else "false"
        print(
            f"[END] success={success_str} steps={step_count} score={final_score:.2f} rewards={rewards_str}",
            flush=True,
        )


def main():
    for task in [1, 2, 3]:
        run_task(task)


if __name__ == "__main__":
    main()

