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
import re
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

client = (
    OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )
    if API_KEY
    else None
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
        prompt = self._build_prompt(state)
        directives: List[Dict[str, Any]] = []

        if API_KEY and client is not None:
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    temperature=0,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert disaster response AI. "
                                "Return only valid JSON."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                response_text = self._extract_response_text(completion)
                payload = self._parse_directives_payload(response_text)
                directives = self._sanitize_directives(payload, state)
            except Exception:
                directives = []

        if not directives:
            directives = self._fallback_directives(state)

        return IncidentAction(directives=directives)

    def _build_prompt(self, state: Dict) -> str:
        return f"""
You are an expert disaster response AI managing a mass casualty incident.

Your goal is to maximize survival by prioritizing:
1. Critical victims (RED) first
2. Efficient ambulance usage
3. Fast rescue of trapped victims
4. Proper hospital assignment based on severity

-------------------------
RULES:
- RED = life-threatening -> highest priority
- YELLOW = serious but stable
- GREEN = minor injuries
- Always triage before dispatch if not tagged
- Use available teams efficiently
- Do NOT assign busy teams
- Prefer nearest or suitable hospitals
-------------------------

AVAILABLE ACTION TYPES:

1. TRIAGE:
{{"type": "triage", "victim_id": "<id>", "tag": "RED|YELLOW|GREEN"}}

2. DISPATCH AMBULANCE:
{{"type": "dispatch", "team_id": "<id>", "victim_id": "<id>", "hospital_id": "<id>"}}

3. ASSIGN SEARCH & RESCUE:
{{"type": "assign_sar", "team_id": "<id>", "victim_id": "<id>"}}

-------------------------

STRICT OUTPUT FORMAT:
- Return ONLY valid JSON
- NO explanations
- NO extra text
- Must be parseable using json.loads()

FORMAT:
{{
  "directives": [
    {{"type": "triage", "victim_id": 1, "tag": "RED"}},
    {{"type": "dispatch", "team_id": 1, "victim_id": 1, "hospital_id": 1}},
    {{"type": "assign_sar", "team_id": 2, "victim_id": 2}}
  ]
}}

-------------------------

CURRENT STATE:
{json.dumps(state)}

Now decide the best actions.
"""

    def _extract_response_text(self, completion: Any) -> str:
        content = completion.choices[0].message.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts).strip()
        return str(content or "").strip()

    def _parse_directives_payload(self, response_text: str) -> Dict[str, Any]:
        text = (response_text or "").strip()
        if not text:
            return {"directives": []}

        if text.startswith("```"):
            text = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {"directives": []}
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return {"directives": []}
            candidate = text[start:end + 1]
            try:
                parsed = json.loads(candidate)
                return parsed if isinstance(parsed, dict) else {"directives": []}
            except json.JSONDecodeError:
                return {"directives": []}

    def _to_int(self, value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None
            if s.lstrip("-").isdigit():
                return int(s)
            match = re.search(r"-?\d+", s)
            if match:
                return int(match.group(0))
        return None

    def _normalize_tag(self, value: Any) -> Optional[str]:
        if isinstance(value, TriageTag):
            return value.name if value.name in ("RED", "YELLOW", "GREEN") else None

        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            mapping = {
                int(TriageTag.RED): "RED",
                int(TriageTag.YELLOW): "YELLOW",
                int(TriageTag.GREEN): "GREEN",
            }
            return mapping.get(value)
        if isinstance(value, float) and value.is_integer():
            return self._normalize_tag(int(value))
        if isinstance(value, str):
            tag = value.strip().upper()
            if tag in ("RED", "YELLOW", "GREEN"):
                return tag
        return None

    def _choose_hospital(
        self,
        hospitals: List[Dict[str, Any]],
        tag: str,
        preferred_hospital_id: Optional[int],
    ) -> Optional[int]:
        accepting = [h for h in hospitals if h.get("is_accepting")]
        if not accepting:
            return None

        if preferred_hospital_id is not None:
            for hosp in accepting:
                if hosp.get("id") == preferred_hospital_id:
                    return preferred_hospital_id

        if tag == "RED":
            accepting.sort(
                key=lambda h: (
                    h.get("trauma_level", 99),
                    h.get("travel_time_minutes", 9999),
                )
            )
        else:
            accepting.sort(
                key=lambda h: (
                    h.get("travel_time_minutes", 9999),
                    h.get("trauma_level", 99),
                )
            )
        return self._to_int(accepting[0].get("id"))

    def _sanitize_directives(self, payload: Dict[str, Any], state: Dict) -> List[Dict[str, Any]]:
        raw_directives = payload.get("directives", [])
        if not isinstance(raw_directives, list):
            return []

        victims = state.get("victims", [])
        hospitals = state.get("hospitals", [])
        teams = state.get("teams", [])

        victims_by_id = {
            self._to_int(v.get("id")): v
            for v in victims
            if self._to_int(v.get("id")) is not None
        }

        free_ambulances = {
            self._to_int(t.get("id"))
            for t in teams
            if (
                t.get("type") == "AMBULANCE"
                and t.get("is_free")
                and t.get("transport_victim") is None
                and self._to_int(t.get("id")) is not None
            )
        }
        free_sar = {
            self._to_int(t.get("id"))
            for t in teams
            if (
                t.get("type") == "SEARCH_RESCUE"
                and t.get("is_free")
                and self._to_int(t.get("id")) is not None
            )
        }

        sanitized: List[Dict[str, Any]] = []
        planned_tags: Dict[int, str] = {}
        triaged_this_step = set()
        used_ambulances = set()
        used_sar = set()
        dispatched_victims = set()
        sar_targets = set()

        for raw in raw_directives:
            if not isinstance(raw, dict):
                continue

            dtype = str(raw.get("type", "")).strip().lower()

            if dtype == "triage":
                victim_id = self._to_int(raw.get("victim_id"))
                tag_name = self._normalize_tag(raw.get("tag"))
                if victim_id is None or tag_name is None:
                    continue

                victim = victims_by_id.get(victim_id)
                if victim is None:
                    continue
                if victim.get("status") not in ("TRAPPED", "TRIAGED"):
                    continue
                if victim.get("assigned_tag") is not None:
                    continue
                if victim_id in triaged_this_step:
                    continue

                triaged_this_step.add(victim_id)
                planned_tags[victim_id] = tag_name
                sanitized.append(
                    {
                        "type": "triage",
                        "victim_id": victim_id,
                        "tag": int(TriageTag[tag_name]),
                    }
                )
                continue

            if dtype == "dispatch":
                team_id = self._to_int(raw.get("team_id"))
                victim_id = self._to_int(raw.get("victim_id"))
                hospital_id = self._to_int(raw.get("hospital_id"))

                if team_id is None or victim_id is None:
                    continue
                if team_id not in free_ambulances or team_id in used_ambulances:
                    continue
                if victim_id in dispatched_victims:
                    continue

                victim = victims_by_id.get(victim_id)
                if victim is None:
                    continue
                if victim.get("status") not in ("TRAPPED", "TRIAGED"):
                    continue

                current_tag = victim.get("assigned_tag")
                normalized_tag = self._normalize_tag(current_tag)
                if normalized_tag is None:
                    normalized_tag = planned_tags.get(victim_id)
                if normalized_tag is None:
                    continue

                selected_hospital_id = self._choose_hospital(
                    hospitals=hospitals,
                    tag=normalized_tag,
                    preferred_hospital_id=hospital_id,
                )
                if selected_hospital_id is None:
                    continue

                used_ambulances.add(team_id)
                dispatched_victims.add(victim_id)
                sanitized.append(
                    {
                        "type": "dispatch",
                        "team_id": team_id,
                        "victim_id": victim_id,
                        "hospital_id": selected_hospital_id,
                    }
                )
                continue

            if dtype == "assign_sar":
                team_id = self._to_int(raw.get("team_id"))
                victim_id = self._to_int(raw.get("victim_id"))
                if team_id is None or victim_id is None:
                    continue
                if team_id not in free_sar or team_id in used_sar:
                    continue
                if victim_id in sar_targets or victim_id in dispatched_victims:
                    continue

                victim = victims_by_id.get(victim_id)
                if victim is None:
                    continue
                if victim.get("status") != "TRAPPED":
                    continue

                used_sar.add(team_id)
                sar_targets.add(victim_id)
                sanitized.append(
                    {
                        "type": "assign_sar",
                        "team_id": team_id,
                        "victim_id": victim_id,
                    }
                )

        return sanitized

    def _fallback_directives(self, state: Dict) -> List[Dict[str, Any]]:
        directives: List[Dict[str, Any]] = []
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

