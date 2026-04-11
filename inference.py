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
import math
from typing import List, Dict, Any, Optional, Set, Tuple

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
MAX_STEPS = 60  # Reduced from 120 to stay well under 30min limit
LLM_TIMEOUT_SECONDS = 10  # Timeout per LLM call

client = (
    OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
        timeout=LLM_TIMEOUT_SECONDS,
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
        self._llm_failures = 0
        # Force pure-heuristic mode. LLM is never called.
        self._llm_disabled = True

    @staticmethod
    def _distance(point_a, point_b) -> float:
        return math.sqrt(
            (point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2
        )

    def _estimate_trip_time(
        self,
        team: Dict[str, Any],
        victim: Dict[str, Any],
        hospital: Dict[str, Any],
        road_blocked: bool = False,
    ) -> float:
        team_loc = team.get("location", [100.0, 100.0])
        victim_loc = victim.get("location", team_loc)
        hospital_loc = hospital.get("location", victim_loc)
        trip = self._distance(team_loc, victim_loc) + self._distance(victim_loc, hospital_loc)
        if road_blocked:
            trip *= 1.5
        return trip

    def act(self, state: Dict) -> IncidentAction:
        """
        Generate action for current state using pure heuristic logic only.
        """
        return IncidentAction(directives=self._fallback_directives(state))

    def _build_prompt(self, state: Dict) -> str:
        """
        Build comprehensive system prompt with full rules, rewards, and decision algorithm.
        State is summarized to essential info only.
        """
        compact_state = self._summarize_state_for_llm(state)

        return f"""You are an elite AI incident commander for an Urban Mass Casualty Incident (MCI).
You control triage, ambulance dispatch, SAR teams, and fire teams in real-time.

════════════════════════════════════════════════════════
CORE RULES (never violate these)
════════════════════════════════════════════════════════
ONE STEP = 1 simulated minute. You have 60 steps before the golden hour ends.
After step 60, ALL rewards are multiplied by exp(-overtime/30) — they decay fast.

VICTIM STATUSES (pipeline order):
  TRAPPED → needs SAR team first (if under debris) → becomes TRIAGED after extraction
  TRIAGED → ready for ambulance dispatch
  IN_TRANSIT → ambulance en route to hospital
  AT_HOSPITAL → delivered, reward collected
  DECEASED → death occurred, no reward possible

TRIAGE TAGS:
  RED    = 1 = immediate, life-threatening
  YELLOW = 2 = serious but stable
  GREEN  = 3 = minor, walking wounded
  BLACK  = 0 = expectant / deceased, do NOT transport

════════════════════════════════════════════════════════
EXACT REWARD VALUES (these drive every decision)
════════════════════════════════════════════════════════
TRIAGE REWARDS:
  Correct RED triage (you tag RED, true tag is RED)     → +1.0
  Correct non-RED triage                                → +0.5
  WRONG: you tag non-RED but victim is truly RED        → -2.0  ← CATASTROPHIC
  Minor mistag (e.g. YELLOW when GREEN)                 → -0.3

DISPATCH REWARDS:
  Any ambulance dispatched                              → +0.3
  RED victim → Level 1 trauma hospital                  → +0.5 bonus (total +0.8)
  RED victim → lower tier hospital                      → -0.3 penalty (total 0.0)
  Dispatching to a full hospital                        → -0.5
  Dispatching without an assigned triage tag            → -0.3

DELIVERY REWARDS (the biggest rewards — worth 10x triage):
  RED delivered:    +10.0 × exp(-max(0, t-60) / 30)
  YELLOW delivered: +4.0  × exp(-max(0, t-120) / 60)
  GREEN delivered:  +1.0 flat (no decay)
  BLACK delivered:  -1.0 (wasted resource, never do this)

SAR ASSIGNMENT:
  Assigning a SAR team to a trapped victim              → +0.2
  CRITICAL: After SAR completes, victim.assigned_tag is WIPED TO NULL.
  You MUST re-triage SAR-extracted victims immediately.
  A post-extraction victim with high minutes_since_injury is almost always RED.

FIRE TEAM ASSIGNMENT:
  Assigning a fire team to any on-scene victim          → +0.15
  Fire teams can assist with trapped OR triaged victims.
  Never leave fire teams idle when victims need help.

════════════════════════════════════════════════════════
DECISION ALGORITHM (follow this order each step)
════════════════════════════════════════════════════════
1. RE-TRIAGE POST-SAR VICTIMS FIRST
   - Any victim with status="TRIAGED" but assigned_tag=null was just extracted
   - Triage these IMMEDIATELY before any other action
   - High minutes_since_injury → almost certainly RED

2. TRIAGE ALL OTHER UNTRIAGED VICTIMS
   - Use minutes_since_injury: >10min=RED, >5min=YELLOW, else GREEN
   - When in doubt, err toward RED (under-triage kills)

3. DISPATCH AMBULANCES (triaged victims only)
   - RED victims first → Level 1 trauma center
   - YELLOW victims → closest available hospital
   - GREEN victims → only if ambulances spare
   - NEVER dispatch without a triage tag (-0.3 penalty)
   - NEVER dispatch to a full hospital (-0.5 penalty)
   - If road_blocked=true → ignore trauma level, use closest hospital

4. ASSIGN SAR TEAMS to trapped victims
   - Prioritize victims with highest minutes_since_injury

5. ASSIGN FIRE TEAMS to any on-scene victim
   - Trapped victims first, then triaged victims waiting

════════════════════════════════════════════════════════
OUTPUT FORMAT — RETURN ONLY THIS JSON, NOTHING ELSE
════════════════════════════════════════════════════════
{{"directives": [
  {{"type": "triage",     "victim_id": <int>, "tag": "<RED|YELLOW|GREEN>"}},
  {{"type": "dispatch",   "team_id": <int>, "victim_id": <int>, "hospital_id": <int>}},
  {{"type": "assign_sar", "team_id": <int>, "victim_id": <int>}},
  {{"type": "assign_fire","team_id": <int>, "victim_id": <int>}}
]}}

RULES FOR OUTPUT:
  - Return ONLY the JSON object. No explanation, no preamble, no markdown fences.
  - Include triage AND dispatch for the same victim in the same directives array.
  - Never reuse the same team_id twice in one step.
  - Never dispatch a victim already IN_TRANSIT or AT_HOSPITAL.
  - Never assign a team that is_free = False.
  - Never dispatch to a hospital with is_accepting = False.
  - If nothing to do (all victims resolved), return: {{"directives": []}}

════════════════════════════════════════════════════════
CURRENT STATE
════════════════════════════════════════════════════════
{json.dumps(compact_state, separators=(',', ':'))}

Return ONLY the JSON:
"""

    def _summarize_state_for_llm(self, state: Dict) -> Dict[str, Any]:
        """
        Summarize state to essential info only:
        - untriaged victims (max 20)
        - triaged waiting victims (max 15)
        - free teams only
        - hospitals
        - key globals: step, road_blocked, secondary_collapse_risk, golden_hour_remaining
        """
        victims = state.get("victims", [])
        hospitals = state.get("hospitals", [])
        teams = state.get("teams", [])

        # Untriaged victims: TRAPPED or TRIAGED with assigned_tag=None (max 20)
        untriaged = [
            v for v in victims
            if v.get("status") in ("TRAPPED", "TRIAGED") and v.get("assigned_tag") is None
        ]
        untriaged.sort(key=lambda v: v.get("minutes_since_injury", 0), reverse=True)

        # Post-SAR victims (status=TRIAGED but assigned_tag=None) get flagged
        for v in untriaged:
            v["_post_sar"] = v.get("status") == "TRIAGED"

        # Waiting victims: already triaged, ready for dispatch (max 15)
        waiting = [
            v for v in victims
            if v.get("status") in ("TRAPPED", "TRIAGED") and v.get("assigned_tag") is not None
        ]
        waiting.sort(
            key=lambda v: (
                0 if v.get("assigned_tag") == "RED" else (1 if v.get("assigned_tag") == "YELLOW" else 2),
                -v.get("minutes_since_injury", 0),
            )
        )

        # Free teams only
        free_ambulances = [
            {"id": t.get("id")}
            for t in teams
            if t.get("type") == "AMBULANCE" and t.get("is_free") and t.get("transport_victim") is None
        ]
        free_sar = [
            {"id": t.get("id")}
            for t in teams
            if t.get("type") == "SEARCH_RESCUE" and t.get("is_free")
        ]
        free_fire = [
            {"id": t.get("id")}
            for t in teams
            if t.get("type") == "FIRE" and t.get("is_free")
        ]

        return {
            "step": state.get("step"),
            "golden_hour_remaining": state.get("golden_hour_remaining"),
            "road_blocked": state.get("road_blocked"),
            "secondary_collapse_risk": state.get("secondary_collapse_risk"),
            "hospitals": [
                {
                    "id": h.get("id"),
                    "available_beds": h.get("available_beds"),
                    "total_capacity": h.get("total_capacity"),
                    "trauma_level": h.get("trauma_level"),
                    "travel_time_minutes": h.get("travel_time_minutes"),
                    "is_accepting": h.get("is_accepting"),
                }
                for h in hospitals
            ],
            "free_ambulances": free_ambulances,
            "free_sar": free_sar,
            "free_fire": free_fire,
            "untriaged": [
                {
                    "id": v.get("id"),
                    "status": v.get("status"),
                    "minutes_since_injury": v.get("minutes_since_injury"),
                    "post_sar": v.get("_post_sar", False),
                }
                for v in untriaged[:20]
            ],
            "waiting": [
                {
                    "id": v.get("id"),
                    "tag": v.get("assigned_tag"),
                    "status": v.get("status"),
                    "minutes_since_injury": v.get("minutes_since_injury"),
                }
                for v in waiting[:15]
            ],
        }

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
        preferred_hospital_id: Optional[int] = None,
        road_blocked: bool = False,
        in_flight_by_hospital: Optional[Dict[int, int]] = None,
    ) -> Optional[int]:
        # `preferred_hospital_id` is accepted for backward compatibility, but
        # projected bed availability and routing rules decide final selection.
        in_flight_by_hospital = in_flight_by_hospital or {}
        accepting_with_capacity = []
        for hospital in hospitals:
            if not hospital.get("is_accepting"):
                continue
            hospital_id = self._to_int(hospital.get("id"))
            if hospital_id is None:
                continue
            projected_beds = hospital.get("available_beds", 0) - in_flight_by_hospital.get(hospital_id, 0)
            if projected_beds <= 0:
                continue
            accepting_with_capacity.append(hospital)

        if not accepting_with_capacity:
            return None

        if road_blocked:
            accepting_with_capacity.sort(
                key=lambda h: (
                    h.get("travel_time_minutes", 9999),
                    h.get("trauma_level", 99),
                )
            )
            return self._to_int(accepting_with_capacity[0].get("id"))

        if tag == "RED":
            accepting_with_capacity.sort(
                key=lambda h: (
                    h.get("trauma_level", 99),
                    h.get("travel_time_minutes", 9999),
                )
            )
        else:
            accepting_with_capacity.sort(
                key=lambda h: h.get("travel_time_minutes", 9999)
            )
        return self._to_int(accepting_with_capacity[0].get("id"))

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

        # Identify post-SAR victims (status=TRIAGED but assigned_tag=None = just extracted, need re-triage)
        post_sar_ids = {
            vid for vid, v in victims_by_id.items()
            if v.get("status") == "TRIAGED" and v.get("assigned_tag") is None
        }

        # Sort directives: post-SAR triage first, then other triage, then dispatch/assignments
        triage_directives = []
        post_sar_triage_directives = []
        other_directives = []

        for raw in raw_directives:
            if not isinstance(raw, dict):
                continue
            dtype = str(raw.get("type", "")).strip().lower()
            victim_id = self._to_int(raw.get("victim_id"))

            if dtype == "triage" and victim_id in post_sar_ids:
                post_sar_triage_directives.append(raw)
            elif dtype == "triage":
                triage_directives.append(raw)
            else:
                other_directives.append(raw)

        # Process post-SAR triage first (these victims have been waiting through extraction)
        for raw in post_sar_triage_directives + triage_directives:
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
                    road_blocked=bool(state.get("road_blocked", False)),
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

    def _post_sar_tag(self, minutes_since_injury: float) -> str:
        return "RED" if minutes_since_injury > 8 else "YELLOW"

    def _regular_tag(self, minutes_since_injury: float) -> str:
        if minutes_since_injury > 12:
            return "RED"
        if minutes_since_injury >= 5:
            return "YELLOW"
        return "GREEN"

    def _count_in_flight_by_hospital(self, state: Dict[str, Any]) -> Dict[int, int]:
        in_flight: Dict[int, int] = {}
        for team in state.get("teams", []):
            if team.get("type") != "AMBULANCE":
                continue
            if team.get("transport_victim") is None:
                continue
            hospital_id = self._to_int(team.get("assigned_hospital"))
            if hospital_id is None:
                continue
            in_flight[hospital_id] = in_flight.get(hospital_id, 0) + 1
        return in_flight

    def _fallback_directives(self, state: Dict) -> List[Dict[str, Any]]:
        """
        Pure heuristic control loop.
        Order:
          a) post-SAR re-triage
          b) triage all other untriaged victims
          c) dispatch ambulances (same-step with planned tags)
          d) assign SAR teams
          e) assign fire teams
        """
        directives: List[Dict[str, Any]] = []
        planned_tags: Dict[int, str] = {}

        # a) POST-SAR RE-TRIAGE FIRST
        post_sar_victims = [
            victim for victim in state.get("victims", [])
            if victim.get("status") == "TRIAGED" and victim.get("assigned_tag") is None
        ]
        post_sar_victims.sort(key=lambda victim: victim.get("minutes_since_injury", 0), reverse=True)
        for victim in post_sar_victims:
            victim_id = self._to_int(victim.get("id"))
            if victim_id is None:
                continue
            tag_name = self._post_sar_tag(victim.get("minutes_since_injury", 0))
            planned_tags[victim_id] = tag_name
            directives.append(
                {
                    "type": "triage",
                    "victim_id": victim_id,
                    "tag": int(TriageTag[tag_name]),
                }
            )

        # b) TRIAGE ALL OTHER UNTRIAGED VICTIMS
        other_untriaged = [
            victim for victim in state.get("victims", [])
            if victim.get("status") in ("TRAPPED", "TRIAGED")
            and victim.get("assigned_tag") is None
            and self._to_int(victim.get("id")) not in planned_tags
        ]
        other_untriaged.sort(key=lambda victim: victim.get("minutes_since_injury", 0), reverse=True)
        for victim in other_untriaged:
            victim_id = self._to_int(victim.get("id"))
            if victim_id is None:
                continue
            tag_name = self._regular_tag(victim.get("minutes_since_injury", 0))
            planned_tags[victim_id] = tag_name
            directives.append(
                {
                    "type": "triage",
                    "victim_id": victim_id,
                    "tag": int(TriageTag[tag_name]),
                }
            )

        # c) DISPATCH AMBULANCES (same-step with planned_tags)
        dispatch_directives, dispatched_victim_ids = self._dispatch_ambulances(state, planned_tags)
        directives.extend(dispatch_directives)

        # d) ASSIGN SAR TEAMS
        sar_directives, sar_target_ids = self._assign_sar_teams(state, dispatched_victim_ids)
        directives.extend(sar_directives)

        # e) ASSIGN FIRE TEAMS
        directives.extend(
            self._assign_fire_teams(
                state=state,
                sar_target_ids=sar_target_ids,
                dispatched_victim_ids=dispatched_victim_ids,
            )
        )

        return directives

    def _dispatch_ambulances(
        self,
        state: Dict[str, Any],
        planned_tags: Optional[Dict[int, str]] = None,
    ) -> Tuple[List[Dict[str, Any]], Set[int]]:
        directives: List[Dict[str, Any]] = []
        dispatched_victim_ids: Set[int] = set()
        planned_tags = planned_tags or {}

        free_ambulances = [
            team
            for team in state.get("teams", [])
            if team.get("type") == "AMBULANCE"
            and bool(team.get("is_free"))
            and team.get("transport_victim") is None
        ]
        free_ambulances.sort(key=lambda team: team.get("id", 0))
        if not free_ambulances:
            return directives, dispatched_victim_ids

        candidates: List[Tuple[Dict[str, Any], str]] = []
        for victim in state.get("victims", []):
            status = victim.get("status")
            if status in ("IN_TRANSIT", "AT_HOSPITAL", "DECEASED"):
                continue

            victim_id = self._to_int(victim.get("id"))
            if victim_id is None:
                continue

            tag_name = self._normalize_tag(victim.get("assigned_tag"))
            if tag_name is None:
                tag_name = planned_tags.get(victim_id)
            if tag_name not in ("RED", "YELLOW", "GREEN"):
                continue

            candidates.append((victim, tag_name))

        if not candidates:
            return directives, dispatched_victim_ids

        hospitals = [hospital for hospital in state.get("hospitals", []) if hospital.get("is_accepting")]
        if not hospitals:
            return directives, dispatched_victim_ids

        road_blocked = bool(state.get("road_blocked", False))
        collapse_fired = float(state.get("secondary_collapse_risk", 0.0)) >= 1.0
        in_flight_by_hospital = self._count_in_flight_by_hospital(state)

        # Throughput-focused dispatch:
        # prioritize RED/YELLOW while minimizing trip distance to keep ambulances cycling.
        priority_penalty = {"RED": 0.0, "YELLOW": 10.0, "GREEN": 20.0}
        if collapse_fired:
            # Secondary collapse can abruptly worsen YELLOW victims.
            priority_penalty["YELLOW"] = 6.0

        for ambulance in free_ambulances:
            best_choice = None

            for victim, tag_name in candidates:
                victim_id = self._to_int(victim.get("id"))
                if victim_id is None or victim_id in dispatched_victim_ids:
                    continue

                best_hospital_id = None
                best_trip_distance = None
                for hospital in hospitals:
                    hospital_id = self._to_int(hospital.get("id"))
                    if hospital_id is None:
                        continue

                    projected_beds = hospital.get("available_beds", 0) - in_flight_by_hospital.get(hospital_id, 0)
                    if projected_beds <= 0:
                        continue

                    trip_distance = self._estimate_trip_time(
                        team=ambulance,
                        victim=victim,
                        hospital=hospital,
                        road_blocked=road_blocked,
                    )
                    if best_trip_distance is None or trip_distance < best_trip_distance:
                        best_trip_distance = trip_distance
                        best_hospital_id = hospital_id

                if best_hospital_id is None or best_trip_distance is None:
                    continue

                urgency = victim.get("minutes_since_injury", 0)
                score = best_trip_distance + priority_penalty[tag_name] - (0.1 * urgency)
                if best_choice is None or score < best_choice[0]:
                    best_choice = (score, victim_id, best_hospital_id)

            if best_choice is None:
                continue

            _, victim_id, hospital_id = best_choice
            directives.append(
                {
                    "type": "dispatch",
                    "team_id": ambulance.get("id"),
                    "victim_id": victim_id,
                    "hospital_id": hospital_id,
                }
            )
            dispatched_victim_ids.add(victim_id)
            in_flight_by_hospital[hospital_id] = in_flight_by_hospital.get(hospital_id, 0) + 1

        return directives, dispatched_victim_ids

    def _assign_sar_teams(
        self,
        state: Dict[str, Any],
        dispatched_victim_ids: Optional[Set[int]] = None,
    ) -> Tuple[List[Dict[str, Any]], Set[int]]:
        directives: List[Dict[str, Any]] = []
        sar_target_ids: Set[int] = set()
        dispatched_victim_ids = dispatched_victim_ids or set()

        free_sar = [
            team
            for team in state.get("teams", [])
            if team.get("type") == "SEARCH_RESCUE" and bool(team.get("is_free"))
        ]
        free_sar.sort(key=lambda team: team.get("id", 0))

        trapped_victims = [
            victim
            for victim in state.get("victims", [])
            if victim.get("status") == "TRAPPED"
            and self._to_int(victim.get("id")) not in dispatched_victim_ids
        ]
        trapped_victims.sort(key=lambda victim: victim.get("minutes_since_injury", 0), reverse=True)

        for team, victim in zip(free_sar, trapped_victims):
            victim_id = self._to_int(victim.get("id"))
            if victim_id is None:
                continue
            directives.append(
                {
                    "type": "assign_sar",
                    "team_id": team.get("id"),
                    "victim_id": victim_id,
                }
            )
            sar_target_ids.add(victim_id)

        return directives, sar_target_ids

    def _assign_fire_teams(
        self,
        state: Dict[str, Any],
        sar_target_ids: Optional[Set[int]] = None,
        dispatched_victim_ids: Optional[Set[int]] = None,
    ) -> List[Dict[str, Any]]:
        directives: List[Dict[str, Any]] = []
        sar_target_ids = sar_target_ids or set()
        dispatched_victim_ids = dispatched_victim_ids or set()

        free_fire = [
            team
            for team in state.get("teams", [])
            if team.get("type") == "FIRE" and bool(team.get("is_free"))
        ]
        free_fire.sort(key=lambda team: team.get("id", 0))
        if not free_fire:
            return directives

        targets = [
            victim
            for victim in state.get("victims", [])
            if victim.get("status") == "TRAPPED"
            and self._to_int(victim.get("id")) not in sar_target_ids
            and self._to_int(victim.get("id")) not in dispatched_victim_ids
        ]
        targets.sort(key=lambda victim: victim.get("minutes_since_injury", 0), reverse=True)
        if not targets:
            return directives

        # Assign every free fire team to trapped victims (reusing targets if needed).
        for index, team in enumerate(free_fire):
            victim = targets[index % len(targets)]
            victim_id = self._to_int(victim.get("id"))
            if victim_id is None:
                continue
            directives.append(
                {
                    "type": "assign_fire",
                    "team_id": team.get("id"),
                    "victim_id": victim_id,
                }
            )
        return directives


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

