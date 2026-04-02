"""
================================================
 OpenEnv: Urban Mass Casualty Incident Command
 Problem #15 â€” Live Incident Command Agent
================================================

A building collapse with 200+ casualties.
The agent plays incident commander in real time:

  - Triages victims via START protocol
  - Dispatches ambulances to capacity-aware hospitals
  - Coordinates fire / search-and-rescue teams
  - Manages a secondary collapse risk clock
  - All under exponential time-decay rewards

One env step == 60 seconds of simulated time.
The "golden hour" is 60 steps.  After that,
survival probability decays exponentially.

Three graded tasks
  Task 1 (Easy)   â€” 40 victims, 8 amb, 2 hospitals, no secondary collapse
  Task 2 (Medium) â€” 120 victims, 5 amb, 3 hospitals (one near capacity),
                    secondary collapse at t=30
  Task 3 (Hard)   â€” 240 victims, 4 amb, 4 hospitals (mixed capacity/distance),
                    secondary collapse at t=20, media incident at t=15,
                    road blockage at t=10
"""

from __future__ import annotations

import copy
import random
import math
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

# Pydantic imports for OpenEnv compliance
from pydantic import BaseModel, Field, validator


def _print_safe(text: str):
    """Print text safely even on narrow Windows terminal encodings."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe_text)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Enumerations
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TriageTag(IntEnum):
    """START triage categories."""
    BLACK  = 0   # Deceased / expectant
    RED    = 1   # Immediate â€” life-threatening, salvageable
    YELLOW = 2   # Delayed â€” serious but stable
    GREEN  = 3   # Minor â€” walking wounded


class TeamType(IntEnum):
    AMBULANCE       = 0
    FIRE            = 1
    SEARCH_RESCUE   = 2


class VictimStatus(IntEnum):
    TRAPPED    = 0   # Not yet reached
    TRIAGED    = 1   # Assessed, awaiting transport
    IN_TRANSIT = 2   # In an ambulance
    AT_HOSPITAL= 3   # Delivered
    DECEASED   = 4


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Pydantic Models (OpenEnv Specification)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class VictimObservation(BaseModel):
    """Observable victim state (without true_tag)."""
    id: int
    status: str
    assigned_tag: Optional[str] = None
    location: Tuple[float, float]
    minutes_since_injury: float


class HospitalObservation(BaseModel):
    """Hospital state."""
    id: int
    name: str
    total_capacity: int
    available_beds: int
    is_accepting: bool
    location: Tuple[float, float]
    trauma_level: int
    travel_time_minutes: int


class TeamObservation(BaseModel):
    """Resource team state."""
    id: int
    type: str
    location: Tuple[float, float]
    is_free: bool
    assigned_victim: Optional[int] = None
    transport_victim: Optional[int] = None
    free_at_step: int


class SummaryObservation(BaseModel):
    """Summary statistics."""
    total_victims: int
    trapped: int
    triaged: int
    in_transit: int
    at_hospital: int
    deceased: int


class EnvironmentState(BaseModel):
    """Full observable state."""
    step: int
    golden_hour_remaining: int
    secondary_collapse_risk: float
    road_blocked: bool
    media_pressure: bool
    victims: List[VictimObservation]
    hospitals: List[HospitalObservation]
    teams: List[TeamObservation]
    summary: SummaryObservation


class TriageDirective(BaseModel):
    """Triage action directive."""
    type: str = "triage"
    victim_id: int
    tag: int  # TriageTag value


class DispatchDirective(BaseModel):
    """Ambulance dispatch directive."""
    type: str = "dispatch"
    team_id: int
    victim_id: int
    hospital_id: int


class SARDirective(BaseModel):
    """Search and rescue assignment."""
    type: str = "assign_sar"
    team_id: int
    victim_id: int


class FireDirective(BaseModel):
    """Fire team assignment."""
    type: str = "assign_fire"
    team_id: int
    victim_id: int


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Data classes
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class Victim:
    id: int
    true_tag: TriageTag           # Ground truth (hidden from agent)
    assigned_tag: Optional[TriageTag] = None  # What agent has declared
    status: VictimStatus = VictimStatus.TRAPPED
    location: Tuple[float, float] = (0.0, 0.0)
    minutes_since_injury: float = 0.0
    survived: Optional[bool] = None

    @property
    def is_reachable(self) -> bool:
        return self.status == VictimStatus.TRAPPED

    @property
    def is_triaged(self) -> bool:
        return self.assigned_tag is not None

    def deteriorate(self, minutes: float):
        """RED victims deteriorate faster. Chance of dying increases with time."""
        self.minutes_since_injury += minutes
        if self.true_tag == TriageTag.RED:
            # After 60 min, 80% mortality if not at hospital
            p_death = 1 - math.exp(-self.minutes_since_injury / 75.0)
        elif self.true_tag == TriageTag.YELLOW:
            p_death = 1 - math.exp(-self.minutes_since_injury / 180.0)
        else:
            p_death = 0.0

        if self.status not in (VictimStatus.AT_HOSPITAL, VictimStatus.DECEASED):
            if random.random() < p_death * 0.02:   # per-step chance
                self.status = VictimStatus.DECEASED
                self.survived = False


@dataclass
class Hospital:
    id: int
    name: str
    location: Tuple[float, float]
    total_capacity: int
    current_occupancy: int = 0
    trauma_level: int = 1          # 1 = highest capability

    @property
    def available_beds(self) -> int:
        return max(0, self.total_capacity - self.current_occupancy)

    @property
    def is_accepting(self) -> bool:
        return self.available_beds > 0


@dataclass
class ResourceTeam:
    id: int
    team_type: TeamType
    location: Tuple[float, float]
    assigned_victim_id: Optional[int] = None
    assigned_hospital_id: Optional[int] = None
    busy_until_step: int = 0       # step when team becomes free again
    transport_victim_id: Optional[int] = None  # victim currently in ambulance


@dataclass
class IncidentEvent:
    step: int
    event_type: str   # 'secondary_collapse', 'road_blockage', 'media_incident'
    description: str
    applied: bool = False


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Action space
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class IncidentAction:
    """
    The agent submits a list of directives each step.

    Each directive is a dict with keys:
      {
        'type': 'triage'    | 'dispatch' | 'assign_sar' | 'assign_fire',
        # triage:
        'victim_id': int,
        'tag': TriageTag,
        # dispatch (ambulance â†’ hospital):
        'team_id': int,
        'victim_id': int,
        'hospital_id': int,
        # assign_sar / assign_fire:
        'team_id': int,
        'victim_id': int,
      }
    """
    directives: List[Dict]


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Grading Function (OpenEnv Requirement)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def grade(env: UrbanMCIEnv) -> float:
    """
    Grade the environment performance.
    
    Returns a normalized score between 0.0 and 1.0.
    
    Score = (lives_saved) / (total_victims - expectant_black)
    
    Where:
    - lives_saved = victims who survived (reached hospital alive)
    - total_victims = all victims in the scenario
    - expectant_black = victims who were BLACK (expectant) from start
    
    This normalizes the score to 0-1 range as required by OpenEnv.
    """
    total = len(env._victims)
    
    # Count non-expectant victims (RED, YELLOW, GREEN)
    saveable = sum(
        1 for v in env._victims 
        if v.true_tag != TriageTag.BLACK
    )
    
    # Count lives saved (survived = True)
    saved = sum(
        1 for v in env._victims 
        if v.survived is True
    )
    
    if saveable == 0:
        return 1.0 if saved == 0 else 0.0
    
    return saved / saveable


def grade_task(env: UrbanMCIEnv, task: int) -> float:
    """
    Grade for a specific task with task-specific normalization.
    
    Task 1: 40 victims (32 saveable) - baseline 20 saves = 0.625
    Task 2: 120 victims (114 saveable) - baseline 50 saves = 0.439
    Task 3: 240 victims (228 saveable) - baseline 80 saves = 0.351
    """
    return grade(env)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Core Environment
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class UrbanMCIEnv:
    """
    Urban Mass Casualty Incident Command environment.

    step(action) -> (state, reward, done, info)
    reset(task)  -> state
    state()      -> dict (current observable state)
    """

    GOLDEN_HOUR_STEPS = 60   # 60 steps Ã— 1 min/step = 60 minutes
    MAX_STEPS         = 120

    def __init__(self, task: int = 1):
        self.task = task
        self._validate_task(task)
        self._current_step = 0
        self._victims: List[Victim] = []
        self._hospitals: List[Hospital] = []
        self._teams: List[ResourceTeam] = []
        self._events: List[IncidentEvent] = []
        self._reward_log: List[float] = []
        self._incident_log: List[str] = []
        self._secondary_collapse_occurred = False
        self._road_blocked = False
        self._media_incident = False
        self._done = False

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Public API
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def reset(self, task: Optional[int] = None) -> Dict:
        """Reset environment to initial state for given task."""
        if task is not None:
            self.task = task
            self._validate_task(task)

        self._current_step = 0
        self._reward_log = []
        self._incident_log = []
        self._secondary_collapse_occurred = False
        self._road_blocked = False
        self._media_incident = False
        self._done = False

        cfg = self._task_config()
        self._victims   = self._generate_victims(cfg)
        self._hospitals = self._generate_hospitals(cfg)
        self._teams     = self._generate_teams(cfg)
        self._events    = self._generate_events(cfg)

        self._log(f"[t=0] Incident declared. {len(self._victims)} casualties reported. "
                  f"{cfg['ambulances']} ambulances, {cfg['sar_teams']} SAR teams, "
                  f"{cfg['fire_teams']} fire teams available.")

        return self.state()

    def step(self, action: IncidentAction) -> Tuple[Dict, float, bool, Dict]:
        """
        Advance one simulated minute.

        Returns
        -------
        state  : dict
        reward : float
        done   : bool
        info   : dict
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self._current_step += 1

        # 1. Apply triggered events
        self._apply_events()

        # 2. Process agent directives
        step_reward = 0.0
        step_reward += self._process_directives(action.directives)

        # 3. Advance in-transit victims â†’ hospitals
        step_reward += self._advance_transports()

        # 4. Deteriorate victims still on scene
        self._deteriorate_victims()

        # 5. Compute base survival reward for newly delivered victims
        #    (counted in _advance_transports)

        # 6. Shape reward with time pressure
        shaped = self._apply_time_shaping(step_reward)
        self._reward_log.append(shaped)

        # 7. Check terminal conditions
        done = self._check_done()

        info = {
            "step": self._current_step,
            "lives_saved": self._lives_saved(),
            "lives_lost": self._lives_lost(),
            "still_trapped": self._count_trapped(),
            "still_triaged_waiting": self._count_triaged_waiting(),
            "in_transit": self._count_in_transit(),
            "ambulances_free": self._count_free_ambulances(),
            "log": self._incident_log[-5:],
            "total_reward_so_far": sum(self._reward_log),
            "grade": grade(self),  # Include current grade in info
        }

        return self.state(), shaped, done, info

    def state(self) -> Dict:
        """
        Returns the full observable state as a structured dict.

        In a real deployment this would be partially observable
        (trapped victims are not fully known until triaged).
        """
        return {
            "step": self._current_step,
            "golden_hour_remaining": max(0, self.GOLDEN_HOUR_STEPS - self._current_step),
            "secondary_collapse_risk": self._secondary_collapse_risk(),
            "road_blocked": self._road_blocked,
            "media_pressure": self._media_incident,

            "victims": [
                {
                    "id": v.id,
                    "status": v.status.name,
                    "assigned_tag": v.assigned_tag.name if v.assigned_tag else None,
                    # Agent cannot see true_tag â€” must triage to discover
                    "location": v.location,
                    "minutes_since_injury": v.minutes_since_injury,
                }
                for v in self._victims
            ],

            "hospitals": [
                {
                    "id": h.id,
                    "name": h.name,
                    "total_capacity": h.total_capacity,
                    "available_beds": h.available_beds,
                    "is_accepting": h.is_accepting,
                    "location": h.location,
                    "trauma_level": h.trauma_level,
                    "travel_time_minutes": self._hospital_travel_time(h),
                }
                for h in self._hospitals
            ],

            "teams": [
                {
                    "id": t.id,
                    "type": t.team_type.name,
                    "location": t.location,
                    "is_free": self._current_step >= t.busy_until_step,
                    "assigned_victim": t.assigned_victim_id,
                    "transport_victim": t.transport_victim_id,
                    "free_at_step": t.busy_until_step,
                }
                for t in self._teams
            ],

            "summary": {
                "total_victims": len(self._victims),
                "trapped":   self._count_trapped(),
                "triaged":   self._count_triaged_waiting(),
                "in_transit": self._count_in_transit(),
                "at_hospital": self._count_at_hospital(),
                "deceased":  self._count_deceased(),
            }
        }

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Reward components
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _apply_time_shaping(self, raw_reward: float) -> float:
        """
        Survival reward decays exponentially after golden hour.
        Reward inside golden hour is full value.
        """
        if self._current_step <= self.GOLDEN_HOUR_STEPS:
            return raw_reward
        overtime = self._current_step - self.GOLDEN_HOUR_STEPS
        decay = math.exp(-overtime / 30.0)
        return raw_reward * decay

    def _process_directives(self, directives: List[Dict]) -> float:
        reward = 0.0
        for d in directives:
            dtype = d.get("type")
            if dtype == "triage":
                reward += self._do_triage(d)
            elif dtype == "dispatch":
                reward += self._do_dispatch(d)
            elif dtype == "assign_sar":
                reward += self._do_assign_sar(d)
            elif dtype == "assign_fire":
                reward += self._do_assign_fire(d)
        return reward

    def _do_triage(self, d: Dict) -> float:
        """
        Triage a victim with START protocol.
        Correct tagging = +1.0 for RED, +0.5 for others.
        Wrong tag for RED (missed critical) = -2.0.
        """
        victim = self._get_victim(d.get("victim_id"))
        if victim is None:
            return -0.1   # invalid action penalty
        # Can triage TRAPPED or TRIAGED (if SAR just extracted them)
        if victim.status not in (VictimStatus.TRAPPED, VictimStatus.TRIAGED):
            return -0.1   # already in transit or at hospital
        if victim.assigned_tag is not None:
            return 0.0    # already triaged

        assigned = TriageTag(d.get("tag", TriageTag.GREEN))
        victim.assigned_tag = assigned

        # Reward shaping
        if victim.true_tag == TriageTag.RED and assigned == TriageTag.RED:
            r = 1.0
        elif victim.true_tag == TriageTag.RED and assigned != TriageTag.RED:
            r = -2.0   # missed critical â€” major penalty
            self._log(f"[t={self._current_step}] âš  WRONG TRIAGE: victim {victim.id} is RED, tagged {assigned.name}")
        elif victim.true_tag == assigned:
            r = 0.5
        else:
            r = -0.3   # minor mistag
        return r

    def _do_dispatch(self, d: Dict) -> float:
        """
        Assign a free ambulance to pick up a triaged victim and
        deliver to specified hospital.
        """
        team   = self._get_team(d.get("team_id"), TeamType.AMBULANCE)
        victim = self._get_victim(d.get("victim_id"))
        hosp   = self._get_hospital(d.get("hospital_id"))

        if team is None or victim is None or hosp is None:
            return -0.1
        if self._current_step < team.busy_until_step:
            return -0.2   # team not free yet
        if victim.status not in (VictimStatus.TRAPPED, VictimStatus.TRIAGED):
            return -0.1
        if not hosp.is_accepting:
            return -0.5   # sent to full hospital
        if victim.assigned_tag is None:
            return -0.3   # dispatched without triage

        travel = self._ambulance_travel_time(team, victim, hosp)
        if self._road_blocked:
            travel = int(travel * 1.5)   # road blockage slows response

        team.assigned_victim_id = victim.id
        team.transport_victim_id = victim.id
        team.assigned_hospital_id = hosp.id
        team.busy_until_step = self._current_step + travel
        victim.status = VictimStatus.IN_TRANSIT

        # Reward for correct hospital choice (trauma level matching severity)
        tag = victim.assigned_tag
        if tag == TriageTag.RED and hosp.trauma_level == 1:
            dispatch_bonus = 0.5
        elif tag == TriageTag.RED and hosp.trauma_level > 1:
            dispatch_bonus = -0.3   # sent critical to lower-tier hospital
        else:
            dispatch_bonus = 0.2

        self._log(f"[t={self._current_step}] Ambulance {team.id} dispatched: "
                  f"victim {victim.id} â†’ {hosp.name} (ETA {travel} min)")
        return 0.3 + dispatch_bonus

    def _do_assign_sar(self, d: Dict) -> float:
        """Assign SAR team to reach and extract a trapped victim."""
        team   = self._get_team(d.get("team_id"), TeamType.SEARCH_RESCUE)
        victim = self._get_victim(d.get("victim_id"))

        if team is None or victim is None:
            return -0.1
        if self._current_step < team.busy_until_step:
            return -0.2
        if victim.status != VictimStatus.TRAPPED:
            return 0.0

        # Extraction time: 5-15 min depending on debris severity (random)
        extract_time = random.randint(5, 15)
        if self._secondary_collapse_occurred:
            extract_time += 10   # harder after secondary collapse

        team.assigned_victim_id = victim.id
        team.busy_until_step = self._current_step + extract_time
        
        # Mark victim as awaiting extraction - will be triaged when SAR arrives
        # Note: victim stays TRAPPED until extraction complete
        self._log(f"[t={self._current_step}] SAR team {team.id} assigned to victim {victim.id} "
                  f"(extraction ~{extract_time} min)")
        return 0.2

    def _do_assign_fire(self, d: Dict) -> float:
        """Assign fire team to suppress hazard near victim location."""
        team   = self._get_team(d.get("team_id"), TeamType.FIRE)
        victim = self._get_victim(d.get("victim_id"))

        if team is None or victim is None:
            return -0.1
        if self._current_step < team.busy_until_step:
            return -0.2

        team.assigned_victim_id = victim.id
        team.busy_until_step = self._current_step + random.randint(8, 20)
        self._log(f"[t={self._current_step}] Fire team {team.id} securing area near victim {victim.id}")
        return 0.15

    def _advance_transports(self) -> float:
        """Move in-transit victims to hospital when travel time elapsed."""
        reward = 0.0
        
        # First, check for completed SAR operations
        for team in self._teams:
            if (team.team_type == TeamType.SEARCH_RESCUE
                    and team.assigned_victim_id is not None
                    and self._current_step >= team.busy_until_step):
                
                victim = self._get_victim(team.assigned_victim_id)
                if victim and victim.status == VictimStatus.TRAPPED:
                    # SAR completed - victim is now accessible for triage
                    victim.status = VictimStatus.TRIAGED
                    # Keep tag unset so agent can triage after extraction.
                    victim.assigned_tag = None
                    self._log(f"[t={self._current_step}] SAR completed: victim {victim.id} "
                              f"now accessible for triage")
                
                # Free the SAR team
                team.assigned_victim_id = None
        
        # Then, check for ambulance arrivals
        for team in self._teams:
            if (team.team_type == TeamType.AMBULANCE
                    and team.transport_victim_id is not None
                    and self._current_step >= team.busy_until_step):

                victim = self._get_victim(team.transport_victim_id)
                hosp   = self._get_hospital(team.assigned_hospital_id)

                if victim and hosp and victim.status == VictimStatus.IN_TRANSIT:
                    victim.status = VictimStatus.AT_HOSPITAL
                    hosp.current_occupancy += 1

                    # Survival reward based on tag and time
                    survival_r = self._survival_reward(victim)
                    victim.survived = survival_r > 0
                    reward += survival_r

                    self._log(f"[t={self._current_step}] âœ“ Victim {victim.id} "
                              f"({victim.true_tag.name}) delivered to {hosp.name}. "
                              f"Reward: {survival_r:.2f}")

                    # Free the ambulance
                    team.transport_victim_id = None
                    team.assigned_victim_id  = None
                    team.assigned_hospital_id = None

        return reward

    def _survival_reward(self, victim: Victim) -> float:
        """
        Reward for delivering a victim to hospital.
        RED: high reward, heavily time-penalized.
        YELLOW: moderate reward.
        GREEN/BLACK: minimal.
        """
        t = victim.minutes_since_injury
        if victim.true_tag == TriageTag.RED:
            # Full reward if within 60 min, decays after
            return 10.0 * math.exp(-max(0, t - 60) / 30.0)
        elif victim.true_tag == TriageTag.YELLOW:
            return 4.0 * math.exp(-max(0, t - 120) / 60.0)
        elif victim.true_tag == TriageTag.GREEN:
            return 1.0
        else:   # BLACK
            return -1.0   # wasted resource on expectant

    def _deteriorate_victims(self):
        for v in self._victims:
            if v.status in (VictimStatus.TRAPPED, VictimStatus.TRIAGED):
                v.deteriorate(1.0)   # 1 minute per step

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Events
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _apply_events(self):
        for evt in self._events:
            if evt.applied or evt.step != self._current_step:
                continue
            evt.applied = True

            if evt.event_type == "secondary_collapse":
                self._secondary_collapse_occurred = True
                # Trap 20% more victims under debris
                new_trapped = [v for v in self._victims
                               if v.status == VictimStatus.TRAPPED]
                count = max(1, len(new_trapped) // 5)
                for v in random.sample(new_trapped, min(count, len(new_trapped))):
                    v.minutes_since_injury += 10   # time penalty
                self._log(f"[t={self._current_step}] ðŸš¨ SECONDARY COLLAPSE â€” "
                          f"debris shifted, {count} victims harder to reach. "
                          f"SAR operations +10 min per extraction.")

            elif evt.event_type == "road_blockage":
                self._road_blocked = True
                self._log(f"[t={self._current_step}] ðŸš§ ROAD BLOCKAGE â€” "
                          f"main access route blocked. Ambulance travel +50%.")

            elif evt.event_type == "media_incident":
                self._media_incident = True
                self._log(f"[t={self._current_step}] ðŸ“º MEDIA PRESSURE â€” "
                          f"press on scene. Command decisions under scrutiny.")

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Task configurations
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _task_config(self) -> Dict:
        configs = {
            1: dict(
                name="Task 1 â€” Easy: Day drill",
                victims=40,
                red_fraction=0.20,
                yellow_fraction=0.35,
                ambulances=8,
                sar_teams=4,
                fire_teams=2,
                hospitals=2,
                events=[],
                description=(
                    "40 victims, 8 ambulances, 2 hospitals (both accepting), "
                    "no secondary collapse. Learn basic triage-dispatch loop."
                )
            ),
            2: dict(
                name="Task 2 â€” Medium: Partial collapse",
                victims=120,
                red_fraction=0.25,
                yellow_fraction=0.40,
                ambulances=5,
                sar_teams=3,
                fire_teams=3,
                hospitals=3,
                events=[
                    {"step": 30, "type": "secondary_collapse",
                     "desc": "North wing collapses further"},
                ],
                description=(
                    "120 victims, scarce ambulances, one hospital near capacity, "
                    "secondary collapse at t=30. Must prioritize RED victims correctly."
                )
            ),
            3: dict(
                name="Task 3 â€” Hard: Full urban MCI",
                victims=240,
                red_fraction=0.30,
                yellow_fraction=0.40,
                ambulances=4,
                sar_teams=4,
                fire_teams=4,
                hospitals=4,
                events=[
                    {"step": 10, "type": "road_blockage",
                     "desc": "Main road debris blockage"},
                    {"step": 15, "type": "media_incident",
                     "desc": "Media arrives on scene"},
                    {"step": 20, "type": "secondary_collapse",
                     "desc": "Structural failure in east wing"},
                ],
                description=(
                    "240 victims, minimal resources, 3 cascading events "
                    "(road blockage t=10, media t=15, secondary collapse t=20). "
                    "Hospitals have mixed capacity and trauma levels. "
                    "This is the full real-world scenario."
                )
            ),
        }
        return configs[self.task]

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Generators
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _generate_victims(self, cfg: Dict) -> List[Victim]:
        n = cfg["victims"]
        red_n    = int(n * cfg["red_fraction"])
        yellow_n = int(n * cfg["yellow_fraction"])
        green_n  = n - red_n - yellow_n - max(0, int(n * 0.05))
        black_n  = n - red_n - yellow_n - green_n

        tags = (
            [TriageTag.RED]    * red_n    +
            [TriageTag.YELLOW] * yellow_n +
            [TriageTag.GREEN]  * green_n  +
            [TriageTag.BLACK]  * black_n
        )
        random.shuffle(tags)

        victims = []
        for i, tag in enumerate(tags):
            x = random.uniform(0, 200)
            y = random.uniform(0, 200)
            # RED victims have more initial injury time (already been down longer)
            init_min = random.uniform(0, 15) if tag == TriageTag.RED else random.uniform(0, 5)
            victims.append(Victim(
                id=i,
                true_tag=tag,
                location=(x, y),
                minutes_since_injury=init_min,
            ))
        return victims

    def _generate_hospitals(self, cfg: Dict) -> List[Hospital]:
        hospital_templates = [
            Hospital(0, "City Trauma Centre",    (300, 100), 80, trauma_level=1),
            Hospital(1, "St Mary's Hospital",    (50,  300), 60, trauma_level=2),
            Hospital(2, "North General",         (400, 350), 40, trauma_level=2),
            Hospital(3, "Eastside Medical",      (500, 50),  30, trauma_level=3),
        ]
        hospitals = hospital_templates[: cfg["hospitals"]]

        # Task 2: one hospital near capacity
        if self.task == 2:
            hospitals[1].current_occupancy = hospitals[1].total_capacity - 5

        # Task 3: mixed capacity
        if self.task == 3:
            hospitals[2].current_occupancy = hospitals[2].total_capacity - 8
            hospitals[3].current_occupancy = hospitals[3].total_capacity - 4

        return hospitals

    def _generate_teams(self, cfg: Dict) -> List[ResourceTeam]:
        teams = []
        tid = 0
        base = (100.0, 100.0)   # incident command post

        for _ in range(cfg["ambulances"]):
            teams.append(ResourceTeam(tid, TeamType.AMBULANCE, base))
            tid += 1
        for _ in range(cfg["sar_teams"]):
            teams.append(ResourceTeam(tid, TeamType.SEARCH_RESCUE, base))
            tid += 1
        for _ in range(cfg["fire_teams"]):
            teams.append(ResourceTeam(tid, TeamType.FIRE, base))
            tid += 1

        return teams

    def _generate_events(self, cfg: Dict) -> List[IncidentEvent]:
        events = []
        for e in cfg.get("events", []):
            events.append(IncidentEvent(
                step=e["step"],
                event_type=e["type"],
                description=e["desc"],
            ))
        return events

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Helpers
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _get_victim(self, vid) -> Optional[Victim]:
        if vid is None:
            return None
        for v in self._victims:
            if v.id == vid:
                return v
        return None

    def _get_hospital(self, hid) -> Optional[Hospital]:
        if hid is None:
            return None
        for h in self._hospitals:
            if h.id == hid:
                return h
        return None

    def _get_team(self, tid, expected_type: TeamType) -> Optional[ResourceTeam]:
        if tid is None:
            return None
        for t in self._teams:
            if t.id == tid and t.team_type == expected_type:
                return t
        return None

    def _dist(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def _ambulance_travel_time(
        self,
        team: ResourceTeam,
        victim: Victim,
        hosp: Hospital,
    ) -> int:
        """Round trip estimate in minutes (pickup + delivery)."""
        d_pickup  = self._dist(team.location, victim.location)
        d_deliver = self._dist(victim.location, hosp.location)
        # Tuned for simulation scale so ambulances can complete multiple trips per episode.
        speed = 30.0
        return max(3, int((d_pickup + d_deliver) / speed))

    def _hospital_travel_time(self, hosp: Hospital) -> int:
        incident_centre = (100.0, 100.0)
        d = self._dist(incident_centre, hosp.location)
        return max(3, int(d / 3.0))

    def _secondary_collapse_risk(self) -> float:
        if self._secondary_collapse_occurred:
            return 1.0
        cfg = self._task_config()
        for e in self._events:
            if e.event_type == "secondary_collapse" and not e.applied:
                steps_away = e.step - self._current_step
                if steps_away >= 0:
                    return max(0.0, 1.0 - steps_away / 20.0)
        return 0.0

    def _count_trapped(self)   -> int:
        return sum(1 for v in self._victims if v.status == VictimStatus.TRAPPED)

    def _count_triaged_waiting(self) -> int:
        return sum(1 for v in self._victims if v.status == VictimStatus.TRIAGED)

    def _count_in_transit(self) -> int:
        return sum(1 for v in self._victims if v.status == VictimStatus.IN_TRANSIT)

    def _count_at_hospital(self) -> int:
        return sum(1 for v in self._victims if v.status == VictimStatus.AT_HOSPITAL)

    def _count_deceased(self) -> int:
        return sum(1 for v in self._victims if v.status == VictimStatus.DECEASED)

    def _count_free_ambulances(self) -> int:
        return sum(
            1 for t in self._teams
            if t.team_type == TeamType.AMBULANCE
            and self._current_step >= t.busy_until_step
        )

    def _lives_saved(self) -> int:
        return sum(
            1 for v in self._victims
            if v.survived is True
        )

    def _lives_lost(self) -> int:
        return sum(
            1 for v in self._victims
            if v.survived is False or v.status == VictimStatus.DECEASED
        )

    def _check_done(self) -> bool:
        if self._current_step >= self.MAX_STEPS:
            self._done = True
            return True
        all_resolved = all(
            v.status in (VictimStatus.AT_HOSPITAL, VictimStatus.DECEASED)
            for v in self._victims
        )
        if all_resolved:
            self._done = True
        return self._done

    def _log(self, msg: str):
        self._incident_log.append(msg)

    def _validate_task(self, task: int):
        if task not in (1, 2, 3):
            raise ValueError(f"task must be 1, 2, or 3 â€” got {task}")

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Rendering
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def render(self, mode="text"):
        s = self.state()
        print(f"\n{'='*60}")
        print(f" INCIDENT COMMAND  step={s['step']:03d}  "
              f"golden hour remaining: {s['golden_hour_remaining']} min")
        print(f"{'='*60}")
        sm = s["summary"]
        print(f"  Victims   | trapped={sm['trapped']:3d}  triaged={sm['triaged']:3d}  "
              f"transit={sm['in_transit']:3d}  hospital={sm['at_hospital']:3d}  "
              f"deceased={sm['deceased']:3d}")
        print(f"  Resources | ambulances free: {self._count_free_ambulances()}")
        print(f"  Hospitals |", end="")
        for h in s["hospitals"]:
            print(f" {h['name']}: {h['available_beds']} beds  ", end="")
        print()
        if s["secondary_collapse_risk"] > 0.5:
            print(f"  âš   SECONDARY COLLAPSE RISK: {s['secondary_collapse_risk']:.0%}")
        if s["road_blocked"]:
            print("  ðŸš§ Road blocked â€” travel time +50%")
        print(f"  Cumulative reward: {sum(self._reward_log):.2f}")
        print(f"  Current grade: {grade(self):.2f}")
        print(f"{'='*60}\n")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Reward Shaping Strategy (documented)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

REWARD_SHAPING = """
Reward Shaping Strategy
=======================

Component                     Value       Notes
---------------------------------------------------------------------
Triage RED correctly          +1.0        Identify critical victims
Triage non-RED correctly      +0.5
Triage RED as non-RED         -2.0        Missed critical = heavy penalty
Minor mistag                  -0.3

Dispatch (ambulance sent)     +0.3
  -> RED to L1 trauma          +0.5 bonus  Correct routing
  -> RED to L2+ hospital       -0.3        Under-resourced for severity
  -> any non-full hospital     +0.2

Victim delivered to hospital  +10.0 (RED)  Full reward inside golden hour
                              * exp(-max(0, t-60)/30)  decays after
                              +4.0 (YELLOW) decays after t=120
                              +1.0 (GREEN)
                              -1.0 (BLACK)  Wasted resource

SAR assignment                +0.2
Fire assignment               +0.15

Hospital full (sent anyway)   -0.5
Invalid action                -0.1 / -0.2

Time shaping (after golden hour):
  All rewards * exp(-overtime/30)
  where overtime = step - 60

This creates a dense, shaped signal that:
  1. Rewards correct triage classification
  2. Rewards matching victim severity to hospital capability
  3. Heavily penalizes delaying RED victims past golden hour
  4. Discourages wasting resources on BLACK (expectant) victims
  5. Penalizes invalid/wasteful actions softly
"""


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Example usage / smoke test
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_random_agent(task: int = 1, verbose: bool = True):
    """
    A random agent that makes legal moves each step.
    Useful for smoke-testing the environment.
    """
    env = UrbanMCIEnv(task=task)
    obs = env.reset()

    total_reward = 0.0
    for step_i in range(env.MAX_STEPS):
        directives = []

        # 1. Triage any untriaged on-scene victims (random tag assignment)
        for v in obs["victims"]:
            if v["status"] in ("TRAPPED", "TRIAGED") and v["assigned_tag"] is None:
                directives.append({
                    "type": "triage",
                    "victim_id": v["id"],
                    "tag": random.choice(list(TriageTag)),
                })

        # 2. Dispatch free ambulances to triaged victims
        free_ambs = [t for t in obs["teams"]
                     if t["type"] == "AMBULANCE" and t["is_free"] and t["transport_victim"] is None]
        triaged_victims = [
            v for v in obs["victims"]
            if v["status"] == "TRIAGED" and v["assigned_tag"] is not None
        ]
        accepting_hospitals = [h for h in obs["hospitals"] if h["is_accepting"]]

        for amb in free_ambs[:len(triaged_victims)]:
            if not triaged_victims or not accepting_hospitals:
                break
            victim = triaged_victims.pop(0)
            hosp   = random.choice(accepting_hospitals)
            directives.append({
                "type": "dispatch",
                "team_id": amb["id"],
                "victim_id": victim["id"],
                "hospital_id": hosp["id"],
            })

        # 3. Assign free SAR teams to trapped victims
        free_sar = [t for t in obs["teams"]
                    if t["type"] == "SEARCH_RESCUE" and t["is_free"]]
        still_trapped = [v for v in obs["victims"] if v["status"] == "TRAPPED"]
        for sar in free_sar[:len(still_trapped)]:
            if not still_trapped:
                break
            victim = random.choice(still_trapped)
            directives.append({
                "type": "assign_sar",
                "team_id": sar["id"],
                "victim_id": victim["id"],
            })

        action = IncidentAction(directives=directives)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if verbose and step_i % 10 == 0:
            env.render()

        if done:
            break

    print(f"\n{'='*60}")
    print(f"  EPISODE COMPLETE â€” Task {task}")
    print(f"  Total reward  : {total_reward:.2f}")
    print(f"  Lives saved   : {info['lives_saved']}")
    print(f"  Lives lost    : {info['lives_lost']}")
    print(f"  Steps taken   : {info['step']}")
    print(f"  Final grade   : {grade(env):.3f}")
    print(f"{'='*60}\n")
    _print_safe(REWARD_SHAPING)
    return total_reward, info


if __name__ == "__main__":
    print("Running smoke test: Task 1 (Easy)")
    run_random_agent(task=1, verbose=False)

    print("\nRunning smoke test: Task 2 (Medium)")
    run_random_agent(task=2, verbose=False)

    print("\nRunning smoke test: Task 3 (Hard)")
    run_random_agent(task=3, verbose=False)

