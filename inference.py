"""
================================================
 Inference Script for Urban MCI Environment
================================================

This script supports two modes:
1. Heuristic: Simple START triage protocol (default)
2. LLM: Uses OpenAI API for decision making (requires API_BASE_URL, MODEL_NAME, HF_TOKEN)

Grading:
- Scores normalized to 0.0-1.0
- Score = lives_saved / saveable_victims
"""

import os
import sys
import random
import json
from typing import List, Dict, Any, Optional

from urban_mci_env import (
    UrbanMCIEnv,
    IncidentAction,
    TriageTag,
    TeamType,
    VictimStatus,
    grade,
)


# Try to import OpenAI client
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


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
        """Generate action based on current state."""
        directives = []
        
        # 1. First, triage any untriaged trapped victims
        directives.extend(self._triage_victims(state))
        
        # 2. Dispatch ambulances to triaged victims
        directives.extend(self._dispatch_ambulances(state))
        
        # 3. Assign SAR teams to trapped victims
        directives.extend(self._assign_sar_teams(state))
        
        # 4. Assign fire teams to secure areas
        directives.extend(self._assign_fire_teams(state))
        
        return IncidentAction(directives=directives)
    
    def _triage_victims(self, state: Dict) -> List[Dict]:
        """Triage untriaged victims. Use simple heuristics."""
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
                
            directives.append({
                "type": "triage",
                "victim_id": victim["id"],
                "tag": tag,
            })
            
        return directives
    
    def _dispatch_ambulances(self, state: Dict) -> List[Dict]:
        """Dispatch free ambulances to triaged victims."""
        directives = []
        
        free_ambs = [
            t for t in state["teams"]
            if t["type"] == "AMBULANCE" 
            and t["is_free"] 
            and t["transport_victim"] is None
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
        """Assign SAR teams to trapped victims to enable triage."""
        directives = []
        
        free_sar = [
            t for t in state["teams"]
            if t["type"] == "SEARCH_RESCUE" and t["is_free"]
        ]
        
        if not free_sar:
            return directives
            
        trapped = [
            v for v in state["victims"]
            if v["status"] == "TRAPPED"
        ]
        
        for sar in free_sar[:len(trapped)]:
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
        """Assign fire teams (currently a placeholder)."""
        return []


class LLMAgent:
    """
    LLM-based agent that uses OpenAI API for decision making.
    
    Requires environment variables:
    - API_BASE_URL: The API endpoint
    - MODEL_NAME: The model identifier
    - HF_TOKEN: HuggingFace / API key
    """
    
    def __init__(self, env: UrbanMCIEnv):
        self.env = env
        
        # Get API configuration from environment
        self.api_base_url = os.environ.get("API_BASE_URL", "")
        self.model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        self.hf_token = os.environ.get("HF_TOKEN", "")
        
        if not self.api_base_url:
            raise ValueError("API_BASE_URL environment variable not set")
        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable not set")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.hf_token,
            base_url=self.api_base_url,
        )
        
    def act(self, state: Dict) -> IncidentAction:
        """Generate action using LLM."""
        # Build prompt from current state
        prompt = self._build_prompt(state)
        
        # Call LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
            )
            
            llm_response = response.choices[0].message.content
            
            # Parse LLM response into directives
            directives = self._parse_llm_response(llm_response, state)
            
        except Exception as e:
            print(f"LLM call failed: {e}, falling back to heuristic")
            # Fallback to heuristic
            heuristic = HeuristicAgent(self.env)
            return heuristic.act(state)
        
        return IncidentAction(directives=directives)
    
    def _get_system_prompt(self) -> str:
        """System prompt for the LLM."""
        return """You are an incident commander for a mass casualty incident.
        
TRIAGE CATEGORIES (START Protocol):
- RED (1): Immediate - life-threatening, requires immediate transport
- YELLOW (2): Delayed - serious but stable, can wait
- GREEN (3): Minor - walking wounded
- BLACK (0): Expectant - deceased or unsalvageable, do not transport

RESOURCE TYPES:
- Ambulances (type=0): Transport victims to hospitals
- SAR Teams (type=1): Extract trapped victims from debris
- Fire Teams (type=2): Secure hazard areas

HOSPITAL CONSIDERATIONS:
- Level 1 trauma centers (trauma_level=1) can handle RED victims
- Lower trauma levels should receive YELLOW/GREEN

RESPONSE FORMAT:
Return a JSON array of directives. Each directive must have:
- type: "triage" | "dispatch" | "assign_sar" | "assign_fire"
- For triage: victim_id, tag (0-3)
- For dispatch: team_id, victim_id, hospital_id
- For assign_sar: team_id, victim_id
- For assign_fire: team_id, victim_id

Example: [{"type": "triage", "victim_id": 0, "tag": 1}, {"type": "dispatch", "team_id": 0, "victim_id": 0, "hospital_id": 0}]

Prioritize:
1. Triage RED victims first
2. Dispatch RED victims to Level 1 trauma centers
3. Use SAR teams to reach trapped victims
4. Do not send ambulances to full hospitals
5. Do not dispatch without triage

Respond ONLY with the JSON array, no other text."""

    def _build_prompt(self, state: Dict) -> str:
        """Build prompt from current state."""
        prompt = f"""Current step: {state['step']}
Golden hour remaining: {state['golden_hour_remaining']} minutes
Secondary collapse risk: {state['secondary_collapse_risk']:.2f}
Road blocked: {state['road_blocked']}

SUMMARY:
- Total victims: {state['summary']['total_victims']}
- Trapped: {state['summary']['trapped']}
- Triaged (waiting): {state['summary']['triaged']}
- In transit: {state['summary']['in_transit']}
- At hospital: {state['summary']['at_hospital']}
- Deceased: {state['summary']['deceased']}

VICTIMS (id, status, assigned_tag, minutes_since_injury):
"""
        
        # Add victim info (limited to first 20 for context)
        for v in state['victims'][:20]:
            tag_str = v['assigned_tag'] if v['assigned_tag'] else 'UNTRIAGED'
            prompt += f"  Victim {v['id']}: {v['status']}, tag={tag_str}, {v['minutes_since_injury']:.1f}min\n"
        
        prompt += f"\nHOSPITALS (id, name, available_beds, trauma_level):\n"
        for h in state['hospitals']:
            prompt += f"  Hospital {h['id']}: {h['name']}, {h['available_beds']} beds, level {h['trauma_level']}\n"
        
        prompt += f"\nTEAMS (id, type, is_free):\n"
        for t in state['teams']:
            prompt += f"  Team {t['id']}: {t['type']}, free={t['is_free']}\n"
        
        prompt += "\nWhat directives should be issued this step? (JSON array)"
        
        return prompt
    
    def _parse_llm_response(self, response: str, state: Dict) -> List[Dict]:
        """Parse LLM response into directives."""
        try:
            # Try to extract JSON from response
            # Handle potential markdown code blocks
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            elif response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            directives = json.loads(response.strip())
            
            # Validate directives
            valid_directives = []
            for d in directives:
                if not isinstance(d, dict):
                    continue
                if 'type' not in d:
                    continue
                # Basic validation
                if d['type'] == 'triage' and 'victim_id' in d and 'tag' in d:
                    valid_directives.append(d)
                elif d['type'] == 'dispatch' and all(k in d for k in ['team_id', 'victim_id', 'hospital_id']):
                    valid_directives.append(d)
                elif d['type'] == 'assign_sar' and 'team_id' in d and 'victim_id' in d:
                    valid_directives.append(d)
                elif d['type'] == 'assign_fire' and 'team_id' in d and 'victim_id' in d:
                    valid_directives.append(d)
            
            return valid_directives
            
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Failed to parse LLM response: {e}")
            # Fallback to heuristic
            heuristic = HeuristicAgent(self.env)
            return heuristic.act(state).directives


def run_inference(
    task: int = 1,
    max_steps: int = 120,
    verbose: bool = True,
    seed: int = 42,
    use_llm: bool = False,
) -> float:
    """
    Run the agent on the given task.
    
    Args:
        task: Task number (1, 2, or 3)
        max_steps: Maximum steps to run
        verbose: Print progress
        seed: Random seed
        use_llm: Whether to use LLM agent (requires API credentials)
    
    Returns:
        float: Normalized score (0.0-1.0)
    """
    random.seed(seed)
    
    env = UrbanMCIEnv(task=task)
    state = env.reset()
    
    # Choose agent based on configuration
    if use_llm:
        print(f"\nUsing LLM Agent with model: {os.environ.get('MODEL_NAME', 'unknown')}")
        agent = LLMAgent(env)
    else:
        print(f"\nUsing Heuristic Agent (START protocol)")
        agent = HeuristicAgent(env)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f" Starting Inference: Task {task}")
        print(f" Agent: {'LLM' if use_llm else 'Heuristic'}")
        print(f" Seed: {seed}")
        print(f"{'='*60}\n")
    
    for step in range(max_steps):
        action = agent.act(state)
        state, reward, done, info = env.step(action)
        
        if verbose and step % 20 == 0:
            print(f"Step {step:3d} | Saved: {info['lives_saved']:3d} | "
                  f"Lost: {info['lives_lost']:3d} | "
                  f"Reward: {reward:7.2f} | Grade: {info['grade']:.3f}")
        
        if done:
            break
    
    final_grade = grade(env)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f" Inference Complete: Task {task}")
        print(f" Lives Saved: {info['lives_saved']}")
        print(f" Lives Lost: {info['lives_lost']}")
        print(f" Steps Taken: {info['step']}")
        print(f" Final Grade: {final_grade:.4f}")
        print(f"{'='*60}\n")
    
    return final_grade


def main():
    """Run inference on all three tasks."""
    print("\n" + "="*60)
    print(" URBAN MCI ENVIRONMENT - INFERENCE BENCHMARK")
    print("="*60 + "\n")
    
    # Check if LLM mode is requested
    use_llm = os.environ.get("API_BASE_URL") and os.environ.get("HF_TOKEN")
    
    if use_llm:
        print(f"LLM Mode: API_BASE_URL={os.environ.get('API_BASE_URL')}")
        print(f"         MODEL_NAME={os.environ.get('MODEL_NAME', 'not set')}")
    else:
        print("Heuristic Mode: Using START triage protocol")
        print("(Set API_BASE_URL and HF_TOKEN for LLM mode)")
    
    results = {}
    
    for task in [1, 2, 3]:
        print(f"\n>>> Running Task {task}...")
        score = run_inference(task=task, verbose=True, seed=42, use_llm=use_llm)
        results[task] = score
    
    # Summary
    print("\n" + "="*60)
    print(" FINAL RESULTS")
    print("="*60)
    for task, score in results.items():
        print(f"  Task {task}: {score:.4f}")
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    main()
