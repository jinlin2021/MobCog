import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import deque


class Persona:

    def __init__(self, persona_data: Dict[str, Any], state_id: Optional[str] = None):
 
        self.id = persona_data.get("id")
        self.profile_id = persona_data.get("profile_id")
        self.profile_label = persona_data.get("profile_label")

        self.cognitive_level = persona_data.get("cognitive_level")
        self.mobility_profile = persona_data.get("mobility")
        self.affect_profile = persona_data.get("affect_profile")
        self.expected_mmse_range = persona_data.get("expected_mmse_range")
        self.expected_adl_range = persona_data.get("expected_adl_range")
        self.expected_gds_range = persona_data.get("expected_gds_range")
        self.description = persona_data.get("description")

        self.base_profile = persona_data.get("base_profile")
        if not self.base_profile:
            self.base_profile = {
                "name": persona_data.get("name", "unknown"),
                "age": persona_data.get("age", "unknown"),
                "gender": persona_data.get("gender", "unknown"),
            }

        states = persona_data.get("states", [])
        if not states and (
            self.cognitive_level is not None
            or self.mobility_profile is not None
            or self.affect_profile is not None
            or self.expected_mmse_range is not None
            or self.expected_adl_range is not None
            or self.expected_gds_range is not None
        ):
            states = [self._build_state_from_flat_profile(persona_data)]

        self.states: List[Dict[str, Any]] = states

        self.current_state: Optional[Dict[str, Any]] = None
        if state_id:
            for s in self.states:
                if s.get("state_id") == state_id:
                    self.current_state = s
                    break

        if not self.current_state and self.states:
            self.current_state = self.states[0]

        self.experiences = deque(maxlen=10)
        self.memory_state: Optional[Dict[str, Any]] = None
        self.memory: Dict[str, Any] = {}

    @staticmethod
    def _midpoint(range_list: Optional[List[int]], default: int) -> int:
        """Return the midpoint of [min, max] as a default score."""
        if isinstance(range_list, list) and len(range_list) == 2:
            return int((range_list[0] + range_list[1]) / 2)
        return default

    def _build_state_from_flat_profile(self, persona_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a normalized state structure from flat persona fields:
        {
          "state_id": ...,
          "health": {...},
          "cognition": {...},
          "function": {...},
          "mental_health": {...}
        }
        """
        health = persona_data.get("health") or {}

        cognition = {
            "mmse_score": persona_data.get(
                "mmse_score",
                self._midpoint(self.expected_mmse_range, 27),
            ),
            "status": persona_data.get("cognitive_status", self.cognitive_level),
        }

        function = {
            "adl_score": persona_data.get(
                "adl_score",
                self._midpoint(self.expected_adl_range, 90),
            ),
            "uses_mobility_aid": persona_data.get("uses_mobility_aid", False),
            "mobility_profile": self.mobility_profile,
        }

        mental_health = {
            "gds_score": persona_data.get(
                "gds_score",
                self._midpoint(self.expected_gds_range, 1),
            ),
            "depression_level": persona_data.get("depression_level", self.affect_profile),
        }

        state = {
            "state_id": persona_data.get("state_id", self.profile_id or "default"),
            "health": health,
            "cognition": cognition,
            "function": function,
            "mental_health": mental_health,
        }
        return state

    def add_experience(self, experience: Dict[str, Any]):

        if isinstance(experience, dict):
            experience["timestamp"] = datetime.now().isoformat()
            self.experiences.append(experience)

    def get_recent_experiences(self, count: int = 3) -> List[Dict[str, Any]]:
  
        return list(self.experiences)[-count:] if self.experiences else []

    def get_health_status(self) -> str:
  
        if not self.current_state:
            parts = []
            if self.cognitive_level:
                parts.append(f"Cognitive status: {self.cognitive_level}")
            if self.mobility_profile:
                parts.append(f"Physical function: {self.mobility_profile}")
            if self.affect_profile:
                parts.append(f"Mood / social engagement: {self.affect_profile}")
            if self.expected_mmse_range:
                parts.append(
                    f"Expected MMSE range: {self.expected_mmse_range[0]}–{self.expected_mmse_range[1]}"
                )
            if self.expected_adl_range:
                parts.append(
                    f"Expected ADL range: {self.expected_adl_range[0]}–{self.expected_adl_range[1]}"
                )
            if self.expected_gds_range:
                parts.append(
                    f"Expected GDS range: {self.expected_gds_range[0]}–{self.expected_gds_range[1]}"
                )
            return "; ".join(parts) if parts else "Unknown health status"

        cognition = self.current_state.get("cognition", {})
        function = self.current_state.get("function", {})
        mental_health = self.current_state.get("mental_health", {})

        status_parts = []

        mmse = cognition.get("mmse_score")
        cog_status = cognition.get("status", self.cognitive_level)
        if mmse is not None:
            status_parts.append(f"Cognitive status: {cog_status} (MMSE ≈ {mmse})")
        else:
            status_parts.append(f"Cognitive status: {cog_status}")

        adl = function.get("adl_score")
        uses_aid = function.get("uses_mobility_aid", False)
        mobility = function.get("mobility_profile", self.mobility_profile)
        if uses_aid:
            mobility_str = f"{mobility}, requires mobility aid" if mobility else "Requires mobility aid"
        else:
            mobility_str = mobility or "Independent walking"
        if adl is not None:
            status_parts.append(f"Physical function: {mobility_str} (ADL ≈ {adl})")
        else:
            status_parts.append(f"Physical function: {mobility_str}")

        gds = mental_health.get("gds_score")
        depression = mental_health.get("depression_level", self.affect_profile)
        if depression or gds is not None:
            if gds is not None:
                status_parts.append(f"Mood / depression: {depression} (GDS ≈ {gds})")
            else:
                status_parts.append(f"Mood / depression: {depression}")

        return "; ".join(status_parts) if status_parts else "Unknown health status"

    # def get_cognitive_load_threshold(self) -> int:
    #     """
    #     Map cognitive status (MMSE) to an approximate cognitive load threshold in [1, 7].
    #     Higher value means higher tolerance for cognitive load.
    #     """
    #     mmse = 25
    #     if self.current_state:
    #         cognition = self.current_state.get("cognition", {})
    #         mmse = cognition.get("mmse_score", mmse)
    #     elif self.expected_mmse_range:
    #         mmse = self._midpoint(self.expected_mmse_range, mmse)

    #     if mmse >= 27:
    #         return 6
    #     elif mmse >= 21:
    #         return 4
    #     elif mmse >= 10:
    #         return 2
    #     else:
    #         return 1

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert persona to a dictionary.
        """
        return {
            "id": self.id,
            "profile_id": self.profile_id,
            "profile_label": self.profile_label,
            "base_profile": self.base_profile,
            "current_state": self.current_state,
            "experiences": list(self.experiences),
            "description": self.description,
        }

    def __str__(self) -> str:
        name = self.base_profile.get("name", "unknown")
        age = self.base_profile.get("age", "unknown")
        gender = self.base_profile.get("gender", "unknown")
        label = f"{self.profile_label}; " if self.profile_label else ""
        return f"{name}, {age} years old, {gender}, {label}{self.get_health_status()}"
