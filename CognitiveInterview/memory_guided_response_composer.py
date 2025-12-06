# CognitiveInterview/memory_guided_response_composer.py
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Callable, Optional

class WorkingMemory:
    """
    Short-term goals and action plans (M_W).
    """

    def __init__(
        self,
        goals: Optional[List[str]] = None,
        action_plans: Optional[List[str]] = None,
        recent_obstacles: Optional[List[str]] = None,
        recent_observations: Optional[List[str]] = None,
    ) -> None:
        self.goals = goals or []
        self.action_plans = action_plans or []
        self.recent_obstacles = recent_obstacles or []
        self.recent_observations = recent_observations or []

    def __repr__(self) -> str:
        return (
            "WorkingMemory("
            f"goals={self.goals!r}, "
            f"action_plans={self.action_plans!r}, "
            f"recent_obstacles={self.recent_obstacles!r}, "
            f"recent_observations={self.recent_observations!r})"
        )


class EpisodicMemory:
    """
    Episodic memory (M_E) as a task-level log.
    """

    def __init__(
        self,
        events: Optional[List[Dict[str, Any]]] = None,
        decision_points: Optional[List[str]] = None,
    ) -> None:
        self.events = events or []
        self.decision_points = decision_points or []

    def __repr__(self) -> str:
        return (
            "EpisodicMemory("
            f"events={self.events!r}, "
            f"decision_points={self.decision_points!r})"
        )


class SpatialCognitiveMemory:
    """
    Spatial-cognitive memory (M_S).
    """

    def __init__(
        self,
        stops: Optional[List[Dict[str, Any]]] = None,
        landmarks: Optional[List[Dict[str, Any]]] = None,
        routes: Optional[List[str]] = None,
        environmental_cues: Optional[List[str]] = None,
    ) -> None:
        self.stops = stops or []
        self.landmarks = landmarks or []
        self.routes = routes or []
        self.environmental_cues = environmental_cues or []

    def __repr__(self) -> str:
        return (
            "SpatialCognitiveMemory("
            f"stops={self.stops!r}, "
            f"landmarks={self.landmarks!r}, "
            f"routes={self.routes!r}, "
            f"environmental_cues={self.environmental_cues!r})"
        )


class StructuredMemoryState:
    """
    Structured memory state M_t, combining three memory systems.
    """

    def __init__(
        self,
        working: WorkingMemory,
        episodic: EpisodicMemory,
        spatial: SpatialCognitiveMemory,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.working = working
        self.episodic = episodic
        self.spatial = spatial
        self.meta = meta or {}

    def __repr__(self) -> str:
        return (
            "StructuredMemoryState("
            f"working={self.working!r}, "
            f"episodic={self.episodic!r}, "
            f"spatial={self.spatial!r}, "
            f"meta={self.meta!r})"
        )


# -----------------------------------------------------------------------------
# Memory-Guided Response Composer
# -----------------------------------------------------------------------------
class MemoryGuidedResponseComposer:


    def __init__(
        self,
        llm_fn: Callable[[str], str],
        system_instruction: Optional[str] = None,
    ) -> None:
        self.llm_fn = llm_fn
        if system_instruction is None:
            system_instruction = (
                "You are now in the post-task cognitive interview phase. "
                "You must answer as the elderly agent, reflecting on your own "
                "mental processes and emotions during the completed trip. "
                "Focus on why you made certain decisions and how you perceived "
                "the environment. Do not propose new plans; only describe what "
                "actually happened and how you felt. Always answer in the first "
                "person (“I ...”) and stay fully in character as the specified "
                "older adult."
            )
        self.system_instruction = system_instruction

    def answer_question(
        self,
        question: str,
        memory_state: StructuredMemoryState,
        return_prompt: bool = False,
    ) -> Any:
        """
        Generate an answer for a single interview question.
        """
        prompt = self._build_prompt(question, memory_state)
        answer = self.llm_fn(prompt).strip()

        if return_prompt:
            return answer, prompt
        return answer

    def answer_question_list(
        self,
        questions: List[str],
        memory_state: StructuredMemoryState,
    ) -> List[str]:
        """
        Generate answers for a list of interview questions.
        """
        answers = []
        for q in questions:
            a = self.answer_question(q, memory_state, return_prompt=False)
            answers.append(a)
        return answers

    def _build_prompt(
        self,
        question: str,
        memory_state: StructuredMemoryState,
    ) -> str:
        """
        Construct the full text prompt for f_interview(q_i, M_t).
        """
        memory_summary = self._format_memory_summary(memory_state)

        prompt = (
            self.system_instruction
            + "\n\n"
            + "Below is the structured memory of the completed trip.\n"
            + memory_summary
            + "\nNow answer the following interview question in the first "
              "person (“I ...”), providing a coherent and subjectively "
              "expressive response.\n"
            + "Interview question:\n"
            + question
            + "\n\nAnswer:"
        )
        return prompt

    def _format_memory_summary(self, memory_state: StructuredMemoryState) -> str:
        """
        Convert the structured memory state into readable text for the LLM.
        """
        wm = memory_state.working
        em = memory_state.episodic
        sm = memory_state.spatial

        lines = []

        # Working memory
        lines.append("[Working memory: short-term goals, plans, and obstacles]")
        lines.append(f"- Goals: {wm.goals}")
        lines.append(f"- Action plans: {wm.action_plans}")
        lines.append(f"- Recent obstacles: {wm.recent_obstacles}")
        lines.append(f"- Recent observations: {wm.recent_observations}")

        # Episodic memory
        lines.append("\n[Episodic memory: key events and decision points]")
        if em.events:
            lines.append(f"- Events: {em.events}")
        else:
            lines.append("- Events: []")
        if em.decision_points:
            lines.append(f"- Decision points: {em.decision_points}")
        else:
            lines.append("- Decision points: []")

        # Spatial-cognitive memory
        lines.append("\n[Spatial-cognitive memory: environmental and spatial cues]")
        lines.append(f"- Stops: {sm.stops}")
        lines.append(f"- Landmarks: {sm.landmarks}")
        lines.append(f"- Routes: {sm.routes}")
        lines.append(f"- Environmental cues: {sm.environmental_cues}")

        # Optional meta-information
        if memory_state.meta:
            lines.append("\n[Additional meta information]")
            for k, v in memory_state.meta.items():
                lines.append(f"- {k}: {v}")

        return "\n".join(lines)
