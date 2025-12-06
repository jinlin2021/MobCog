"""
PromptScriptGenerator

Constructs a set of structured and context-aware interview questions
for elderly travel cognitive assessment.
"""

from typing import Any, Dict, List, Optional
import random


class AgentProfile:

    def __init__(
        self,
        agent_id: str,
        health_condition: str,
        task_completion_status: str,
        memory_consistency: str,
        extra_attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.agent_id = agent_id
        self.health_condition = health_condition
        self.task_completion_status = task_completion_status
        self.memory_consistency = memory_consistency
        self.extra_attributes = extra_attributes or {}

    def __repr__(self) -> str:
        return (
            f"AgentProfile(agent_id={self.agent_id!r}, "
            f"health_condition={self.health_condition!r}, "
            f"task_completion_status={self.task_completion_status!r}, "
            f"memory_consistency={self.memory_consistency!r})"
        )


class TaskContext:

    def __init__(
        self,
        route_description: str,
        environment_description: str,
        difficulty_level: str,
        time_pressure: str,
    ) -> None:
        self.route_description = route_description
        self.environment_description = environment_description
        self.difficulty_level = difficulty_level
        self.time_pressure = time_pressure

    def __repr__(self) -> str:
        return (
            "TaskContext("
            f"route_description={self.route_description!r}, "
            f"environment_description={self.environment_description!r}, "
            f"difficulty_level={self.difficulty_level!r}, "
            f"time_pressure={self.time_pressure!r})"
        )


# ---------------------------------------------------------------------
# Prompt Script Generator
# ---------------------------------------------------------------------

class PromptScriptGenerator:

    def __init__(
        self,
        random_seed: Optional[int] = 42,
        max_questions_per_dimension: int = 2,
    ) -> None:
        """
        max_questions_per_dimension : int
            Maximum number of questions to sample from each dimension
            (difficulty, disorientation, time pressure, emotion).
            The route-recall questions for metrics are always appended
            and are not randomly dropped.
        """
        self._rng = random.Random(random_seed)
        self.max_questions_per_dimension = max_questions_per_dimension
        self._init_templates()

    def _init_templates(self) -> None:
        self._templates: Dict[str, List[str]] = {
            "perceived_difficulty": [
                "During this trip, at which moments did the route feel more "
                "complicated than you expected?",
                "Which part of the journey felt the most difficult for you, and why?",
                "Were there specific segments where you felt the task became "
                "mentally demanding?",
            ],
            "spatial_disorientation": [
                "At any point did you feel unsure about where you were or "
                "which direction to go?",
                "Were there transfers or stops where you felt disoriented in space?",
                "Did you ever feel that the surroundings looked unfamiliar or "
                "confusing during the route?",
            ],
            "time_pressure": [
                "Did you feel rushed or worried about missing a bus or transfer "
                "during the trip?",
                "How much time pressure did you experience while navigating "
                "this route?",
                "Were there moments when you felt there was not enough time "
                "to make decisions comfortably?",
            ],
            "emotional_fluctuation": [
                "Did you experience noticeable changes in your mood during the trip?",
                "Were there moments when you felt especially anxious, frustrated, "
                "or relieved?",
                "Can you recall any part of the journey that made you feel "
                "worried, stressed, or unusually calm?",
            ],
        }

    def generate_questions(
        self,
        agent: AgentProfile,
        task: TaskContext,
        extra_dims: Optional[Dict[str, List[str]]] = None,
    ) -> List[str]:
        """
        Generate a list of interview questions for the given agent and task.

        - First, sample a small number of questions from each predefined dimension.
        - Then, append dynamic questions that depend on agent/task context.
        - Finally, append specially tagged route-recall questions
          ([IRRA], [DRRA], [RAC]) for metric computation.
        """
        dims: Dict[str, List[str]] = dict(self._templates)
        if extra_dims:
            for dim, tmpl_list in extra_dims.items():
                if dim in dims:
                    dims[dim] = dims[dim] + list(tmpl_list)
                else:
                    dims[dim] = list(tmpl_list)

        questions: List[str] = []

        for dim, candidates in dims.items():
            if not candidates:
                continue
            k = min(self.max_questions_per_dimension, len(candidates))
            sampled = self._rng.sample(candidates, k)
            questions.extend(sampled)

        dynamic_questions = self._generate_dynamic_questions(agent, task)
        questions.extend(dynamic_questions)

        recall_questions = self._generate_recall_questions(agent, task)
        questions.extend(recall_questions)

        return questions

    def _generate_dynamic_questions(
        self,
        agent: AgentProfile,
        task: TaskContext,
    ) -> List[str]:
        qs: List[str] = []

        qs.append(
            "Given that this route was labeled as {difficulty} difficulty, "
            "did it feel easier or harder than you anticipated? Please explain."
            .format(difficulty=task.difficulty_level)
        )
        qs.append(
            "Considering your health condition ({health}), were there any parts "
            "of the route that you found especially demanding physically or "
            "mentally?"
            .format(health=agent.health_condition)
        )
        qs.append(
            "The overall task outcome was {status}. From your own perspective, "
            "what were the main reasons that led to this outcome?"
            .format(status=agent.task_completion_status)
        )

        qs.append(
            "During the trip, did you notice any inconsistencies between what "
            "you originally planned and what you actually remembered at "
            "different stops? Please describe these moments if they occurred."
        )

        qs.append(
            "The environment was described as: {env}. In what ways did this "
            "environment influence your concentration, emotions, or decision "
            "making along the route?"
            .format(env=task.environment_description)
        )

        return qs

    def _generate_recall_questions(
        self,
        agent: AgentProfile,
        task: TaskContext,
    ) -> List[str]:
        """
         DRRA / RAC 
        """
        qs: List[str] = []

        # Delayed Route Recall (DRRA)
        qs.append(
            "[DRRA] Now, after the whole trip is over, please again list the "
            "bus stops you remember from that same main part of the route. "
            "Use the same format:\n"
            "STOP1 > STOP2 > STOP3 > ..."
        )

        # Route Reproduction Accuracy (RAC)
        qs.append(
            "[RAC] Please reconstruct your entire trip from departure to arrival "
            "as a sequence of stops in order. Answer using the format:\n"
            "STOP1 > STOP2 > ... > STOPN"
        )

        return qs


