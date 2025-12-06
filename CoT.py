import json
import os
import re
import requests
from typing import List, Dict, Any, Optional, Union
import time
import openai

try:
    from openai import OpenAI
    USE_OPENAI = True
except ImportError:
    USE_OPENAI = False
    print("OpenAI library not installed, will use local model")


THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)


def _extract_block(pattern, text):
    m = pattern.search(text or "")
    if not m:
        return ""
    return m.group(1).strip()


class CoTReasoner:
    """Chain-of-thought reasoner simulating an older adult's cognitive decisions during a bus trip."""

    def __init__(self, tools, api_key=None, model="qwen2-vl"):
        self.tools = tools
        self.model = model
        self.tools_schema = self.create_tools_schema()
        self.vllm_endpoint = os.environ.get("VLLM_ENDPOINT", "http://localhost:8000/v1")
        self.client = openai.OpenAI(
            base_url=self.vllm_endpoint,
            api_key="EMPTY"
        )

    @staticmethod
    def create_tools_schema():
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a given date and location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "Date in the format YYYY-MM-DD"
                            }
                        },
                        "required": ["date"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_traffic",
                    "description": "Get traffic / crowding information for a given line and time.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "line": {
                                "type": "string",
                                "description": "Transit line identifier"
                            },
                            "time": {
                                "type": "string",
                                "description": "Time in the format HH:MM"
                            }
                        },
                        "required": ["line", "time"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_street_view",
                    "description": "Get street-view images and description for a given stop.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "station": {
                                "type": "string",
                                "description": "Stop/station name"
                            }
                        },
                        "required": ["station"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "finish",
                    "description": "Finish analysis for the current segment and submit a summary.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "string",
                                "description": "Short summary of the experience for this segment"
                            },
                            "score": {
                                "type": "object",
                                "description": "Scores (1–7) for different aspects",
                                "properties": {
                                    "fatigue": {
                                        "type": "number",
                                        "description": "Mental fatigue level"
                                    },
                                    "anxiety": {
                                        "type": "number",
                                        "description": "Anxiety level"
                                    },
                                    "info_load": {
                                        "type": "number",
                                        "description": "Information processing load"
                                    },
                                    "risk": {
                                        "type": "number",
                                        "description": "Perceived risk of getting lost"
                                    },
                                    "environment": {
                                        "type": "number",
                                        "description": "Environmental influence"
                                    },
                                    "overall": {
                                        "type": "number",
                                        "description": "Overall cognitive load"
                                    }
                                }
                            }
                        },
                        "required": ["summary", "score"]
                    }
                }
            }
        ]

    def _call_model(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools_schema,
                tool_choice="auto",
                temperature=0.7,
                parallel_tool_calls=True,
            )
            return response.choices[0].message
        except Exception as e:
            print(f"LLM API call failed: {e}")
            return {"content": f"Error: {e}"}

    def run_stepwise(
        self,
        persona,
        step,
        trajectory,
        history=None,
        max_inner_steps=4,
        use_llm_in_tools=False,
    ):
        """
        Multi-step CoT + Tool-Use with explicit <think>, <tool_call>, <observation> segments.

        Returns:
            {
                "step_results": [...],
                "history": [...],
            }
        """
        import json

        step_results = []
        step_history = []

        profile = persona.base_profile
        health_status = persona.get_health_status()
        cognitive_threshold = persona.get_cognitive_load_threshold()

        def log_event(event_type, role, content, metadata=None):
            step_history.append(
                {
                    "step": step + 1,
                    "inner_step": len(step_history) + 1,
                    "role": role,
                    "type": event_type,
                    "content": content,
                    "metadata": metadata or {},
                }
            )

        history = history if history is not None else persona.get_recent_experiences(3)
        history_text = json.dumps(history, ensure_ascii=False) if history else "no prior experience"

        system_prompt = (
            f"You are role-playing a {profile.get('age', '70')}-year-old older adult "
            f"with gender {profile.get('gender', 'male')}. "
            f"You are currently experiencing a bus trip. Always describe your thoughts "
            f"and feelings in the first person as \"I\".\n\n"
            f"Your health status is as follows:\n{health_status}\n\n"
            f"Based on your health condition, your tolerable cognitive load is roughly "
            f"{cognitive_threshold}/7. When the environment is complex, noisy, or transfers "
            f"are complicated, you may feel nervous, confused, or uneasy. When the route "
            f"is familiar and the environment is friendly, you feel relaxed and reassured.\n\n"
            f"In each reasoning round, you must follow this output format:\n"
            f"1. First output a <think>...</think> block that describes in detail your current "
            f"thoughts, subgoals, and feelings.\n"
            f"2. Then output a <tool_call>...</tool_call> block containing a JSON object used "
            f"to call a tool or to finish the analysis.\n"
            f"   - If you need a tool, set name to 'get_weather', 'get_traffic', or 'get_street_view' "
            f"and specify the required args.\n"
            f"   - If you think this segment can be summarized now, set name to 'finish' and provide "
            f"'summary' (string) and 'score' (dict, for example {{\"overall\": 5}}) in args."
        )
        log_event("context", "system", system_prompt, {"kind": "system_prompt"})

        segment_info = (
            f"This is ride segment {step + 1}:\n"
            f"Date: {trajectory.get('date', 'unknown')}\n"
            f"Card type: {trajectory.get('card_type', 'unknown card type')}\n"
            f"Boarding stop: {trajectory.get('start_station', 'unknown')}\n"
            f"Boarding time: {trajectory.get('start_time', 'unknown')}\n"
            f"Alighting stop: {trajectory.get('end_station', 'unknown')}\n"
            f"Alighting time: {trajectory.get('end_time', 'unknown')}\n"
            f"Line: {trajectory.get('line', 'unknown')}\n"
        )
        log_event(
            "context",
            "agent",
            segment_info,
            {
                "kind": "segment_info",
                "line": trajectory.get("line"),
                "start_station": trajectory.get("start_station"),
                "end_station": trajectory.get("end_station"),
                "start_time": trajectory.get("start_time"),
                "end_time": trajectory.get("end_time"),
                "date": trajectory.get("date"),
                "card_type": trajectory.get("card_type"),
            },
        )

        reasoning_fragments = []
        observations = []

        finished = False        # whether this segment is finished
        inner_round = 0

        while not finished and inner_round < max_inner_steps:
            inner_round += 1

            think_history_text = "\n".join(
                [f"[t={i+1}] {r}" for i, r in enumerate(reasoning_fragments)]
            ) or "none"
            obs_history_text = "\n".join(
                [f"[t={i+1}] {o}" for i, o in enumerate(observations)]
            ) or "none"

            user_content = (
                f"Here are some of your past travel experiences (for reference only):\n"
                f"{history_text}\n\n"
                f"Current ride segment information:\n{segment_info}\n\n"
                f"In this segment so far, your previous <think> history is:\n"
                f"{think_history_text}\n\n"
                f"Your previous <observation> history from tools is:\n"
                f"{obs_history_text}\n\n"
                f"Now continue with reasoning round {inner_round} for this segment. "
                f"You must output:\n"
                f"1. A <think>...</think> block that describes your current thoughts, level of "
                f"confusion or sense of safety, and what information you think you need next.\n"
                f"2. A <tool_call>{{...}}</tool_call> block that contains a JSON object with fields "
                f"'name' and 'args'.\n"
                f"   - If you need to query the weather, call name='get_weather' and provide 'date' in args.\n"
                f"   - If you need to query crowding for the bus, call name='get_traffic' and provide "
                f"'line' and 'time' in args.\n"
                f"   - If you need to view street scenes at a stop, call name='get_street_view' and "
                f"provide 'station' in args.\n"
                f"   - If you think this segment can already be summarized, call name='finish' and provide "
                f"'summary' and 'score' in args (for example {{\"overall\": 5}}).\n"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            response_msg = self._call_model(messages)
            content = getattr(response_msg, "content", "") or ""

            log_event(
                "raw_model_output",
                "agent",
                content,
                {"inner_round": inner_round},
            )

            think_block = _extract_block(THINK_RE, content)
            tool_block = _extract_block(TOOL_RE, content)

            if think_block:
                reasoning_fragments.append(think_block)
                log_event(
                    "thought",
                    "agent",
                    think_block,
                    {"inner_round": inner_round},
                )
            else:
                reasoning_fragments.append(content)
                log_event(
                    "thought",
                    "agent",
                    content,
                    {"inner_round": inner_round, "missing_think_tag": True},
                )

            if not tool_block:
                result_obj = {
                    "station": trajectory.get("start_station", "unknown station"),
                    "summary": think_block or content or "Unable to obtain a description of this ride segment.",
                    "score": {"overall": 4},
                }
                step_results.append(result_obj)
                persona.add_experience(result_obj)

                log_event(
                    "subtask_complete",
                    "agent",
                    result_obj["summary"],
                    {
                        "station": result_obj["station"],
                        "score": result_obj["score"],
                        "fallback": True,
                    },
                )
                finished = True
                break

            try:
                tool_json = json.loads(tool_block)
            except Exception:
                result_obj = {
                    "station": trajectory.get("start_station", "unknown station"),
                    "summary": think_block or content or f"Tool call parsing failed: {tool_block}",
                    "score": {"overall": 4},
                }
                step_results.append(result_obj)
                persona.add_experience(result_obj)

                log_event(
                    "subtask_complete",
                    "agent",
                    result_obj["summary"],
                    {
                        "station": result_obj["station"],
                        "score": result_obj["score"],
                        "tool_parse_error": True,
                    },
                )
                finished = True
                break

            func_name = tool_json.get("name")
            args = tool_json.get("args") or {}

            log_event(
                "tool_call",
                "tool",
                "",
                {"name": func_name, "args": args, "inner_round": inner_round},
            )

            if func_name == "finish":
                result_obj = {
                    "station": trajectory.get("start_station", "unknown station"),
                    "summary": args.get("summary", "").strip() or (think_block or "No explicit summary was provided for this segment."),
                    "score": args.get("score", {}) or {"overall": 4},
                }
                step_results.append(result_obj)
                persona.add_experience(result_obj)

                log_event(
                    "subtask_complete",
                    "agent",
                    result_obj["summary"],
                    {
                        "station": result_obj["station"],
                        "score": result_obj["score"],
                        "from_finish_tool": True,
                    },
                )
                finished = True
                break

            if func_name == "get_weather":
                result = self.tools["get_weather"](**args)
            elif func_name == "get_traffic":
                result = self.tools["get_traffic"](**args)
            elif func_name == "get_street_view":
                if use_llm_in_tools:
                    sv = self.tools["get_street_view"](
                        args.get("station"),
                        use_llm_vision=True,
                        llm_client=self.client,
                        model_name=self.model,
                    )
                else:
                    sv = self.tools["get_street_view"](args.get("station"))
                result = {k: v for k, v in sv.items() if k != "image_b64_list"}
                c = result.get("orc_correct")
                t = result.get("orc_total")
                if isinstance(c, int) and isinstance(t, int):
                    orc_correct_step += c
                    orc_total_step += t
            else:
                result = {"error": f"Unimplemented tool: {func_name}"}

            log_event(
                "tool_result",
                "tool",
                json.dumps(result, ensure_ascii=False),
                {"name": func_name, "inner_round": inner_round},
            )

            obs_text = (
                f"<observation>Tool {func_name} result: "
                f"{json.dumps(result, ensure_ascii=False)}</observation>"
            )
            observations.append(obs_text)
            log_event(
                "observation",
                "env",
                obs_text,
                {"tool_name": func_name, "inner_round": inner_round},
            )

        if not step_results:
            default_summary = (
                reasoning_fragments[-1]
                if reasoning_fragments
                else "This ride segment ended without an explicit summary."
            )
            result_obj = {
                "station": trajectory.get("start_station", "unknown station"),
                "summary": default_summary,
                "score": {"overall": 4},
            }
            step_results.append(result_obj)
            persona.add_experience(result_obj)

            log_event(
                "subtask_complete",
                "agent",
                result_obj["summary"],
                {
                    "station": result_obj["station"],
                    "score": result_obj["score"],
                    "post_loop_fallback": True,
                },
            )

        return {
            "step_results": step_results,
            "history": step_history,
        }
