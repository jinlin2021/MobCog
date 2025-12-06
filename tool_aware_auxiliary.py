# -*- coding: utf-8 -*-

import json


class ToolAwareAuxiliaryMechanism(object):
    def __init__(self, max_summary_len=200):
        self.max_summary_len = max_summary_len

    def process(self, interaction_history):
      
        events_by_step = self._group_by_step(interaction_history)
        compressed_all = self._build_compressed_trajectory(events_by_step)
        tool_traj = self._build_tool_trajectory(interaction_history)

        return {
            "compressed_trajectory": compressed_all,
            "tool_trajectory": tool_traj,
        }

    def process_one_step(self, step_history):
      
        if not step_history:
            return {
                "compressed_trajectory": [],
                "tool_trajectory": [],
            }

        step = int(step_history[0].get("step", 0) or 0)
        events_by_step = {
            step: sorted(step_history, key=lambda e: e.get("inner_step", 0))
        }
        compressed_list = self._build_compressed_trajectory(events_by_step)
        tool_traj = self._build_tool_trajectory(step_history)

        return {
            "compressed_trajectory": compressed_list,
            "tool_trajectory": tool_traj,
        }

    def _group_by_step(self, interaction_history):
        events_by_step = {}
        for ev in interaction_history:
            step = int(ev.get("step", 0) or 0)
            events_by_step.setdefault(step, []).append(ev)

        for step in events_by_step:
            events_by_step[step] = sorted(
                events_by_step[step],
                key=lambda e: e.get("inner_step", 0),
            )
        return events_by_step

    def _build_compressed_trajectory(self, events_by_step):
      
        compressed = []

        for step in sorted(events_by_step.keys()):
            events = events_by_step[step]

            summary = self._extract_summary_for_step(events)
            scores = self._extract_scores_for_step(events)
            key_tools = self._extract_key_tools_for_step(events)
            affect_tags = self._extract_affect_tags(summary)

            summary = (summary or "").strip()
            if len(summary) > self.max_summary_len:
                summary = summary[: self.max_summary_len] + "..."

            compressed.append(
                {
                    "step": step,
                    "summary": summary,
                    "scores": scores,
                    "key_tools": key_tools,
                    "affect_tags": affect_tags,
                }
            )

        return compressed

    def _extract_summary_for_step(self, events):
        for ev in reversed(events):
            if ev.get("type") == "subtask_complete":
                return ev.get("content", "")

        last_thought = None
        last_obs = None
        for ev in reversed(events):
            if ev.get("type") == "thought" and last_thought is None:
                last_thought = ev.get("content")
            if ev.get("type") == "observation" and last_obs is None:
                last_obs = ev.get("content")
            if last_thought is not None and last_obs is not None:
                break

        if last_thought or last_obs:
            pieces = []
            if last_thought:
                pieces.append(last_thought)
            if last_obs:
                pieces.append(last_obs)
            return " ".join(pieces)

        return "There is no explicit summary for this segment; only process events were recorded."

    def _extract_scores_for_step(self, events):
        for ev in reversed(events):
            if ev.get("type") == "subtask_complete":
                md = ev.get("metadata") or {}
                sc = md.get("score")
                if isinstance(sc, dict):
                    return sc
        return {}

    def _extract_key_tools_for_step(self, events):
        tools = []
        for ev in events:
            if ev.get("type") == "tool_call":
                md = ev.get("metadata") or {}
                name = md.get("name")
                if name and name not in tools:
                    tools.append(name)
        return tools

    def _extract_affect_tags(self, summary_text):
       
        tags = []
        text = (summary_text or "").lower()

        positive_words = ["relaxed", "smooth", "comfortable", "at ease", "familiar"]
        high_load_words = [
            "nervous",
            "worried",
            "confused",
            "crowded",
            "noisy",
            "afraid",
            "lost",
            "uneasy",
        ]

        if any(w in text for w in positive_words):
            tags.append("positive")
        if any(w in text for w in high_load_words):
            tags.append("high_load")

        return tags

    def _build_tool_trajectory(self, interaction_history):
        tool_trajectory = []
        pending = {}

        for ev in interaction_history:
            ev_type = ev.get("type")
            md = ev.get("metadata") or {}
            step = int(ev.get("step", 0) or 0)
            inner_round = md.get("inner_round") or ev.get("inner_step")

            if ev_type == "tool_call":
                name = md.get("name")
                if not name:
                    continue
                args = md.get("args")

                rec = {
                    "step": step,
                    "inner_step": inner_round,
                    "tool_name": name,
                    "args": args,
                    "result": None,
                }
                tool_trajectory.append(rec)
                key = (step, inner_round, name)
                pending[key] = len(tool_trajectory) - 1

            elif ev_type == "tool_result":
                name = md.get("name")
                content_str = ev.get("content") or ""
                try:
                    result = json.loads(content_str)
                except Exception:
                    result = {"raw": content_str}

                key = (step, inner_round, name)
                idx = pending.get(key)
                if idx is not None:
                    tool_trajectory[idx]["result"] = result
                else:
                    rec = {
                        "step": step,
                        "inner_step": inner_round,
                        "tool_name": name,
                        "args": None,
                        "result": result,
                    }
                    tool_trajectory.append(rec)

        return tool_trajectory
