# CognitiveTagger.py
# -*- coding: utf-8 -*-

import json


class CognitiveTagger(object):
    """Rule-based lightweight cognitive load tagger."""

    def __init__(self):
        # Simple English keywords for detecting high cognitive load / time pressure / spatial disorientation
        self.high_load_words_zh = [
            "nervous", "worried", "anxious", "afraid", "scared", "stressed",
            "overwhelmed", "uncomfortable", "tired", "exhausted", "dizzy",
            "crowded", "noisy", "loud", "very loud", "uneasy",
        ]
        self.high_load_words_en = [
            "nervous", "worried", "anxious", "afraid", "scared", "stressed",
            "overwhelmed", "uncomfortable", "crowded", "noisy",
        ]

        self.time_pressure_words_zh = [
            "tight on time", "in a rush", "afraid of being late",
            "afraid of missing", "running out of time", "very urgent",
        ]
        self.time_pressure_words_en = [
            "time pressure", "in a hurry", "rushed", "rush", "afraid of missing",
            "afraid to miss", "worried about missing",
        ]

        self.spatial_disorientation_words_zh = [
            "got lost", "detoured", "could not find the way",
            "did not recognize the way", "confused about direction",
            "could not figure it out", "not sure where I was",
        ]
        self.spatial_disorientation_words_en = [
            "disoriented", "lost", "not sure where", "confused about where",
            "uncertain about where", "did not know where",
        ]

    def tag_single_task(self, sim_record, interview_record=None):

        interaction_history = sim_record.get("interaction_history") or []
        step_events = self._group_by_step(interaction_history)

        per_step_tags = []

        for step_id in sorted(step_events.keys()):
            events = step_events[step_id]
            tags_for_step = self._tag_single_step(events)
            per_step_tags.append(tags_for_step)

        global_summary = self._summarize_global(per_step_tags)

        return {
            "per_step_tags": per_step_tags,
            "global_summary": global_summary,
        }


    def _group_by_step(self, interaction_history):
        """
        Group interaction_history by step into a dict:
            { step_id: [event1, event2, ...], ... }
        """
        step_to_events = {}
        for ev in interaction_history:
            step = ev.get("step")
            if step is None:
                # Skip events without step
                continue
            step = int(step)
            if step not in step_to_events:
                step_to_events[step] = []
            step_to_events[step].append(ev)
        return step_to_events


    def _tag_single_step(self, events):
       
        step_id = int(events[0].get("step", 0) or 0)

        thoughts = []
        observations = []
        tool_calls = []
        tool_results = []

        # Collect text and tool information
        for ev in events:
            ev_type = ev.get("type")
            content = ev.get("content") or ""
            meta = ev.get("metadata") or {}

            if ev_type == "thought":
                thoughts.append(content)
            elif ev_type == "observation":
                observations.append(content)
            elif ev_type == "tool_call":
                tool_calls.append({
                    "name": (meta.get("name") or meta.get("tool_name")),
                    "args": meta.get("args"),
                })
            elif ev_type == "tool_result":
                tool_results.append({
                    "name": (meta.get("name") or meta.get("tool_name")),
                    "raw": content,
                })


        all_text = "\n".join(thoughts + observations)
        all_text_lower = all_text.lower()

        tags = []


        if self._contains_any(all_text, self.high_load_words_zh) \
           or self._contains_any(all_text_lower, self.high_load_words_en):
            tags.append("high_load")

        if self._contains_any(all_text, self.time_pressure_words_zh) \
           or self._contains_any(all_text_lower, self.time_pressure_words_en):
            tags.append("time_pressure")

        if self._contains_any(all_text, self.spatial_disorientation_words_zh) \
           or self._contains_any(all_text_lower, self.spatial_disorientation_words_en):
            tags.append("spatial_disorientation")

        crowding_flag = False        # inferred from tool results
        env_high_load_flag = False   # inferred from environmental description

        for tr in tool_results:
            parsed = self._safe_json_loads(tr.get("raw") or "")
            if isinstance(parsed, dict):
                crowd_str = parsed.get("crowd") or parsed.get("crowding_level")
                if isinstance(crowd_str, str):
                    if any(w in crowd_str.lower() for w in ["very crowded", "packed", "crowded"]):
                        crowding_flag = True

                desc = parsed.get("description") or parsed.get("station_desc") or ""
                if isinstance(desc, str):
                    desc_lower = desc.lower()
                    if (
                        "noise" in desc_lower
                        or "noisy" in desc_lower
                        or "obstacle" in desc_lower
                        or "unsafe" in desc_lower
                        or "poor lighting" in desc_lower
                    ):
                        env_high_load_flag = True

        if crowding_flag:
            if "high_load" not in tags:
                tags.append("high_load")
            tags.append("crowding")

        if env_high_load_flag:
            if "high_load" not in tags:
                tags.append("high_load")
            tags.append("environmental_stress")

        score = self._estimate_cog_load_score(tags)

        evidence = {
            "thoughts": thoughts,
            "observations": observations,
            "tools": {
                "calls": tool_calls,
                "results": tool_results,
            },
        }

        return {
            "step": step_id,
            "cog_load_score": score,
            "tags": sorted(list(set(tags))),
            "evidence": evidence,
        }


    def _estimate_cog_load_score(self, tags):
    
        base = 0.3
        if "high_load" in tags:
            base += 0.3
        if "time_pressure" in tags:
            base += 0.2
        if "spatial_disorientation" in tags:
            base += 0.2
        if base > 1.0:
            base = 1.0
        if base < 0.0:
            base = 0.0
        return base


    def _summarize_global(self, per_step_tags):

        if not per_step_tags:
            return {
                "avg_cog_load": 0.0,
                "max_cog_load": 0.0,
                "num_steps": 0,
                "num_high_load_steps": 0,
                "fraction_high_load": 0.0,
                "high_load_steps": [],
                "all_tags": [],
            }

        scores = [x.get("cog_load_score", 0.0) for x in per_step_tags]
        num_steps = len(scores)
        avg_score = sum(scores) / float(num_steps)
        max_score = max(scores)

        high_load_steps = []
        for x in per_step_tags:
            if x.get("cog_load_score", 0.0) >= 0.6:
                high_load_steps.append(x.get("step"))

        num_high = len(high_load_steps)
        fraction_high = num_high / float(num_steps) if num_steps > 0 else 0.0

        all_tags = set()
        for x in per_step_tags:
            for t in x.get("tags", []):
                all_tags.add(t)

        return {
            "avg_cog_load": avg_score,
            "max_cog_load": max_score,
            "num_steps": num_steps,
            "num_high_load_steps": num_high,
            "fraction_high_load": fraction_high,
            "high_load_steps": sorted(high_load_steps),
            "all_tags": sorted(list(all_tags)),
        }


    def _contains_any(self, text, word_list):
        """Return True if text contains any substring from word_list."""
        if not text:
            return False
        for w in word_list:
            if w and (w in text):
                return True
        return False

    def _safe_json_loads(self, text):
        """Try to parse a string as JSON; if it fails, return the original text."""
        if not text:
            return text
        try:
            return json.loads(text)
        except Exception:
            return text



def _build_dummy_history():
    return [
        {
            "step": 1,
            "inner_step": 1,
            "role": "agent",
            "type": "thought",
            "content": "I want to arrive at the hospital early today and do not want to be late.",
            "metadata": {},
        },
        {
            "step": 1,
            "inner_step": 2,
            "role": "agent",
            "type": "thought",
            "content": "There are many people at the bus stop, and I am worried I might not get on the bus.",
            "metadata": {},
        },
        {
            "step": 1,
            "inner_step": 3,
            "role": "tool",
            "type": "tool_result",
            "content": json.dumps(
                {
                    "crowd": "very crowded",
                    "description": "The bus is very crowded, and I have to stand during the ride.",
                },
                ensure_ascii=False,
            ),
            "metadata": {"name": "get_traffic"},
        },
        {
            "step": 2,
            "inner_step": 1,
            "role": "agent",
            "type": "thought",
            "content": "When transferring, I was a bit confused about the direction and almost went to the wrong platform.",
            "metadata": {},
        },
        {
            "step": 2,
            "inner_step": 2,
            "role": "agent",
            "type": "thought",
            "content": "Fortunately, the signs helped, and I eventually found the correct platform.",
            "metadata": {},
        },
    ]


def _run_demo():
    sim_record = {
        "interaction_history": _build_dummy_history(),
        "step_results": [],
        "agent": {},
        "task": {},
    }

    tagger = CognitiveTagger()
    result = tagger.tag_single_task(sim_record)

    print("=== Per-step tags ===")
    for x in result["per_step_tags"]:
        print(x)

    print("\n=== Global summary ===")
    print(result["global_summary"])


if __name__ == "__main__":
    _run_demo()
