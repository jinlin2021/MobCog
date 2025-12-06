# -*- coding: utf-8 -*-

from tool_aware_auxiliary import ToolAwareAuxiliaryMechanism


class WorkingMemoryState(object):

    def __init__(
        self,
        current_goal,
        current_subtask,
        recent_obstacles,
        recent_observations,
        cognitive_load,
        last_step,
    ):
        self.current_goal = current_goal
        self.current_subtask = current_subtask
        self.recent_obstacles = recent_obstacles
        self.recent_observations = recent_observations
        self.cognitive_load = cognitive_load
        self.last_step = last_step

    def to_dict(self):
        return {
            "current_goal": self.current_goal,
            "current_subtask": self.current_subtask,
            "recent_obstacles": self.recent_obstacles,
            "recent_observations": self.recent_observations,
            "cognitive_load": self.cognitive_load,
            "last_step": self.last_step,
        }


class EpisodicMemoryState(object):

    def __init__(self, episodes):
        self.episodes = episodes or []

    def to_dict(self):
        return {"episodes": self.episodes}


class SpatialCognitiveMemoryState(object):


    def __init__(self, locations):
        self.locations = locations or []

    def to_dict(self):
        return {"locations": self.locations}


class StructuredMemoryState(object):

    def __init__(self, working, episodic, spatial, meta):
        self.working = working
        self.episodic = episodic
        self.spatial = spatial
        self.meta = meta or {}

    def to_dict(self):
        return {
            "working": self.working.to_dict() if self.working else None,
            "episodic": self.episodic.to_dict() if self.episodic else None,
            "spatial": self.spatial.to_dict() if self.spatial else None,
            "meta": self.meta,
        }


class HierarchicalMemoryManager(object):


    def __init__(self, auxiliary_mechanism=None):
        self.aux = auxiliary_mechanism or ToolAwareAuxiliaryMechanism()

    def consolidate(self, interaction_history, agent_profile, env_state=None, last_memory_state=None):
      
        env_state = env_state or {}

        aux_out = self.aux.process(interaction_history)
        compressed_traj = aux_out["compressed_trajectory"]
        tool_traj = aux_out["tool_trajectory"]

        working = self._build_working_memory(compressed_traj, agent_profile)
        episodic = self._build_episodic_memory(
            compressed_traj=compressed_traj,
            tool_traj=tool_traj,
            env_state=env_state,
            last_memory_state=None,
        )
        spatial = self._build_spatial_memory(
            interaction_history=interaction_history,
            tool_traj=tool_traj,
            last_memory_state=None,
        )

        meta = self._build_meta(interaction_history, agent_profile, env_state)

        return StructuredMemoryState(working=working, episodic=episodic, spatial=spatial, meta=meta)

    def update_after_step(
        self,
        interaction_history_so_far,
        agent_profile,
        env_state_step,
        last_memory_state=None,
    ):
      
        env_state_step = env_state_step or {}

        latest_step = 0
        for ev in interaction_history_so_far:
            s = int(ev.get("step", 0) or 0)
            if s > latest_step:
                latest_step = s

        step_history = [ev for ev in interaction_history_so_far if int(ev.get("step", 0) or 0) == latest_step]

        aux_out = self.aux.process_one_step(step_history)
        compressed_list = aux_out["compressed_trajectory"]
        tool_traj_step = aux_out["tool_trajectory"]

        if compressed_list:
            compressed_current = compressed_list[0]
        else:
            compressed_current = None

        working = self._build_working_memory_incremental(
            compressed_current=compressed_current,
            agent_profile=agent_profile,
            last_memory_state=last_memory_state,
        )

        episodic = self._build_episodic_memory_incremental(
            compressed_current=compressed_current,
            tool_traj_step=tool_traj_step,
            env_state_step=env_state_step,
            last_memory_state=last_memory_state,
        )

        spatial = self._build_spatial_memory_incremental(
            step_history=step_history,
            tool_traj_step=tool_traj_step,
            last_memory_state=last_memory_state,
        )

        meta = self._build_meta(interaction_history_so_far, agent_profile, env_state_step)

        return StructuredMemoryState(working=working, episodic=episodic, spatial=spatial, meta=meta)

    def _build_meta(self, interaction_history, agent_profile, env_state):
        steps = set()
        for ev in interaction_history:
            steps.add(int(ev.get("step", 0) or 0))

        if isinstance(agent_profile, dict):
            agent_id = agent_profile.get("agent_id") or agent_profile.get("profile_id")
        else:
            agent_id = getattr(agent_profile, "agent_id", None)

        meta = {
            "agent_id": agent_id,
            "num_steps": len(steps),
            "num_events": len(interaction_history),
            "auxiliary_version": "tool_aware_v2",
            "env_state": env_state,
        }
        return meta

    def _build_working_memory(self, compressed_traj, agent_profile):
      
        if not compressed_traj:
            return WorkingMemoryState(
                current_goal="Complete the current trip safely with manageable cognitive load.",
                current_subtask="No active subtask.",
                recent_obstacles=[],
                recent_observations=[],
                cognitive_load=4.0,
                last_step=0,
            )

        latest = compressed_traj[-1]
        last_step = int(latest.get("step", 0) or 0)

        if isinstance(agent_profile, dict):
            name = agent_profile.get("name", "the traveler")
        else:
            name = getattr(agent_profile, "name", "the traveler")

        current_goal = "Ensure that {} reaches the destination safely with manageable cognitive load.".format(name)
        current_subtask = "Reflect on and complete step {} of the trip.".format(last_step)

        recent_obstacles = []
        recent_observations = []
        for item in compressed_traj[-2:]:
            summ = item.get("summary") or ""
            tags = item.get("affect_tags") or []
            if "high_load" in tags:
                recent_obstacles.append(summ)
            else:
                recent_observations.append(summ)

        cognitive_load = self._estimate_cognitive_load(latest.get("scores") or {})

        return WorkingMemoryState(
            current_goal=current_goal,
            current_subtask=current_subtask,
            recent_obstacles=recent_obstacles,
            recent_observations=recent_observations,
            cognitive_load=cognitive_load,
            last_step=last_step,
        )

    def _build_working_memory_incremental(self, compressed_current, agent_profile, last_memory_state):
       
        if not compressed_current:
            if last_memory_state and last_memory_state.working:
                return last_memory_state.working
            return WorkingMemoryState(
                current_goal="Complete the current trip safely with manageable cognitive load.",
                current_subtask="No active subtask.",
                recent_obstacles=[],
                recent_observations=[],
                cognitive_load=4.0,
                last_step=0,
            )

        step_id = int(compressed_current.get("step", 0) or 0)

        if isinstance(agent_profile, dict):
            name = agent_profile.get("name", "the traveler")
        else:
            name = getattr(agent_profile, "name", "the traveler")

        current_goal = "Ensure that {} reaches the destination safely with manageable cognitive load.".format(name)
        current_subtask = "Reflect on and complete step {} of the trip.".format(step_id)

        recent_obstacles = []
        recent_observations = []

        summ_now = compressed_current.get("summary") or ""
        tags_now = compressed_current.get("affect_tags") or []
        if "high_load" in tags_now:
            recent_obstacles.append(summ_now)
        else:
            recent_observations.append(summ_now)

        if last_memory_state and last_memory_state.episodic:
            episodes = last_memory_state.episodic.episodes
            if episodes:
                prev = episodes[-1]
                summ_prev = prev.get("summary") or ""
                tags_prev = prev.get("affect_tags") or []
                if "high_load" in tags_prev:
                    recent_obstacles.append(summ_prev)
                else:
                    recent_observations.append(summ_prev)

        cognitive_load = self._estimate_cognitive_load(compressed_current.get("scores") or {})

        return WorkingMemoryState(
            current_goal=current_goal,
            current_subtask=current_subtask,
            recent_obstacles=recent_obstacles,
            recent_observations=recent_observations,
            cognitive_load=cognitive_load,
            last_step=step_id,
        )

    def _estimate_cognitive_load(self, scores):
     
        if not scores:
            return 4.0

        overall = scores.get("overall")
        if overall is None:
            for v in scores.values():
                if isinstance(v, (int, float)):
                    overall = v
                    break

        if overall is None:
            return 4.0

        try:
            val = float(overall)
        except Exception:
            return 4.0

        if 1.0 <= val <= 7.0:
            return max(1.0, min(7.0, val))

        if 0.0 <= val <= 100.0:
            return max(1.0, min(7.0, 1.0 + (val / 100.0) * 6.0))

        return 4.0

    def _build_episodic_memory(self, compressed_traj, tool_traj, env_state, last_memory_state):
        
        episodes = []
        tools_by_step = {}
        for t in tool_traj:
            s = int(t.get("step", 0) or 0)
            tools_by_step.setdefault(s, []).append(t)

        for item in compressed_traj:
            step_id = int(item.get("step", 0) or 0)
            episode = {
                "step": step_id,
                "summary": item.get("summary"),
                "scores": item.get("scores") or {},
                "key_tools": item.get("key_tools") or [],
                "affect_tags": item.get("affect_tags") or [],
                "tool_events": tools_by_step.get(step_id, []),
                "env_context": env_state,
            }
            episodes.append(episode)

        return EpisodicMemoryState(episodes)

    def _build_episodic_memory_incremental(self, compressed_current, tool_traj_step, env_state_step, last_memory_state):
    
        if last_memory_state and last_memory_state.episodic:
            episodes = list(last_memory_state.episodic.episodes)
        else:
            episodes = []

        if compressed_current:
            step_id = int(compressed_current.get("step", 0) or 0)
            episode = {
                "step": step_id,
                "summary": compressed_current.get("summary"),
                "scores": compressed_current.get("scores") or {},
                "key_tools": compressed_current.get("key_tools") or [],
                "affect_tags": compressed_current.get("affect_tags") or [],
                "tool_events": tool_traj_step or [],
                "env_context": env_state_step or {},
            }
            episodes.append(episode)

        return EpisodicMemoryState(episodes)

    def _build_spatial_memory(self, interaction_history, tool_traj, last_memory_state):
        """
        Used in consolidate: build locations across all steps based on segment_info
        context events and tool trajectories.
        """
        base_by_step = self._extract_base_spatial_from_context(interaction_history)
        self._inject_tool_info_into_spatial(base_by_step, tool_traj)

        locations = self._build_locations_list(base_by_step)
        return SpatialCognitiveMemoryState(locations)

    def _build_spatial_memory_incremental(self, step_history, tool_traj_step, last_memory_state):
        """
        Used in update_after_step: update the spatial entry for the current step,
        keeping previous steps from the last memory state.
        """
        if last_memory_state and last_memory_state.spatial:
            old_locations = list(last_memory_state.spatial.locations)
        else:
            old_locations = []

        base_by_step = self._extract_base_spatial_from_context(step_history)
        self._inject_tool_info_into_spatial(base_by_step, tool_traj_step)
        new_locations = self._build_locations_list(base_by_step)

        new_by_step = {}
        for loc in new_locations:
            s = int(loc.get("step", 0) or 0)
            new_by_step[s] = loc

        merged = []
        used_steps = set()

        for loc in old_locations:
            s = int(loc.get("step", 0) or 0)
            if s in new_by_step:
                merged.append(new_by_step[s])
                used_steps.add(s)
            else:
                merged.append(loc)

        for s, loc in new_by_step.items():
            if s not in used_steps:
                merged.append(loc)

        merged = sorted(merged, key=lambda x: int(x.get("step", 0) or 0))

        return SpatialCognitiveMemoryState(merged)

    def _extract_base_spatial_from_context(self, interaction_history):
        base_by_step = {}
        for ev in interaction_history:
            if ev.get("type") != "context":
                continue
            md = ev.get("metadata") or {}
            if md.get("kind") != "segment_info":
                continue

            step = int(ev.get("step", 0) or 0)
            base_by_step.setdefault(step, {})
            base_by_step[step].update(
                {
                    "step": step,
                    "line": md.get("line"),
                    "start_station": md.get("start_station"),
                    "end_station": md.get("end_station"),
                    "start_time": md.get("start_time"),
                    "end_time": md.get("end_time"),
                    "notes": [],
                }
            )
        return base_by_step

    def _inject_tool_info_into_spatial(self, base_by_step, tool_traj):
        for t in tool_traj:
            name = t.get("tool_name")
            step = int(t.get("step", 0) or 0)
            result = t.get("result") or {}

            if step not in base_by_step:
                base_by_step[step] = {"step": step, "notes": []}

            slot = base_by_step[step]

            if name == "get_street_view":
                info = result if isinstance(result, dict) else {}
                crowd = info.get("crowding_level") or info.get("crowd") or info.get("load_factor")
                safety = info.get("perceived_safety")
                barrier = info.get("barrier_free") or info.get("barrier_free_score")
                signage = info.get("signage_visibility")

                if crowd is not None:
                    slot["crowding_level"] = crowd
                if safety is not None:
                    slot["perceived_safety"] = safety
                if barrier is not None:
                    slot["barrier_free"] = barrier
                if signage is not None:
                    slot["signage_visibility"] = signage

                slot["notes"].append("street_view_analyzed")

            elif name == "get_traffic":
                info = result if isinstance(result, dict) else {}
                crowd = info.get("crowd") or info.get("crowding_level") or info.get("load_factor")
                if crowd is not None and "crowding_level" not in slot:
                    slot["crowding_level"] = crowd
                slot["notes"].append("traffic_checked")

    def _build_locations_list(self, base_by_step):
        locations = []

        for step in sorted(base_by_step.keys()):
            entry = base_by_step[step]

            perceived_safety = entry.get("perceived_safety")
            crowd = entry.get("crowding_level")

            if perceived_safety is None:
                if isinstance(crow, (int, float)):
                    perceived_safety = max(0.0, min(1.0, 1.0 - float(crow)))
                else:
                    perceived_safety = 0.7

            complexity = entry.get("complexity")
            if complexity is None:
                complexity = 0.5

            familiarity = entry.get("familiarity")
            if familiarity is None:
                familiarity = 0.7 if step == 1 else 0.5

            locations.append(
                {
                    "step": step,
                    "line": entry.get("line"),
                    "start_station": entry.get("start_station"),
                    "end_station": entry.get("end_station"),
                    "familiarity": familiarity,
                    "complexity": complexity,
                    "perceived_safety": perceived_safety,
                    "barrier_free": entry.get("barrier_free"),
                    "crowding_level": entry.get("crowding_level"),
                    "signage_visibility": entry.get("signage_visibility"),
                    "notes": entry.get("notes", []),
                }
            )

        return locations
