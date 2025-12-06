import os
import json
import argparse
import random
from datetime import datetime
import traceback

from tools import ToolBox
from CoT import CoTReasoner
from persona import Persona
# from config.multi_prompt_engine import build_prompt
from hierarchical_memory_manager import HierarchicalMemoryManager
from elder_assessment import run_elder_assessment
from evaluation import run_immediate_recall_probe


def load_personas(filepath):
    """Load elder persona profiles."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_trips(filepath):
    """Load trip data."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_streetview_map(filepath):
    """Load station–image mapping."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


class Trajectory:

    def __init__(self, trip):
        self.trip = trip
        self.segments = trip.get("segments", [])

    def get_context(self):
        return self.segments

    def get_segment(self, step):
        if 0 <= step < len(self.segments):
            return self.segments[step]
        return None


def build_agent_block(persona: Persona) -> dict:

    base = persona.base_profile

    agent_block = {
        "agent_id": base.get("profile_id", base.get("id", "unknown")),
        "profile_label": base.get("profile_label"),
        "name": base.get("name"),
        "age": base.get("age"),
        "gender": base.get("gender"),
        "health_condition": base.get("cognitive_level") or base.get("health_condition"),
        "mobility": base.get("mobility"),
        "affect_profile": base.get("affect_profile"),
        "task_completion_status": "success",
        "memory_consistency": "unknown",
        "extra_attributes": {
            "expected_mmse_range": base.get("expected_mmse_range"),
            "expected_adl_range": base.get("expected_adl_range"),
            "expected_gds_range": base.get("expected_gds_range"),
            "description": base.get("description"),
            "profile_id": base.get("profile_id"),
        },
    }
    return agent_block


def build_task_block(trip: dict) -> dict:
   
    segments = trip.get("segments", [])
    if segments:
        first = segments[0]
        last = segments[-1]
        if len(segments) == 1:
            route_desc = (
                f"Single-bus trip on line {first.get('line', '')} "
                f"from {first.get('start_station', '')} to {first.get('end_station', '')} "
                f"on {trip.get('date', '')}."
            )
        else:
            route_desc = (
                f"Multi-segment bus trip with {len(segments)} segments "
                f"starting from {first.get('start_station', '')} "
                f"and ending at {last.get('end_station', '')} "
                f"on {trip.get('date', '')}."
            )
    else:
        route_desc = "Unknown trip."

    env_desc = "Bus trip simulated without a global environment summary yet."

    total_duration = trip.get("total_duration_min", 0)
    if len(segments) == 1 and total_duration <= 30:
        difficulty = "easy"
    elif len(segments) <= 2 and total_duration <= 60:
        difficulty = "medium"
    else:
        difficulty = "hard"

    if total_duration <= 20:
        time_pressure = "low"
    elif total_duration <= 45:
        time_pressure = "medium"
    else:
        time_pressure = "high"

    task_block = {
        "route_description": route_desc,
        "environment_description": env_desc,
        "difficulty_level": difficulty,
        "time_pressure": time_pressure,
        "raw_trip": trip,
    }
    return task_block


def build_env_state(trip: dict) -> dict:

    segments = trip.get("segments", [])
    if segments:
        start_time_str = segments[0].get("start_time")
        try:
            dt = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            dt = None
    else:
        dt = None

    if dt is not None:
        day_of_week = dt.strftime("%A")
        hour = dt.hour
        if 6 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 18:
            time_of_day = "afternoon"
        elif 18 <= hour < 22:
            time_of_day = "evening"
        else:
            time_of_day = "night"
    else:
        day_of_week = "unknown"
        time_of_day = "unknown"

    env_state = {
        "day_of_week": day_of_week,
        "time_of_day": time_of_day,
        "weather": "unknown",
        "temperature_celsius": None,
        "avg_crowding_level": "unknown",
        "max_delay_min": None,
        "num_incidents": 0,
    }
    return env_state


def flatten_histories(all_step_histories):

    interaction_history = []
    for step_history in all_step_histories:
        interaction_history.extend(step_history)
    return interaction_history


def classify_trip_pattern(trip: dict) -> str:

    if "pattern_type" in trip:
        return trip["pattern_type"]

    segments = trip.get("segments") or []
    n = len(segments)
    if n <= 1:
        return "Single ride"
    elif n == 2:
        return "One transfer"
    else:
        return "Multiple transfers"


def save_results(results, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def run_simulation(args):

    personas = load_personas(args.persona_file)
    trips = load_trips(args.trips_file)
    streetview_map = load_streetview_map(args.streetview_file)

    toolbox = ToolBox(
        streetview_map,
        image_base_path=args.image_base_path,
        model=args.model,
    )
    cot_reasoner = CoTReasoner(
        tools=toolbox.get_tools(),
        api_key=args.openai_api_key,
        model=args.model,
    )
    memory_manager = HierarchicalMemoryManager()

    if args.persona_id:
        candidates = [
            p for p in personas
            if str(p.get("profile_id")) == str(args.persona_id)
        ]
        persona_data = random.choice(candidates or personas)
    else:
        persona_data = random.choice(personas)

    persona = Persona(persona_data)
    agent_block = build_agent_block(persona)

    persona_json = persona.base_profile
    pre_assessment = run_elder_assessment(
        persona=persona_json,
        tag="pre-task",
        model=args.model,
    )

    buckets = {
        "Single ride": [],
        "One transfer": [],
        "Multiple transfers": [],
    }
    for trip in trips:
        pat = classify_trip_pattern(trip)
        if pat in buckets:
            buckets[pat].append(trip)

    max_per_pattern = 100

    all_outputs = []

    for pattern_name, trip_list in buckets.items():
        if not trip_list:
            continue

        selected_trips = trip_list[:max_per_pattern]
        print(f"\n====== Pattern: {pattern_name}, {len(selected_trips)} trips will be simulated ======")

        for trip_idx, trip in enumerate(selected_trips):
            print(f"\n------ {pattern_name} trip {trip_idx + 1}/{len(selected_trips)} ------")

            trajectory = Trajectory(trip)

            task_block = build_task_block(trip)
            task_block["pattern_type"] = pattern_name

            env_state_trip = build_env_state(trip)
            env_state_trip["pattern_type"] = pattern_name
            trip_orc_correct = 0
            trip_orc_total = 0

            memory_state = None
            interaction_history = []
            all_step_histories = []
            step_results_all = []

            num_segments = len(trajectory.segments)

            probed_step = num_segments - 1

            for step_idx in range(num_segments):
                current_segment = trajectory.get_segment(step_idx)
                print(
                    f"Step {step_idx + 1}/{num_segments}: "
                    f"{current_segment.get('line')} | "
                    f"{current_segment.get('start_station')} -> {current_segment.get('end_station')}"
                )

                result = cot_reasoner.run_stepwise(
                    persona=persona,
                    step=step_idx,
                    trajectory=current_segment,
                )
                trip_orc_correct += result.get("orc_correct", 0)
                trip_orc_total += result.get("orc_total", 0)

                step_results = result.get("step_results", [])
                step_results_all.extend(step_results)

                step_history = result.get("history", [])
                all_step_histories.append(step_history)
                interaction_history.extend(step_history)

                env_state_step = {
                    "segment_index": step_idx,
                    "line": current_segment.get("line"),
                    "start_station": current_segment.get("start_station"),
                    "end_station": current_segment.get("end_station"),
                    "start_time": current_segment.get("start_time"),
                    "end_time": current_segment.get("end_time"),
                    "trip_date": trip.get("date"),
                    "pattern_type": pattern_name,
                }

                memory_state = memory_manager.update_after_step(
                    interaction_history_so_far=interaction_history,
                    agent_profile=agent_block,
                    env_state_step=env_state_step,
                    last_memory_state=memory_state,
                )

                if step_idx == probed_step:
                    imm_record = run_immediate_recall_probe(
                        persona=persona,
                        memory_state=memory_state,
                        trip=trip,
                        segment_index=step_idx,
                        model=args.model,
                    )

            persona.memory_state = memory_state

            memory_dict = memory_state.to_dict() if memory_state is not None else None
            interaction_history_flat = flatten_histories(all_step_histories)

            simulation_output = {
                "agent": agent_block,
                "task": task_block,
                "env_state": env_state_trip,
                "interaction_history": interaction_history_flat,
                "memory_state": memory_dict,
                "step_results": step_results_all,
                "pattern_type": pattern_name,
                "trip_index_within_pattern": trip_idx,
                "orc_correct": trip_orc_correct,
                "orc_total": trip_orc_total,
                "immediate_recall": imm_record,
            }

            all_outputs.append(simulation_output)

        post_results = run_elder_assessment(
            persona=persona,
            tag="post-task",
            model=args.model,
        )

    save_results(all_outputs, args.output_file)
    print(
        f"\nAll simulations finished, total {len(all_outputs)} records, "
        f"results saved to {args.output_file}"
    )

    if isinstance(persona, Persona):
        persona_json = persona.base_profile
    else:
        persona_json = persona

    assessment = run_elder_assessment(
        persona=persona_json,
        tag="post-task",
        model=args.model,
    )

    assess_path = os.path.join(
        os.path.dirname(args.output_file),
        "elder_assessment_post_task.json",
    )
    os.makedirs(os.path.dirname(assess_path), exist_ok=True)
    with open(assess_path, "w", encoding="utf-8") as f:
        json.dump(assessment, f, ensure_ascii=False, indent=2)

    print(f"Post-task cognitive assessment saved to {assess_path}")


def main():
    """Main entry: parse CLI arguments and run the simulation."""
    parser = argparse.ArgumentParser(description="Elder cognitive travel simulation system")

    parser.add_argument(
        "--persona_file",
        default="/config/personas/elder_personas_P1-P8.json",
        help="Path to the elder persona profile file.",
    )
    parser.add_argument(
        "--trips_file",
        default="data/trip_data.json",
        help="Path to the trip data file.",
    )
    parser.add_argument(
        "--streetview_file",
        default="/data/station_sv_map_nearest10.json",
        help="Path to the station–street-view mapping file.",
    )
    parser.add_argument(
        "--output_file",
        default="results/simulation_outputs.jsonl",
        help="Path to the output results file.",
    )
    parser.add_argument(
        "--profile_id",
        default=None,
        type=str,
        help="Specific elder profile ID; if not provided, a random persona is selected.",
    )
    parser.add_argument(
        "--openai_api_key",
        default=None,
        help="OpenAI API key; if not provided, a local model will be used.",
    )
    parser.add_argument(
        "--model",
        default="/Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model name to use (vLLM endpoint model).",
    )
    parser.add_argument(
        "--vllm_endpoint",
        default="http://localhost:8002/v1",
        help="vLLM service endpoint.",
    )
    parser.add_argument(
        "--image_base_path",
        default="/data/imgs",
        help="Base path to images.",
    )

    args = parser.parse_args()

    os.environ["VLLM_ENDPOINT"] = args.vllm_endpoint

    run_simulation(args)


if __name__ == "__main__":
    main()
