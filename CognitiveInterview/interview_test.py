# -*- coding: utf-8 -*-

import json
import os
from typing import Dict, Any, List

import openai

from prompt_script_generator import (
    PromptScriptGenerator,
    AgentProfile,
    TaskContext,
)
from hierarchical_memory_manager import HierarchicalMemoryManager
from memory_guided_response_composer import (
    MemoryGuidedResponseComposer,
    StructuredMemoryState,
)


def load_records(path: str) -> List[Dict[str, Any]]:
    """
    Support two formats:
    - .json  : file contains a dict or list[dict], always converted to a list
    - .jsonl : each line is one sample, all lines are loaded
    """
    if path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        else:
            return [data]


def build_agent_from_json(agent_cfg: Dict[str, Any]) -> AgentProfile:
    """
    Map the 'agent' field in simulation_output to an AgentProfile.
    """
    return AgentProfile(
        agent_id=agent_cfg.get("agent_id", "unknown"),
        health_condition=agent_cfg.get(
            "health_condition",
            agent_cfg.get("cognitive_level", "unknown"),
        ),
        task_completion_status=agent_cfg.get("task_completion_status", "unknown"),
        memory_consistency=agent_cfg.get("memory_consistency", "unknown"),
        extra_attributes={
            "profile_label": agent_cfg.get("profile_label"),
            "name": agent_cfg.get("name"),
            "age": agent_cfg.get("age"),
            "gender": agent_cfg.get("gender"),
            "mobility": agent_cfg.get("mobility"),
            "affect_profile": agent_cfg.get("affect_profile"),
            "expected_mmse_range": agent_cfg.get("extra_attributes", {}).get(
                "expected_mmse_range"
            )
            or agent_cfg.get("expected_mmse_range"),
            "expected_adl_range": agent_cfg.get("extra_attributes", {}).get(
                "expected_adl_range"
            )
            or agent_cfg.get("expected_adl_range"),
            "expected_gds_range": agent_cfg.get("extra_attributes", {}).get(
                "expected_gds_range"
            )
            or agent_cfg.get("expected_gds_range"),
            "description": agent_cfg.get("extra_attributes", {}).get("description")
            or agent_cfg.get("description"),
            "profile_id": agent_cfg.get("extra_attributes", {}).get("profile_id")
            or agent_cfg.get("profile_id"),
        },
    )


def build_task_from_json(task_cfg: Dict[str, Any]) -> TaskContext:
    """
    Map the 'task' field in simulation_output to a TaskContext.
    """
    return TaskContext(
        route_description=task_cfg.get("route_description", ""),
        environment_description=task_cfg.get("environment_description", ""),
        difficulty_level=task_cfg.get("difficulty_level", "medium"),
        time_pressure=task_cfg.get("time_pressure", "medium"),
    )



def build_llm_client(model: str):

    vllm_endpoint = os.environ.get("VLLM_ENDPOINT", "http://localhost:8000/v1")
    client = openai.OpenAI(
        base_url=vllm_endpoint,
        api_key="EMPTY",
    )
    return client, model


def make_interview_llm_fn(model: str):
    """
    Return a closure llm_fn(prompt) -> str,
    internally using the client above to call a local Qwen / GPT model.
    """
    client, model_name = build_llm_client(model)

    def llm_fn(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an elderly bus traveler. Answer in the first person "
                        "as this older adult, based on the provided memory summary."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        return resp.choices[0].message.content

    return llm_fn


def run_interview_for_sample(
    sample: Dict[str, Any],
    memory_manager: HierarchicalMemoryManager,
    prompt_gen: PromptScriptGenerator,
    composer: MemoryGuidedResponseComposer,
) -> Dict[str, Any]:

    agent_cfg = sample["agent"]
    task_cfg = sample["task"]
    env_state = sample.get("env_state", {})
    interaction_history = sample["interaction_history"]

    # 1) Agent / Task
    agent = build_agent_from_json(agent_cfg)
    task = build_task_from_json(task_cfg)

    # 2) history -> structured memory M_t
    M_t: StructuredMemoryState = memory_manager.consolidate(
        interaction_history=interaction_history,
        agent_profile=agent,
        env_state=env_state,
        last_memory_state=None,
    )


    questions = prompt_gen.generate_questions(agent, task)

    qa_list = []
    for q in questions:
        answer, prompt_debug = composer.answer_question(q, M_t, return_prompt=True)
        qa_list.append(
            {
                "question": q,
                "answer": answer,
                "debug_prompt": prompt_debug,  # can be dropped later if too long
            }
        )


    out = {
        "pattern_type": sample.get("pattern_type", "Unknown"),
        "trip_index_within_pattern": sample.get("trip_index_within_pattern"),
        "agent_id": agent.agent_id,
        "agent_extra": agent.extra_attributes,
        "task_brief": {
            "route_description": task.route_description,
            "environment_description": task.environment_description,
            "difficulty_level": task.difficulty_level,
            "time_pressure": task.time_pressure,
        },
       
        "memory_state": {
            "working": M_t.working.__dict__,
            "episodic": M_t.episodic.__dict__,
            "spatial": M_t.spatial.__dict__,
            "meta": M_t.meta,
        },
        "qa": qa_list,
    }
    return out


def run_batch_interviews(
    input_path: str,
    output_dir: str,
    model: str,
    max_per_pattern: int = 20,
) -> None:
    os.makedirs(output_dir, exist_ok=True)


    all_samples = load_records(input_path)
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for s in all_samples:
        pat = s.get("pattern_type", "Unknown")
        buckets.setdefault(pat, []).append(s)

    memory_manager = HierarchicalMemoryManager()
    prompt_gen = PromptScriptGenerator(
        random_seed=0,
        max_questions_per_dimension=1,
    )
    llm_fn = make_interview_llm_fn(model)
    composer = MemoryGuidedResponseComposer(llm_fn=llm_fn)

    for pattern_name, samples in buckets.items():
        slug = pattern_name.lower().replace(" ", "_")
        out_path = os.path.join(
            output_dir, f"{slug}_interviews.jsonl"
        )

        selected = samples[:max_per_pattern]

        print(
            f"\n=== Pattern: {pattern_name} | "
            f"{len(selected)} / {len(samples)} trips will be interviewed ==="
        )

        with open(out_path, "w", encoding="utf-8") as f_out:
            for idx, s in enumerate(selected):
                print(f"  - interviewing trip {idx + 1}/{len(selected)}")
                interview_record = run_interview_for_sample(
                    s, memory_manager, prompt_gen, composer
                )
                f_out.write(json.dumps(interview_record, ensure_ascii=False) + "\n")

        print(f"Saved interviews for pattern [{pattern_name}] to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="results/simulation_outputs.jsonl",
        help="Combined simulation output file produced by run_simulation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/interviews",
        help="Directory to save interview results for different patterns",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/xx/Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model name used ",
    )
    parser.add_argument(
        "--max_per_pattern",
        type=int,
        default=10,
        help="Maximum number of samples to interview per travel pattern",
    )
    args = parser.parse_args()

    run_batch_interviews(
        input_path=args.input_file,
        output_dir=args.output_dir,
        model=args.model,
        max_per_pattern=args.max_per_pattern,
    )
