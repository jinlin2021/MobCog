# elder_assessment.py
# -*- coding: utf-8 -*-

import json
from typing import Dict, Any, Tuple, List, Optional
import os

import openai

def load_scale(path: str) -> Dict[str, Any]:
    """Load a scale configuration from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_personas(path: str = "data/elder_personas_P1-P8.json") -> List[Dict[str, Any]]:
    """Load elder personas (P1–P8) from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        personas = json.load(f)
    if not isinstance(personas, list):
        raise ValueError("elder_personas_P1-P8.json must contain a list of persona objects.")
    return personas



def build_persona_context(persona: Dict[str, Any]) -> str:

    name = persona.get("name", "Unknown name")
    age = persona.get("age", "unknown age")
    gender = persona.get("gender", "unknown gender")
    profile_label = persona.get("profile_label", "")
    cognitive = persona.get("cognitive_level", "")
    mobility = persona.get("mobility", "")
    affect = persona.get("affect_profile", "")

    lines = [
        f"Name: {name}",
        f"Age: {age}",
        f"Gender: {gender}",
        f"Profile: {profile_label}",
        f"Cognitive status: {cognitive}",
        f"Mobility: {mobility}",
        f"Mood / affect: {affect}",
    ]
    return "\n".join(lines)


def ask_agent_with_llm(persona: Dict[str, Any], question: str, model) -> str:

    persona_context = build_persona_context(persona)

    system_prompt = (
        "You are an older adult with the given health profile. "
        "Answer in the first person, realistically and briefly, "
        "staying fully in character. Do not mention being an AI or taking a test. "
        "For multiple-choice or Yes/No questions, clearly choose exactly ONE option "
        "(e.g., 'A.' or 'Yes') in your answer."
    )

    user_prompt = (
        "Below is your persona description. You must stay in character when answering.\n\n"
        f"{persona_context}\n\n"
        "Now answer the following question as this older adult.\n\n"
        f"Question:\n{question}"
    )

    vllm_endpoint = os.environ.get("VLLM_ENDPOINT", "http://localhost:8000/v1")
    client = openai.OpenAI(
        base_url=vllm_endpoint,
        api_key="EMPTY",
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )

    answer = response.choices[0].message.content.strip()
    return answer


def score_item(item: Dict[str, Any], answer: str) -> Tuple[int, bool]:

    strategy = item.get("scoring_strategy", "manual")
    max_score = int(item.get("max_score", 0))

    if strategy == "manual":
        hint = item.get("scoring_hint", "")
        print(f"[INFO] Item {item['id']} requires manual scoring. Hint: {hint}")
        return 0, True

    if strategy == "choice_map":
        options = item.get("options", {})
        normalized = answer.upper().strip()
        first_char = normalized[0] if normalized else ""
        score = int(options.get(first_char, 0))
        return score, False

    if strategy == "yes_no":
        depressed_answer = item.get("depressed_answer", "Yes").strip()
        normalized = answer.replace(" ", "").lower()
        target = depressed_answer.replace(" ", "").lower()
        if target in normalized:
            return max_score, False
        else:
            return 0, False

    print(f"[WARN] Unknown scoring_strategy={strategy} for item {item['id']}. Treating as manual.")
    return 0, True



def run_single_scale(
    scale_cfg: Dict[str, Any],
    persona: Dict[str, Any],
    model,
    scale_tag: str = ""
) -> Dict[str, Any]:

    meta = scale_cfg.get("meta", {})
    items = scale_cfg.get("items", [])
    total_score = 0
    max_score = int(meta.get("total_score", 0))
    item_results = []
    manual_items = []

    scale_name = meta.get("scale_name", "Unknown scale")
    description = meta.get("description", "")

    header = f"{scale_name}" if not scale_tag else f"{scale_name} ({scale_tag})"
    print(f"\n========== Start scale: {header} ==========")
    if description:
        print(description)

    for item in items:
        qid = item["id"]
        prompt = item.get("prompt", "")

        print("\n----------------------------------------")
        print(f"[ITEM] {qid}")
        print(f"Prompt: {prompt}")

        answer = ask_agent_with_llm(persona, prompt, model=model)
        print(f"[AGENT ANSWER] {answer}")

        score, needs_manual = score_item(item, answer)
        total_score += score

        result = {
            "id": qid,
            "prompt": prompt,
            "answer": answer,
            "score": score,
            "max_score": item.get("max_score", 0),
            "needs_manual_review": needs_manual,
        }
        item_results.append(result)
        if needs_manual:
            manual_items.append(result)

        print(f"[RESULT] {qid}: {score}/{item.get('max_score', 0)}")

    if max_score > 0 and total_score > max_score:
        total_score = max_score

    print(f"\n>>> {header} total score: {total_score} / {max_score}\n")

    return {
        "scale_name": scale_name,
        "tag": scale_tag,
        "total_score": total_score,
        "max_score": max_score,
        "item_results": item_results,
        "manual_items": manual_items,
    }



def run_elder_assessment(
    persona: Dict[str, Any],
    mmse_path: Optional[str] = "MMSE.json",
    adl_path: Optional[str] = "ADL.json",
    gds_path: Optional[str] = "GDS.json",
    run_mmse: bool = True,
    run_adl: bool = True,
    run_gds: bool = True,
    tag: str = "",
    model=None,
) -> Dict[str, Any]:
    """
    Run MMSE-like, ADL-like, and GDS-like scales for one persona.
    """
    results = {
        "persona_id": persona.get("id"),
        "profile_id": persona.get("profile_id"),
        "tag": tag,
        "mmse_like": None,
        "adl_like": None,
        "gds_like": None,
    }

    if run_mmse and mmse_path is not None:
        mmse_cfg = load_scale(mmse_path)
        results["mmse_like"] = run_single_scale(mmse_cfg, persona, model=model, scale_tag=tag)

    if run_adl and adl_path is not None:
        adl_cfg = load_scale(adl_path)
        results["adl_like"] = run_single_scale(adl_cfg, persona, model=model, scale_tag=tag)

    if run_gds and gds_path is not None:
        gds_cfg = load_scale(gds_path)
        results["gds_like"] = run_single_scale(gds_cfg, persona, model=model, scale_tag=tag)

    return results



if __name__ == "__main__":
    personas = load_personas("data/elder_personas_P1-P8.json")
    print(f"Loaded {len(personas)} personas from data/elder_personas_P1-P8.json.\n")

    example_persona = personas[0]
    print("Using persona:")
    print(json.dumps(example_persona, ensure_ascii=False, indent=2))

    all_results = run_elder_assessment(
        persona=example_persona,
        mmse_path="MMSE.json",
        adl_path="ADL.json",
        gds_path="GDS.json",
        run_mmse=True,
        run_adl=True,
        run_gds=True,
        tag="pre-task",
        model="xxx",  # 这里写你自己的模型名字
    )

    with open("results_pre_task_persona.json", "w", encoding="utf-8") as f_out:
        json.dump(all_results, f_out, ensure_ascii=False, indent=2)
