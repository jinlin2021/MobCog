import os
import json
import glob
import argparse
from typing import List, Dict, Any

import openai


def run_immediate_recall_probe(persona, memory_state, trip, segment_index, model):
    """
    Run an immediate route recall probe right after a given segment.

    Returns a dict:
        {
            "segment_index": int,
            "line": str,
            "date": str,
            "gt_stops": [str, ...],
            "predicted_stops_raw": str,
            "prompt": str,
        }
    """
    segments = trip.get("segments", [])
    if not segments or segment_index < 0 or segment_index >= len(segments):
        return {
            "segment_index": segment_index,
            "gt_stops": [],
            "predicted_stops_raw": "",
            "prompt": "",
        }

    seg = segments[segment_index]
    line = seg.get("line", "")
    start_station = seg.get("start_station", "")
    end_station = seg.get("end_station", "")
    trip_date = trip.get("date", "")

    # Here we use a simple [start, end] as the ground-truth sequence
    gt_stops = []
    if start_station:
        gt_stops.append(start_station)
    if end_station and end_station != start_station:
        gt_stops.append(end_station)

    try:
        mem_dict = memory_state.to_dict()
    except Exception:
        mem_dict = {}

    # Truncate to avoid overly long prompts
    mem_text = json.dumps(mem_dict, ensure_ascii=False)[:1500]

    base = getattr(persona, "base_profile", {}) or {}
    name = base.get("name", "Unknown name")
    age = base.get("age", "70")
    gender = base.get("gender", "unknown gender")
    # These may not exist in base_profile; fall back to persona-level fields if needed
    health = base.get("cognitive_level", "") or base.get("health_condition", "")
    mobility = base.get("mobility", "")
    affect = base.get("affect_profile", "")

    persona_brief = (
        f"Name: {name}\n"
        f"Age: {age}\n"
        f"Gender: {gender}\n"
        f"Cognitive / health status: {health}\n"
        f"Mobility: {mobility}\n"
        f"Mood / affect: {affect}\n"
    )

    system_prompt = (
        "You are an older adult who has just completed one segment of a bus trip. "
        "You must always answer in the first person (starting with “I ...”). "
        "Based on your own memory, you need to recall all bus stops passed during "
        "this specific segment, in the correct order. "
        "Only output the stop names, connected by ' > '. "
        "Do not add any explanation, comments, or extra text."
    )

    user_prompt = (
        "Below is your persona profile:\n"
        f"{persona_brief}\n\n"
        "You have just completed the following bus segment:\n"
        f"- Date: {trip_date}\n"
        f"- Line: {line}\n"
        f"- Start stop: {start_station}\n"
        f"- End stop: {end_station}\n\n"
        "Below is a structured summary of your current internal memory state:\n"
        f"{mem_text}\n\n"
        "Now, immediately after completing this segment, please recall the names "
        "of all stops you passed during THIS segment, in the exact order.\n\n"
        "Output format (very important, this is only an example):\n"
        "StopA > StopB > StopC\n"
        "Do not add any explanation or extra sentences.\n"
    )

    vllm_endpoint = os.environ.get("VLLM_ENDPOINT", "http://localhost:8000/v1")
    client = openai.OpenAI(
        base_url=vllm_endpoint,
        api_key="EMPTY",  # or your actual key if needed
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=128,
    )

    answer = resp.choices[0].message.content.strip()

    out = {
        "segment_index": segment_index,
        "line": line,
        "date": trip_date,
        "gt_stops": gt_stops,
        "predicted_stops_raw": answer,
        "prompt": user_prompt,
    }
    return out


# =========================
# Metrics computation part
# =========================

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_simulation_records(sim_file):
    if sim_file.endswith(".jsonl"):
        return load_jsonl(sim_file)
    else:
        with open(sim_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        else:
            return [data]


def load_interview_records(interview_dir):
    all_records = []
    pattern = os.path.join(interview_dir, "*.jsonl")
    for path in glob.glob(pattern):
        recs = load_jsonl(path)
        all_records.extend(recs)
    return all_records


def parse_stop_sequence(text):
    if not text:
        return []

    text = text.replace("->", ">")
    parts = text.split(">")

    stops = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = p.strip(" .,:;，。；！!？?\"'")
        if not p:
            continue
        stops.append(p)

    return stops


def overlap_ratio(gt_seq, pred_seq):
    if not gt_seq:
        return None
    gt_set = set(gt_seq)
    pred_set = set(pred_seq)
    if not gt_set:
        return None
    inter = gt_set & pred_set
    return float(len(inter)) / float(len(gt_set))


def build_full_route_from_trip(raw_trip):
    segments = raw_trip.get("segments") or []
    if not segments:
        return []

    route = []
    for seg in segments:
        start = seg.get("start_station")
        if start:
            route.append(start)

    last_end = segments[-1].get("end_station")
    if last_end:
        route.append(last_end)

    deduped = []
    for s in route:
        if not deduped or deduped[-1] != s:
            deduped.append(s)
    return deduped


def extract_task_stats_from_sim(sim_record):
    task = sim_record.get("task", {})
    agent = sim_record.get("agent", {})

    # Task success flag
    success_flag = None
    if "success" in task:
        success_flag = bool(task["success"])
    elif "task_success" in sim_record:
        success_flag = bool(sim_record["task_success"])
    else:
        status = agent.get("task_completion_status", "").lower()
        if status in ["success", "succeeded", "完成", "成功"]:
            success_flag = True
        elif status in ["fail", "failed", "失败"]:
            success_flag = False

    if success_flag is None:
        success_flag = True

    # Duration (minutes)
    raw_trip = task.get("raw_trip", {})
    duration = raw_trip.get("total_duration_min")
    if duration is None:
        duration = 0.0
    else:
        duration = float(duration)

    # ORC stats (object recognition)
    orc_correct = sim_record.get("orc_correct")
    orc_total = sim_record.get("orc_total")

    if orc_correct is not None and orc_total is not None:
        try:
            orc_correct = int(orc_correct)
            orc_total = int(orc_total)
        except Exception:
            orc_correct = None
            orc_total = None

    # IRRA (immediate recall) from simulation_output["immediate_recall"]
    irra = None
    imm_gt_seq = []
    imm_rec_seq = []

    imm = sim_record.get("immediate_recall") or {}
    gt_seq = imm.get("gt_sequence") or []
    rec_seq = imm.get("recalled_sequence") or []

    if gt_seq:
        imm_gt_seq = list(gt_seq)
        imm_rec_seq = list(rec_seq) if rec_seq else []
        irra = overlap_ratio(imm_gt_seq, imm_rec_seq)

    # Full ground-truth route for RAC
    full_route_gt = build_full_route_from_trip(raw_trip)

    pattern_type = sim_record.get("pattern_type", "Unknown")
    trip_index = sim_record.get("trip_index_within_pattern")

    return {
        "success": success_flag,
        "duration": duration,
        "orc_correct": orc_correct,
        "orc_total": orc_total,
        "irra": irra,
        "pattern_type": pattern_type,
        "trip_index": trip_index,
        "full_route_gt": full_route_gt,
        "imm_gt_seq": imm_gt_seq,
    }


def build_interview_index(interview_records):
    index = {}
    for r in interview_records:
        pat = r.get("pattern_type", "Unknown")
        idx = r.get("trip_index_within_pattern")
        if idx is None:
            continue
        key = (pat, idx)
        index[key] = r
    return index


def extract_drra_rac_from_interview(interview_record):
    qa_list = interview_record.get("qa") or []
    drra_seq = []
    rac_seq = []

    for qa in qa_list:
        q = qa.get("question") or ""
        a = qa.get("answer") or ""
        if q.startswith("[DRRA]"):
            drra_seq = parse_stop_sequence(a)
        elif q.startswith("[RAC]"):
            rac_seq = parse_stop_sequence(a)

    return {
        "drra_seq": drra_seq,
        "rac_seq": rac_seq,
    }


def compute_metrics(sim_file, interview_dir):
    sim_records = load_simulation_records(sim_file)
    interview_records = load_interview_records(interview_dir)
    interview_index = build_interview_index(interview_records)

    task_stats = []

    for sim in sim_records:
        stats = extract_task_stats_from_sim(sim)

        key = (stats["pattern_type"], stats["trip_index"])
        interview = interview_index.get(key)

        drra = None
        rac = None

        if interview is not None:
            ir = extract_drra_rac_from_interview(interview)

            drra_seq = ir["drra_seq"]
            if stats["imm_gt_seq"]:
                drra = overlap_ratio(stats["imm_gt_seq"], drra_seq)

            rac_seq = ir["rac_seq"]
            if stats["full_route_gt"]:
                rac = overlap_ratio(stats["full_route_gt"], rac_seq)

        stats["drra"] = drra
        stats["rac"] = rac

        task_stats.append(stats)

    if not task_stats:
        print("No tasks found. Please check sim_file and interview_dir.")
        return None

    n = len(task_stats)

    # TSR
    succ_sum = sum(1 for s in task_stats if s["success"])
    tsr = succ_sum / float(n)

    # TCT
    tct = sum(s["duration"] for s in task_stats) / float(n)

    # ORC (global correct / total)
    total_correct = 0
    total_obj = 0
    for s in task_stats:
        c = s.get("orc_correct")
        t = s.get("orc_total")
        if c is None or t is None:
            continue
        total_correct += c
        total_obj += t

    if total_obj > 0:
        orc = total_correct / float(total_obj)
    else:
        orc = None

    # IRRA / DRRA / RAC
    irra_vals = [s["irra"] for s in task_stats if s["irra"] is not None]
    drra_vals = [s["drra"] for s in task_stats if s["drra"] is not None]
    rac_vals = [s["rac"] for s in task_stats if s["rac"] is not None]

    irra_mean = sum(irra_vals) / float(len(irra_vals)) if irra_vals else None
    drra_mean = sum(drra_vals) / float(len(drra_vals)) if drra_vals else None
    rac_mean = sum(rac_vals) / float(len(rac_vals)) if rac_vals else None

    # STFR
    if irra_mean is not None and drra_mean is not None and irra_mean > 0:
        stfr = (irra_mean - drra_mean) / irra_mean
    else:
        stfr = None

    metrics = {
        "num_tasks": n,
        "TSR": tsr,
        "TCT": tct,
        "ORC": orc,
        "IRRA": irra_mean,
        "DRRA": drra_mean,
        "RAC": rac_mean,
        "STFR": stfr,
    }

    return {
        "metrics": metrics,
        "per_task": task_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute CityCog-Agent metrics (TSR, TCT, ORC, IRRA, DRRA, RAC, STFR)."
    )
    parser.add_argument(
        "--sim_file",
        default="results/simulation_outputs.jsonl",
        help="Simulation result file (JSONL or JSON).",
    )
    parser.add_argument(
        "--interview_dir",
        default="results/interviews",
        help="Directory containing interview results (*.jsonl).",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Optional: path to save metrics and per-task stats as a JSON file.",
    )

    args = parser.parse_args()

    res = compute_metrics(args.sim_file, args.interview_dir)
    if res is None:
        return

    metrics = res["metrics"]

    print("\n========== Summary Metrics ==========")
    print(f"Number of tasks: {metrics['num_tasks']}")
    print(f"TSR : {metrics['TSR']:.4f}")
    print(f"TCT : {metrics['TCT']:.4f}")
    if metrics["ORC"] is not None:
        print(f"ORC : {metrics['ORC']:.4f}")
    else:
        print("ORC : N/A (no orc_correct/orc_total)")

    if metrics["IRRA"] is not None:
        print(f"IRRA: {metrics['IRRA']:.4f}")
    else:
        print("IRRA: N/A")
    if metrics["DRRA"] is not None:
        print(f"DRRA: {metrics['DRRA']:.4f}")
    else:
        print("DRRA: N/A")
    if metrics["RAC"] is not None:
        print(f"RAC : {metrics['RAC']:.4f}")
    else:
        print("RAC : N/A")
    if metrics["STFR"] is not None:
        print(f"STFR: {metrics['STFR']:.4f}")
    else:
        print("STFR: N/A")

    if args.output_file:
        out_obj = {
            "metrics": metrics,
            "per_task": res["per_task"],
        }
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        print(f"\nMetrics saved to {args.output_file}")


if __name__ == "__main__":
    main()
