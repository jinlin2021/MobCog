import os
import io
import random
import base64
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO
import openai
import torch
import yaml
import json

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def load_config_and_prompts(prompts_dir="prompts", locale=None):
    base = Path(prompts_dir)
    cfg_path = base / "manifest.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    locale = locale or cfg.get("default_locale", "zh")
    prompt_paths = cfg["prompts"][locale]

    system_prompt = (base / prompt_paths["system"]).read_text(encoding="utf-8").strip()
    user_prompt = (base / prompt_paths["user"]).read_text(encoding="utf-8").strip()

    return cfg, system_prompt, user_prompt


def image_to_base(image_path):
    """Convert an image to a base64-encoded string."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_string}"


def images_to_base64(image_paths):
    """Convert multiple images to a list of base64-encoded strings."""
    return [image_to_base(path) for path in image_paths if os.path.exists(path)]


def to_data_url(paths, mime="image/jpeg"):

    if isinstance(paths, (str, Path)):
        paths = [paths]

    urls = []
    for p in paths:
        p = str(p)
        if p.startswith(("http://", "https://", "data:")):
            urls.append(p)
            continue
        data = Path(p).read_bytes()
        b64 = base64.b64encode(data).decode("utf-8")
        urls.append(f"data:{mime};base64,{b64}")
    return urls


def path_to_resized_data_url(paths, max_side=768, quality=80, mime="image/jpeg"):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    fmt = {"image/png": "PNG", "image/webp": "WEBP"}.get(mime.lower(), "JPEG")
    out = []
    for p in map(str, paths):
        if p.startswith(("http://", "https://", "data:")):
            out.append(p)
            continue
        img = Image.open(p).convert("RGB")
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.BILINEAR)
        buf = io.BytesIO()
        kwargs = {"quality": quality} if fmt in ("JPEG", "WEBP") else {}
        if fmt == "JPEG":
            kwargs["optimize"] = True
        img.save(buf, format=fmt, **kwargs)
        out.append(f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}")
    return out


class ToolBox:
    """
    Toolbox:
      - get_weather(date): cache weather results per date
      - get_traffic(line, time): simple peak/off-peak crowding simulation
      - get_street_view(station): YOLO + SAM-based street-view analysis
    """

    def __init__(
        self,
        station_streetview_map: Dict[str, List[str]],
        image_base_path: str,
        model: str = None,
        sam_checkpoint: str = None,
        sam_device: str = "cuda",
    ):
        self.station_streetview_map = station_streetview_map
        self.image_base_path = image_base_path
        self.model = model

        with open("./gpt4o_manual _abeling_ocr.json", "r", encoding="utf-8") as f:
            self.ocr_gt = json.load(f)


        self.weather_options = [
            {"weather": "sunny", "temperature": 28, "description": "Clear and sunny."},
            {"weather": "cloudy", "temperature": 24, "description": "Cloudy with mild temperature."},
            {"weather": "light rain", "temperature": 20, "description": "Light rain with slightly wet roads."},
            {"weather": "overcast", "temperature": 22, "description": "Overcast with relatively low light."},
            {"weather": "windy", "temperature": 21, "description": "Noticeably windy."},
            {"weather": "snow", "temperature": -1, "description": "Snowy, cold and slippery roads."},
        ]
        self.traffic_options = [
            {"crowd": "empty", "description": "Very few passengers and plenty of space."},
            {"crowd": "moderate", "description": "Some passengers, mild crowding."},
            {"crowd": "crowded", "description": "Standing room only but still manageable."},
            {"crowd": "packed", "description": "Extremely crowded and difficult to board or alight."},
        ]

        self._weather_cache: Dict[str, Dict[str, Any]] = {}

        self._yolo_det = None
        self._sam_predictor = None
        self._sam_checkpoint = sam_checkpoint
        self._sam_device = sam_device

    def _load_yolo_det(self):
        if self._yolo_det is not None:
            return self._yolo_det
        if YOLO is None:
            return None
        self._yolo_det = YOLO("yolov8n.pt")
        return self._yolo_det

    def _load_sam(self):
        if self._sam_predictor is not None:
            return self._sam_predictor
        if sam_model_registry is None:
            return None
        if not self._sam_checkpoint or not os.path.exists(self._sam_checkpoint):
            return None
        sam = sam_model_registry["vit_b"](checkpoint=self._sam_checkpoint)
        sam.to(self._sam_device)
        self._sam_predictor = SamAutomaticMaskGenerator(sam)
        return self._sam_predictor

    def get_weather(self, date: str) -> Dict[str, Any]:
   
        if date in self._weather_cache:
            return self._weather_cache[date]

        info = random.choice(self.weather_options).copy()
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
            m = dt.month
            if 3 <= m <= 5:
                info["temperature"] += random.randint(-2, 2)
            elif 6 <= m <= 8:
                info["temperature"] += random.randint(1, 4)
            elif 9 <= m <= 11:
                info["temperature"] += random.randint(-2, 2)
            else:
                info["temperature"] -= random.randint(3, 8)
        except Exception:
            pass

        self._weather_cache[date] = info
        return info

    def set_fixed_weather(self, date: str, weather_info: Dict[str, Any]) -> None:
   
        self._weather_cache[date] = weather_info

    def get_traffic(self, line: str, time: str) -> Dict[str, Any]:
        """
        Roughly simulate crowding level based on time of day.
        """
        info = random.choice(self.traffic_options).copy()
        try:
            t = datetime.strptime(time, "%H:%M")
            hour = t.hour
            if (7 <= hour <= 9) or (17 <= hour <= 19):
                if random.random() < 0.7:
                    info = random.choice(self.traffic_options[2:])
            elif (11 <= hour <= 13) or (hour >= 21):
                if random.random() < 0.7:
                    info = random.choice(self.traffic_options[:2])
        except Exception:
            pass

        info["line"] = line
        info["time"] = time
        return info

    def _analyze_single_image(self, image_path: str) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            return {"error": f"Image not found: {image_path}"}

        img = np.array(Image.open(image_path).convert("RGB"))
        h, w = img.shape[:2]

        det_model = self._load_yolo_det()
        sam_predictor = self._load_sam()

        objects: List[str] = []
        comfort_score = 5.0  # 1–10

        if det_model is not None:
            results = det_model(image_path)
            det_objs = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                name = results[0].names[cls_id]
                det_objs.append(name)
            objects = list(set(det_objs))

            if "bench" in det_objs:
                comfort_score += 1.0
            if "person" in det_objs and len([o for o in det_objs if o == "person"]) > 5:
                comfort_score -= 0.5

        sam_mask_ratio = None
        if sam_predictor is not None:
            masks = sam_predictor.generate(img)
            if masks:
                union_mask = np.zeros((h, w), dtype=bool)
                for m in masks:
                    union_mask |= m["segmentation"]
                sam_mask_ratio = float(union_mask.sum() / (h * w))
                if sam_mask_ratio > 0.6:
                    comfort_score -= 0.5

        comfort_score = max(1.0, min(10.0, comfort_score))

        return {
            "image_path": image_path,
            "objects": objects,
            "comfort_score": comfort_score,
            "sam_mask_ratio": sam_mask_ratio,
        }

    def get_street_view(self, station: str, use_llm_vision: bool = False) -> Dict[str, Any]:
   
        photo_ids = self.station_streetview_map.get(station, [])
        if not photo_ids:
            return {
                "error": f"No street-view images found for station {station}",
                "image_b64_list": [],
            }

        n_sample = min(3, len(photo_ids))
        sample_ids = random.sample(photo_ids, n_sample)
        image_paths = [os.path.join(self.image_base_path, pid) for pid in sample_ids]
        image_paths = [p for p in image_paths if os.path.exists(p)]

        if not image_paths:
            return {"error": "No valid images found.", "image_b64_list": []}

        b64_list = images_to_base64(image_paths)

        analyses = []
        comfort_scores = []
        for path in image_paths:
            a = self._analyze_single_image(path)
            analyses.append(a)
            if "comfort_score" in a:
                comfort_scores.append(a["comfort_score"])

        avg_comfort = sum(comfort_scores) / len(comfort_scores) if comfort_scores else 5.0

        combined_objects = set()
        for a in analyses:
            for o in a.get("objects") or []:
                combined_objects.add(o)

        if combined_objects:
            description = f"The street view around {station} includes: " + ", ".join(combined_objects) + "."
        else:
            description = (
                f"The street view around {station} contains some objects that are hard to identify."
            )

        if avg_comfort >= 7:
            description += " Overall, the environment makes me feel comfortable and relaxed."
        elif avg_comfort >= 5:
            description += " Overall, the environment feels acceptable, but I still need to pay some attention."
        else:
            description += " Overall, the environment makes me feel a bit uneasy, so I need to be more cautious."

        orc_correct = 0
        orc_total = 0

        for img_id in sample_ids:
            gt_info = self.ocr_gt.get(img_id, {})
            gt_cats = gt_info.get("categories") or []
            gt_set = set(gt_cats)

            if not gt_set:
                continue

            pred_info = self.ocr_pred.get(img_id, {})
            pred_cats = pred_info.get("categories") or []
            pred_set = set(pred_cats)

            orc_correct += len(gt_set & pred_set)
            orc_total += len(gt_set)

        return {
            "station": station,
            "image_b64_list": b64_list,
            "photo_ids": sample_ids,
            "description": description,
            "comfort_score": avg_comfort,
            "analyses": analyses,
            "orc_correct": orc_correct,
            "orc_total": orc_total,
        }

    def get_tools(self):
        return {
            "get_weather": self.get_weather,
            "get_traffic": self.get_traffic,
            "get_street_view": self.get_street_view,
        }
