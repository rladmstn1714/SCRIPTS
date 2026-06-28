#!/usr/bin/env python3
"""Build a small examples JSON for the SCRIPTS project website."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "web" / "data" / "examples.json"
SLOTS_PER_PAGE = 4

SPLITS = {
    "en": [
        ROOT / "dataset" / "english_combined_577_matched.csv",
        ROOT / "dataset" / "english_combined.csv",
    ],
    "ko": [
        ROOT / "dataset" / "korean_combined_matched.csv",
        ROOT / "dataset" / "korean_combined.csv",
    ],
}

EXAMPLE_SCENE_IDS = {
    "en": ["scene317", "scene332", "scene485", "scene744"],
    "ko": ["89", "239", "381", "285"],
}


def resolve_csv(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No dataset file found. Tried: {', '.join(str(p) for p in candidates)}")


def parse_labels(raw: str) -> list[str]:
    value = (raw or "").strip()
    if not value:
        return []
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                seen: set[str] = set()
                labels: list[str] = []
                for item in parsed:
                    label = str(item).strip()
                    if not label or label.lower() == "nan" or label in seen:
                        continue
                    seen.add(label)
                    labels.append(label)
                return labels
        except (SyntaxError, ValueError):
            pass
    return [value]


def load_rows(lang: str) -> dict[str, dict[str, str]]:
    csv_path = resolve_csv(SPLITS[lang])
    with csv_path.open(encoding="utf-8", newline="") as handle:
        return {row["scene_id"].strip(): row for row in csv.DictReader(handle)}


def build_example(row: dict[str, str]) -> dict:
    return {
        "scene_id": row.get("scene_id", "").strip(),
        "dialogue": row.get("dialogue", "").strip(),
        "highly_likely": parse_labels(row.get("relation_high_probable_gold", "")),
        "unlikely": parse_labels(row.get("relation_impossible_gold", "")),
    }


def main() -> None:
    payload = {"slots_per_page": SLOTS_PER_PAGE, "examples": {}}

    for lang, scene_ids in EXAMPLE_SCENE_IDS.items():
        rows = load_rows(lang)
        examples = []
        for scene_id in scene_ids:
            if scene_id not in rows:
                raise KeyError(f"Example scene {scene_id!r} not found in {lang} dataset")
            examples.append(build_example(rows[scene_id]))
        payload["examples"][lang] = examples

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
