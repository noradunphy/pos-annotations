"""Render ICL prompts from Jinja2 templates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "prompts"


def mark_sentence(words: list[str], target_index: int) -> str:
    out: list[str] = []
    for i, w in enumerate(words):
        if i == target_index:
            out.append(f"[[{w}]]")
        else:
            out.append(w)
    return " ".join(out)


def load_icl_examples(path: Path | None, max_examples: int = 3) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows[:max_examples]


def render_distinction_prompt(
    *,
    distinction: dict[str, Any],
    words: list[str],
    target_index: int,
    icl_path: Path | None = None,
    max_icl: int = 3,
) -> str:
    env = Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        autoescape=select_autoescape(enabled_extensions=()),
    )
    tpl = env.get_template("distinction.jinja2")
    icl = load_icl_examples(icl_path, max_examples=max_icl) if icl_path else []
    icl_block = ""
    if icl:
        parts = []
        for ex in icl:
            sent = ex.get("sentence") or ""
            surf = ex.get("surface", "")
            lab = ex.get("gold_subclass", "")
            ti = int(ex.get("token_index", -1))
            if sent and ti >= 0:
                wds = sent.split()
                marked = mark_sentence(wds, ti)
                parts.append(f'- Sentence: "{marked}" -> label: {lab}')
            else:
                parts.append(f'- Word "{surf}" -> label: {lab}')
        icl_block = "In-context examples (gold label):\n" + "\n".join(parts) + "\n\n"
    marked = mark_sentence(words, target_index)
    return tpl.render(
        distinction_id=distinction["id"],
        labels=distinction["labels"],
        guidelines=distinction.get("guidelines", "").strip(),
        marked_sentence=icl_block + marked,
    )


def label_json_schema(labels: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": list(labels)},
        },
        "required": ["label"],
        "additionalProperties": False,
    }
