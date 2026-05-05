"""Render ICL prompts from Jinja2 templates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "prompts"
_GEMINI_ICL_PATH = PROMPTS_DIR / "gemini_icl.yaml"

_gemini_icl_cache: dict[str, list[dict[str, str]]] | None = None


def load_gemini_icl() -> dict[str, list[dict[str, str]]]:
    """Load handcrafted ICL examples for all distinctions (cached)."""
    global _gemini_icl_cache
    if _gemini_icl_cache is None:
        _gemini_icl_cache = yaml.safe_load(_GEMINI_ICL_PATH.read_text(encoding="utf-8")) or {}
    return _gemini_icl_cache


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


def render_distinction_messages(
    *,
    distinction: dict[str, Any],
    words: list[str],
    target_index: int,
) -> tuple[str, str]:
    """Return (system_message, user_message) for chat API calls (e.g. Gemini).

    The system message contains the task definition, expanded guidelines, and
    handcrafted ICL examples. The user message contains only the marked
    sentence and the JSON output contract. No bootstrapped ICL files are used.
    """
    icl_data = load_gemini_icl()
    did = distinction["id"]
    labels = list(distinction["labels"])
    guidelines = distinction.get("guidelines", "").strip()
    examples = icl_data.get(did, [])

    icl_lines: list[str] = []
    for ex in examples:
        icl_lines.append(f'  Sentence: "{ex["sentence"]}" → label: {ex["label"]}')
    icl_block = ""
    if icl_lines:
        icl_block = "\n\nIn-context examples (gold labels):\n" + "\n".join(icl_lines)

    system_msg = (
        "You are a careful computational linguist performing morphosyntactic annotation. "
        "Reply with a single JSON object and nothing else — no markdown fences, no explanation.\n\n"
        f"Task: {did}\n"
        f"Allowed labels: {', '.join(labels)}\n\n"
        f"Guidelines: {guidelines}"
        f"{icl_block}"
    )

    marked = mark_sentence(words, target_index)
    user_msg = (
        f'Sentence (target word in [[double brackets]]): "{marked}"\n\n'
        'Return JSON exactly in this form: {"label": "<one_of_allowed>"}'
    )

    return system_msg, user_msg


def label_json_schema(labels: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": list(labels)},
        },
        "required": ["label"],
        "additionalProperties": False,
    }
