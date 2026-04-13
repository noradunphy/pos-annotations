"""PTB -> Tier A coarse; eligible Tier-B distinction ids from taxonomy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PTB = PROJECT_ROOT / "configs" / "ptb_to_coarse.yaml"
DEFAULT_TAX = PROJECT_ROOT / "configs" / "taxonomy.yaml"


@dataclass
class CollapseResult:
    ptb_pos: str
    high_level: str | None
    eligible_distinction_ids: list[str]


def load_ptb_to_coarse(path: Path | None = None) -> dict[str, str]:
    p = path or DEFAULT_PTB
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    m = data.get("tag_to_coarse") or {}
    return {str(k): str(v) for k, v in m.items()}


def load_taxonomy(path: Path | None = None) -> dict[str, Any]:
    p = path or DEFAULT_TAX
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def enabled_distinctions(tax: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for d in tax.get("distinctions", []):
        if d.get("enabled", True):
            out.append(d)
    return out


def collapse_token(
    ptb_pos: str,
    *,
    ptb_map: dict[str, str] | None = None,
    tax: dict[str, Any] | None = None,
) -> CollapseResult:
    ptb = str(ptb_pos).strip()
    pmap = ptb_map if ptb_map is not None else load_ptb_to_coarse()
    tax = tax if tax is not None else load_taxonomy()
    high = pmap.get(ptb) or pmap.get(ptb.upper()) or "UNK"
    if high in ("UNK", "PUNCT", None):
        return CollapseResult(ptb, high if high else None, [])
    tier_a = set(tax.get("tier_a", []))
    if high not in tier_a:
        return CollapseResult(ptb, high, [])
    eligible: list[str] = []
    for d in enabled_distinctions(tax):
        if d.get("parent_high_level") == high:
            eligible.append(str(d["id"]))
    return CollapseResult(ptb, high, eligible)
