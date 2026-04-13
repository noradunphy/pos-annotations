"""Shared token pool loading for mining and batch export."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

from src.mapping.collapse import collapse_token


def load_tokens(path: Path, splits: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") in splits:
                rows.append(r)
    return rows


def build_sentences(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    by: dict[str, dict[int, str]] = {}
    for r in rows:
        sid = r["sentence_id"]
        by.setdefault(sid, {})[int(r["token_index"])] = r["word"]
    out: dict[str, list[str]] = {}
    for sid, mp in by.items():
        out[sid] = [mp[i] for i in sorted(mp)]
    return out


def enrich_rows(
    rows: list[dict[str, Any]],
    pmap: dict[str, str],
    tax: dict[str, Any],
) -> list[tuple[str, int, str, str, str]]:
    """(sentence_id, token_index, high_level, ptb_pos, genre_prefix) for subclass-eligible tokens."""
    out: list[tuple[str, int, str, str, str]] = []
    for r in rows:
        c = collapse_token(r["ptb_pos"], ptb_map=pmap, tax=tax)
        if not c.high_level or not c.eligible_distinction_ids:
            continue
        g = str(r.get("genre_prefix", "unk"))
        out.append(
            (
                r["sentence_id"],
                int(r["token_index"]),
                c.high_level,
                str(r["ptb_pos"]),
                g,
            )
        )
    return out


def pool_for_parent(
    enriched: list[tuple[str, int, str, str, str]],
    parent_high_level: str,
) -> list[tuple[str, int, str]]:
    """Pool tuples (sentence_id, token_index, high_level) for one Tier-A parent."""
    return [(sid, idx, high) for sid, idx, high, _, _ in enriched if high == parent_high_level]


def stratified_sample_indices(
    enriched_for_parent: list[tuple[str, int, str, str, str]],
    *,
    total: int,
    rng: random.Random,
) -> list[int]:
    """Return indices into enriched_for_parent in stratified round-robin order by genre_prefix."""
    if not enriched_for_parent:
        return []
    buckets: dict[str, list[int]] = {}
    for i, row in enumerate(enriched_for_parent):
        g = row[4]
        buckets.setdefault(g, []).append(i)
    for g in buckets:
        rng.shuffle(buckets[g])
    genres = list(buckets.keys())
    rng.shuffle(genres)
    order: list[int] = []
    ptr = {g: 0 for g in genres}
    while len(order) < total and any(ptr[g] < len(buckets[g]) for g in genres):
        for g in genres:
            if len(order) >= total:
                break
            bi = buckets[g]
            p = ptr[g]
            if p < len(bi):
                order.append(bi[p])
                ptr[g] += 1
    if len(order) < total:
        remainder = [i for i in range(len(enriched_for_parent)) if i not in set(order)]
        rng.shuffle(remainder)
        for i in remainder:
            if len(order) >= total:
                break
            order.append(i)
    return order[:total]
