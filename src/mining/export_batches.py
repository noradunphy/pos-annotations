"""Export per-distinction mining prompts + JSON batches for subagent annotation."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.mapping.collapse import load_ptb_to_coarse, load_taxonomy
from src.mining.pool_utils import (
    build_sentences,
    enrich_rows,
    load_tokens,
    stratified_sample_indices,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_MINING = PROJECT_ROOT / "prompts" / "mining"


def render_mining_prompt(distinction: dict[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(PROMPTS_MINING)),
        autoescape=select_autoescape(enabled_extensions=()),
    )
    tpl = env.get_template("distinction_mine.md.jinja2")
    return tpl.render(
        distinction_id=distinction["id"],
        parent_high_level=distinction["parent_high_level"],
        labels=distinction["labels"],
        guidelines=str(distinction.get("guidelines", "")).strip(),
    )


def example_record(
    row: tuple[str, int, str, str, str],
    distinction: dict[str, Any],
    sentences: dict[str, list[str]],
) -> dict[str, Any] | None:
    sid, tok_idx, high, _ptb, _genre = row
    words = sentences.get(sid)
    if not words or tok_idx >= len(words):
        return None
    eid = f"{sid}#{tok_idx}"
    return {
        "example_id": eid,
        "sentence_id": sid,
        "token_index": tok_idx,
        "sentence": " ".join(words),
        "surface": words[tok_idx],
        "high_level": high,
        "distinction_id": distinction["id"],
        "labels": list(distinction["labels"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export mining batches for subagents")
    ap.add_argument("--tokens", type=Path, default=Path("data/interim/tokens.jsonl"))
    ap.add_argument("--taxonomy", type=Path, default=None)
    ap.add_argument("--ptb-map", type=Path, default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "validation"])
    ap.add_argument("--out-root", type=Path, default=Path("data/mining_batches"))
    ap.add_argument("--total-examples", type=int, default=100, help="Max examples sampled per distinction")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--distinction-ids",
        nargs="*",
        default=None,
        help="If set, only export these distinction ids",
    )
    args = ap.parse_args()
    rng = random.Random(args.seed)
    tax = load_taxonomy(args.taxonomy)
    pmap = load_ptb_to_coarse(args.ptb_map)
    rows = load_tokens(args.tokens, set(args.splits))
    if not rows:
        print("No rows loaded; run ingest first.", file=sys.stderr)
        sys.exit(1)
    sentences = build_sentences(rows)
    enriched = enrich_rows(rows, pmap, tax)
    distinctions = [d for d in tax["distinctions"] if d.get("enabled", True)]
    if args.distinction_ids:
        wanted = set(args.distinction_ids)
        distinctions = [d for d in distinctions if d["id"] in wanted]

    args.out_root.mkdir(parents=True, exist_ok=True)

    for d in distinctions:
        did = d["id"]
        parent = d["parent_high_level"]
        pool_rows = [t for t in enriched if t[2] == parent]
        if len(pool_rows) < 5:
            print(f"skip {did}: tiny pool ({len(pool_rows)})", file=sys.stderr)
            continue
        n = min(args.total_examples, len(pool_rows))
        idx_order = stratified_sample_indices(pool_rows, total=n, rng=rng)
        selected = [pool_rows[i] for i in idx_order]
        out_dir = args.out_root / did
        out_dir.mkdir(parents=True, exist_ok=True)
        prompt_md = render_mining_prompt(d)
        (out_dir / "MINING_PROMPT.md").write_text(prompt_md, encoding="utf-8")

        examples: list[dict[str, Any]] = []
        for row in selected:
            rec = example_record(row, d, sentences)
            if rec:
                examples.append(rec)
        n_batches = max(1, math.ceil(len(examples) / args.batch_size))
        for bi in range(n_batches):
            chunk = examples[bi * args.batch_size : (bi + 1) * args.batch_size]
            path = out_dir / f"batch_{bi:03d}.json"
            path.write_text(json.dumps(chunk, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{did}: wrote {len(examples)} examples in {n_batches} batches -> {out_dir}")


if __name__ == "__main__":
    main()
