"""Mine ICL + held-out silver examples per distinction_id."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.mapping.collapse import load_ptb_to_coarse, load_taxonomy
from src.mining.backends import MinerBackend, OpenAIMiner, StubMiner
from src.mining.pool_utils import build_sentences, enrich_rows, load_tokens


def pick_backend(name: str) -> MinerBackend:
    if name == "stub":
        return StubMiner()
    if name == "openai":
        return OpenAIMiner()
    raise ValueError(f"Unknown miner backend: {name}")


def mine_distinction(
    distinction: dict[str, Any],
    pool: list[tuple[str, int, str]],
    sentences: dict[str, list[str]],
    backend: MinerBackend,
    rng: random.Random,
    icl_per_label: int,
    heldout_per_label: int,
    max_passes: int,
) -> tuple[list[dict], list[dict]]:
    labels = list(distinction["labels"])
    icl_counts: dict[str, int] = defaultdict(int)
    hd_counts: dict[str, int] = defaultdict(int)

    def done() -> bool:
        return all(
            icl_counts[l] >= icl_per_label and hd_counts[l] >= heldout_per_label for l in labels
        )

    icl_rows: list[dict] = []
    hd_rows: list[dict] = []
    passes = 0

    while not done() and passes < max_passes:
        passes += 1
        shuffled = pool[:]
        rng.shuffle(shuffled)
        moved = False
        for sid, tok_idx, high in shuffled:
            if done():
                break
            words = sentences.get(sid)
            if not words or tok_idx >= len(words):
                continue
            dist_payload = dict(distinction)
            lab = backend.label(
                words=words,
                target_index=tok_idx,
                distinction=dist_payload,
                rng=rng,
            )
            if lab not in labels:
                continue
            if hd_counts[lab] < heldout_per_label:
                split = "heldout"
            elif icl_counts[lab] < icl_per_label:
                split = "icl"
            else:
                continue
            rec = {
                "sentence_id": sid,
                "token_index": tok_idx,
                "surface": words[tok_idx],
                "sentence": " ".join(words),
                "high_level": high,
                "gold_subclass": lab,
                "distinction_id": distinction["id"],
                "provenance": {"miner": backend.__class__.__name__, "pass": passes},
            }
            if split == "heldout":
                hd_rows.append(rec)
                hd_counts[lab] += 1
            else:
                icl_rows.append(rec)
                icl_counts[lab] += 1
            moved = True
        if not moved:
            break
    return icl_rows, hd_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Mine ICL and held-out examples")
    ap.add_argument("--tokens", type=Path, default=Path("data/interim/tokens.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/mined_examples"))
    ap.add_argument("--taxonomy", type=Path, default=None)
    ap.add_argument("--ptb-map", type=Path, default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "validation"])
    ap.add_argument("--miner", choices=["stub", "openai"], default="stub")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--icl-per-label", type=int, default=6)
    ap.add_argument("--heldout-per-label", type=int, default=3)
    ap.add_argument("--max-passes", type=int, default=500)
    ap.add_argument(
        "--distinction-ids",
        nargs="*",
        default=None,
        help="If set, only mine these ids (default: all enabled in taxonomy)",
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

    backend = pick_backend(args.miner)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    icl_dir = args.out_dir / "icl"
    hd_dir = args.out_dir / "heldout"
    icl_dir.mkdir(parents=True, exist_ok=True)
    hd_dir.mkdir(parents=True, exist_ok=True)

    for d in distinctions:
        did = d["id"]
        parent = d["parent_high_level"]
        pool = [(sid, idx, high) for sid, idx, high, _, _ in enriched if high == parent]
        if len(pool) < 10:
            print(f"skip {did}: tiny pool ({len(pool)})", file=sys.stderr)
            continue
        icl_rows, hd_rows = mine_distinction(
            d,
            pool,
            sentences,
            backend,
            rng,
            args.icl_per_label,
            args.heldout_per_label,
            args.max_passes,
        )
        with (icl_dir / f"{did}.jsonl").open("w", encoding="utf-8") as f:
            for row in icl_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (hd_dir / f"{did}.jsonl").open("w", encoding="utf-8") as f:
            for row in hd_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{did}: icl={len(icl_rows)} heldout={len(hd_rows)}")


if __name__ == "__main__":
    main()
