"""Annotate unseen sentences (e.g. 1K from test split) with the tagger.

Supports two backends:
  --backend vllm    (default) Local vLLM inference with bootstrapped ICL prompts.
  --backend gemini  Google Gemini API with handcrafted ICL examples baked into
                    the system prompt. Requires GEMINI_API_KEY in the environment.
                    Use --gemini-model to select 'flash', 'frontier', or a literal
                    model name (default: frontier).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.mapping.collapse import collapse_token, load_ptb_to_coarse, load_taxonomy
from src.tagging.vllm_tagger import TaggerConfig, VLLMTagger


def load_tokens(path: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == split:
                rows.append(r)
    return rows


def build_sentences(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    by: dict[str, dict[int, str]] = {}
    for r in rows:
        sid = r["sentence_id"]
        by.setdefault(sid, {})[int(r["token_index"])] = r["word"]
    return {sid: [d[i] for i in sorted(d)] for sid, d in by.items()}


def load_excluded_sentence_ids(mined_root: Path) -> set[str]:
    s: set[str] = set()
    for sub in ("icl", "heldout"):
        d = mined_root / sub
        if not d.is_dir():
            continue
        for fp in d.glob("*.jsonl"):
            with fp.open(encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    s.add(r["sentence_id"])
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="Annotate unseen corpus slice")
    ap.add_argument("--tokens", type=Path, default=Path("data/interim/tokens.jsonl"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-sentences", type=int, default=1000)
    ap.add_argument("--subset-size", type=int, default=None, help="If set with num-sentences=1000, compare subset (e.g. 500)")
    ap.add_argument("--mined-root", type=Path, default=Path("data/mined_examples"))
    ap.add_argument("--taxonomy", type=Path, default=None)
    ap.add_argument("--ptb-map", type=Path, default=None)
    ap.add_argument("--tagger-config", type=Path, default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/annotations/corpus_1k.jsonl"))
    # Gemini backend options
    ap.add_argument(
        "--backend",
        choices=["vllm", "gemini"],
        default="vllm",
        help="Inference backend (default: vllm)",
    )
    ap.add_argument(
        "--gemini-model",
        default="frontier",
        help="Gemini model alias ('flash' or 'frontier') or literal model name "
             "(used when --backend gemini, default: frontier)",
    )
    ap.add_argument(
        "--gemini-cfg",
        type=Path,
        default=None,
        help="Path to configs/gemini.yaml (default: auto-detected)",
    )
    args = ap.parse_args()

    tax = load_taxonomy(args.taxonomy)
    pmap = load_ptb_to_coarse(args.ptb_map)
    rows = load_tokens(args.tokens, args.split)
    if not rows:
        print("No rows for split", args.split, file=sys.stderr)
        sys.exit(1)
    ptb_index = {(r["sentence_id"], int(r["token_index"])): r["ptb_pos"] for r in rows}
    sents = build_sentences(rows)
    excluded = load_excluded_sentence_ids(args.mined_root)
    candidates = [sid for sid in sents if sid not in excluded]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    pick = candidates[: args.num_sentences]

    if args.backend == "gemini":
        from src.tagging.gemini_tagger import GeminiConfig, GeminiTagger

        gcfg = GeminiConfig.load(args.gemini_cfg)
        tagger: Any = GeminiTagger(model=args.gemini_model, cfg=gcfg)
        print(f"Backend: Gemini ({gcfg.resolve_model(args.gemini_model)})", file=sys.stderr)
    else:
        cfg = TaggerConfig.load(args.tagger_config)
        if args.mock:
            cfg = replace(cfg, mock=True)
        tagger = VLLMTagger(cfg)

    distinctions = [d for d in tax["distinctions"] if d.get("enabled", True)]
    by_id = {d["id"]: d for d in distinctions}
    icl_root = args.mined_root / "icl"

    # Collect all work items grouped by distinction for batched inference
    from collections import defaultdict
    work: list[dict[str, Any]] = []
    batch_by_did: dict[str, list[int]] = defaultdict(list)

    for sid in pick:
        words = sents[sid]
        for ti, w in enumerate(words):
            ptb = ptb_index.get((sid, ti))
            if ptb is None:
                continue
            c = collapse_token(ptb, ptb_map=pmap, tax=tax)
            for did in c.eligible_distinction_ids:
                idx = len(work)
                work.append({
                    "sentence_id": sid,
                    "token_index": ti,
                    "word": w,
                    "ptb_pos": ptb,
                    "high_level": c.high_level,
                    "distinction_id": did,
                    "words": words,
                })
                batch_by_did[did].append(idx)

    print(f"Collected {len(work)} tagging requests across {len(batch_by_did)} distinctions", file=sys.stderr)
    preds: list[str] = [""] * len(work)

    for did, idxs in sorted(batch_by_did.items()):
        dist = by_id[did]
        # icl_path is only used by the vLLM backend; GeminiTagger ignores it
        icl_path = icl_root / f"{did}.jsonl" if args.backend == "vllm" else None
        items = [(work[i]["words"], work[i]["token_index"]) for i in idxs]
        CHUNK = 256
        for start in range(0, len(items), CHUNK):
            chunk_items = items[start : start + CHUNK]
            chunk_idxs = idxs[start : start + CHUNK]
            results = tagger.predict_batch(
                distinction=dist, items=chunk_items, icl_path=icl_path,
            )
            for i, lab in zip(chunk_idxs, results):
                preds[i] = lab
        print(f"  {did}: {len(idxs)} items done", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fout:
        for i, item in enumerate(work):
            rec = {
                "sentence_id": item["sentence_id"],
                "token_index": item["token_index"],
                "word": item["word"],
                "ptb_pos": item["ptb_pos"],
                "high_level": item["high_level"],
                "distinction_id": item["distinction_id"],
                "pred_subclass": preds[i],
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    meta = {
        "num_sentences": len(pick),
        "split": args.split,
        "subset_size_note": args.subset_size,
        "output": str(args.out),
    }
    Path("data/annotations").mkdir(parents=True, exist_ok=True)
    (Path("data/annotations") / "last_annotate_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Wrote annotations for {len(pick)} sentences to {args.out}")


if __name__ == "__main__":
    main()
