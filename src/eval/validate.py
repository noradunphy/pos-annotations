"""Run tagger on held-out mined rows; report per-distinction accuracy."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.mapping.collapse import load_taxonomy
from src.tagging.vllm_tagger import TaggerConfig, VLLMTagger


def words_from_record(rec: dict[str, Any]) -> tuple[list[str], int]:
    if "sentence" in rec:
        w = rec["sentence"].split()
        return w, int(rec["token_index"])
    raise ValueError("Record needs sentence field")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate tagger on held-out mined rows")
    ap.add_argument("--heldout-dir", type=Path, default=Path("data/mined_examples/heldout"))
    ap.add_argument("--taxonomy", type=Path, default=None)
    ap.add_argument("--tagger-config", type=Path, default=None)
    ap.add_argument("--mock", action="store_true", help="Force mock tagger (no vLLM)")
    ap.add_argument("--out-dir", type=Path, default=Path("data/eval/runs"))
    args = ap.parse_args()

    tax = load_taxonomy(args.taxonomy)
    by_id = {d["id"]: d for d in tax["distinctions"]}
    cfg = TaggerConfig.load(args.tagger_config)
    if args.mock:
        cfg = replace(cfg, mock=True)
    tagger = VLLMTagger(cfg)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_run = args.out_dir / run_id
    out_run.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"run_id": run_id, "distinctions": {}}

    icl_dir = args.heldout_dir.parent / "icl"

    for hf in sorted(args.heldout_dir.glob("*.jsonl")):
        did = hf.stem
        if did not in by_id:
            continue
        dist = by_id[did]
        if not dist.get("enabled", True):
            continue
        icl_path = icl_dir / f"{did}.jsonl"
        correct = 0
        total = 0
        preds_path = out_run / f"preds_{did}.jsonl"
        with hf.open(encoding="utf-8") as fin, preds_path.open("w", encoding="utf-8") as fout:
            for line in fin:
                rec = json.loads(line)
                words, ti = words_from_record(rec)
                gold = rec["gold_subclass"]
                pred, _prompt = tagger.predict_label(
                    distinction=dist,
                    words=words,
                    target_index=ti,
                    icl_path=icl_path,
                )
                ok = pred == gold
                correct += int(ok)
                total += 1
                out = {**rec, "pred_subclass": pred, "correct": ok}
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        acc = correct / total if total else 0.0
        summary["distinctions"][did] = {"accuracy": acc, "n": total, "correct": correct}

    summary_path = out_run / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path.read_text())


if __name__ == "__main__":
    main()
