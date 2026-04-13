"""Merge subagent `batch_*.labeled.json` files into `data/mined_examples/{icl,heldout}/*.jsonl`."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.mapping.collapse import load_taxonomy


def load_input_batch(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return data


def load_labeled_batch(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return data


def route_splits(
    rows: list[dict[str, Any]],
    labels: list[str],
    icl_per_label: int,
    heldout_per_label: int,
) -> tuple[list[dict], list[dict]]:
    """Greedy routing in row order: fill held-out per label first, then ICL per label."""
    icl_counts: dict[str, int] = defaultdict(int)
    hd_counts: dict[str, int] = defaultdict(int)
    icl_out: list[dict] = []
    hd_out: list[dict] = []
    for rec in rows:
        lab = rec.get("gold_subclass")
        if lab not in labels:
            continue
        if hd_counts[lab] < heldout_per_label:
            hd_out.append(rec)
            hd_counts[lab] += 1
        elif icl_counts[lab] < icl_per_label:
            icl_out.append(rec)
            icl_counts[lab] += 1
        else:
            continue
    return icl_out, hd_out


def merge_one_distinction_dir(
    in_dir: Path,
    distinction: dict[str, Any],
    icl_dir: Path,
    hd_dir: Path,
    *,
    icl_per_label: int,
    heldout_per_label: int,
    dry_run: bool,
) -> tuple[int, int, list[str]]:
    did = distinction["id"]
    labels = list(distinction["labels"])
    errors: list[str] = []
    merged: list[dict[str, Any]] = []
    labeled_files = sorted(in_dir.glob("batch_*.labeled.json"))
    if not labeled_files:
        errors.append(f"{in_dir}: no batch_*.labeled.json files")
        return 0, 0, errors

    for lf in labeled_files:
        m = re.match(r"batch_(\d+)\.labeled\.json$", lf.name)
        if not m:
            continue
        base = f"batch_{int(m.group(1)):03d}.json"
        inp_path = in_dir / base
        if not inp_path.is_file():
            errors.append(f"missing input {inp_path} for {lf.name}")
            continue
        inputs = load_input_batch(inp_path)
        labeled = load_labeled_batch(lf)
        in_by_id = {str(x["example_id"]): x for x in inputs}
        if len(labeled) != len(inputs):
            errors.append(f"{lf.name}: length mismatch labeled={len(labeled)} input={len(inputs)}")
        for i, lab_row in enumerate(labeled):
            eid = str(lab_row.get("example_id", ""))
            if eid not in in_by_id:
                errors.append(f"{lf.name}: unknown example_id {eid}")
                continue
            src = in_by_id[eid]
            gold = lab_row.get("gold_subclass")
            if gold not in labels:
                errors.append(f"{lf.name}: invalid gold_subclass {gold!r} for {eid}")
                continue
            out = {
                "sentence_id": src["sentence_id"],
                "token_index": int(src["token_index"]),
                "surface": src["surface"],
                "sentence": src["sentence"],
                "high_level": src["high_level"],
                "gold_subclass": gold,
                "distinction_id": did,
                "provenance": {
                    "source": "subagent_batch",
                    "batch_file": lf.name,
                    "input_batch": base,
                    "notes": lab_row.get("notes", ""),
                },
            }
            merged.append(out)

    icl_rows, hd_rows = route_splits(merged, labels, icl_per_label, heldout_per_label)
    if not dry_run:
        icl_dir.mkdir(parents=True, exist_ok=True)
        hd_dir.mkdir(parents=True, exist_ok=True)
        with (icl_dir / f"{did}.jsonl").open("w", encoding="utf-8") as f:
            for row in icl_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (hd_dir / f"{did}.jsonl").open("w", encoding="utf-8") as f:
            for row in hd_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(icl_rows), len(hd_rows), errors


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge subagent labeled batches into mined_examples")
    ap.add_argument(
        "--in-dir",
        type=Path,
        default=None,
        help="One distinction folder under data/mining_batches/<id>",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Process every subdirectory of --batches-root that contains labeled batches",
    )
    ap.add_argument("--batches-root", type=Path, default=Path("data/mining_batches"))
    ap.add_argument("--taxonomy", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("data/mined_examples"))
    ap.add_argument("--icl-per-label", type=int, default=6)
    ap.add_argument("--heldout-per-label", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tax = load_taxonomy(args.taxonomy)
    by_id = {d["id"]: d for d in tax["distinctions"]}
    icl_dir = args.out_dir / "icl"
    hd_dir = args.out_dir / "heldout"

    dirs: list[Path] = []
    if args.all:
        if not args.batches_root.is_dir():
            print(f"No batches root: {args.batches_root}", file=sys.stderr)
            sys.exit(1)
        for sub in sorted(args.batches_root.iterdir()):
            if sub.is_dir() and any(sub.glob("batch_*.labeled.json")):
                dirs.append(sub)
    elif args.in_dir:
        dirs = [args.in_dir]
    else:
        print("Provide --in-dir or --all", file=sys.stderr)
        sys.exit(1)

    for dpath in dirs:
        did = dpath.name
        if did not in by_id:
            print(f"skip unknown distinction folder: {did}", file=sys.stderr)
            continue
        dist = by_id[did]
        if not dist.get("enabled", True):
            continue
        n_ic, n_hd, errs = merge_one_distinction_dir(
            dpath,
            dist,
            icl_dir,
            hd_dir,
            icl_per_label=args.icl_per_label,
            heldout_per_label=args.heldout_per_label,
            dry_run=args.dry_run,
        )
        for e in errs:
            print(e, file=sys.stderr)
        mode = "dry-run" if args.dry_run else "wrote"
        print(f"{did}: {mode} icl={n_ic} heldout={n_hd}")


if __name__ == "__main__":
    main()
