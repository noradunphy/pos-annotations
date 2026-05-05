"""Inspect trial_100 predictions: confusion matrices and error examples.

Usage:
    python -m src.eval.inspect_trial --model frontier
    python -m src.eval.inspect_trial --model flash
    python -m src.eval.inspect_trial --model frontier --distinction verb_transitivity
    python -m src.eval.inspect_trial --model frontier --errors-only
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "data" / "trial" / "results"


def load_preds(model_tag: str, results_dir: Path) -> list[dict]:
    path = results_dir / f"preds_{model_tag}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No preds file found at {path}")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def confusion_matrix(rows: list[dict]) -> dict[tuple[str, str], int]:
    """Returns {(gold, pred): count}."""
    cm: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        cm[(r["gold_subclass"], r["pred_subclass"])] += 1
    return dict(cm)


def print_confusion_matrix(did: str, rows: list[dict]) -> None:
    labels = sorted({r["gold_subclass"] for r in rows} | {r["pred_subclass"] for r in rows})
    cm = confusion_matrix(rows)
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    acc = correct / n if n else 0.0

    print(f"\n{'─'*60}")
    print(f"  {did}   acc={acc:.3f}  ({correct}/{n})")
    print(f"{'─'*60}")

    # Header row
    col_w = max(len(lb) for lb in labels) + 2
    row_label_w = max(len(lb) for lb in labels) + 2
    header = " " * (row_label_w + 2) + "".join(f"  {lb:<{col_w}}" for lb in labels)
    print(f"  {'gold \\ pred':<{row_label_w}}  " + "  ".join(f"{lb:<{col_w}}" for lb in labels))
    print(f"  {'':─<{row_label_w}}  " + "  ".join("─" * col_w for _ in labels))
    for gold in labels:
        row_vals = []
        for pred in labels:
            count = cm.get((gold, pred), 0)
            cell = f"{count:>{col_w}}" if count > 0 else " " * col_w
            # Mark diagonal (correct) with a dot
            if gold == pred and count > 0:
                cell = f"[{count}]".rjust(col_w)
            row_vals.append(cell)
        print(f"  {gold:<{row_label_w}}  " + "  ".join(row_vals))


def print_errors(did: str, rows: list[dict], max_examples: int = 5) -> None:
    errors = [r for r in rows if not r["correct"]]
    if not errors:
        print("  (no errors)")
        return

    # Group by (gold, pred) to show representative examples per error type
    by_type: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in errors:
        by_type[(r["gold_subclass"], r["pred_subclass"])].append(r)

    shown = 0
    for (gold, pred), group in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"\n  gold={gold} → pred={pred}  ({len(group)} errors)")
        for r in group[:max_examples]:
            # Highlight the target token in the sentence
            sentence = r["sentence"]
            token = r["token"]
            tok_idx = r["token_index"]
            words = [t for t in sentence.split()]
            if tok_idx < len(words):
                words[tok_idx] = f"[[{words[tok_idx]}]]"
            highlighted = " ".join(words)
            print(f"    ex{r['example_id']}  \"{highlighted}\"")
        shown += len(group)

    remaining = len(errors) - shown
    if remaining > 0:
        print(f"\n  ... {remaining} more errors not shown")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect trial_100 predictions")
    ap.add_argument("--model", default="frontier", help="Model alias (flash or frontier)")
    ap.add_argument(
        "--distinction", default=None,
        help="Only show this distinction (e.g. verb_transitivity). Default: all with errors."
    )
    ap.add_argument(
        "--errors-only", action="store_true",
        help="Skip confusion matrix, only show error examples."
    )
    ap.add_argument(
        "--max-examples", type=int, default=5,
        help="Max error examples to show per (gold, pred) pair."
    )
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()

    rows = load_preds(args.model, args.out_dir)
    model_name = rows[0].get("model", args.model) if rows else args.model
    total = len(rows)
    correct = sum(r["correct"] for r in rows)
    print(f"\nModel: {model_name}")
    print(f"Total predictions: {total}  |  Correct: {correct}  |  "
          f"Micro-acc: {correct/total:.3f}" if total else "No predictions.")

    # Group by distinction
    by_did: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_did[r["distinction_id"]].append(r)

    # Filter to requested distinction or all with at least one error
    if args.distinction:
        dids = [args.distinction] if args.distinction in by_did else []
        if not dids:
            print(f"No predictions found for distinction '{args.distinction}'.")
            return
    else:
        dids = sorted(
            (did for did, rs in by_did.items() if not all(r["correct"] for r in rs)),
            key=lambda d: sum(r["correct"] for r in by_did[d]) / len(by_did[d])
        )
        if not dids:
            print("\nAll distinctions are 100% correct!")
            return
        print(f"\nShowing {len(dids)} distinctions with at least one error "
              f"(sorted by accuracy, lowest first):\n")

    for did in dids:
        dist_rows = by_did[did]
        if not args.errors_only:
            print_confusion_matrix(did, dist_rows)
        else:
            n = len(dist_rows)
            acc = sum(r["correct"] for r in dist_rows) / n
            errors = sum(1 for r in dist_rows if not r["correct"])
            print(f"\n{'─'*60}")
            print(f"  {did}   acc={acc:.3f}  ({n - errors}/{n})  —  {errors} error(s)")
        print_errors(did, dist_rows, max_examples=args.max_examples)

    print(f"\n{'─'*60}\n")


if __name__ == "__main__":
    main()
