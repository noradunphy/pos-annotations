"""Bootstrap confidence intervals on per-example binary accuracy vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import bootstrap


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap 95% CIs from validate preds")
    ap.add_argument("--preds-dir", type=Path, required=True, help="Directory with preds_*.jsonl from validate")
    ap.add_argument("--n-resamples", type=int, default=10000)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Use only the first N examples per preds file (e.g. compare 500 vs 1K pool sizes)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    results: dict[str, Any] = {}
    for fp in sorted(args.preds_dir.glob("preds_*.jsonl")):
        did = fp.stem.removeprefix("preds_")
        ys: list[float] = []
        with fp.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                ys.append(1.0 if r.get("correct") else 0.0)
        if not ys:
            continue
        if args.max_examples is not None:
            ys = ys[: args.max_examples]
        yv = np.array(ys, dtype=np.float64)
        res = bootstrap(
            (yv,),
            np.mean,
            n_resamples=args.n_resamples,
            confidence_level=args.confidence,
            random_state=0,
        )
        lo, hi = res.confidence_interval.low, res.confidence_interval.high
        results[did] = {
            "n": len(ys),
            "accuracy_point": float(np.mean(ys)),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "confidence_level": args.confidence,
            "n_resamples": args.n_resamples,
        }

    text = json.dumps(results, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(args.out)
    else:
        print(text)


if __name__ == "__main__":
    main()
