"""Download and flatten OntoNotes CoNLL-2012 (english_v12) to token-level JSONL."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

from src.data.pos_tagset import int_to_ptb_pos

DATASET = "ontonotes/conll2012_ontonotesv5"
CONFIG = "english_v12"


def _ensure_datasets() -> None:
    import datasets as ds

    major = int(ds.__version__.split(".")[0])
    if major >= 4:
        print(
            "This dataset uses a loading script; install datasets<4 "
            f"(found {ds.__version__}). See requirements.txt.",
            file=sys.stderr,
        )
        sys.exit(1)


def genre_from_doc_id(document_id: str) -> str:
    parts = document_id.split("/")
    return parts[0] if len(parts) >= 1 else "unk"


def iter_token_rows(split: str, max_documents: int | None = None):
    _ensure_datasets()
    from datasets import load_dataset

    kwargs: dict = {
        "path": DATASET,
        "name": CONFIG,
        "split": split,
        "streaming": True,
        "trust_remote_code": True,
    }
    ds = load_dataset(**kwargs)
    n_docs = 0
    for ex in ds:
        if max_documents is not None and n_docs >= max_documents:
            break
        n_docs += 1
        doc_id = ex["document_id"]
        genre = genre_from_doc_id(doc_id)
        for si, sent in enumerate(ex["sentences"]):
            words = sent["words"]
            pos_tags = sent["pos_tags"]
            for wi, w in enumerate(words):
                if wi >= len(pos_tags):
                    continue
                pt = pos_tags[wi]
                if isinstance(pt, int):
                    ptb = int_to_ptb_pos(pt)
                else:
                    ptb = str(pt)
                sid = f"{doc_id}#{si}"
                yield {
                    "split": split,
                    "document_id": doc_id,
                    "sentence_index": si,
                    "sentence_id": sid,
                    "token_index": wi,
                    "token_id": f"{sid}#{wi}",
                    "word": w,
                    "ptb_pos": ptb,
                    "genre_prefix": genre,
                }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest OntoNotes english_v12 to JSONL")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/interim"),
        help="Directory for tokens.jsonl and manifest",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Dataset splits to export",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Cap documents per split (for smoke tests)",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=None,
        help="Optional HF_HOME cache directory (default: env or ~/.cache)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to tokens.jsonl instead of overwriting (for adding splits incrementally)",
    )
    args = parser.parse_args()
    if args.hf_home:
        os.environ["HF_HOME"] = str(args.hf_home.resolve())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    out_path = args.out_dir / "tokens.jsonl"
    counts: dict[str, int] = {}
    total = 0
    mode = "a" if args.append else "w"
    with out_path.open(mode, encoding="utf-8") as fout:
        for split in args.splits:
            counts[split] = 0
            for row in tqdm(
                iter_token_rows(split, max_documents=args.max_documents),
                desc=f"ingest:{split}",
            ):
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[split] += 1
                total += 1

    import datasets as ds

    man_path = raw_dir / "ingest_manifest.json"
    prev: dict | None = None
    if args.append and man_path.is_file():
        prev = json.loads(man_path.read_text(encoding="utf-8"))
    merged_counts = dict(prev.get("row_counts_by_split", {})) if prev else {}
    merged_counts.update(counts)
    merged_splits = list(dict.fromkeys((prev.get("splits") or []) + list(args.splits)))
    manifest = {
        "dataset": DATASET,
        "config": CONFIG,
        "datasets_library_version": ds.__version__,
        "splits": merged_splits,
        "row_counts_by_split": merged_counts,
        "total_rows": (prev.get("total_rows", 0) if prev else 0) + total,
        "max_documents_per_split": args.max_documents,
        "output": str(out_path.resolve()),
        "append": args.append,
    }
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    cum = manifest["total_rows"]
    print(f"Wrote {total} rows this run ({mode}) -> {out_path} (cumulative rows in manifest: {cum})")
    print(f"Manifest: {man_path}")


if __name__ == "__main__":
    main()
