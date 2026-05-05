"""Trial evaluation: run GeminiTagger on pos_subclass_dataset_100.json.

Annotates 100 hand-labeled sentences without exposing ground truth to the model,
then computes per-distinction agreement with the gold labels.

Usage:
    # Flash (fast, cheap):
    python -m src.eval.trial_100 --model flash --out-dir data/trial/results

    # Frontier (best quality):
    python -m src.eval.trial_100 --model frontier --out-dir data/trial/results

    # Resume an interrupted run (automatic if preds file exists):
    python -m src.eval.trial_100 --model frontier --out-dir data/trial/results

    # Force a fresh start, discarding prior results:
    python -m src.eval.trial_100 --model frontier --out-dir data/trial/results --restart

Outputs (in --out-dir):
    preds_<model>.jsonl       — one row per (token, distinction): word, sentence,
                                distinction_id, pred_subclass, gold_subclass
    agreement_<model>.json    — per-distinction accuracy + macro average
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRIAL_DATA = PROJECT_ROOT / "data" / "trial" / "pos_subclass_dataset_100.json"
TAXONOMY_PATH = PROJECT_ROOT / "configs" / "taxonomy.yaml"

# ---------------------------------------------------------------------------
# Tier A mapping: 100.json POS name → taxonomy tier_a tag
# ---------------------------------------------------------------------------
POS_NAME_TO_TIER_A: dict[str, str] = {
    "Noun": "NOUN",
    "Verb": "VERB",
    "Adjective": "ADJ",
    "Adverb": "ADV",
    "Preposition": "PREP",
    "Determiner": "DET",
    "Pronoun": "PRON",
    "Conjunction": "CONJ",
    "Particle": "PART",
    "Interjection": "INT",
}

# ---------------------------------------------------------------------------
# Schema mapper: 100.json subclass fields → (distinction_id, pipeline_label)
#
# Returns None for the label when the value is ambiguous / not representable
# in the taxonomy label set (those tokens are skipped from agreement scoring).
# ---------------------------------------------------------------------------

def _lower(s: str) -> str:
    return s.strip().lower()


def map_subclass_fields(
    pos_name: str,
    subclass: dict[str, Any],
) -> list[tuple[str, str | None]]:
    """Convert 100.json subclass dict into [(distinction_id, gold_label), …].

    gold_label is None when the field value is ambiguous (e.g. compound like
    'Finite_or_Non-finite(dependent)'). Such pairs are excluded from scoring.
    """
    result: list[tuple[str, str | None]] = []

    if pos_name == "Noun":
        # proper_common
        pc = subclass.get("proper_common")
        if pc is not None:
            result.append(("noun_proper_common", _lower(str(pc)) if str(pc) in ("Common", "Proper") else None))

        # count_mass
        cm = subclass.get("count_mass")
        if cm is not None:
            if str(cm) in ("Count", "Mass"):
                result.append(("noun_count_mass", _lower(str(cm))))
            else:
                result.append(("noun_count_mass", None))  # ambiguous

        # concrete_abstract
        ca = subclass.get("concrete_abstract")
        if ca is not None:
            result.append(("noun_concrete_abstract", _lower(str(ca))))

        # collective (boolean in JSON, stringified as 'True'/'False')
        coll = subclass.get("collective")
        if coll is not None:
            coll_str = str(coll)
            if coll_str in ("True", "true", "1"):
                result.append(("noun_collective", "collective"))
            elif coll_str in ("False", "false", "0"):
                result.append(("noun_collective", "non-collective"))
            else:
                result.append(("noun_collective", None))

    elif pos_name == "Verb":
        # verb_lexical_aux
        lt = subclass.get("lexical_type")
        if lt is not None:
            mapping = {
                "Lexical": "lexical",
                "Auxiliary": "auxiliary",
                "Copular": "auxiliary",  # copular treated as auxiliary in taxonomy
            }
            result.append(("verb_lexical_aux", mapping.get(str(lt))))

        # verb_finite
        fin = subclass.get("finiteness")
        if fin is not None:
            fin_map = {"Finite": "finite", "Non-finite": "non-finite"}
            result.append(("verb_finite", fin_map.get(str(fin))))  # None for ambiguous

        # verb_stative_eventive
        se = subclass.get("stative_eventive")
        if se is not None:
            se_map = {"Stative": "stative", "Eventive": "eventive"}
            result.append(("verb_stative_eventive", se_map.get(str(se))))

        # verb_transitivity
        tr = subclass.get("transitivity")
        if tr is not None:
            tr_map = {
                "Transitive": "transitive",
                "Intransitive": "intransitive",
                "Ditransitive": "ditransitive",
                # Compound/ambi forms → ambitransitive
                "Intransitive_or_Ambitransitive": "ambitransitive",
                "Transitive_or_Ambitransitive": "ambitransitive",
                "Intransitive_or_Transitive": "ambitransitive",
                "Transitive_or_Intransitive": "ambitransitive",
            }
            result.append(("verb_transitivity", tr_map.get(str(tr))))

        # verb_copular_prog_pass — not encoded in 100.json, skip

    elif pos_name == "Adjective":
        # adj_attributive_predicative
        ap = subclass.get("attributive_predicative")
        if ap is not None:
            ap_map = {"Attributive": "attributive", "Predicative": "predicative"}
            result.append(("adj_attributive_predicative", ap_map.get(str(ap))))

        # adj_gradable
        gr = subclass.get("gradable")
        if gr is not None:
            gr_map = {"Gradable": "gradable", "Non-gradable": "non-gradable"}
            result.append(("adj_gradable", gr_map.get(str(gr))))

        # adj_intersective
        ix = subclass.get("intersective")
        if ix is not None:
            ix_map = {"Intersective": "intersective", "Non-intersective": "non-intersective"}
            result.append(("adj_intersective", ix_map.get(str(ix))))

    elif pos_name == "Adverb":
        t = subclass.get("type")
        if t is not None:
            adv_map = {
                "Manner": "manner",
                "Time": "time",
                "Place": "place",
                "Degree": "degree",
                "Frequency": "frequency",
                "Sentence": "sentence_discourse",
            }
            result.append(("adv_type", adv_map.get(str(t))))

    elif pos_name == "Determiner":
        t = subclass.get("type")
        if t is not None:
            det_map = {
                "Article": "articles",
                "Demonstrative": "demonstratives",
                "Possessive": "possessives",
                "Quantifier": "quantifiers",
                "Numeral": "numerals",
            }
            result.append(("det_type", det_map.get(str(t))))

    elif pos_name == "Pronoun":
        t = subclass.get("type")
        if t is not None:
            pron_map = {
                "Personal": "personal",
                "Possessive": "possessive",
                "Reflexive": "reflexive",
                "Demonstrative": "demonstrative",
                "Relative": "relative",
                "Interrogative": "interrogative",
                "Indefinite": "indefinite",
            }
            result.append(("pron_type", pron_map.get(str(t))))

    elif pos_name == "Conjunction":
        t = subclass.get("type")
        if t is not None:
            conj_map = {
                "Coordinating": "coordinating",
                "Subordinating": "subordinating",
            }
            result.append(("conj_type", conj_map.get(str(t))))

    elif pos_name == "Particle":
        t = subclass.get("type")
        if t is not None:
            part_map = {
                "PhrasalVerbParticle": "phrasal_verb_particle",
                "Standard": "standard",
            }
            result.append(("part_type", part_map.get(str(t))))

    return result


# ---------------------------------------------------------------------------
# Taxonomy helpers
# ---------------------------------------------------------------------------

def load_taxonomy_distinctions(path: Path | None = None) -> dict[str, dict[str, Any]]:
    p = path or TAXONOMY_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    return {
        d["id"]: d
        for d in raw.get("distinctions", [])
        if d.get("enabled", True)
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Trial: annotate 100 sentences with Gemini and measure agreement")
    ap.add_argument(
        "--model",
        default="flash",
        help="Model alias: 'flash' (gemini-2.5-flash) or 'frontier' (gemini-2.5-pro), "
             "or a literal Gemini model name.",
    )
    ap.add_argument(
        "--data",
        type=Path,
        default=TRIAL_DATA,
        help="Path to pos_subclass_dataset_100.json",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "trial" / "results",
    )
    ap.add_argument("--taxonomy", type=Path, default=None)
    ap.add_argument("--gemini-cfg", type=Path, default=None)
    ap.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key (overrides GEMINI_API_KEY env var)",
    )
    ap.add_argument(
        "--restart",
        action="store_true",
        help="Discard any existing partial results and start fresh.",
    )
    ap.add_argument(
        "--log-responses",
        action="store_true",
        help="Print raw API response text to stderr for every call (useful for debugging parser issues).",
    )
    args = ap.parse_args()

    # Resolve model label for filenames
    model_tag = args.model  # 'flash', 'frontier', or literal name

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load tagger
    from src.tagging.gemini_tagger import GeminiConfig, GeminiTagger

    cfg = GeminiConfig.load(args.gemini_cfg)
    tagger = GeminiTagger(model=model_tag, cfg=cfg, api_key=args.api_key)
    resolved_model = cfg.resolve_model(model_tag)
    print(f"Model: {resolved_model}", file=sys.stderr)

    # Load taxonomy distinctions
    distinctions = load_taxonomy_distinctions(args.taxonomy)

    # Load 100-sentence dataset
    data = json.loads(args.data.read_text(encoding="utf-8"))
    examples = data["examples"]
    print(f"Loaded {len(examples)} sentences from {args.data}", file=sys.stderr)

    preds_path = args.out_dir / f"preds_{model_tag}.jsonl"
    agreement_path = args.out_dir / f"agreement_{model_tag}.json"

    # ------------------------------------------------------------------
    # Resume: load any previously written predictions so we can skip them
    # and seed the accumulators correctly.
    # ------------------------------------------------------------------
    done: set[tuple[int, int, str]] = set()   # (example_id, token_index, distinction_id)
    correct_by_did: dict[str, list[bool]] = defaultdict(list)
    total_items = 0
    skipped_ambiguous = 0

    if preds_path.exists() and not args.restart:
        with preds_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (rec["example_id"], rec["token_index"], rec["distinction_id"])
                done.add(key)
                correct_by_did[rec["distinction_id"]].append(rec["correct"])
                total_items += 1
        print(
            f"Resuming: {total_items} predictions already done, "
            f"{len(done)} (example, token, distinction) triples skipped.",
            file=sys.stderr,
        )
    elif args.restart and preds_path.exists():
        preds_path.unlink()
        print("--restart: deleted existing predictions, starting fresh.", file=sys.stderr)

    file_mode = "a" if done else "w"

    with preds_path.open(file_mode, encoding="utf-8") as fout:
        for ex in examples:
            sentence = ex["sentence"]
            tokens = ex["tokens"]

            # Rebuild word list from tokens (preserves exact tokenization)
            token_words = [t["token"] for t in tokens]

            for tok_idx, tok in enumerate(tokens):
                pos_name = tok["label"]["pos"]
                subclass = tok["label"].get("subclass", {})

                # Get ground-truth pairs for all applicable distinctions
                gold_pairs = map_subclass_fields(pos_name, subclass)

                for did, gold_label in gold_pairs:
                    if did not in distinctions:
                        continue  # distinction disabled or not in taxonomy

                    if gold_label is None:
                        skipped_ambiguous += 1
                        continue

                    dist = distinctions[did]
                    labels = list(dist["labels"])

                    if gold_label not in labels:
                        skipped_ambiguous += 1
                        continue

                    # Skip if already predicted in a prior run
                    if (ex["id"], tok_idx, did) in done:
                        continue

                    # Run tagger (blind to gold_label)
                    pred_label, _ = tagger.predict_label(
                        distinction=dist,
                        words=token_words,
                        target_index=tok_idx,
                        log_response=args.log_responses,
                    )

                    is_correct = pred_label == gold_label
                    correct_by_did[did].append(is_correct)
                    total_items += 1

                    record = {
                        "example_id": ex["id"],
                        "sentence": sentence,
                        "token": tok["token"],
                        "token_index": tok_idx,
                        "distinction_id": did,
                        "pred_subclass": pred_label,
                        "gold_subclass": gold_label,
                        "correct": is_correct,
                        "model": resolved_model,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()  # ensure each prediction survives an interrupt

                    status = "✓" if is_correct else "✗"
                    print(
                        f"  [{status}] ex{ex['id']} tok{tok_idx} '{tok['token']}' "
                        f"{did}: pred={pred_label} gold={gold_label}",
                        file=sys.stderr,
                    )

    print(f"\nTotal scored: {total_items} | Skipped (ambiguous): {skipped_ambiguous}", file=sys.stderr)

    # Compute agreement
    per_dist: dict[str, dict[str, Any]] = {}
    all_accs: list[float] = []
    total_correct = 0
    for did, bools in sorted(correct_by_did.items()):
        n_correct = sum(bools)
        acc = n_correct / len(bools) if bools else 0.0
        per_dist[did] = {"n": len(bools), "n_correct": n_correct, "accuracy": round(acc, 4)}
        all_accs.append(acc)
        total_correct += n_correct
        print(f"  {did}: {acc:.3f} ({n_correct}/{len(bools)})", file=sys.stderr)

    macro_avg = sum(all_accs) / len(all_accs) if all_accs else 0.0
    micro_avg = total_correct / total_items if total_items else 0.0
    print(f"\nMacro-average accuracy: {macro_avg:.3f}  (equal weight per distinction)", file=sys.stderr)
    print(f"Micro-average accuracy: {micro_avg:.3f}  (equal weight per prediction)", file=sys.stderr)

    report = {
        "model": resolved_model,
        "model_alias": model_tag,
        "total_scored": total_items,
        "total_correct": total_correct,
        "skipped_ambiguous": skipped_ambiguous,
        "macro_avg_accuracy": round(macro_avg, 4),
        "micro_avg_accuracy": round(micro_avg, 4),
        "per_distinction": per_dist,
    }
    agreement_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote predictions → {preds_path}", file=sys.stderr)
    print(f"Wrote agreement report → {agreement_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
