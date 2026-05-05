"""Sample N random sentences from a HuggingFace text dataset.

Default: SNLI premises — clean, declarative English sentences ideal for
POS annotation (varied structure, moderate length, no markup artifacts).

Usage:
    python -m src.data.sample_sentences
    python -m src.data.sample_sentences --n 1000 --out data/sampled/snli_1k.txt
    python -m src.data.sample_sentences --dataset simple_wikipedia
    python -m src.data.sample_sentences --dataset wikitext
    python -m src.data.sample_sentences --dataset ag_news
    python -m src.data.sample_sentences --seed 99
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Sentence cleaning
# ---------------------------------------------------------------------------

def clean_wikitext(text: str) -> str:
    """Fix WikiText-103 tokenisation artifacts before further processing."""
    # @-@ → hyphen,  @.@ → decimal point,  @,@ → comma
    text = re.sub(r"\s@-@\s", "-", text)
    text = re.sub(r"\s@\.@\s", ".", text)
    text = re.sub(r"\s@,@\s", ",", text)
    # Remove any remaining lone @ tokens
    text = re.sub(r"\s@\S+@\s?", " ", text)
    # Fix spaced contractions: don 't → don't,  it 's → it's, etc.
    text = re.sub(r"\b(\w+) (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1\2", text)
    # Fix spaced punctuation:   word , word → word, word
    text = re.sub(r" ([,;:!?])", r"\1", text)
    # Fix space before period at end of sentence
    text = re.sub(r" \.$", ".", text)
    text = re.sub(r" \. ([A-Z\"])", r". \1", text)
    # Fix spaced quotes:  " word " → "word"
    text = re.sub(r'" ([^"]+) "', r'"\1"', text)
    # Collapse multiple spaces
    text = re.sub(r"  +", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_ABBREV = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|al|Fig|fig|No|no|St|Ave|Blvd|Dept|"
    r"Gov|Gen|Lt|Sgt|Cpl|Pvt|Capt|Col|Rep|Sen|Rev|Hon|approx|est|vol|pp)\."
)


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences using punctuation heuristics."""
    text = clean_wikitext(text)
    masked = _ABBREV.sub(lambda m: m.group().replace(".", "\x00"), text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", masked)
    sentences = []
    for p in parts:
        s = p.replace("\x00", ".").strip()
        if s:
            sentences.append(s)
    return sentences


# ---------------------------------------------------------------------------
# Validity filters
# ---------------------------------------------------------------------------

# Characters that suggest template/table/markup noise
_NOISE = re.compile(r"[|{}\[\]<>\\]|={2,}|@\S*@|\bFile:|colspan|rowspan|thumb|px\b")
# Bullet / list markers at start
_BULLET = re.compile(r"^\s*[*#;:]")


def is_valid(sentence: str, min_words: int, max_words: int) -> bool:
    """Return True for clean, well-formed prose sentences."""
    # Must end with sentence-final punctuation
    if not re.search(r"[.!?][\"']?\s*$", sentence):
        return False
    # Must start with a capital letter or opening quote
    if not re.match(r'^[A-Z"\u201c]', sentence):
        return False
    # Reject headings, bullets, markup noise
    if _BULLET.match(sentence):
        return False
    if _NOISE.search(sentence):
        return False
    # Reject wikitext headings (= Title =)
    if re.match(r"=+\s", sentence):
        return False
    words = sentence.split()
    if len(words) < min_words or len(words) > max_words:
        return False
    # Require majority alphabetic content
    alpha = sum(c.isalpha() for c in sentence)
    if alpha < len(sentence) * 0.65:
        return False
    # Reject if too many numbers (often table rows or stats)
    digits = sum(c.isdigit() for c in sentence)
    if digits > len(sentence) * 0.15:
        return False
    # Reject if sentence contains lone @ (leftover wikitext tokens)
    if "@" in sentence:
        return False
    # Reject sentences that look like fragments (no verb-like word)
    # Simple heuristic: must have at least one word ending in common verb suffixes
    if not re.search(r"\b\w+(?:ed|ing|es|ize|ise|ified|ate)\b", sentence, re.I):
        # Allow short declarative sentences that have 'is/are/was/were/has/have'
        if not re.search(r"\b(?:is|are|was|were|has|have|had|did|does|do|will|would|can|could|should|may|might|must|shall)\b", sentence, re.I):
            return False
    return True


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def iter_texts_snli(split: str = "train"):
    """Yield premise sentences from SNLI.

    SNLI premises are clean, grammatical, declarative English sentences
    written specifically to be unambiguous — ideal for POS annotation.
    Each is already a single sentence; no splitting needed.
    """
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/snli", split=split, streaming=True)
    for row in ds:
        text = (row.get("premise") or "").strip()
        if text:
            yield text


def iter_texts_simple_wikipedia(split: str = "train"):
    """Yield article text from Simple English Wikipedia.

    Articles are written with simpler grammar and shorter sentences
    than regular Wikipedia, while still covering broad topics.
    """
    from datasets import load_dataset
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.simple",
        split=split, streaming=True, trust_remote_code=True,
    )
    for row in ds:
        text = (row.get("text") or "").strip()
        if text:
            yield text


def iter_texts_wikitext(split: str = "train"):
    """Yield paragraph strings from WikiText-103."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
    for row in ds:
        text = (row.get("text") or "").strip()
        if text:
            yield text


def iter_texts_ag_news(split: str = "train"):
    """Yield description strings from AG News."""
    from datasets import load_dataset
    ds = load_dataset("ag_news", split=split, streaming=True)
    for row in ds:
        text = (row.get("text") or "").strip()
        if text:
            yield text


def iter_texts_wikipedia(split: str = "train"):
    """Yield article text from full English Wikipedia (slow to initialise)."""
    from datasets import load_dataset
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.en",
        split=split, streaming=True, trust_remote_code=True,
    )
    for row in ds:
        text = (row.get("text") or "").strip()
        if text:
            yield text


LOADERS = {
    "snli": iter_texts_snli,
    "simple_wikipedia": iter_texts_simple_wikipedia,
    "wikitext": iter_texts_wikitext,
    "ag_news": iter_texts_ag_news,
    "wikipedia": iter_texts_wikipedia,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Sample N sentences from a HuggingFace dataset")
    ap.add_argument("--dataset", default="snli", choices=list(LOADERS),
                    help="Source dataset (default: snli)")
    ap.add_argument("--split", default="train", help="Dataset split (default: train)")
    ap.add_argument("-n", "--n", type=int, default=1000,
                    help="Number of sentences to sample (default: 1000)")
    ap.add_argument("--min-words", type=int, default=8,
                    help="Minimum sentence length in words (default: 8)")
    ap.add_argument("--max-words", type=int, default=50,
                    help="Maximum sentence length in words (default: 50)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out", type=Path,
        default=PROJECT_ROOT / "data" / "sampled" / "snli_1k.txt",
        help="Output path (default: data/sampled/snli_1k.txt)",
    )
    # Reservoir sampling uses a pool multiplier to get variety
    ap.add_argument("--pool", type=int, default=50_000,
                    help="Candidate pool size for reservoir sampling (default: 50000)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Datasets that are already one-sentence-per-row (no splitting needed)
    SENTENCE_LEVEL = {"snli", "ag_news"}

    loader = LOADERS[args.dataset]
    print(f"Streaming {args.dataset} ({args.split} split)…", file=sys.stderr)

    # Reservoir sampling: fill a pool then shuffle-sample
    pool: list[str] = []
    total_seen = 0

    for text in loader(args.split):
        candidates = [text] if args.dataset in SENTENCE_LEVEL else split_sentences(text)
        for sent in candidates:
            if not is_valid(sent, args.min_words, args.max_words):
                continue
            total_seen += 1
            if len(pool) < args.pool:
                pool.append(sent)
            else:
                # Replace a random element (standard reservoir sampling)
                idx = rng.randint(0, total_seen - 1)
                if idx < args.pool:
                    pool[idx] = sent
            if total_seen % 10_000 == 0:
                print(f"  scanned {total_seen:,} candidates, pool={len(pool):,}",
                      file=sys.stderr)
            if len(pool) >= args.pool:
                break  # pool full — sample from it

    print(f"Total candidates scanned: {total_seen:,}", file=sys.stderr)

    if len(pool) < args.n:
        print(
            f"Warning: only {len(pool)} valid sentences found, "
            f"requested {args.n}. Writing all of them.",
            file=sys.stderr,
        )
        sample = pool
    else:
        sample = rng.sample(pool, args.n)

    args.out.write_text("\n".join(sample) + "\n", encoding="utf-8")
    print(f"Wrote {len(sample)} sentences → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
