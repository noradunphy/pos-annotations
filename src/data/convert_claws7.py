"""Convert data/snli_CLAWS7.txt → data/sampled/snli_claws7.json.

Input format
------------
Lines of space-separated ``word_TAG`` tokens.
Sentence boundaries are marked by a ``._. `` token (period tagged as ``.``).

Ditto tags encode multi-word units, e.g.:
    ``in_II31  the_AT  air_NN1``  →  a 3-word preposition starting at position 1.

Usage
-----
    python -m src.data.convert_claws7 \\
        --input data/snli_CLAWS7.txt \\
        --out   data/sampled/snli_claws7.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Project root on sys.path so the package can be imported directly.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.mapping.claws7_to_json import CLAWS7_ATTRIBUTES, CLAWS7_POS

# Punctuation characters that get pos=PUNC (not looked up in the mapping dicts).
_PUNC_RE = re.compile(r'^[^\w]+$')

# Pattern that identifies the end of a ditto sequence in a tag, e.g.
#   II31  → base=II, total=3, position=1
#   PPX121 → base=PPX1, total=2, position=1
#   RR21  → base=RR, total=2, position=1
# General form: <base_tag><N><M>  where N=length (1 digit) and M=position (1 digit).
_DITTO_RE = re.compile(r'^(.+?)(\d)(\d)$')


def strip_ditto(raw_tag: str) -> tuple[str, dict | None]:
    """Return (base_tag, multiword_unit_dict_or_None).

    For a plain tag like ``NN1`` the second element is ``None``.
    For a ditto tag like ``II31`` it is ``{"length": 3, "position": 1}``.
    """
    m = _DITTO_RE.match(raw_tag)
    if m:
        base, length_str, pos_str = m.group(1), m.group(2), m.group(3)
        length = int(length_str)
        position = int(pos_str)
        # Validate that the stripped base tag is known; if not, it might just
        # happen to end in two digits (unlikely but possible for foreign words).
        if base in CLAWS7_POS or base in (".", ",", "``", "''"):
            return base, {"length": length, "position": position}
        # Fall back: treat whole string as base tag
        return raw_tag, None
    return raw_tag, None


def is_punctuation(word: str, tag: str) -> bool:
    """True when the token should be labelled pos=PUNC."""
    if tag in (".", ",", "``", "''", ":", ";", "!", "?", "-", "--",
               "(", ")", "[", "]", "{", "}", "\"", "'", "/"):
        return True
    # Tags starting with punctuation character
    if tag and not tag[0].isalpha():
        return True
    # Words that are purely non-word characters and unrecognised tags
    if _PUNC_RE.match(word):
        return True
    return False


def parse_claws7_file(path: Path) -> list[list[tuple[str, str]]]:
    """Read the CLAWS7 file and return a list of sentences.

    Each sentence is a list of ``(word, raw_tag)`` pairs.
    The sentence-boundary token ``._. `` is consumed and not included in
    the returned tokens.
    """
    raw_text = path.read_text(encoding="utf-8")
    # Normalise newlines and collapse multiple spaces
    tokens_flat = raw_text.split()

    sentences: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []

    for tok in tokens_flat:
        # Tokens look like  word_TAG  (last underscore is the separator).
        # Some tokens may contain underscores in the word part (e.g. proper
        # names in some corpora), so we split on the *last* underscore.
        if "_" not in tok:
            # Bare word with no tag – skip (should not happen in clean CLAWS7)
            continue
        sep = tok.rfind("_")
        word = tok[:sep]
        tag  = tok[sep + 1:]

        # Sentence boundary: period tagged as "."
        if tag == ".":
            if current:
                sentences.append(current)
                current = []
            # Do NOT add the full stop to the token list (it is metadata).
            continue

        current.append((word, tag))

    # Flush any trailing sentence that had no period
    if current:
        sentences.append(current)

    return sentences


def build_token(word: str, raw_tag: str) -> dict:
    """Build a single token dict according to the output schema."""
    base_tag, mwu = strip_ditto(raw_tag)

    if is_punctuation(word, base_tag):
        entry: dict = {"token": word, "claws7_tag": raw_tag, "pos": "PUNC"}
        if mwu is not None:
            entry["base_tag"] = base_tag
            entry["multiword_unit"] = mwu
        return entry

    pos = CLAWS7_POS.get(base_tag, "X")
    attrs = dict(CLAWS7_ATTRIBUTES.get(base_tag, {}))  # copy so we don't mutate

    entry = {"token": word, "claws7_tag": raw_tag, "pos": pos}
    if mwu is not None:
        entry["base_tag"] = base_tag
        entry["multiword_unit"] = mwu
    if attrs:
        entry["attributes"] = attrs
    return entry


def reconstruct_text(tokens: list[tuple[str, str]]) -> str:
    """Best-effort reconstruction of the surface sentence string."""
    words = [w for w, _ in tokens]
    # Simple join; attach punctuation-like tokens without extra space.
    parts: list[str] = []
    for i, w in enumerate(words):
        if i == 0:
            parts.append(w)
        elif w in (",", ".", ":", ";", "!", "?", "'s", "'", "n't", ")", "]", "}"):
            parts.append(w)
        elif parts and parts[-1] in ("(", "[", "{", "``"):
            parts.append(w)
        else:
            parts.append(" " + w)
    return "".join(parts)


def convert(input_path: Path, output_path: Path) -> None:
    sentences_raw = parse_claws7_file(input_path)
    print(f"Parsed {len(sentences_raw)} sentences from {input_path}", file=sys.stderr)

    sentences_out = []
    unknown_tags: set[str] = set()

    for idx, sent_tokens in enumerate(sentences_raw, start=1):
        token_dicts = []
        for word, raw_tag in sent_tokens:
            base_tag, _ = strip_ditto(raw_tag)
            if (
                not is_punctuation(word, base_tag)
                and base_tag not in CLAWS7_POS
            ):
                unknown_tags.add(base_tag)
            token_dicts.append(build_token(word, raw_tag))

        sentences_out.append({
            "id": idx,
            "text": reconstruct_text(sent_tokens),
            "tokens": token_dicts,
        })

    if unknown_tags:
        print(
            f"WARNING: {len(unknown_tags)} unrecognised base tag(s): "
            f"{sorted(unknown_tags)}",
            file=sys.stderr,
        )

    output = {
        "schema_version": "1.0",
        "source": "SNLI",
        "tag_system": "CLAWS7",
        "n_sentences": len(sentences_out),
        "sentences": sentences_out,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(sentences_out)} sentences → {output_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/snli_CLAWS7.txt"),
        help="Path to the CLAWS7-tagged text file",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/sampled/snli_claws7.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()
    convert(args.input, args.out)


if __name__ == "__main__":
    main()
