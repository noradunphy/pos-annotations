# pos-annotations

ICL-backed domain POS tagging pipeline. Ingests OntoNotes, maps PTB tags to a
two-tier taxonomy, mines silver-standard subclass labels with a frontier model
(or LLM-powered subagents), builds ICL prompts, runs a local vLLM tagger, and
produces validated corpus annotations with bootstrap confidence intervals.

## Overview

```
OntoNotes (PTB tags)
  │
  ▼
Ingest ─► data/interim/tokens.jsonl        (flat token rows)
  │
  ▼
Collapse ─► Tier A coarse POS              (deterministic PTB → {NOUN,VERB,ADJ,...})
  │
  ▼
Mine ─► data/mined_examples/{icl,heldout}  (frontier-model silver labels)
  │
  ▼
Tag  ─► vLLM + ICL prompts + constrained   (local 14B model)
  │       JSON decoding
  ▼
Validate ─► data/eval/runs/<run_id>/       (per-distinction accuracy + preds)
  │
  ▼
Annotate ─► data/annotations/corpus_1k.jsonl  (1K unseen sentences)
  │
  ▼
Bootstrap CIs ─► bootstrap_ci.json         (95% confidence intervals)
```

## Taxonomy

**Tier A** (coarse POS, deterministic from PTB): `NOUN`, `VERB`, `ADJ`, `ADV`,
`PREP`, `DET`, `PRON`, `CONJ`, `PART`, `INT`.

**Tier B** (fine subclass distinctions, 17 enabled). Each distinction applies to
one Tier A class and has a small closed label set:

| Distinction | Parent | Labels |
|---|---|---|
| `noun_proper_common` | NOUN | proper, common |
| `noun_count_mass` | NOUN | count, mass |
| `noun_concrete_abstract` | NOUN | concrete, abstract |
| `noun_collective` | NOUN | collective, non-collective |
| `verb_lexical_aux` | VERB | lexical, auxiliary |
| `verb_finite` | VERB | finite, non-finite |
| `verb_stative_eventive` | VERB | stative, eventive |
| `verb_copular_prog_pass` | VERB | copular, progressive, passive, none_of_these |
| `verb_transitivity` | VERB | transitive, intransitive, ditransitive, ambitransitive |
| `adj_attributive_predicative` | ADJ | attributive, predicative |
| `adj_gradable` | ADJ | gradable, non-gradable |
| `adj_intersective` | ADJ | intersective, non-intersective |
| `pron_type` | PRON | personal, possessive, reflexive, demonstrative, relative, interrogative, indefinite |
| `adv_type` | ADV | manner, time, place, degree, frequency, sentence_discourse |
| `det_type` | DET | articles, demonstratives, possessives, quantifiers, numerals |
| `conj_type` | CONJ | coordinating, subordinating |
| `part_type` | PART | standard, phrasal_verb_particle |

Optional grammatical-feature extensions (disabled by default) are defined in
`configs/taxonomy.yaml`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

OntoNotes on Hugging Face uses a dataset script; this repo pins
`datasets>=3.0,<4`. Point the cache at a writable location:

```bash
export HF_HOME="$(pwd)/.hf_cache"
```

## Pipeline

All commands assume the venv is active and the repo root is on `PYTHONPATH`
(handled by `pip install -e .`).

### Step 1 — Ingest

Download OntoNotes `english_v12` from Hugging Face and flatten to token-level
JSONL.

```bash
python -m src.data.ingest --splits train validation test
```

| Output | Description |
|---|---|
| `data/interim/tokens.jsonl` | One JSON object per token (sentence_id, token_index, word, ptb_pos, split, genre) |
| `data/raw/ingest_manifest.json` | Per-split document/sentence/token counts |

Use `--max-documents N` for a quick smoke test; `--append` to add splits
incrementally.

### Step 2 — Mine ICL examples

Two workflows are available. Both produce the same output layout.

#### 2a. API-based mining (sequential)

```bash
python -m src.mining.mine --miner openai   # requires OPENAI_API_KEY
python -m src.mining.mine --miner stub     # deterministic round-robin for testing
```

#### 2b. Subagent-oriented mining (parallel batches)

This is the workflow used in this project. It dispatches LLM-powered subagents
in parallel, one per distinction.

**Export** prompts and JSON batches:

```bash
python -m src.mining.export_batches \
    --out-root data/mining_batches \
    --total-examples 100 \
    --batch-size 8
```

Each distinction gets `data/mining_batches/<distinction_id>/` containing:

- `MINING_PROMPT.md` — instructions, allowed labels, output schema
- `batch_000.json` … `batch_012.json` — example arrays to label

**Annotate** each batch (done by subagents or humans). For every
`batch_NNN.json`, write `batch_NNN.labeled.json` in the same directory:

```json
[{"example_id": "…", "gold_subclass": "<label>", "notes": ""}, …]
```

**Merge** labeled batches into the standard mined layout:

```bash
python -m src.mining.merge_batches --all \
    --batches-root data/mining_batches \
    --out-dir data/mined_examples \
    --icl-per-label 6 --heldout-per-label 3
```

| Output | Description |
|---|---|
| `data/mined_examples/icl/<did>.jsonl` | ICL training examples (≤6 per label) |
| `data/mined_examples/heldout/<did>.jsonl` | Held-out validation examples (≤3 per label) |

If some labels are underrepresented, run a second round with more examples and
a separate `--out-root`, then copy the improved files over:

```bash
python -m src.mining.export_batches \
    --out-root data/mining_batches_round2 \
    --total-examples 300 --batch-size 10 \
    --distinction-ids pron_type adv_type verb_transitivity
# (label the batches)
python -m src.mining.merge_batches --all \
    --batches-root data/mining_batches_round2 \
    --out-dir data/mined_examples_round2
cp data/mined_examples_round2/icl/pron_type.jsonl data/mined_examples/icl/
cp data/mined_examples_round2/heldout/pron_type.jsonl data/mined_examples/heldout/
```

### Step 3 — Validate tagger

Run the vLLM tagger on held-out examples and report per-distinction accuracy.

```bash
# real GPU inference (requires CUDA):
python -m src.eval.validate \
    --heldout-dir data/mined_examples/heldout \
    --out-dir data/eval/runs

# CPU / no-GPU smoke test:
python -m src.eval.validate --mock
```

| Output | Description |
|---|---|
| `data/eval/runs/<run_id>/metrics.json` | Per-distinction accuracy summary |
| `data/eval/runs/<run_id>/preds_<did>.jsonl` | Per-example predictions with `correct` flag |

### Step 4 — Annotate unseen corpus

Tag every eligible token in 1,000 unseen sentences across all 17 distinctions.
Uses batched vLLM inference (256 prompts per batch).

```bash
python -m src.eval.annotate \
    --split test \
    --num-sentences 1000 \
    --out data/annotations/corpus_1k.jsonl
```

| Output | Description |
|---|---|
| `data/annotations/corpus_1k.jsonl` | One row per (token, distinction) prediction |
| `data/annotations/last_annotate_meta.json` | Run metadata |

### Step 5 — Bootstrap confidence intervals

Compute 95% CIs from the validation predictions.

```bash
python -m src.eval.report_ci \
    --preds-dir data/eval/runs/<run_id> \
    --out data/eval/runs/<run_id>/bootstrap_ci.json
```

## Directory layout

```
pos-annotations/
├── configs/
│   ├── ptb_to_coarse.yaml      # PTB tag → Tier A mapping
│   ├── taxonomy.yaml           # Tier A classes, Tier B distinctions, labels, guidelines
│   └── tagger.yaml             # vLLM model config (model id, quantization, mock flag)
├── data/
│   ├── raw/                    # ingest manifest
│   ├── interim/
│   │   └── tokens.jsonl        # flat token rows from OntoNotes
│   ├── mining_batches/         # exported prompts + JSON batches (round 1)
│   │   └── <distinction_id>/
│   │       ├── MINING_PROMPT.md
│   │       ├── batch_000.json
│   │       └── batch_000.labeled.json
│   ├── mining_batches_round2/  # second-round batches for sparse distinctions
│   ├── mined_examples/
│   │   ├── icl/                # ICL training examples per distinction
│   │   └── heldout/            # held-out validation examples per distinction
│   ├── eval/
│   │   └── runs/<run_id>/
│   │       ├── metrics.json
│   │       ├── preds_*.jsonl
│   │       └── bootstrap_ci.json
│   └── annotations/
│       ├── corpus_1k.jsonl     # 36K+ annotations over 1K sentences
│       └── last_annotate_meta.json
├── prompts/
│   ├── distinction.jinja2              # vLLM tagger prompt template
│   └── mining/
│       └── distinction_mine.md.jinja2  # subagent mining prompt template
├── src/
│   ├── data/
│   │   ├── ingest.py           # OntoNotes download + flatten
│   │   └── pos_tagset.py       # integer → PTB string mapping
│   ├── mapping/
│   │   └── collapse.py         # PTB → Tier A + eligible distinctions
│   ├── mining/
│   │   ├── mine.py             # API-based sequential mining
│   │   ├── export_batches.py   # export prompts + JSON batches for subagents
│   │   ├── merge_batches.py    # merge labeled batches → mined_examples
│   │   ├── pool_utils.py       # shared data loading and sampling
│   │   └── backends.py         # MinerBackend interface (Stub, OpenAI)
│   ├── tagging/
│   │   ├── vllm_tagger.py      # vLLM integration with constrained JSON decoding
│   │   └── prompt_render.py    # Jinja2 prompt rendering + JSON schema
│   └── eval/
│       ├── validate.py         # run tagger on held-out, report accuracy
│       ├── annotate.py         # annotate unseen corpus slice (batched)
│       └── report_ci.py        # bootstrap 95% CIs
├── tests/
│   ├── test_collapse.py
│   └── test_merge_batches.py
├── requirements.txt
├── pyproject.toml
└── pytest.ini
```

## Model

The default tagger is **Qwen/Qwen2.5-14B-Instruct-AWQ** (4-bit AWQ
quantization) served by vLLM with constrained JSON decoding. Each prompt
includes the distinction guidelines, up to 3 ICL examples with marked target
words, and the test sentence. The model outputs `{"label": "<one_of_allowed>"}`
enforced by a JSON schema.

Configure in `configs/tagger.yaml`. Set `mock: true` to skip vLLM for CPU-only
testing (returns the first label deterministically).

## Known limitations

- **conj_type**: PTB tags subordinating conjunctions as `IN` (→ `PREP`), so the
  `CONJ` pool contains only coordinating conjunctions. The `subordinating`
  label has no ICL or held-out examples.
- **det_type**: Possessive determiners (my, your, his…) are tagged `PRP$` (→
  `PRON`) in PTB, so the `possessives` label is absent from the `DET` pool.
- **pron_type / verb_transitivity**: Some rare labels (demonstrative, reflexive,
  ditransitive) are underrepresented in the mined ICL set despite two mining
  rounds.
- Validation CIs are wide due to small held-out sizes (3–18 examples per
  distinction).

## Tests

```bash
pytest
```
