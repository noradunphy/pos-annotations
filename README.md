# pos-annotations

ICL-backed domain POS tagging pipeline. Ingests OntoNotes, maps PTB tags to a
two-tier taxonomy, mines silver-standard subclass labels with a frontier model
(or LLM-powered subagents), and produces validated corpus annotations.

Two annotation backends are supported:

- **vLLM (original)** — local 14B model with bootstrapped ICL prompts from mined
  silver examples.
- **Gemini API (new)** — frontier Google Gemini model with proper system/user chat
  roles and handcrafted ICL examples baked into the system prompt; no local GPU
  required.

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
  ├── vLLM backend ──────────────────────────────────────────────────────┐
  │   Tag ─► vLLM + ICL prompts + constrained JSON decoding             │
  │          (local 14B model, bootstrapped ICL from mined examples)     │
  │                                                                      ▼
  └── Gemini API backend ─────────────────────────────────────────────►  │
      Tag ─► Gemini API + system/user chat roles                         │
             (handcrafted ICL in system prompt, no local GPU)            │
                                                                         │
  ▼                                                                      │
Annotate ─► data/annotations/corpus_1k.jsonl  (1K unseen sentences) ◄──┘
  │
  ▼
Validate / Bootstrap CIs ─► data/eval/runs/<run_id>/
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

### Gemini API key

The Gemini backend and trial evaluation require a Google AI API key:

```bash
export GEMINI_API_KEY="your-key-here"
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

#### vLLM backend (original)

Uses batched vLLM inference (256 prompts per batch) with bootstrapped ICL prompts
from `data/mined_examples/icl/`.

```bash
python -m src.eval.annotate \
    --backend vllm \
    --split test \
    --num-sentences 1000 \
    --out data/annotations/corpus_1k.jsonl
```

#### Gemini API backend (new)

Uses the Google Gemini API with handcrafted ICL examples and proper system/user
chat roles. No local GPU required. Requires `GEMINI_API_KEY` in the environment.

```bash
export GEMINI_API_KEY="your-key-here"

python -m src.eval.annotate \
    --backend gemini \
    --gemini-model frontier \
    --split test \
    --num-sentences 1000 \
    --out data/annotations/corpus_1k_gemini.jsonl
```

`--gemini-model` accepts `flash` (gemini-2.5-flash), `frontier` (gemini-2.5-pro),
or any literal Gemini model name. Model names are configured in `configs/gemini.yaml`.

| Output | Description |
|---|---|
| `data/annotations/corpus_1k.jsonl` | One row per (token, distinction) prediction (vLLM) |
| `data/annotations/corpus_1k_gemini.jsonl` | Same format, Gemini backend |
| `data/annotations/last_annotate_meta.json` | Run metadata |

### Step 5 — Bootstrap confidence intervals

Compute 95% CIs from the validation predictions.

```bash
python -m src.eval.report_ci \
    --preds-dir data/eval/runs/<run_id> \
    --out data/eval/runs/<run_id>/bootstrap_ci.json
```

## Trial evaluation (100-sentence agreement test)

`src/eval/trial_100.py` annotates the 100 hand-labeled sentences in
`data/trial/pos_subclass_dataset_100.json` with Gemini (without exposing ground
truth to the model) and computes per-distinction agreement.

This serves two purposes:
- Verify that system and user parts of the prompt are being routed correctly
  through the Gemini chat API.
- Measure how well the model agrees with the human gold-standard annotations.

```bash
export GEMINI_API_KEY="your-key-here"

# Test with the small/fast model
python -m src.eval.trial_100 --model flash --out-dir data/trial/results

# Test with the frontier model
python -m src.eval.trial_100 --model frontier --out-dir data/trial/results
```

| Output | Description |
|---|---|
| `data/trial/results/preds_flash.jsonl` | Per-(token, distinction) predictions + gold labels |
| `data/trial/results/agreement_flash.json` | Per-distinction accuracy + macro average |
| `data/trial/results/preds_frontier.jsonl` | Same for the frontier model |
| `data/trial/results/agreement_frontier.json` | Same for the frontier model |

The agreement JSON looks like:

```json
{
  "model": "gemini-2.5-flash",
  "model_alias": "flash",
  "total_scored": 312,
  "skipped_ambiguous": 14,
  "macro_avg_accuracy": 0.871,
  "per_distinction": {
    "noun_proper_common": {"n": 48, "accuracy": 0.958},
    "noun_count_mass":    {"n": 48, "accuracy": 0.833},
    ...
  }
}
```

Tokens with ambiguous gold labels (e.g. `"Finite_or_Non-finite(dependent)"`) are
excluded from scoring (`skipped_ambiguous`).

## CLAWS7 annotation scheme

In addition to the PTB-based pipeline, this repo includes a separate annotation
dataset derived from the **CLAWS7** tagset (UCREL, Lancaster University). 998
SNLI premise sentences were tagged with CLAWS7 and converted to structured JSON.

### Tag-to-attribute mapping

`src/mapping/claws7_to_json.py` exports two dicts:

**`CLAWS7_POS`** — maps every CLAWS7 base tag to one of the following high-level
POS strings:

| POS | Example tags |
|---|---|
| `NOUN` | `NN1` `NN2` `NP1` `NNT1` `NNU` |
| `VERB` | `VBZ` `VVG` `VVD` `VM` `VHD` |
| `ADJ` | `JJ` `JJR` `JJT` |
| `ADV` | `RR` `RG` `RL` `RRR` `RRT` |
| `PREP` | `II` `IF` `IO` `IW` |
| `DET` | `AT` `AT1` `DD1` `DD2` `APPGE` `DA2` |
| `PRON` | `PPHS1` `PPH1` `PPX1` `PN1` `PNQS` |
| `CONJ` | `CC` `CCB` `CS` `CSA` `CST` |
| `PART` | `RP` `TO` |
| `NUM` | `MC` `MC1` `MD` `MF` |
| `INTJ` | `UH` |
| `EXPL` | `EX` (existential *there*) |
| `NEG` | `XX` (*not*, *n't*) |
| `GEN` | `GE` (germanic genitive *'s*) |
| `X` | `FW` `FO` `ZZ1` (foreign / unclassified) |

**`CLAWS7_ATTRIBUTES`** — maps each base tag to a dict of subclass attributes.
The attributes encode the linguistic information already present in the CLAWS7
tag, for example:

```python
# Nouns
'NN1':  {'proper_common': 'common',  'number': 'singular'}
'NP1':  {'proper_common': 'proper',  'number': 'singular'}
'NNT1': {'proper_common': 'common',  'number': 'singular', 'noun_subtype': 'temporal'}

# Verbs
'VBZ':  {'verb_class': 'copular',   'lemma': 'be',   'finiteness': 'finite',
         'verb_form': 'present', 'person': '3rd', 'number': 'singular'}
'VVG':  {'verb_class': 'lexical',   'finiteness': 'non-finite',
         'verb_form': 'present_participle'}
'VM':   {'verb_class': 'modal',     'finiteness': 'finite'}

# Determiners
'AT1':  {'det_type': 'article',       'number': 'singular'}
'DD1':  {'det_type': 'demonstrative', 'number': 'singular'}

# Pronouns
'PPHS1': {'pron_type': 'personal', 'person': '3rd', 'number': 'singular', 'case': 'subjective'}
'PPX1':  {'pron_type': 'reflexive', 'number': 'singular'}
```

### Ditto tags for multi-word units

CLAWS7 uses *ditto tags* to mark multi-word expressions that function as a
single unit. A tag like `II31` means: preposition `II`, 3-word unit, position 1.
The converter strips the trailing digits to get the base tag for lookup, and
records the original ditto information in the output as a `multiword_unit` field:

```json
{"token": "in",    "claws7_tag": "II31", "base_tag": "II",
 "pos": "PREP", "multiword_unit": {"length": 3, "position": 1},
 "attributes": {"prep_subtype": "general"}},
{"token": "front", "claws7_tag": "II32", "base_tag": "II",
 "pos": "PREP", "multiword_unit": {"length": 3, "position": 2}, ...},
{"token": "of",    "claws7_tag": "II33", "base_tag": "II",
 "pos": "PREP", "multiword_unit": {"length": 3, "position": 3}, ...}
```

### Conversion

`src/data/convert_claws7.py` reads `data/snli_CLAWS7.txt` and writes
`data/sampled/snli_claws7.json`:

```bash
python -m src.data.convert_claws7 \
    --input data/snli_CLAWS7.txt \
    --out   data/sampled/snli_claws7.json
```

### Output format

```json
{
  "schema_version": "1.0",
  "source": "SNLI",
  "tag_system": "CLAWS7",
  "n_sentences": 998,
  "sentences": [
    {
      "id": 1,
      "text": "A cute toddler is playing in the grass in the park",
      "tokens": [
        {"token": "A",       "claws7_tag": "AT1", "pos": "DET",
         "attributes": {"det_type": "article", "number": "singular"}},
        {"token": "cute",    "claws7_tag": "JJ",  "pos": "ADJ",
         "attributes": {"degree": "positive"}},
        {"token": "toddler", "claws7_tag": "NN1", "pos": "NOUN",
         "attributes": {"proper_common": "common", "number": "singular"}},
        {"token": "is",      "claws7_tag": "VBZ", "pos": "VERB",
         "attributes": {"verb_class": "copular", "lemma": "be",
                        "finiteness": "finite", "verb_form": "present",
                        "person": "3rd", "number": "singular"}},
        ...
      ]
    }
  ]
}
```

Key schema rules:
- `claws7_tag` — original tag from the file (preserves ditto info)
- `base_tag` — present only when the tag is a ditto variant
- `multiword_unit` — `{length, position}`, present only for ditto-tagged tokens
- `attributes` — present for all non-punctuation tokens; may be `{}` for tags like `XX` and `GE`
- Punctuation tokens carry `"pos": "PUNC"` and no `attributes` key

The 998-sentence dataset contains **14,593 tokens** across 15 POS classes.

## Directory layout

```
pos-annotations/
├── configs/
│   ├── gemini.yaml             # Gemini model aliases and generation config
│   ├── ptb_to_coarse.yaml      # PTB tag → Tier A mapping
│   ├── taxonomy.yaml           # Tier A classes, Tier B distinctions, labels, guidelines
│   └── tagger.yaml             # vLLM model config (model id, quantization, mock flag)
├── data/
│   ├── raw/                    # ingest manifest
│   ├── interim/
│   │   └── tokens.jsonl        # flat token rows from OntoNotes
│   ├── trial/
│   │   ├── pos_subclass_dataset_100.json  # 100 hand-labeled sentences (gold)
│   │   └── results/            # trial_100.py output (preds + agreement JSON)
│   ├── sampled/
│   │   └── snli_claws7.json    # 998 SNLI sentences with CLAWS7-derived attributes
│   ├── mining_batches/         # exported prompts + JSON batches (round 1)
│   │   └── <distinction_id>/
│   │       ├── MINING_PROMPT.md
│   │       ├── batch_000.json
│   │       └── batch_000.labeled.json
│   ├── mining_batches_round2/  # second-round batches for sparse distinctions
│   ├── mined_examples/
│   │   ├── icl/                # ICL training examples per distinction (vLLM backend)
│   │   └── heldout/            # held-out validation examples per distinction
│   ├── eval/
│   │   └── runs/<run_id>/
│   │       ├── metrics.json
│   │       ├── preds_*.jsonl
│   │       └── bootstrap_ci.json
│   └── annotations/
│       ├── corpus_1k.jsonl         # 36K+ annotations over 1K sentences (vLLM)
│       ├── corpus_1k_gemini.jsonl  # same format, Gemini backend
│       └── last_annotate_meta.json
├── prompts/
│   ├── distinction.jinja2              # vLLM tagger prompt template (single string)
│   ├── gemini_icl.yaml                 # handcrafted ICL examples for Gemini system prompt
│   └── mining/
│       └── distinction_mine.md.jinja2  # subagent mining prompt template
├── src/
│   ├── data/
│   │   ├── ingest.py           # OntoNotes download + flatten
│   │   ├── convert_claws7.py   # CLAWS7 txt → structured JSON converter
│   │   ├── sample_sentences.py # sample N sentences from HuggingFace datasets
│   │   └── pos_tagset.py       # integer → PTB string mapping
│   ├── mapping/
│   │   ├── claws7_to_json.py   # CLAWS7_POS and CLAWS7_ATTRIBUTES dicts
│   │   └── collapse.py         # PTB → Tier A + eligible distinctions
│   ├── mining/
│   │   ├── mine.py             # API-based sequential mining
│   │   ├── export_batches.py   # export prompts + JSON batches for subagents
│   │   ├── merge_batches.py    # merge labeled batches → mined_examples
│   │   ├── pool_utils.py       # shared data loading and sampling
│   │   └── backends.py         # MinerBackend interface (Stub, OpenAI)
│   ├── tagging/
│   │   ├── gemini_tagger.py    # Gemini API tagger (system/user roles, no local GPU)
│   │   ├── vllm_tagger.py      # vLLM integration with constrained JSON decoding
│   │   └── prompt_render.py    # prompt rendering: vLLM (Jinja2) + Gemini (messages)
│   └── eval/
│       ├── trial_100.py        # trial: annotate 100 gold sentences, compute agreement
│       ├── validate.py         # run tagger on held-out, report accuracy
│       ├── annotate.py         # annotate unseen corpus slice (vLLM or Gemini backend)
│       └── report_ci.py        # bootstrap 95% CIs
├── tests/
│   ├── test_collapse.py
│   └── test_merge_batches.py
├── requirements.txt
├── pyproject.toml
└── pytest.ini
```

## Models

### vLLM (original backend)

The default tagger is **Qwen/Qwen2.5-14B-Instruct-AWQ** (4-bit AWQ
quantization) served by vLLM with constrained JSON decoding. Each prompt
includes the distinction guidelines, up to 3 ICL examples (from mined silver
data) with marked target words, and the test sentence.

Configure in `configs/tagger.yaml`. Set `mock: true` to skip vLLM for CPU-only
testing.

### Gemini API (new backend)

Two models are configured in `configs/gemini.yaml`:

| Alias | Model | Use case |
|---|---|---|
| `flash` | `gemini-2.5-flash` | Fast, cost-efficient annotation and trial runs |
| `frontier` | `gemini-2.5-pro` | Highest quality annotations for the full 1K corpus |

The Gemini tagger uses **actual chat roles**:
- **System instruction** — task definition, expanded distinction guidelines, and
  handcrafted ICL examples from `prompts/gemini_icl.yaml` (2–4 per label, per
  distinction).
- **User message** — the marked sentence with the target word in `[[brackets]]`
  and the JSON output contract.

This is distinct from the vLLM approach which embeds pseudo-`### System` /
`### User` headings in a single prompt string.

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
- **verb_copular_prog_pass**: Not encoded in `pos_subclass_dataset_100.json`, so
  this distinction is excluded from trial_100 agreement scoring.
- **Ambiguous gold labels**: Some tokens in the 100-sentence dataset have
  compound labels (e.g. `"Finite_or_Non-finite(dependent)"`). These are excluded
  from trial agreement scoring (`skipped_ambiguous` field in output).

## Tests

```bash
pytest
```
