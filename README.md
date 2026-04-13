# pos-annotations

ICL-backed domain POS tagging: OntoNotes (PTB) → coarse Tier A → frontier silver for Tier B → local vLLM tagger → validation and bootstrap CIs.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

OntoNotes on Hugging Face uses a **dataset script**; this repo pins `datasets>=3.0,<4`. Set a writable cache if needed:

```bash
export HF_HOME="$(pwd)/.hf_cache"
```

## Pipeline (from repo root, `PYTHONPATH=.` or `pip install -e .`)

1. **Ingest** OntoNotes `english_v12` to `data/interim/tokens.jsonl` + manifest under `data/raw/`:

```bash
python -m src.data.ingest --splits train validation test
# smoke test:
python -m src.data.ingest --splits train --max-documents 20
# add another split later without wiping train rows:
python -m src.data.ingest --splits test --max-documents 50 --append
```

2. **Mine** ICL + held-out JSONL per `distinction_id` under `data/mined_examples/{icl,heldout}/`:

```bash
python -m src.mining.mine --miner stub
# frontier (requires OPENAI_API_KEY and `pip install openai`):
python -m src.mining.mine --miner openai
```

### 2b. Subagent-oriented mining (prompt + JSON batches)

Use this when you want **Cursor / CLI subagents** (or humans) to label ~**100 examples per distinction** in small batches instead of one big API loop.

1. **Export** prompts + batch files (default: `train` + `validation`, excludes `test`):

```bash
python -m src.mining.export_batches --out-root data/mining_batches --total-examples 100 --batch-size 8
# one distinction only:
python -m src.mining.export_batches --distinction-ids noun_proper_common
```

Each distinction gets a folder `data/mining_batches/<distinction_id>/` containing:

- `MINING_PROMPT.md` — paste this into the subagent as the system/instructions.
- `batch_000.json`, `batch_001.json`, … — each file is a JSON array of examples to label.

2. **Annotate** each batch: for every `batch_NNN.json`, the subagent writes **`batch_NNN.labeled.json`** in the **same folder**, a JSON array of `{ "example_id": "...", "gold_subclass": "<one of allowed>", "notes": "" }` in the **same order** as the input (see the prompt file for the exact contract).

3. **Merge** into the standard mined layout for the rest of the pipeline:

```bash
python -m src.mining.merge_batches --all --batches-root data/mining_batches
# or one distinction:
python -m src.mining.merge_batches --in-dir data/mining_batches/noun_proper_common
```

Use `--dry-run` to print counts without writing. Quotas default to `--icl-per-label 6` and `--heldout-per-label 3` (same idea as `mine.py`).

4. **Validate** tagger on held-out rows (use `--mock` without GPU):

```bash
python -m src.eval.validate --mock
```

5. **Bootstrap CIs** from a validate run directory:

```bash
python -m src.eval.report_ci --preds-dir data/eval/runs/<run_id> --out data/eval/runs/<run_id>/bootstrap_ci.json
# subset (e.g. simulate smaller held-out pool):
python -m src.eval.report_ci --preds-dir data/eval/runs/<run_id> --max-examples 500 --out data/eval/runs/<run_id>/bootstrap_ci_n500.json
```

6. **Annotate** ~1K unseen test sentences (excludes any `sentence_id` seen in mined data):

```bash
python -m src.eval.annotate --split test --num-sentences 1000 --mock
```

## Config

- [`configs/ptb_to_coarse.yaml`](configs/ptb_to_coarse.yaml) — PTB → Tier A.
- [`configs/taxonomy.yaml`](configs/taxonomy.yaml) — distinctions, labels, `enabled` flags (core + extensions).
- [`configs/tagger.yaml`](configs/tagger.yaml) — vLLM model id, quantization, `mock: true` for CPU tests.

## Tests

```bash
pytest
```
