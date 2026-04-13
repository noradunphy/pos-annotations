# Mining task: `part_type`

You are labeling **one closed-class answer** per example for a single linguistic distinction.

## Distinction

- **Task id:** `part_type`
- **Parent POS (Tier A):** `PART`
- **Allowed labels (exact strings):** standard, phrasal_verb_particle

## Guidelines

phrasal_verb_particle = particle in verb-particle construction; else standard.

## Instructions

1. For **each** object in the input JSON batch file, read the `sentence` and the target token (`surface` at `token_index`).
2. Assign exactly **`gold_subclass`** — it **must** be one of the allowed labels above (character-for-character).
3. If the target is clearly **not a good instance** for this distinction (e.g. punctuation, tagging error), still pick the **closest** label and add a short `notes` field explaining why.
4. **Output:** a JSON **array** with **one object per input row, in the same order** as the input file.
5. Each output object **must** include:
   - `example_id` (copy from input unchanged)
   - `gold_subclass` (one of the allowed labels)
   - `notes` (optional string; may be empty)

## Output schema (per element)

```json
{
  "example_id": "<same as input>",
  "gold_subclass": "<one of allowed labels>",
  "notes": ""
}
```

## Save as

Write your completed annotations next to the batch file:

- Input: `batch_NNN.json`
- Your output: **`batch_NNN.labeled.json`** (same directory)

Do not rename `example_id` values; they are used to merge your work into the project pipeline.