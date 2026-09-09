# Same SQL, Different Risk

Do text-to-SQL correctness verifiers transfer across languages?

The paired design holds the candidate SQL, the database, the schema, and the
execution-derived correctness label fixed, and varies **only the language the
question is written in**. Any change in verifier confidence is therefore
attributable to the language of the request rather than to the query itself.

## Status

Vertical slice: 20 parallel MultiSpider dev questions, K=3 candidates each,
scored in `en`/`de`/`zh`. This exists to prove the design produces signal before
scaling to the full 300-400 question study described in the plan.

## Pipeline

```
uv run python scripts/01_prepare_slice.py   # sample items with executable gold SQL
uv run python scripts/02_generate.py        # local Qwen2.5-Coder-7B -> K candidates + exec labels
uv run python scripts/03_verify.py [backend]  # score each SQL under each language
uv run python scripts/04_report.py [backend]  # AUROC / ECE / paired deltas / threshold transfer
uv run python scripts/05_analyze.py [backend ...]  # bootstrapped CIs and significance
```

`backend` is `api` (Claude), or any `local*` label (uses `LOCAL_VERIFIER_MODEL`).
It names the output file, so runs from different verifiers sit side by side.

### Scaled runs

`scripts/build_colab_notebook.py` emits `notebooks/xsql_colab.ipynb`, which runs
the GPU half (generation + local verifiers, 7 languages) on Colab and exports
JSONL. Drop the JSONL into `data/slice/` and the local report/analysis scripts
work on it unchanged.

**Verification runs at batch size 1 on purpose.** Batched bf16 forward passes are
not numerically identical to unbatched ones — on this workload batching shifted
confidences by up to 6e-2, the same magnitude as the cross-language effects being
measured. Length-sorting and explicit `position_ids` both made it worse, so it is
batch-shape kernel nondeterminism rather than padding. Batch 1 reproduces
unbatched scoring exactly.

Generation is local and seeded; verification is API-based and disk-cached in
`data/verifier_cache.sqlite`, so re-running analysis costs nothing.

## Data

`dreamerdeo/multispider` supplies both the parallel questions and the 166 Spider
SQLite databases.

Use the **`with_english_value`** variant (set in `src/xsql/config.py`). The
`with_original_value` variant localizes literal values inside the questions
("JetBlue Airways" becomes "深圳航空公司") while leaving the gold SQL in English,
so a translated question no longer asks what the gold query answers. Under that
variant the verifier correctly assigns low confidence to non-English questions,
which would be easy to misread as cross-lingual verifier failure.

## Layout

| path | role |
|---|---|
| `src/xsql/config.py` | paths, language set, model names, slice sizes |
| `src/xsql/data.py` | parallel item loading, schema DDL extraction |
| `src/xsql/execute.py` | sandboxed SQLite execution, execution-match labeling |
| `src/xsql/generate.py` | local candidate generation |
| `src/xsql/verify.py` | LLM verifier with disk cache |
| `src/xsql/metrics.py` | AUROC, Brier, ECE, risk-coverage, threshold transfer |
