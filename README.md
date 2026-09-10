# Same SQL, Different Decision

Code and data for "Same SQL, Different Decision" (ORACLE 2026): do text-to-SQL
correctness verifiers make the same execute/defer decision when only the
language of the question changes?

## Study in one paragraph

1,200 MultiSpider questions (`with_english_value` variant, train and dev, at
most 10 per database) over 163 Spider databases. Qwen2.5-Coder-7B-Instruct
generates K=5 SQL candidates per question from the English question only,
giving 6,000 candidates, each executed against its SQLite database and labeled
correct or incorrect by result-set match with the gold query (5,053 correct,
947 incorrect). Two verifiers, Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct,
score every candidate under each of seven question languages (en, de, es, fr,
ja, vi, zh) while the SQL, schema, database, and label stay fixed: 42,000 scores
per verifier, 84,000 in total. Additional score files cover two mitigations
(pivot translation and bilingual prompting, both verifiers, six non-English
languages) and an architecture-screening subset (300 questions, 39 databases,
eight verifier and verbalizer configurations). Every number in the paper (kept in the separate `sql_oracle_paper` repository) is
recomputed from these saved scores by a script under `scripts/`.

## Repository layout

| path | contents |
|---|---|
| `data/big/` | The study. `items.jsonl` (1,200 questions with all seven renderings and gold SQL), `candidates.jsonl` (6,000 candidates with execution labels), `scores_big-llama8b.jsonl` and `scores_big-qwen7b.jsonl` (42,000 rows each). |
| `data/mitigation/` | `scores_mitigation_{pivot,bilingual}-{llama8b,qwen7b}.jsonl` (36,000 rows each) and `pivot_translations.jsonl`. |
| `data/architecture/` | Eight `scores_architecture_{yesno,correctincorrect}-<verifier>.jsonl` files on the 300-question screening subset (10,500 rows each). |
| `data/slice/`, `data/colab/` | Pilot runs (20 and 80 questions). Not used in the paper. |
| `results/` | Markdown tables and the risk-coverage figure, one file per analysis script. |
| `scripts/` | Numbered analysis scripts and the `build_*_notebook.py` emitters. |
| `src/xsql/` | Library: config, data loading, sandboxed SQLite execution and labeling, generation, verifiers, metrics. |
| `notebooks/` | Colab notebooks that ran the GPU work (generation and scoring). Emitted by the build scripts. |

`data/verifier_cache.sqlite` is the disk cache for the pilot-era API verifier
and is not needed for anything below.

## Reproducing the paper from saved scores (no GPU)

Every table and figure except one (see the perplexity note) comes from
`data/` alone. Scripts 05, 07, and 08 read the run named by `XSQL_RUN`
(default `slice`, the pilot), so set it to `big`. The others read `data/big`,
`data/mitigation`, and `data/architecture` directly.

```
XSQL_RUN=big uv run python scripts/05_analyze.py big-llama8b big-qwen7b   # results/significance.md
XSQL_RUN=big uv run python scripts/07_schema_shift.py                       # results/schema_shift.md
XSQL_RUN=big uv run python scripts/08_calibration.py                        # results/calibration.md (defaults to big-llama8b big-qwen7b)
uv run python scripts/09_mitigation.py                # results/mitigation.md
uv run python scripts/10_offset_decomposition.py      # results/offset_decomposition.md
uv run python scripts/11_risk_coverage_figure.py      # results/risk_coverage.pdf and .png
uv run python scripts/12_decision_analysis.py         # results/decision_analysis.md
uv run python scripts/13_transport.py                 # results/transport.md
uv run python scripts/14_architecture.py              # results/architecture.md
uv run python scripts/15_within_model_sensitivity.py  # results/within_model_sensitivity.md
uv run python scripts/16_mitigation_decisions.py      # results/mitigation_decisions.md
uv run python scripts/17_translation_audit.py         # results/translation_audit.md
uv run python scripts/18_pivot_validation.py          # results/pivot_validation.md
uv run python scripts/19_final_numbers.py             # results/final_numbers.md
uv run python scripts/21_robustness.py                # results/robustness.md
```

Perplexity note: `scripts/06_perplexity.py` (`results/perplexity_big-llama8b.md`)
loads Llama-3.1-8B-Instruct through `transformers` to measure question
perplexity under the verifier, so it needs an accelerator (the script is set up
for Apple `mps`). It also reads `XSQL_RUN`:

```
XSQL_RUN=big uv run python scripts/06_perplexity.py big-llama8b
```

Its output is checked in, so nothing else depends on rerunning it.

Scripts 01 to 04 are the pilot pipeline (sample, generate, verify, report) and
run inference. They are not part of the paper's reproduction.

### Table and figure to script map

Numbering follows the compiled paper (`acl/main.pdf` in the `sql_oracle_paper` repository) (main-body Tables 1 to 6,
appendix Tables 7 to 23).

| paper item | script | results file |
|---|---|---|
| Table 1 (coverage at the English threshold, test split) | `13_transport.py` (raw row), `19_final_numbers.py` (section 5) | `transport.md`, `final_numbers.md` |
| Figure 1 (risk-coverage curves) | `11_risk_coverage_figure.py` | `risk_coverage.pdf` |
| Tables 2, 3, 14, 15 (harm and transition counts, both verifiers) | `12_decision_analysis.py` | `decision_analysis.md` |
| Tables 4, 10 (calibrators A to D) | `08_calibration.py` | `calibration.md` |
| Tables 5, 11, 12 (score transport, mean and per language) | `13_transport.py` | `transport.md` |
| Tables 6, 13 (bilingual and pivot conditions, mean and per language) | `16_mitigation_decisions.py`, `18_pivot_validation.py` | `mitigation_decisions.md`, `pivot_validation.md` |
| Table 7 (design comparison with Zhou et al.) | none, prose only | |
| Table 8 (full-data, in-sample coverage and AUROC) | `05_analyze.py` | `significance.md` |
| Table 9 (per-language AUROC on the test split) | `19_final_numbers.py` (section 5) | `final_numbers.md` |
| Table 16 (offset decomposition) | `10_offset_decomposition.py` | `offset_decomposition.md` |
| Tables 17, 18, 19 (translation audit, perplexity, schema shift) | `17_translation_audit.py`, `06_perplexity.py`, `07_schema_shift.py` | `translation_audit.md`, `perplexity_big-llama8b.md`, `schema_shift.md` |
| Tables 20, 21 (risk targets, repeated splits) | `21_robustness.py` | `robustness.md` |
| Table 22 (worked example, `train:0#0`) | `19_final_numbers.py` (section 6; re-executes the two queries against the SQLite database) | `results/final_numbers.md` |
| Table 23 (architecture screening) | `14_architecture.py`, `15_within_model_sensitivity.py` | `architecture.md`, `within_model_sensitivity.md` |
| Pivot vs quantile paired bootstrap, quantile-rule spread over ten splits, cross-model English flips (bf16 vs 4-bit, dense vs MoE), Aya saturation, canonical thresholds, AUROC gaps, and risk intervals | `19_final_numbers.py` | `final_numbers.md` |

`scripts/09_mitigation.py` reports the mitigations by coverage spread only; the
paper uses the decision-level versions from scripts 16 and 18.

## Canonical protocol

Stated once here; every primary result uses it.

- The 163 databases are shuffled with `numpy.random.default_rng(0)`. The first
  81 are the calibration split, the remaining 82 the test split. Every
  threshold and every fitted correction is estimated on calibration databases
  and evaluated on test databases.
- The English threshold is the most permissive score whose selective risk on
  calibration-split English scores is within a 10% target, evaluated at distinct
  score values only (`xsql.metrics.threshold_at_risk`), since tied scores cannot
  be separated by any threshold. This puts Qwen's threshold at 0.9999971, the
  score ceiling.
- Test split: 3,005 candidates, 2,519 correct and 486 incorrect.

Repeated-split results (`21_robustness.py`, `19_final_numbers.py`) use seeds
0 to 9 with the same rule and say so where they appear.

## Execution labels

Labels follow the Spider execution-accuracy convention, implemented in
`src/xsql/execute.py` (`rows_match`, `result_match`): the candidate's result set
must equal the gold result set, row order is enforced only when the gold query
has `ORDER BY`, and column order is never enforced (some permutation of the
candidate's columns must match).

Data exported by an older notebook version compared columns positionally, which
mislabels column-reordered SELECT lists as incorrect. The relabeling step
re-executes every candidate and gold query from the cached SQLite databases (no
model inference) and rewrites the `correct` field in `data/big/candidates.jsonl`
and in every `scores_*.jsonl` under `data/big`, `data/mitigation`, and
`data/architecture`, recording each change in `results/relabel_column_order.md`:

```
XSQL_RUN=big python scripts/20_relabel_column_order.py
```

Run it on any data exported by a notebook that predates the column-order
comparator. On the shipped data it has already been applied (159 candidates
across 45 questions moved from incorrect to correct, none the other way).

## Regenerating from scratch (GPU)

The GPU work runs in the Colab notebooks under `notebooks/`, which are emitted
by the build scripts and write JSONL to Drive for copying into `data/`:

| build script | notebook | produces |
|---|---|---|
| `scripts/build_colab_notebook.py` | `notebooks/xsql_colab.ipynb` | `data/big/` items, candidates, and both score files |
| `scripts/build_mitigation_notebook.py` | `notebooks/xsql_mitigation.ipynb` | `data/mitigation/` |
| `scripts/build_architecture_notebook.py` | `notebooks/xsql_architecture.ipynb` | `data/architecture/` |
| `scripts/build_endtoend_notebook.py` | `notebooks/xsql_endtoend.ipynb` | end-to-end condition, not used in the paper |

The mitigation and architecture notebooks re-score the existing
`data/big/items.jsonl` and `data/big/candidates.jsonl` (upload both to Drive
first). They never regenerate SQL.

Checkpoints:

| role | checkpoint |
|---|---|
| generator | `Qwen/Qwen2.5-Coder-7B-Instruct`, temperature 0.8, top_p 0.95, 5 samples, max 256 new tokens |
| verifiers | `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, bf16 |
| pivot translator | `Qwen/Qwen2.5-7B-Instruct` |
| architecture screening | `Qwen/Qwen3-4B-Instruct-2507` (bf16 and nf4), `Qwen/Qwen3-30B-A3B-Instruct-2507`, `CohereLabs/aya-expanse-8b` |

Verifier scoring (paper Section 3.3, `src/xsql/verify_local.py`): the verifier
sees the schema, one language's question, and the candidate SQL, and is asked to
answer with a single word, Yes or No. The score is read from the next-token
distribution as P(Yes) / (P(Yes) + P(No)), with probability mass summed over
casing and whitespace variants of each verdict token. There is no generated
numeric confidence. The `correctincorrect` architecture files change only the
final prompt line to CORRECT or INCORRECT.

Settings that are not negotiable (each one, when wrong, produced plausible
output with a different conclusion; see paper Appendix F):

1. MultiSpider variant `with_english_value`. `with_original_value` localizes
   literals inside the question while the gold SQL stays English, which
   fabricates a large cross-lingual effect.
2. Verifier scoring at batch size 1. Batched bf16 forward passes shifted
   confidences by up to 6e-2, the size of the effect under study. Length
   sorting and explicit `position_ids` made it worse.
3. Score clipping epsilon 1e-12 before any logit transform (`08_calibration.py`,
   `13_transport.py`). At 1e-6 Qwen's distinct scores collapse from 10,845 to
   5,619. `08_calibration.py` asserts AUROC invariance under every monotone
   calibrator as a tripwire.
4. Per-database caps bound the sample: Spider's dev split has 20 databases, so
   the study samples from train and dev with a cap of 10 per database
   (`MAX_PER_DB` in the notebook).

Item ids are strings such as `train:812`; candidate ids append `#k`.

## Environment

Python 3.12 (`.python-version`). Install from the lockfile:

```
uv sync
```

The MultiSpider questions and the Spider SQLite databases are fetched
automatically from the HuggingFace dataset `dreamerdeo/multispider` on first
use (`xsql.config.data_root`), so the relabeling step and the pilot scripts need
network access once. The analysis scripts read only `data/` and `results/`.

Two scripts need the Spider SQLite databases as well as the saved scores:
`scripts/20_relabel_column_order.py` re-executes every candidate and gold query,
and section 6 of `scripts/19_final_numbers.py` re-executes the worked example.
Both use Python's built-in `sqlite3` module against the databases fetched
above (about 170 database directories under the MultiSpider snapshot), read-only
with a 30-second statement timeout. No other analysis script touches the
databases; everything else reads only the JSONL score files.

`src/xsql/config.py` still carries the pilot defaults (`XSQL_RUN=slice`, three
languages, 20 questions); the notebooks set the study-scale values themselves.

## Duplicate candidates

The 6,000 sampled candidates contain 3,138 distinct (database, question, SQL)
triples. Duplicates are kept: they are what the generator produced at K=5, and
the question-clustered bootstrap resamples whole questions, so repeated SQL
within a question does not inflate the intervals.

## License and attribution

Code license: see LICENSE.

MultiSpider, Spider, and the model weights listed above are used under their own
terms and are not redistributed here. `data/` contains only derived artifacts
(sampled question ids and text, generated SQL, execution labels, and verifier
scores).
