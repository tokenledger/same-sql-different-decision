"""Emit the Colab notebook for the ARCHITECTURE / label-word robustness experiment.

Background: `scripts/build_colab_notebook.py` established that AUROC transfers
across languages but the score SCALE does not, so one English-calibrated
threshold flips 12-23% of individual execute/defer decisions on IDENTICAL SQL
(Llama-3.1-8B, Qwen2.5-7B-Instruct -- both dense decoder-only Transformers).
That is family-level replication, not architectural or training-distribution
diversity. This notebook adds three things, in priority order:

  1. LABEL-WORD ROBUSTNESS (cheap, done first, on the EXISTING two verifiers).
     `src/xsql/verify_local.py` already reads P(Yes)/(P(Yes)+P(No)) off the
     next-token logits of a forced-choice prompt -- there is no free-form
     numeric confidence step to interrogate. What IS still open is whether the
     12-23% flip rate is an artifact of the specific verdict tokens "Yes"/"No".
     This re-scores with the prompt's final line changed to
     "Answer with a single word, CORRECT or INCORRECT." -- everything else
     (system message, schema, question, SQL, temperature=implicit-greedy
     single forward pass) byte-identical to `src/xsql/verify_local.py`. This
     also probes whether Qwen2.5-7B's ceiling saturation (82% of English
     scores > 0.99, 10%-risk threshold at 0.99999996, which broke isotonic
     transport in scripts/13_transport.py) is specific to the word "Yes".

  2. DENSE vs SPARSE-MoE, matched within one family, NON-THINKING mode:
       Qwen/Qwen3-4B-Instruct-2507          (dense)
       Qwen/Qwen3-30B-A3B-Instruct-2507     (sparse MoE, ~30.5B total / ~3.3B active)

  3. Aya Expanse 8B (CohereLabs/aya-expanse-8b) as a TRAINING-DISTRIBUTION
     control -- explicitly multilingual post-training, same dense-decoder
     backbone class as the other verifiers.

EFFICIENCY GATE: none of this re-runs all 84,000 (2 verifiers x 42,000)
judgments. Everything here scores a prespecified, seeded, database-disjoint
300-question subset of the existing 1200 items (300 x 5 candidates x 7
languages = 10,500 scorings/model/condition, deduplicated further by shared
SQL text -- see the SUBSET cell for the exact, locally-verified counts). A
model is only a candidate for scaling to the full 1200 if it clears ALL of the
gate checks printed after it is scored (see GATE_CHECK cell): English AUROC
< ~0.95, a clear majority of scores not within 1e-6 of {0,1}, enough incorrect
candidates on both sides of its English threshold, and no missing Yes/No (or
CORRECT/INCORRECT) probability mass.

This notebook reuses the EXISTING items/candidates/labels from `data/big/`
(upload `items.jsonl` and `candidates.jsonl` to Drive `MyDrive/xsql_out/` --
they are already there from the original `build_colab_notebook.py` run) and
only re-scores. It never regenerates SQL, which would silently break
comparability with the established results.

This script only emits the notebook; it does not run the GPU job. Validation
(subset reproducibility/disjointness against the real data/big files, example
prompts under both label schemes, ast.parse of every code cell, a re-read text
audit) is done separately and reported back in the task's final summary, not
embedded here -- mirroring how `scripts/build_mitigation_notebook.py` and its
sibling validator are split.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "xsql_architecture.ipynb"

MD_INTRO = """# Same SQL, Different Risk — ARCHITECTURE stage

Both verifiers used so far (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct) are
dense decoder-only Transformers, so the established result --- an
English-calibrated threshold flips 12-23% of individual execute/defer
decisions on identical SQL when only the question's language changes --- is
so far only shown to replicate at the *model-family* level. This notebook adds:

| # | addition | verifiers | label scheme(s) | status |
|---|---|---|---|---|
| 1 | label-word robustness | Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct | Yes/No (full 42k already in `data/big/`; this notebook also re-runs it on just the 300-subset as a harness sanity check) **and** CORRECT/INCORRECT (**NEW**, subset only) | cheap, done first |
| 2 | dense vs sparse-MoE, precision-matched | Qwen3-4B-Instruct-2507 at bf16 AND nf4 (dense), Qwen3-30B-A3B-Instruct-2507 at auto-detected precision (MoE) | Yes/No (**NEW**) | needs gate check before any scale-up; headline contrast is same-precision dense vs MoE, not bf16-dense vs nf4-MoE |
| 3 | multilingual training-distribution control | CohereLabs/aya-expanse-8b | Yes/No (**NEW**) | needs gate check before any scale-up |

**Why precision-matching matters for item 2.** `Qwen3-30B-A3B-Instruct-2507`
(~30.5B total params, all resident regardless of the ~3.3B active per token --
see the runtime/VRAM markdown) may not fit in bf16 on the attached GPU and can
fall back to 4-bit (nf4) quantization. A dense-vs-MoE comparison is only
interpretable as an ARCHITECTURE effect if both sides run at the SAME
numerical precision: on this exact workload, batching alone -- a *smaller*
perturbation than 4-bit quantization -- was measured to shift verifier
confidences by up to 6e-2, the same magnitude as the cross-lingual decision
flips this whole study is about. A bf16-dense-vs-4bit-MoE comparison would
confound architecture with precision and the resulting number would not mean
what it looks like it means. So this notebook:
- scores the dense model, `Qwen3-4B-Instruct-2507`, at **both** bf16 and nf4
  (`qwen3-4b-bf16`, `qwen3-4b-nf4`) -- required for precision-matching, and
  also a free, independently useful result: how much does quantization ALONE
  perturb cross-lingual decision consistency, architecture held fixed?
- auto-detects whichever precision the MoE model actually runs at (bf16 if
  the GPU has >= 65GB, else nf4), and prints an explicit
  **`ACTIVE PRECISION-MATCHED COMPARISON`** line naming which dense run is the
  correct architecture-only comparator for that MoE run. See the RUN_DENSE_MOE
  cell.

**Both Qwen3-Instruct-2507 checkpoints are non-thinking-only releases** (unlike
the hybrid `Qwen3-*-Base`/plain `Qwen3-*` checkpoints, which support an
`enable_thinking` toggle and default to reasoning mode). The `-Instruct-2507`
suffix specifically denotes the non-thinking variant per the model cards; their
tokenizer chat templates do not emit `<think>` scaffolding and there is no
`enable_thinking` argument to set. We rely on that documented default rather
than on any local verification (no GPU here) --- **the RUN_CHECKS cell below
asserts the tokenizer output contains no `<think>` token before any scoring
happens, and aborts loudly if it does**, so a wrong assumption fails fast
instead of silently contaminating scores with reasoning-mode behavior.

**Efficiency gate.** Every scoring condition here runs on a single
prespecified, seeded, **database-disjoint** 300-question subset of the
existing 1200 items in `data/big/items.jsonl` (see the SUBSET cell for the
selection rule and locally-verified counts: 300 items / 39 databases / 1500
candidates / 79.4% correct, matching the full set's 81.6%). A new verifier is
only a candidate for scaling to the full 1200 if it clears the printed GATE
CHECK: English AUROC below ~0.95, a clear majority of scores NOT within 1e-6
of {0,1}, a nonzero count of incorrect candidates on both sides of its English
threshold, and zero missing Yes/No (or CORRECT/INCORRECT) probability mass.

**This notebook reuses the existing items/candidates/labels from the `big`
run.** Upload `items.jsonl` and `candidates.jsonl` from `data/big/` on your
machine to `My Drive/xsql_out/` before running (see the LOAD cell). It never
regenerates SQL.

**Runtime and VRAM:** see the final markdown cell. The MoE model
(`Qwen3-30B-A3B-Instruct-2507`) needs the FULL ~30.5B parameters resident in
VRAM regardless of the ~3.3B active per token -- in bf16 that is ~61GB of
weights alone, which will NOT fit on an A100-40GB. Read the runtime cell
before starting that model.
"""

SETUP = """!pip -q install transformers accelerate huggingface_hub bitsandbytes
import torch, os, time
from transformers import AutoModelForCausalLM, AutoTokenizer
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE — set Runtime > Change runtime type > GPU")
if torch.cuda.is_available():
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
"""

AUTH = """from huggingface_hub import login
try:
    from google.colab import userdata
    token = userdata.get('HF_TOKEN')
except Exception:
    import getpass
    token = getpass.getpass('HF token (needs Llama-3.1 access): ')
login(token)
"""

CONFIG = '''# Mirrors src/xsql/config.py and scripts/build_colab_notebook.py's CONFIG cell
# so scoring internals are byte-identical to the already-done big run wherever
# unchanged (schema DDL derivation, execution-match labeling rule).
REPO_ID  = "dreamerdeo/multispider"
VARIANT  = "with_english_value"   # with_original_value localizes question literals
                                   # but leaves gold SQL in English -- breaks the paired design.
LANGUAGES = ["en", "de", "es", "fr", "ja", "vi", "zh"]
PIVOT     = "en"

# --- Verifiers -------------------------------------------------------------
# Group 1: existing verifiers. CORRECT/INCORRECT is the new condition; Yes/No
# is also re-run here on just the 300-subset (cheap) as a harness sanity check
# against the full-scale data/big/scores_big-*.jsonl numbers, not because the
# original Yes/No scoring needs replacing.
EXISTING_VERIFIERS = {
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen7b":  "Qwen/Qwen2.5-7B-Instruct",
}
# Group 2: dense vs sparse-MoE, matched family, both non-thinking (see intro
# markdown -- the -Instruct-2507 suffix specifically denotes non-thinking).
#
# PRECISION MATCHING (see intro markdown "Why precision-matching" cell for the
# full argument): Qwen3-30B-A3B may not fit in bf16 on the attached GPU (see
# MOE_MIN_BF16_GB below), so it may end up scored in 4-bit (nf4). A dense-vs-MoE
# contrast is only interpretable as an ARCHITECTURE effect if both sides are at
# the SAME numerical precision -- batching alone (a far smaller perturbation
# than 4-bit quantization) was measured to shift confidences by up to 6e-2 on
# this exact workload, the same magnitude as the cross-lingual effects under
# study, so bf16-vs-4bit noise could swamp or masquerade as an architecture
# effect. This notebook therefore scores the DENSE model at BOTH precisions
# (qwen3-4b-bf16, qwen3-4b-nf4) and, once the MoE's actual precision is known
# at runtime, treats the dense run at that SAME precision as the headline
# comparator. See the RUN_DENSE_MOE cell for the explicit branch and printed
# "ACTIVE PRECISION-MATCHED COMPARISON" line.
DENSE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MOE_MODEL_ID   = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MOE_MIN_BF16_GB = 65.0   # ~61GB of bf16 weights + headroom for activations/KV cache
# Group 3: multilingual-post-trained control, same dense-decoder backbone class.
CONTROL_VERIFIERS = {
    "aya-expanse-8b": "CohereLabs/aya-expanse-8b",
}

LABEL_SCHEMES = ["yesno", "correctincorrect"]

import pathlib
try:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE = pathlib.Path("/content/drive/MyDrive")
except Exception:
    DRIVE = pathlib.Path("/content/out_local")
    DRIVE.mkdir(exist_ok=True, parents=True)

SOURCE_DIR = DRIVE / "xsql_out"                  # where the ORIGINAL run wrote items/candidates/scores
OUT = DRIVE / "xsql_architecture_out"            # this notebook's own outputs, kept separate
OUT.mkdir(exist_ok=True, parents=True)
print("source:", SOURCE_DIR)
print("out:   ", OUT)
'''

DOWNLOAD = '''from huggingface_hub import snapshot_download
import pathlib

# Schema DDL is derived from the sqlite files at scoring time; it is not stored
# in items.jsonl. Question files are not needed: items.jsonl already carries
# every language's question text.
ROOT = pathlib.Path(snapshot_download(
    REPO_ID, repo_type="dataset",
    allow_patterns=["dataset/spider/database/**"],
))
DBDIR = ROOT / "dataset/spider/database"
print(DBDIR, len(list(DBDIR.iterdir())), "databases")
'''

DATA_HARNESS = '''import json, sqlite3

def db_path(db_id): return DBDIR / db_id / f"{db_id}.sqlite"

def schema_ddl(db_id):
    with sqlite3.connect(f"file:{db_path(db_id)}?mode=ro", uri=True) as c:
        rows = c.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL").fetchall()
    return "\\n\\n".join(s.strip() for (s,) in rows)

items_path = SOURCE_DIR / "items.jsonl"
cand_path  = SOURCE_DIR / "candidates.jsonl"
assert items_path.exists() and cand_path.exists(), (
    "items.jsonl / candidates.jsonl not found on Drive. Upload data/big/items.jsonl "
    "and data/big/candidates.jsonl to My Drive/xsql_out/ before running this notebook -- "
    "this run must reuse the existing SQL/labels, never regenerate them."
)
all_items = [json.loads(l) for l in items_path.open()]
all_cands = [json.loads(l) for l in cand_path.open()]
print(f"loaded from Drive: {len(all_items)} items, {len(all_cands)} candidates")
BY_IDX_ALL = {i["idx"]: i for i in all_items}
'''

SUBSET = '''# Prespecified, seeded, DATABASE-DISJOINT 300-question subset of the 1200
# items (efficiency gate: 300 x 5 candidates x 7 languages = 10,500 scorings
# per model/label-scheme instead of 42,000).
#
# "Database-disjoint" here means: databases are sampled first (not
# individual items), and every database that is included contributes ALL of
# its items in data/big/items.jsonl -- no database is ever split between
# "in the subset" and "not in the subset". This keeps the subset compatible
# with the same fit/test database-disjoint-split methodology used by
# scripts/12_decision_analysis.py and scripts/13_transport.py (a threshold
# fit on half the subset's databases and evaluated on the other half never
# leaks a partially-seen database across the split), and makes the subset
# itself independently reproducible from nothing but this seed.
#
# Locally verified against the real data/big files (scripts/14_architecture.py
# reruns this exact function and asserts the same counts):
#   300 items, 39 databases, 1500 candidates, 79.4% correct
#   (full 1200-item set: 81.6% correct -- the subset is representative).
SUBSET_SEED = 20260728
N_TARGET = 300

def select_subset(items, seed=SUBSET_SEED, n_target=N_TARGET):
    from collections import defaultdict
    import random
    by_db = defaultdict(list)
    for it in items:
        by_db[it["db_id"]].append(it)
    dbs = sorted(by_db)
    rng = random.Random(seed)
    rng.shuffle(dbs)
    chosen, chosen_dbs = [], []
    for db in dbs:
        grp = by_db[db]
        if len(chosen) + len(grp) <= n_target:
            chosen.extend(grp)
            chosen_dbs.append(db)
        if len(chosen) >= n_target:
            break
    return chosen, chosen_dbs

chosen, chosen_dbs = select_subset(all_items)
chosen_idx = {it["idx"] for it in chosen}
records = [c for c in all_cands if c["item_idx"] in chosen_idx]
BY_IDX = {it["idx"]: it for it in chosen}

ncorr = sum(c["correct"] for c in records)
print(f"subset: {len(chosen)} items across {len(chosen_dbs)} databases, "
      f"{len(records)} candidates, {ncorr}/{len(records)} = {ncorr/len(records):.1%} correct")
assert len(chosen) == N_TARGET, f"expected exactly {N_TARGET} items, got {len(chosen)}"
# Disjointness check: every chosen database's FULL item set must be present.
by_db_full = {}
for it in all_items: by_db_full.setdefault(it["db_id"], []).append(it["idx"])
for d in chosen_dbs:
    on_disk = set(by_db_full[d])
    in_subset = {it["idx"] for it in chosen if it["db_id"] == d}
    assert on_disk == in_subset, f"database {d} only partially included -- selection bug"
print("database-disjointness check passed: every included database is fully included")
'''

PROMPTS = '''# Two label-word schemes for the SAME forced-choice-logprob mechanism as
# src/xsql/verify_local.py: P(verdict-A) / (P(verdict-A) + P(verdict-B)) off
# the next-token logits of a single greedy forward pass. Nothing else in the
# prompt changes between schemes -- only the final instruction line and the
# surface-form token lists.
VER_SYSTEM = ("You are a meticulous database engineer auditing text-to-SQL output. "
              "You judge whether a candidate SQLite query correctly answers a user's "
              "question against a given schema. The question may be written in any "
              "language; judge the query on its merits, not on the question's language.")

VER_TEMPLATE_YESNO = """Database schema:
{schema}

User question:
{question}

Candidate SQLite query:
{sql}

Does this query correctly answer the user's question? Answer with a single word, Yes or No."""

VER_TEMPLATE_CI = """Database schema:
{schema}

User question:
{question}

Candidate SQLite query:
{sql}

Does this query correctly answer the user's question? Answer with a single word, CORRECT or INCORRECT."""

TEMPLATES = {"yesno": VER_TEMPLATE_YESNO, "correctincorrect": VER_TEMPLATE_CI}

YES_FORMS = ["Yes", " Yes", "yes", " yes", "YES", "Y"]
NO_FORMS  = ["No", " No", "no", " no", "NO", "N"]
# Deliberately NOT including single-letter "C"/"I" abbreviations here: "I" is
# an extremely common token (the pronoun) in English text and would inflate
# INCORRECT's probability mass for reasons that have nothing to do with the
# verdict, unlike "Y"/"N" which are conventional Yes/No abbreviations. This is
# an intentional asymmetry with YES_FORMS/NO_FORMS, documented rather than
# silently copied.
CORRECT_FORMS   = ["Correct", " Correct", "correct", " correct", "CORRECT"]
INCORRECT_FORMS = ["Incorrect", " Incorrect", "incorrect", " incorrect", "INCORRECT"]
VERDICT_FORMS = {
    "yesno":            (YES_FORMS, NO_FORMS),
    "correctincorrect": (CORRECT_FORMS, INCORRECT_FORMS),
}

def first_ids(tok, forms):
    return sorted({tok.encode(f, add_special_tokens=False)[0] for f in forms if tok.encode(f, add_special_tokens=False)})

# Quick sanity print, human-eyeball-able (data/big real schema + real candidate).
_demo_item = chosen[0]
_demo_cand = next(c for c in records if c["item_idx"] == _demo_item["idx"])
_demo_schema = schema_ddl(_demo_item["db_id"])
print("--- YES/NO prompt ---")
print(VER_TEMPLATE_YESNO.format(schema=_demo_schema, question=_demo_item["questions"]["en"], sql=_demo_cand["sql"]))
print()
print("--- CORRECT/INCORRECT prompt ---")
print(VER_TEMPLATE_CI.format(schema=_demo_schema, question=_demo_item["questions"]["en"], sql=_demo_cand["sql"]))
'''

RUN_CHECKS = '''# Fails loudly (rather than silently scoring garbage) if a new model tokenizes
# the verdict words in a way that collides between classes, or if a Qwen3
# checkpoint turns out to emit <think> scaffolding despite the -Instruct-2507
# non-thinking documentation this notebook relies on (see intro markdown).

def assert_no_token_collision(tok, label_scheme):
    pos_forms, neg_forms = VERDICT_FORMS[label_scheme]
    pos_ids, neg_ids = first_ids(tok, pos_forms), first_ids(tok, neg_forms)
    overlap = set(pos_ids) & set(neg_ids)
    assert not overlap, f"{label_scheme} verdict token ids collide: {overlap} -- tokenizer-specific, check surface forms"
    return pos_ids, neg_ids

def assert_non_thinking(tok, model_id):
    """Cheap, no-GPU-forward-pass check: render the chat template and make sure
    it doesn't itself inject <think> scaffolding by default. This cannot prove
    the MODEL won't reason internally, but it does verify our reliance on the
    documented non-thinking default isn't contradicted by the tokenizer's own
    template, and it costs nothing extra."""
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "placeholder"}], tokenize=False, add_generation_prompt=True)
    assert "<think>" not in rendered, (
        f"{model_id}: chat template emitted <think> -- this checkpoint may not be the "
        f"non-thinking variant this notebook assumes. Stop and re-check the model card.")
    print(f"  {model_id}: non-thinking template check passed")
'''

VERIFY_HARNESS = '''# One resumable scorer shared by every (verifier, label_scheme) condition in
# this notebook. Same dedup + batch-1 discipline as the original big run:
# identical (db_id, sql, question, label_scheme) quadruples score identically,
# so unique forward passes are computed once and fanned out; batch size is 1
# because batched bf16 forward passes are not numerically identical to
# unbatched ones on this workload (up to 6e-2 drift, the same magnitude as the
# effects under study). RESUMABLE: raw per-unique-key results are appended to
# Drive as they are computed and reloaded on restart, so a disconnect costs at
# most the in-flight forward pass.
BATCH = 1
SCHEMAS = {d: schema_ddl(d) for d in {r["db_id"] for r in records}}

def build_jobs(label_scheme):
    template = TEMPLATES[label_scheme]
    jobs = {}
    for r in records:
        it = BY_IDX[r["item_idx"]]
        for lang in LANGUAGES:
            q = it["questions"][lang]
            jobs.setdefault((r["db_id"], r["sql"], lang, q), []).append(
                (r["candidate_id"], r["item_idx"], r["split"], lang, r["correct"]))
    return jobs

def load_model_precision(model_id, precision):
    """Force-load `model_id` at exactly `precision` ("bf16" or "nf4") -- no
    auto-fallback inside this function. The CALLER decides precision
    explicitly (see RUN_DENSE_MOE), so which numerical precision every score
    file was produced at is always visible in the run log and in the
    verifier's own name, never silently decided deep in a helper."""
    tok = AutoTokenizer.from_pretrained(model_id)
    if precision == "bf16":
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
    elif precision == "nf4":
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="cuda").eval()
    else:
        raise ValueError(f"unknown precision {precision!r}, expected 'bf16' or 'nf4'")
    return tok, model

def run_verifier(name, model_id, label_scheme, precision="bf16"):
    out_name = f"scores_architecture_{label_scheme}-{name}.jsonl"
    raw_path = OUT / f"raw_{label_scheme}_{name}.jsonl"
    jobs = build_jobs(label_scheme)
    template = TEMPLATES[label_scheme]

    conf = {}
    if raw_path.exists():
        for line in raw_path.open():
            rec = json.loads(line)
            conf[tuple(rec["key"])] = rec["confidence"]
        print(f"  {name}/{label_scheme}: resuming, {len(conf)} unique passes already done")

    keys = [k for k in jobs if k not in conf]
    if keys:
        print(f"  {name}/{label_scheme}: loading at precision={precision}")
        tok, model = load_model_precision(model_id, precision)
        tok.padding_side = "left"
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        assert_non_thinking(tok, model_id)
        pos_ids, neg_ids = assert_no_token_collision(tok, label_scheme)

        raw_f = raw_path.open("a")
        t0 = time.monotonic()
        for s, key in enumerate(keys, 1):
            d, sql, lang, q = key
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": VER_SYSTEM},
                 {"role": "user", "content": template.format(schema=SCHEMAS[d], question=q, sql=sql)}],
                tokenize=False, add_generation_prompt=True)
            enc = tok([prompt], return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                logits = model(**enc).logits[:, -1, :].float()
            p = torch.softmax(logits, dim=-1)
            pos, neg = p[:, pos_ids].sum().item(), p[:, neg_ids].sum().item()
            c = (pos / (pos + neg)) if (pos + neg) > 0 else None
            conf[key] = c
            raw_f.write(json.dumps({"key": list(key), "confidence": c}, ensure_ascii=False) + "\\n")
            raw_f.flush()
            if s % 200 == 0 or s == len(keys):
                el = time.monotonic() - t0
                print(f"  {name}/{label_scheme}: {s}/{len(keys)} new | ETA {(len(keys)-s)*el/max(s,1)/60:.0f}m", flush=True)
        raw_f.close()
        del model; torch.cuda.empty_cache()
    else:
        print(f"  {name}/{label_scheme}: all {len(jobs)} unique passes already done")

    out = []
    for k, fanout in jobs.items():
        d, sql, lang, q = k
        for cid, item_idx, split, l, correct in fanout:
            out.append(dict(candidate_id=cid, item_idx=item_idx, split=split, db_id=d,
                            lang=l, verifier=name, label_scheme=label_scheme,
                            correct=correct, confidence=conf[k]))
    with open(OUT / out_name, "w") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\\n")
    miss = sum(1 for r in out if r["confidence"] is None)
    print(f"{name}/{label_scheme}: wrote {len(out)} scores to {out_name} ({miss} without verdict-token mass)")
    return out
'''

GATE_CHECK = '''# Printed after every NEW model/condition (item 1 rescoring included, though
# item 1's models already passed this gate at full scale in the big run --
# printing it here is a cheap consistency check, not a new decision point).
# A model is a scale-up CANDIDATE only if ALL FOUR checks pass.
from sklearn.metrics import roc_auc_score
import numpy as np

def gate_check(name, label_scheme, scores):
    en = [r for r in scores if r["lang"] == "en"]
    labels = np.array([r["correct"] for r in en], dtype=float)
    confs = np.array([r["confidence"] if r["confidence"] is not None else np.nan for r in en])
    n_missing = int(np.isnan(confs).sum())
    valid = ~np.isnan(confs)

    auroc = roc_auc_score(labels[valid], confs[valid]) if len(set(labels[valid].tolist())) > 1 else float("nan")
    sat_frac = float(((np.abs(confs[valid]) < 1e-6) | (np.abs(1 - confs[valid]) < 1e-6)).mean())
    not_saturated = sat_frac < 0.5

    # Enough incorrect candidates on both sides of the (English, this-scheme's
    # own) risk-feasible threshold. Mirrors threshold_at_risk in
    # src/xsql/metrics.py at target risk 0.10, computed inline to avoid an
    # extra import dependency inside the Colab runtime.
    def threshold_at_risk(labels, scores, target_risk=0.10):
        best = None
        for t in np.unique(scores)[::-1]:
            acc = scores >= t
            if acc.any() and float(1 - labels[acc].mean()) <= target_risk:
                best = float(t)
        return best

    thr = threshold_at_risk(labels[valid], confs[valid])
    if thr is None:
        both_sides = False
        n_below_incorrect = n_above_incorrect = 0
    else:
        incorrect = valid & (labels == 0)
        n_below_incorrect = int(((confs < thr) & incorrect).sum())
        n_above_incorrect = int(((confs >= thr) & incorrect).sum())
        both_sides = n_below_incorrect > 0 and n_above_incorrect > 0

    passes = {
        "auroc_below_0.95": (auroc == auroc) and auroc < 0.95,
        "not_saturated (<50% within 1e-6 of 0/1)": not_saturated,
        "incorrect_on_both_sides_of_threshold": both_sides,
        "no_missing_verdict_mass": n_missing == 0,
    }
    all_pass = all(passes.values())
    print(f"GATE CHECK {name}/{label_scheme}: English AUROC={auroc:.4f}  saturated_frac={sat_frac:.1%}  "
          f"incorrect_below/above_thr={n_below_incorrect}/{n_above_incorrect}  missing={n_missing}")
    for k, v in passes.items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  -> {'CANDIDATE for full-1200 scale-up' if all_pass else 'NOT recommended for scale-up as-is'}")
    return all_pass
'''

RUN_LABEL_WORD = '''# --- Item 1: label-word robustness on the two EXISTING verifiers ----------
# yesno scores for these two verifiers already exist in full at data/big/
# scores_big-{llama8b,qwen7b}.jsonl; only correctincorrect is new here. We
# additionally re-run yesno on just the 300-subset as a byte-for-byte sanity
# check that this notebook's harness reproduces the original scores before
# trusting the new correctincorrect numbers next to them.
for name, model_id in EXISTING_VERIFIERS.items():
    for label_scheme in LABEL_SCHEMES:
        out_name = OUT / f"scores_architecture_{label_scheme}-{name}.jsonl"
        if out_name.exists():
            scores = [json.loads(l) for l in out_name.open()]
        else:
            scores = run_verifier(name, model_id, label_scheme, precision="bf16")
        gate_check(name, label_scheme, scores)
'''

RUN_DENSE_MOE = '''# --- Item 2: dense vs sparse-MoE, same family, non-thinking, Yes/No only,
# PRECISION-MATCHED (see intro markdown "Why precision-matching matters") ---
#
# Step 1: score the dense model at BOTH precisions. This is not optional --
# it is what makes the eventual dense-vs-MoE contrast interpretable as an
# architecture effect rather than a numerical-precision artifact -- and it is
# also a free, independently reportable result: how much does 4-bit
# quantization ALONE (architecture held fixed) perturb cross-lingual decision
# consistency?
dense_scores = {}
for precision in ["bf16", "nf4"]:
    name = f"qwen3-4b-{precision}"
    out_name = OUT / f"scores_architecture_yesno-{name}.jsonl"
    if out_name.exists():
        scores = [json.loads(l) for l in out_name.open()]
    else:
        scores = run_verifier(name, DENSE_MODEL_ID, "yesno", precision=precision)
    gate_check(name, "yesno", scores)
    dense_scores[precision] = scores

# Step 2: auto-detect the precision the MoE model will actually run at on
# THIS GPU (bf16 needs ~61GB of weights resident; see MOE_MIN_BF16_GB above),
# and print, in plain text, exactly which dense run is the correct
# same-precision comparator. Never silently compare across precisions.
free_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
moe_precision = "bf16" if free_gb >= MOE_MIN_BF16_GB else "nf4"
moe_name = f"qwen3-30ba3b-moe-{moe_precision}"
active_dense_name = f"qwen3-4b-{moe_precision}"
print(f"\\nMoE precision auto-selected: {moe_precision}  ({free_gb:.0f}GB VRAM detected, "
      f"bf16 requires >= {MOE_MIN_BF16_GB:.0f}GB)")
print(f"ACTIVE PRECISION-MATCHED COMPARISON: {active_dense_name}  vs  {moe_name}")
if moe_precision == "nf4":
    print("  bf16 did not fit on this GPU. If an A100-80GB/H100 is available, rerun this "
          "cell there instead to get the (preferred) bf16-vs-bf16 comparison.")
else:
    print("  bf16 fit on this GPU -- using the preferred bf16-vs-bf16 comparison.")

out_name = OUT / f"scores_architecture_yesno-{moe_name}.jsonl"
if out_name.exists():
    moe_scores = [json.loads(l) for l in out_name.open()]
else:
    moe_scores = run_verifier(moe_name, MOE_MODEL_ID, "yesno", precision=moe_precision)
gate_check(moe_name, "yesno", moe_scores)

print(f"\\nFor the architecture-only contrast in scripts/14_architecture.py, compare "
      f"'{active_dense_name}' against '{moe_name}' (both {moe_precision}). The OTHER dense "
      f"precision (qwen3-4b-{'nf4' if moe_precision == 'bf16' else 'bf16'}) is kept only for "
      f"the separate quantization-only comparison against qwen3-4b-{moe_precision}.")
'''

RUN_CONTROL = '''# --- Item 3: multilingual training-distribution control, Yes/No only ------
for name, model_id in CONTROL_VERIFIERS.items():
    out_name = OUT / f"scores_architecture_yesno-{name}.jsonl"
    if out_name.exists():
        scores = [json.loads(l) for l in out_name.open()]
    else:
        scores = run_verifier(name, model_id, "yesno", precision="bf16")
    gate_check(name, "yesno", scores)
'''

SAVE = '''import shutil
shutil.make_archive("/content/xsql_architecture_results", "zip", OUT)
print("results:", [p.name for p in OUT.iterdir()])

from google.colab import files
files.download("/content/xsql_architecture_results.zip")
'''

MD_RUNTIME = """## Runtime estimate, VRAM requirements, and caveats

**Volume per (model, label-scheme) condition.** 300 items x 5 candidates x 7
languages = 10,500 scorings. Locally verified against the real
`data/big/candidates.jsonl` (see the SUBSET cell numbers and
`scripts/14_architecture.py`'s reproduction of the same selection): the 300
items carry 1500 candidates, and deduplicating identical `(db_id, sql,
question)` triples across the 7 languages collapses 1500 x 7 = 10,500
scorings to **5,610 unique forward passes** per condition (~46% dedup, in
line with the ~48% figure from the original 6000-candidate run).

**Item 1 (label-word robustness).** 2 verifiers x 1 new label scheme
(correctincorrect; yesno is reused from `data/big/`, plus one sanity
re-run on the subset) = at most 2 x 2 x 5,610 ~= 22,440 forward passes.
At tens of milliseconds per batch-1 forward pass on an A100 (the same rate
observed in the original big run), this is well under an hour of pure compute;
wall-clock with tokenization/schema-lookup overhead should be under an hour.

**Item 2 (dense vs MoE, precision-matched).** The dense model is now scored
TWICE -- `qwen3-4b-bf16` and `qwen3-4b-nf4` -- so 3 forward-pass runs total
(2 dense precisions + 1 MoE run at its auto-detected precision) x 1 label
scheme x 5,610 = **16,830 forward passes**, roughly 50% more than the earlier
single-precision design. `qwen3-4b-bf16` is comparable in cost to the existing
7-8B verifiers; `qwen3-4b-nf4` is a similar number of forward passes but with
extra quantization/dequantization overhead per step, typically somewhat slower
wall-clock despite the smaller weight footprint. **Qwen3-30B-A3B-Instruct-2507
is the VRAM outlier**: it has ~30.5B total parameters, and because
Mixture-of-Experts routing selects a different ~3.3B-parameter subset per
token, ALL experts must stay resident in VRAM -- the ~3.3B "active" figure
describes compute per token, not memory footprint. In bf16 that is roughly
61GB of weights alone, before activations or KV cache, which will NOT fit on
a standard Colab A100-40GB.

The RUN_DENSE_MOE cell auto-detects available VRAM (`MOE_MIN_BF16_GB = 65`)
and picks the MoE's precision accordingly, then prints an explicit
`ACTIVE PRECISION-MATCHED COMPARISON: qwen3-4b-{precision} vs
qwen3-30ba3b-moe-{precision}` line naming the correct same-precision
comparator -- **the dense-vs-MoE architecture claim must only ever use that
pair, never the opposite-precision dense run.** On a 40GB A100 this will read
nf4 vs nf4 (comparable weight footprints of ~2GB dense / ~16-20GB MoE); the
`qwen3-4b-bf16` run still happens regardless, purely to supply the
quantization-only comparison (see next paragraph) and because it's cheap.
**If an A100-80GB, H100, or similar is available, rerun the RUN_DENSE_MOE
cell there** to get the preferred bf16-vs-bf16 architecture comparison instead
of nf4-vs-nf4.

**The `qwen3-4b-bf16` vs `qwen3-4b-nf4` pair is a genuinely separate,
reportable result**, independent of the MoE comparison: it isolates how much
4-bit quantization alone perturbs cross-lingual decision consistency, with
architecture and every other variable held fixed. `scripts/14_architecture.py`
surfaces this pair's flip rate and Jaccard overlap directly (see its
"Quantization-only" section) rather than leaving it implicit in the
per-language tables.

**Item 3 (multilingual control).** 1 verifier (Aya Expanse 8B, dense, ~8B
params, comparable footprint to Llama-3.1-8B) x 1 label scheme x 5,610 =
5,610 forward passes, well under 30 minutes.

**Total estimate: roughly 2.5-5h on an A100-40GB** for items 1 and 3 plus all
three item-2 runs (two dense precisions + one MoE run at its auto-detected
precision) -- about 30-60 minutes more than the earlier single-precision
design because of the extra `qwen3-4b-nf4` pass, plus first-load download time
for each model's weights on top of compute (~61GB for MoE-bf16 or ~16-20GB
for MoE-nf4, on top of the dense model's own bf16 + nf4 downloads).

**Known residual risks this notebook could not eliminate:**
- The `<think>`-token template check in `assert_non_thinking` verifies the
  tokenizer's own default template, not the model's runtime behavior --
  it cannot fully rule out latent reasoning-mode behavior in edge cases.
- If the attached GPU cannot fit the MoE in bf16, the headline dense-vs-MoE
  contrast runs at nf4 for both sides, which is precision-matched (the bug
  this revision fixes) but still not bf16 -- absolute confidence values for
  that pair should be read with the same caution as any 4-bit run, even
  though the ARCHITECTURE comparison itself is now valid.
- Aya Expanse 8B and the Qwen3 family may use different tokenizers whose
  Yes/No or CORRECT/INCORRECT first-token segmentation differs in ways the
  `assert_no_token_collision` check catches for COLLISIONS but not for
  coverage gaps (e.g. a model that never places any mass on the listed surface
  forms at all) beyond what the existing "missing verdict mass" gate check
  reports.
"""


def cell(kind: str, src: str) -> dict:
    base = {"cell_type": kind, "metadata": {}, "source": src.strip().splitlines(keepends=True)}
    if kind == "code":
        base |= {"execution_count": None, "outputs": []}
    return base


def main() -> None:
    cells = [
        cell("markdown", MD_INTRO),
        cell("markdown", "## 1. Setup"),
        cell("code", SETUP),
        cell("code", AUTH),
        cell("markdown", "## 2. Config"),
        cell("code", CONFIG),
        cell("markdown", "## 3. Data (schema DDL derivation)"),
        cell("code", DOWNLOAD),
        cell("code", DATA_HARNESS),
        cell("markdown", "## 4. Prespecified database-disjoint 300-question subset"),
        cell("code", SUBSET),
        cell("markdown", "## 5. Prompts under both label-word schemes"),
        cell("code", PROMPTS),
        cell("markdown", "## 6. Safety checks (token collisions, non-thinking mode)"),
        cell("code", RUN_CHECKS),
        cell("markdown", "## 7. Shared, resumable scoring harness"),
        cell("code", VERIFY_HARNESS),
        cell("markdown", "## 8. Scale-up gate check"),
        cell("code", GATE_CHECK),
        cell("markdown", "## 9. Item 1 — label-word robustness (existing verifiers)"),
        cell("code", RUN_LABEL_WORD),
        cell("markdown", "## 10. Item 2 — dense vs sparse-MoE (Qwen3 family, non-thinking)"),
        cell("code", RUN_DENSE_MOE),
        cell("markdown", "## 11. Item 3 — multilingual training-distribution control (Aya Expanse 8B)"),
        cell("code", RUN_CONTROL),
        cell("markdown", "## 12. Export"),
        cell("code", SAVE),
        cell("markdown", MD_RUNTIME),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
