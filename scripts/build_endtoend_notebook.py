"""Emit the Colab notebook for the end-to-end experiment.

Complement to build_colab_notebook.py (the paired experiment): there, SQL is
generated once from the English question and only the scoring language
varies, isolating the causal effect of language on verifier confidence. Here,
SQL is generated independently from each language's own question, executed,
labeled, and then scored under that same language's question (the deployment
condition). This measures the combined operational effect: language affects
generation, language affects scoring, and calibration determines whether the
system executes or abstains.

Because the candidate sets differ per language here (unlike the paired run,
where the same fixed SQL is rescored under every language), cross-language
comparisons on this output are not paired and must not be analyzed with the
paired bootstrap in scripts/05_analyze.py, which assumes a shared candidate
set. Every output row carries `gen_lang` and `condition` so this is visible
downstream.

Reuses the same variant, prompts, execution-match rule, and Yes/No logprob
scoring as build_colab_notebook.py / src/xsql so the two runs stay comparable
wherever they can be.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "xsql_endtoend.ipynb"

MD_INTRO = """# Same SQL, Different Risk — END-TO-END stage

Complement to the PAIRED experiment (`xsql_colab.ipynb`). There, SQL is generated
**once from the English question** and only the scoring language varies, isolating
the causal effect of language on verifier confidence, holding SQL and label fixed.
Here, SQL is generated **independently from each language's own question**,
executed, labeled, and scored under **that same language's question** (the
deployment condition). The paired run establishes that language affects the
verifier causally; this run measures how large the combined effect is once
language is also allowed to affect what SQL gets generated in the first place.

**What this does**
1. Downloads MultiSpider (`with_english_value`) + the 166 Spider SQLite databases
2. Samples the **same 1200** parallel questions as the paired run (same seed,
   same selection logic, so the two experiments share questions and databases)
3. For each of the 7 languages, generates SQL candidates **from that language's
   own question**: K=5 for English, K=3 for each of the other 6 languages
   (23 candidates/item, 27,600 total — see the note in the config cell for why
   English gets more)
4. Executes every candidate against its database and labels it by result-set match
5. Scores every candidate under **its own generation language's question only**
   (the deployment condition — not the 7-language fan-out of the paired run) with
   Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct

**Not paired.** Each language now has its own, independently sampled set of
candidate SQL, so per-language numbers here (accuracy, AUROC, coverage) CANNOT be
compared with the paired bootstrap statistics in `scripts/05_analyze.py` — that
script assumes every language scores the *same* fixed SQL and candidate ids
overlap 1:1 across languages, which is false here. Every output row carries a
`gen_lang` field (the language SQL was generated from) and a `condition` field
so this cannot be silently conflated with the paired run downstream. Compare
languages here only with per-language marginal statistics (e.g. bootstrap over
questions *within* a language, or an unpaired-groups test across languages).

**Runtime:** pick an A100. This run generates ~4.6x more candidates than the
paired run's 6,000 (27,600 vs 6,000) because SQL is generated 7x per item
instead of once, at K=5/3 instead of a flat K=5. Verifier scoring is on the
other hand *lighter* per candidate than the paired run's 7-language fan-out,
because each candidate is scored under only one language. Net: expect roughly
**5-6h** on an A100 (rough arithmetic in the config cell), noticeably longer
than the paired run's 2-3h. Output is written to Drive and every long cell
(generation, each verifier) resumes from exactly the (item, language) pairs /
forward passes already on disk, so a disconnect costs only the current step,
not the run. Needs an HF token with Llama-3.1 access.
"""

SETUP = """!pip -q install transformers accelerate huggingface_hub
import torch, os
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE — set Runtime > Change runtime type > GPU")
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

CONFIG = '''# Mirrors src/xsql/config.py. SPLITS, N_QUESTIONS, MAX_PER_DB, and SEED are
# identical to the paired notebook's config cell so item selection reproduces
# the same 1200 items.
REPO_ID  = "dreamerdeo/multispider"
# `with_original_value` localizes literals inside questions ("JetBlue Airways" ->
# "深圳航空公司") while leaving gold SQL in English, so a translated question
# would stop asking what the gold query answers.
VARIANT  = "with_english_value"

LANGUAGES = ["en", "de", "es", "fr", "ja", "vi", "zh"]
PIVOT     = "en"
CONDITION = "endtoend_independent_generation"  # stamped on every output row

GENERATOR_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
VERIFIERS = {
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen7b":  "Qwen/Qwen2.5-7B-Instruct",
}

# Spider's dev split has only 20 databases, so a per-database cap silently
# bounds the run (a cap of 4 yields at most 80 questions). Pull from train too:
# 146 more databases. These four values match the paired notebook's config
# cell, so this run samples the same 1200 items over the same 163 databases.
SPLITS       = ["dev", "train"]
N_QUESTIONS  = 1200
MAX_PER_DB   = 10      # 166 databases x 10 = 1660 ceiling, comfortably above N
SEED         = 0

# K is asymmetric. English keeps K=5, matching the paired run's K for the pivot
# language. Each other language gets K=3: generating from all 7 languages
# already multiplies generation work by up to 7x over the paired run, and K=3
# keeps total compute tractable (23 candidates/item = 5 + 6*3) while still
# giving 3 candidates per language per item for per-language execution
# accuracy, AUROC, and calibration.
K_EN     = 5
K_TARGET = 3

_cands_per_item = K_EN + (len(LANGUAGES) - 1) * K_TARGET
print(f"{_cands_per_item} candidates/item x {N_QUESTIONS} items = "
      f"{_cands_per_item * N_QUESTIONS} candidates total "
      f"({_cands_per_item * N_QUESTIONS / (5 * N_QUESTIONS):.1f}x the paired run's "
      f"{5 * N_QUESTIONS})")

import pathlib
# Write to Drive (separate folder from the paired run) so a disconnected
# runtime does not lose hours of work.
try:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = pathlib.Path("/content/drive/MyDrive/xsql_endtoend_out")
except Exception:
    OUT = pathlib.Path("/content/out_endtoend")
OUT.mkdir(exist_ok=True, parents=True)
print("writing to", OUT)
'''

DOWNLOAD = '''from huggingface_hub import snapshot_download
import pathlib

ROOT = pathlib.Path(snapshot_download(
    REPO_ID, repo_type="dataset",
    allow_patterns=[f"dataset/multispider/{VARIANT}/{s}_*.json" for s in SPLITS]
                   + ["dataset/spider/database/**"],
))
QDIR = ROOT / "dataset/multispider" / VARIANT
DBDIR = ROOT / "dataset/spider/database"
print(QDIR, len(list(DBDIR.iterdir())), "databases")
'''

DATA = '''import json, sqlite3, re, time

def db_path(db_id): return DBDIR / db_id / f"{db_id}.sqlite"

def load_parallel(split, languages=LANGUAGES):
    """MultiSpider per-language files are index-aligned within a split."""
    per = {l: json.loads((QDIR / f"{split}_{l}.json").read_text()) for l in languages}
    n = {l: len(r) for l, r in per.items()}
    assert len(set(n.values())) == 1, f"{split} not parallel: {n}"
    base = per[languages[0]]
    items = []
    for i, b in enumerate(base):
        assert all(per[l][i]["db_id"] == b["db_id"] for l in languages), f"db_id mismatch {split}:{i}"
        items.append(dict(idx=f"{split}:{i}", split=split, db_id=b["db_id"],
                          gold_sql=b["query"],
                          questions={l: per[l][i]["question"] for l in languages}))
    return items

def schema_ddl(db_id):
    with sqlite3.connect(f"file:{db_path(db_id)}?mode=ro", uri=True) as c:
        rows = c.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL").fetchall()
    return "\\n\\n".join(s.strip() for (s,) in rows)

items_all = [it for s in SPLITS for it in load_parallel(s)]
import collections
print(len(items_all), "parallel items across",
      len({i["db_id"] for i in items_all}), "databases",
      dict(collections.Counter(i["split"] for i in items_all)))
'''


# The execution comparator is copied verbatim from the repository module at
# build time so the notebook and the analysis code use one implementation.
import inspect
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xsql import execute as _xsql_execute  # noqa: E402

COMPARATOR_SOURCE = "\n\n".join(
    [
        "from itertools import permutations",
        f"MAX_PERMUTED_COLUMNS = {_xsql_execute.MAX_PERMUTED_COLUMNS}",
    ]
    + [
        inspect.getsource(getattr(_xsql_execute, name))
        for name in [
            "_normalize_cell", "_sort_key", "_normalize_rows", "has_order_by",
            "_column_permutations", "_columns", "_column_key", "_wide_match",
            "_assignments", "rows_match",
        ]
    ]
)

EXECUTE = '''TIMEOUT = 30.0

def execute(db_id, sql, timeout=TIMEOUT):
    deadline = time.monotonic() + timeout
    timed_out = False
    def abort():
        nonlocal timed_out
        if time.monotonic() > deadline:
            timed_out = True; return 1
        return 0
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path(db_id)}?mode=ro", uri=True)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        conn.set_progress_handler(abort, 1000)
        return True, conn.execute(sql).fetchall(), None
    except Exception as e:
        return False, None, ("timeout" if timed_out else str(e))
    finally:
        if conn is not None: conn.close()

''' + COMPARATOR_SOURCE + '''

def result_match(db_id, pred_sql, gold_sql):
    """Execution match with the same comparator as src/xsql/execute.py.

    Row order is enforced only when gold has ORDER BY; column order is never
    enforced (Spider convention). The comparator source above is copied from
    the repository module at notebook-build time, so the two cannot drift."""
    ok, prows, err = execute(db_id, pred_sql)
    if not ok: return False, ("timeout" if err == "timeout" else "error")
    gok, grows, gerr = execute(db_id, gold_sql)
    if not gok: return False, "gold_bad"
    return rows_match(prows, grows, has_order_by(gold_sql)), "ok"
'''

COMPARATOR_CHECK = '''# Parity check: the comparator must accept reordered columns, reordered rows
# (when gold has no ORDER BY), duplicate rows, and float noise, and must
# reject recombined rows and row-order changes when ORDER BY is present.
_g = [(1, 2.0, "a"), (3, 4.0, "b"), (3, 4.0, "b")]
assert rows_match([(2.0, 1, "a"), (4.0, 3, "b"), (4.0, 3, "b")], _g, False)   # columns reordered
assert rows_match(list(reversed(_g)), _g, False)                               # rows reordered
assert not rows_match(list(reversed(_g)), _g, True)                            # ...but not under ORDER BY
assert not rows_match(_g[:2], _g, False)                                       # duplicate count matters
assert rows_match([(1, 2.0000001, "a"), (3, 4.0, "b"), (3, 4.0, "b")], _g, False)  # float noise
assert not rows_match([(1, 4.0, "a"), (3, 2.0, "b"), (3, 4.0, "b")], _g, False)    # recombined rows
_w = [tuple(range(7)), tuple(range(10, 17))]
assert rows_match([tuple(reversed(r)) for r in reversed(_w)], _w, False)       # wide result path
assert has_order_by("SELECT a FROM t ORDER BY a") and not has_order_by("SELECT a FROM t")
print("comparator parity check passed")
'''

PREPARE = '''import random
from collections import defaultdict

# Same logic and seed as the paired notebook's item-selection cell, so this
# run samples the same 1200 items over the same 163 databases.
rng = random.Random(SEED)
order = list(items_all); rng.shuffle(order)
per_db, chosen, skipped = defaultdict(int), [], 0
for it in order:
    if len(chosen) >= N_QUESTIONS: break
    if per_db[it["db_id"]] >= MAX_PER_DB: continue
    ok, rows, _ = execute(it["db_id"], it["gold_sql"])
    if not ok or not rows:          # empty gold makes match trivially satisfiable
        skipped += 1; continue
    per_db[it["db_id"]] += 1; chosen.append(it)

with open(OUT / "items.jsonl", "w") as f:
    for it in chosen: f.write(json.dumps(it, ensure_ascii=False) + "\\n")
print(f"{len(chosen)} items across {len(per_db)} databases ({skipped} skipped)")
'''

GENERATE = '''import zlib
from transformers import AutoModelForCausalLM, AutoTokenizer

GEN_PROMPT = """You are an expert at writing SQLite queries.

Database schema:
{schema}

Question: {question}

Write a single SQLite query that answers the question. Respond with only the query inside a ```sql code block."""

def extract_sql(text):
    m = re.search(r"```(?:sql)?\\s*(.*?)```", text, re.DOTALL | re.I)
    sql = (m.group(1) if m else text).strip()
    if ";" in sql: sql = sql.split(";")[0]
    return " ".join(sql.split())

gtok = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
gmodel = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL, dtype=torch.bfloat16, device_map="cuda").eval()

# Resume support, keyed on (item_idx, gen_lang): generation happens once per
# language per item, so that is the unit a restart must be able to skip.
CAND_PATH = OUT / "candidates.jsonl"
records = [json.loads(l) for l in CAND_PATH.open()] if CAND_PATH.exists() else []
done = {(r["item_idx"], r["gen_lang"]) for r in records}
if done:
    print(f"resuming: {len(records)} candidates for {len(done)} (item, lang) pairs already done")

cand_f = CAND_PATH.open("a")
t0 = time.monotonic()
total_pairs = len(chosen) * len(LANGUAGES)
worked = 0
for it in chosen:
    schema = None
    for lang in LANGUAGES:
        if (it["idx"], lang) in done:
            continue
        if schema is None:
            schema = schema_ddl(it["db_id"])
        k = K_EN if lang == PIVOT else K_TARGET
        msgs = [{"role": "user", "content": GEN_PROMPT.format(schema=schema, question=it["questions"][lang])}]
        prompt = gtok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = gtok(prompt, return_tensors="pt").to(gmodel.device)
        # Item ids are strings like "train:812"; hash (idx, lang) to a stable
        # int seed so a resumed run regenerates identical candidates for any
        # (item, lang) pair not yet done.
        seed = zlib.crc32(f"{it['idx']}:{lang}".encode()) % (2**31)
        torch.manual_seed(seed)
        with torch.inference_mode():
            out = gmodel.generate(**inputs, do_sample=True, temperature=0.8, top_p=0.95,
                                  num_return_sequences=k, max_new_tokens=256,
                                  pad_token_id=gtok.eos_token_id)
        texts = gtok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for j, t in enumerate(texts):
            sql = extract_sql(t)
            correct, status = result_match(it["db_id"], sql, it["gold_sql"])
            rec = dict(candidate_id=f"{it['idx']}#{lang}#{j}", item_idx=it["idx"], split=it["split"],
                       db_id=it["db_id"], gen_lang=lang, condition=CONDITION,
                       sql=sql, correct=correct, exec_status=status)
            records.append(rec)
            cand_f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
        cand_f.flush()
        worked += 1
        if worked % 200 == 0 or (len(done) + worked) == total_pairs:
            el = time.monotonic() - t0
            nc = sum(r["correct"] for r in records)
            rate = el / worked
            remaining = total_pairs - len(done) - worked
            print(f"[{len(done)+worked}/{total_pairs} item-lang pairs] {nc}/{len(records)} correct | "
                  f"{rate:.2f}s/pair | ETA {remaining*rate/60:.0f}m", flush=True)

cand_f.close()
nc = sum(r["correct"] for r in records)
print(f"\\n{len(records)} candidates across {len(chosen)} items x {len(LANGUAGES)} languages "
      f"(K={K_EN} en / K={K_TARGET} other): {nc} correct, {len(records)-nc} incorrect")
print("Per-language execution accuracy:")
for lang in LANGUAGES:
    sub = [r for r in records if r["gen_lang"] == lang]
    acc = sum(r["correct"] for r in sub) / len(sub) if sub else float("nan")
    print(f"  {lang}: {sum(r['correct'] for r in sub)}/{len(sub)} = {acc:.1%}")

del gmodel; torch.cuda.empty_cache()
'''

VERIFY = '''VER_SYSTEM = ("You are a meticulous database engineer auditing text-to-SQL output. "
              "You judge whether a candidate SQLite query correctly answers a user's "
              "question against a given schema. The question may be written in any "
              "language; judge the query on its merits, not on the question's language.")

VER_TEMPLATE = """Database schema:
{schema}

User question:
{question}

Candidate SQLite query:
{sql}

Does this query correctly answer the user's question? Answer with a single word, Yes or No."""

YES_FORMS = ["Yes", " Yes", "yes", " yes", "YES", "Y"]
NO_FORMS  = ["No", " No", "no", " no", "NO", "N"]

def first_ids(tok, forms):
    return sorted({tok.encode(f, add_special_tokens=False)[0] for f in forms if tok.encode(f, add_special_tokens=False)})

SCHEMAS = {d: schema_ddl(d) for d in {r["db_id"] for r in records}}
BY_IDX = {i["idx"]: i for i in chosen}
sql_by_cid = {r["candidate_id"]: r["sql"] for r in records}

# Deployment condition: each candidate is scored under only the language it
# was generated from (gen_lang == lang in this run's output), unlike the paired
# run's 7-language fan-out. Per-language numbers are therefore not directly
# comparable to the paired run's, and cross-language comparisons within this
# run are unpaired, since each language has its own candidate set.
jobs = {}
for r in records:
    q = BY_IDX[r["item_idx"]]["questions"][r["gen_lang"]]
    key = (r["db_id"], r["sql"], q)
    jobs.setdefault(key, []).append(
        (r["candidate_id"], r["item_idx"], r["split"], r["gen_lang"], r["correct"]))
print(f"{len(records)} candidates -> {len(jobs)} unique forward passes "
      f"({100*(1 - len(jobs)/len(records)):.0f}% deduped, same-SQL-same-question)")

# Batch size is 1. Batched bf16 forward passes are not numerically identical to
# unbatched ones: on this workload, batching perturbs confidences by up to
# 6e-2, the same magnitude as the effects being measured. Length-sorting and
# explicit position_ids did not remove the drift, so it is batch-shape-dependent
# kernel nondeterminism rather than padding. A forward pass at this prompt
# length takes tens of milliseconds, so the cost is small.
BATCH = 1

def run_verifier(name, model_id):
    out_path = OUT / f"scores_endtoend-{name}.jsonl"
    done_rows = [json.loads(l) for l in out_path.open()] if out_path.exists() else []
    done_cids = {r["candidate_id"] for r in done_rows}

    # Resumable at the level of individual forward passes: a job (unique
    # db_id/sql/question triple) is skipped only once every candidate_id that
    # fans out from it has already been written.
    remaining = {}
    for key, fanout in jobs.items():
        pending = [f for f in fanout if f[0] not in done_cids]
        if pending:
            remaining[key] = pending
    print(f"  {name}: {len(done_rows)} scores already on disk, "
          f"{len(remaining)}/{len(jobs)} forward passes remaining")
    if not remaining:
        print(f"  {name}: nothing left to do")
        return

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
    yes_ids, no_ids = first_ids(tok, YES_FORMS), first_ids(tok, NO_FORMS)
    assert not (set(yes_ids) & set(no_ids)), "Yes/No token ids collide"

    keys = list(remaining)
    out_f = out_path.open("a")
    t0 = time.monotonic()
    for s in range(0, len(keys), BATCH):
        chunk = keys[s:s+BATCH]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": VER_SYSTEM},
             {"role": "user", "content": VER_TEMPLATE.format(schema=SCHEMAS[d], question=q, sql=sql)}],
            tokenize=False, add_generation_prompt=True) for (d, sql, q) in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            logits = model(**enc).logits[:, -1, :].float()   # left padding -> last pos is real
        p = torch.softmax(logits, dim=-1)
        y, n = p[:, yes_ids].sum(-1), p[:, no_ids].sum(-1)
        for k, yy, nn in zip(chunk, y.tolist(), n.tolist()):
            conf = (yy / (yy + nn)) if (yy + nn) > 0 else None
            for cid, item_idx, split, lang, correct in remaining[k]:
                rec = dict(candidate_id=cid, item_idx=item_idx, split=split, db_id=k[0],
                           gen_lang=lang, lang=lang, condition=CONDITION,
                           correct=correct, confidence=conf, stop_reason="logprob")
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
        out_f.flush()
        if s % 500 == 0 or s + BATCH >= len(keys):
            el = time.monotonic() - t0; done_now = s + len(chunk)
            print(f"  {name}: {done_now}/{len(keys)} | ETA {(len(keys)-done_now)*el/max(done_now,1)/60:.0f}m", flush=True)
    out_f.close()

    n_total = len(done_rows) + sum(len(v) for v in remaining.values())
    n_missing = sum(1 for l in out_path.open() if json.loads(l)["confidence"] is None)
    print(f"{name}: {n_total} total scores on disk ({n_missing} without Yes/No mass)")
    del model; torch.cuda.empty_cache()

for name, mid in VERIFIERS.items():
    run_verifier(name, mid)

print()
print("Reminder: scores_endtoend-*.jsonl rows carry gen_lang == lang (own-language")
print("deployment scoring) and condition=%r. Candidate sets differ per language," % CONDITION)
print("so this file must not be fed into scripts/05_analyze.py's paired bootstrap,")
print("which assumes one shared candidate set scored under every language.")
'''

SAVE = '''import shutil
shutil.make_archive("/content/xsql_endtoend_results", "zip", OUT)
print("results:", [p.name for p in OUT.iterdir()])

# Download the archive.
from google.colab import files
files.download("/content/xsql_endtoend_results.zip")

# Optional: also keep a copy in Drive.
# from google.colab import drive; drive.mount('/content/drive')
# shutil.copy("/content/xsql_endtoend_results.zip", "/content/drive/MyDrive/xsql_endtoend_results.zip")
'''


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
        cell("markdown", "## 2. Config — mirrors `src/xsql/config.py` (K split explained here)"),
        cell("code", CONFIG),
        cell("markdown", "## 3. Data"),
        cell("code", DOWNLOAD),
        cell("code", DATA),
        cell("markdown", "## 4. Execution harness"),
        cell("code", EXECUTE),
        cell("code", COMPARATOR_CHECK),
        cell("markdown", "## 5. Sample items with executable gold SQL (same 1200 items as the paired run)"),
        cell("code", PREPARE),
        cell("markdown", "## 6. Generate candidates INDEPENDENTLY per language (K=5 en / K=3 other) + label by execution"),
        cell("code", GENERATE),
        cell("markdown", "## 7. Score every candidate under its OWN generation language only (deployment condition — not the paired 7-language fan-out)"),
        cell("code", VERIFY),
        cell("markdown", "## 8. Export"),
        cell("code", SAVE),
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
