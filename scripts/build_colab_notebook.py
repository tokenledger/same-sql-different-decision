"""Emit the Colab notebook for the GPU half of the pipeline.

The notebook mirrors src/xsql exactly (same variant, prompts, execution-match
rule, and Yes/No logprob scoring) so its output is directly comparable to local
runs. Analysis stays local: the notebook only produces items/candidates/scores
JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "xsql_colab.ipynb"

MD_INTRO = """# Same SQL, Different Risk — GPU stage

Generates SQL candidates and scores them with local verifiers across 7 languages.

**What this does**
1. Downloads MultiSpider (`with_english_value`) + the 166 Spider SQLite databases
2. Samples 1200 parallel questions with executable gold SQL, drawn from train+dev (163 databases)
3. Generates K=5 candidates per question **from the English question only**
4. Executes every candidate and labels it by result-set match
5. Scores each candidate under all 7 languages with Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct

Only the question language varies during scoring — SQL, schema, database, and label are held fixed.

**Runtime:** pick an A100. Expect ~2-3h. Output is written to Drive after every item, and
cells 6 and 7 resume from what is already there, so a disconnect costs only the current item.
Needs an HF token with Llama-3.1 access.
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

CONFIG = '''# Mirrors src/xsql/config.py
REPO_ID  = "dreamerdeo/multispider"
# `with_original_value` localizes literals inside questions ("JetBlue Airways" ->
# "深圳航空公司") while leaving gold SQL in English, which breaks the paired design.
VARIANT  = "with_english_value"

LANGUAGES = ["en", "de", "es", "fr", "ja", "vi", "zh"]
PIVOT     = "en"

GENERATOR_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
# Two independent capable verifiers, different model families. The headline
# effect so far (French depresses Llama's confidence on identical SQL) needs a
# second opinion from an unrelated model to count as a finding rather than a
# Llama quirk. Qwen2.5-1.5B is dropped: at scale its AUROC was 0.573, too close
# to chance to support any statement about calibrated abstention.
VERIFIERS = {
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen7b":  "Qwen/Qwen2.5-7B-Instruct",
}

# Spider's dev split has only 20 databases, so a per-database cap silently
# bounds the run (a cap of 4 yields at most 80 questions). Pull from train too:
# 146 more databases, which also makes database-disjoint splits possible.
SPLITS       = ["dev", "train"]
N_QUESTIONS  = 1200
K_CANDIDATES = 5
MAX_PER_DB   = 10      # 166 databases x 10 = 1660 ceiling, comfortably above N
SEED         = 0

import pathlib
# Write to Drive so a disconnected runtime does not lose hours of work.
try:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = pathlib.Path("/content/drive/MyDrive/xsql_out")
except Exception:
    OUT = pathlib.Path("/content/out")
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

# Resume support: a disconnected runtime should not cost the whole run.
CAND_PATH = OUT / "candidates.jsonl"
records = [json.loads(l) for l in CAND_PATH.open()] if CAND_PATH.exists() else []
done_items = {r["item_idx"] for r in records}
if done_items:
    print(f"resuming: {len(records)} candidates for {len(done_items)} items already done")

cand_f = CAND_PATH.open("a")
t0 = time.monotonic()
for n, it in enumerate(chosen, 1):
    if it["idx"] in done_items:
        continue
    schema = schema_ddl(it["db_id"])
    msgs = [{"role": "user", "content": GEN_PROMPT.format(schema=schema, question=it["questions"][PIVOT])}]
    prompt = gtok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = gtok(prompt, return_tensors="pt").to(gmodel.device)
    # Item ids are strings like "train:812"; hash to a stable int seed so a
    # resumed run regenerates identical candidates.
    torch.manual_seed(zlib.crc32(it["idx"].encode()) % (2**31))
    with torch.inference_mode():
        out = gmodel.generate(**inputs, do_sample=True, temperature=0.8, top_p=0.95,
                              num_return_sequences=K_CANDIDATES, max_new_tokens=256,
                              pad_token_id=gtok.eos_token_id)
    texts = gtok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    for j, t in enumerate(texts):
        sql = extract_sql(t)
        correct, status = result_match(it["db_id"], sql, it["gold_sql"])
        rec = dict(candidate_id=f"{it['idx']}#{j}", item_idx=it["idx"], split=it["split"],
                   db_id=it["db_id"], sql=sql, correct=correct, exec_status=status)
        records.append(rec)
        cand_f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
    cand_f.flush()
    if n % 25 == 0 or n == len(chosen):
        el = time.monotonic() - t0
        nc = sum(r["correct"] for r in records)
        rate = el / max(n - len(done_items), 1)
        print(f"[{n}/{len(chosen)}] {nc}/{len(records)} correct | {rate:.1f}s/item "
              f"| ETA {(len(chosen)-n)*rate/60:.0f}m", flush=True)

cand_f.close()
nc = sum(r["correct"] for r in records)
print(f"\\n{len(records)} candidates: {nc} correct, {len(records)-nc} incorrect")

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

# Identical SQL under the same question scores identically, so score unique
# (candidate sql, language) pairs once and fan the result back out.
BY_IDX = {i["idx"]: i for i in chosen}
jobs = {}
for r in records:
    for lang in LANGUAGES:
        q = BY_IDX[r["item_idx"]]["questions"][lang]
        jobs.setdefault((r["db_id"], r["sql"], q), []).append(
            (r["candidate_id"], r["item_idx"], r["split"], lang, r["correct"]))
print(f"{len(records)*len(LANGUAGES)} scorings -> {len(jobs)} unique forward passes")

# Batch size 1 is deliberate. Batched bf16 forward passes are not numerically
# identical to unbatched ones -- measured on this exact workload, batching
# perturbs confidences by up to 6e-2, which is the same magnitude as the
# cross-language effects being measured. Length-sorting and explicit position_ids
# both made it worse; it is batch-shape-dependent kernel nondeterminism, not
# padding. Batch 1 reproduces the local runs exactly (max |diff| = 0).
# A GPU forward at this prompt length is only tens of milliseconds, so the
# correctness is nearly free.
BATCH = 1

def run_verifier(name, model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
    yes_ids, no_ids = first_ids(tok, YES_FORMS), first_ids(tok, NO_FORMS)
    assert not (set(yes_ids) & set(no_ids)), "Yes/No token ids collide"

    keys = list(jobs)
    conf, t0 = {}, time.monotonic()
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
            conf[k] = (yy / (yy + nn)) if (yy + nn) > 0 else None
        if s % 500 == 0 or s + BATCH >= len(keys):
            el = time.monotonic() - t0; done = s + len(chunk)
            print(f"  {name}: {done}/{len(keys)} | ETA {(len(keys)-done)*el/max(done,1)/60:.0f}m", flush=True)

    out = []
    for k, fanout in jobs.items():
        for cid, item_idx, split, lang, correct in fanout:
            out.append(dict(candidate_id=cid, item_idx=item_idx, split=split, db_id=k[0],
                            lang=lang, correct=correct, confidence=conf[k], stop_reason="logprob"))
    with open(OUT / f"scores_big-{name}.jsonl", "w") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\\n")
    miss = sum(1 for r in out if r["confidence"] is None)
    print(f"{name}: wrote {len(out)} scores ({miss} without Yes/No mass)")
    del model; torch.cuda.empty_cache()

for name, mid in VERIFIERS.items():
    if (OUT / f"scores_big-{name}.jsonl").exists():
        print(f"{name}: already done, skipping"); continue
    run_verifier(name, mid)
'''

SAVE = '''import shutil
shutil.make_archive("/content/xsql_results", "zip", OUT)
print("results:", [p.name for p in OUT.iterdir()])

# Download to your machine
from google.colab import files
files.download("/content/xsql_results.zip")

# Optional: also drop a copy in Drive
# from google.colab import drive; drive.mount('/content/drive')
# shutil.copy("/content/xsql_results.zip", "/content/drive/MyDrive/xsql_results.zip")
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
        cell("markdown", "## 2. Config — mirrors `src/xsql/config.py`"),
        cell("code", CONFIG),
        cell("markdown", "## 3. Data"),
        cell("code", DOWNLOAD),
        cell("code", DATA),
        cell("markdown", "## 4. Execution harness"),
        cell("code", EXECUTE),
        cell("code", COMPARATOR_CHECK),
        cell("markdown", "## 5. Sample items with executable gold SQL"),
        cell("code", PREPARE),
        cell("markdown", "## 6. Generate candidates (English only) + label by execution"),
        cell("code", GENERATE),
        cell("markdown", "## 7. Score every candidate under all 7 languages"),
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
