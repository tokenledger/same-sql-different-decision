"""Emit the Colab notebook for the mitigation experiment.

Background: `scripts/build_colab_notebook.py` produced items/candidates and
scored every candidate under 7 languages, always showing the verifier only the
native-language question. That established that AUROC transfers across
languages but an English-fitted abstention threshold does not (coverage moves
a lot at equal risk). This notebook tests whether showing the verifier English
as well closes that gap, by adding two more scoring conditions on top of the
same fixed (sql, db, schema, label):

  1. en         already scored (data/big/scores_big-*.jsonl); not recomputed
  2. native     already scored (same files, other `lang` values); not recomputed
  3. bilingual  new: native question and English question shown together
  4. pivot      new: an automatic English translation of the native question

Condition 3 needs no translator, since both texts are available, and runs
first. Condition 4 needs an MT step, done with Qwen2.5-7B-Instruct (see the
CONFIG cell). It must translate the native question and never reuse the
original English question, which would make condition 4 identical to
condition 1.

This script only emits the notebook; it does not run the GPU job.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "xsql_mitigation.ipynb"

MD_INTRO = """# Same SQL, Different Risk — MITIGATION stage

Tests whether showing the verifier English *in addition to* the native question
closes the coverage-at-equal-risk gap found when scoring native-only
(Qwen-7B: 36.0% coverage in English vs 14.1% in German at matched risk, n=1200,
2 verifiers).

**Four conditions, same fixed (sql, db, schema, execution-derived label):**

| # | condition   | question shown to verifier                    | status |
|---|-------------|-------------------------------------------------|--------|
| 1 | `en`        | English question only                            | already done — reused from `data/big/scores_big-*.jsonl`, not recomputed here |
| 2 | `native`    | target-language question only                    | already done — same files, `lang != "en"` rows, not recomputed here |
| 3 | `bilingual` | native question and English question, one prompt | **new**, built first — no translator needed |
| 4 | `pivot`     | machine translation of the native question into English | **new**, needs MT (see CONFIG cell) |

Condition 4 translates the *native* question, not the original English one.
Reusing the original English text there would make condition 4 identical to
condition 1 and produce a spurious "perfect mitigation" result. The notebook
asserts against that (see the TRANSLATE cell) and prints a diagnostic if
translations look suspiciously close to the originals.

**This notebook reuses the existing items/candidates/labels from the `big`
run rather than regenerating SQL** (regenerating would break comparability
with conditions 1 and 2, which were scored against a specific candidate set).
Upload `items.jsonl` and `candidates.jsonl` from `data/big/` to
`My Drive/xsql_out/` before running; see the LOAD cell for the best-effort
fallback if they are missing.

**Runtime:** see the estimate and caveats in the last markdown cell. Expect this
to run longer than the original GPU notebook (more language x condition
pairs). Every long cell appends its raw per-unique-job output to Drive as it
goes and skips already-done keys on restart, so a disconnect costs only
whatever was not yet flushed.
"""

SETUP = """!pip -q install transformers accelerate huggingface_hub
import torch, os
# Imported here rather than in the LOAD cell's fallback branch: on the
# recommended path (items/candidates uploaded to Drive) that branch never runs,
# and the verifier cells need these names.
from transformers import AutoModelForCausalLM, AutoTokenizer
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

CONFIG = '''# Mirrors src/xsql/config.py and scripts/build_colab_notebook.py's CONFIG cell,
# so prompt templates, Yes/No token logic, and the execution rule match the
# en/native runs. Only the new conditions are added.
REPO_ID  = "dreamerdeo/multispider"
VARIANT  = "with_english_value"   # with_original_value localizes literals in the
                                   # question but not in the gold SQL.

LANGUAGES = ["en", "de", "es", "fr", "ja", "vi", "zh"]
# Two different things are called "pivot" in this project:
#   - GEN_PIVOT ("en"): candidates were generated from the English question only
#     (`PIVOT` in src/xsql/config.py and build_colab_notebook.py).
#   - condition "pivot" (this notebook): scoring the verifier on a machine
#     translation of the native question back into English.
GEN_PIVOT = "en"
NEW_LANGS = [l for l in LANGUAGES if l != GEN_PIVOT]   # the 6 languages that get new conditions

VERIFIERS = {
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen7b":  "Qwen/Qwen2.5-7B-Instruct",
}

# Translator for condition 4 (pivot). One translator is used for all six
# languages so that translation quality does not become a per-language
# confound on top of the language effect being measured. Qwen2.5-7B-Instruct
# is already downloaded for this run (it is also VERIFIERS["qwen7b"]) and
# translates de/es/fr/ja/vi/zh competently via prompting. The self-preference
# concern that rules out using the SQL generator as a verifier does not apply
# here: translation happens once per (item, language), upstream of and blind
# to which verifier or SQL candidate is later scored against it.
TRANSLATOR_MODEL = "Qwen/Qwen2.5-7B-Instruct"

import pathlib
try:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE = pathlib.Path("/content/drive/MyDrive")
except Exception:
    DRIVE = pathlib.Path("/content/out_local")
    DRIVE.mkdir(exist_ok=True, parents=True)

# Where the original run (build_colab_notebook.py) wrote items/candidates/scores.
SOURCE_DIR = DRIVE / "xsql_out"
# Where this notebook writes its outputs, kept separate so a re-run cannot
# overwrite the original run's files.
OUT = DRIVE / "xsql_mitigation_out"
OUT.mkdir(exist_ok=True, parents=True)
print("source:", SOURCE_DIR, "(existing items/candidates/scores)")
print("out:   ", OUT, "(new bilingual/pivot outputs)")
'''

DOWNLOAD = '''from huggingface_hub import snapshot_download
import pathlib

# Schema DDL is derived from the sqlite files; it is not stored in
# items.jsonl. Question files are also fetched so the fallback regeneration
# path (see the LOAD cell) works if items.jsonl was not uploaded.
ROOT = pathlib.Path(snapshot_download(
    REPO_ID, repo_type="dataset",
    allow_patterns=[f"dataset/multispider/{VARIANT}/*.json"]
                   + ["dataset/spider/database/**"],
))
QDIR = ROOT / "dataset/multispider" / VARIANT
DBDIR = ROOT / "dataset/spider/database"
print(QDIR, len(list(DBDIR.iterdir())), "databases")
'''

DATA_HARNESS = '''import json, sqlite3, re, time

def db_path(db_id): return DBDIR / db_id / f"{db_id}.sqlite"

def schema_ddl(db_id):
    with sqlite3.connect(f"file:{db_path(db_id)}?mode=ro", uri=True) as c:
        rows = c.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL").fetchall()
    return "\\n\\n".join(s.strip() for (s,) in rows)

def load_parallel(split, languages=LANGUAGES):
    """MultiSpider per-language files are index-aligned within a split. Only
    used by the fallback regeneration path below."""
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

TIMEOUT = 30.0

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

def _cell(v):
    if isinstance(v, float): return round(v, 6)
    if isinstance(v, bool):  return int(v)
    if isinstance(v, bytes): return v.decode("utf-8", errors="replace")
    return v

def _norm(rows, ordered):
    n = [tuple(_cell(c) for c in r) for r in rows]
    return n if ordered else sorted(n, key=lambda r: tuple(str(c) for c in r))

def result_match(db_id, pred_sql, gold_sql):
    ok, prows, err = execute(db_id, pred_sql)
    if not ok: return False, ("timeout" if err == "timeout" else "error")
    gok, grows, gerr = execute(db_id, gold_sql)
    if not gok: return False, "gold_bad"
    ordered = re.search(r"\\border\\s+by\\b", gold_sql, re.I) is not None
    return _norm(prows, ordered) == _norm(grows, ordered), "ok"
'''

LOAD = '''# Preferred path: upload data/big/items.jsonl and data/big/candidates.jsonl
# to Drive:xsql_out/{items,candidates}.jsonl before running this notebook.
# This guarantees the same candidates as the en/native scores in
# data/big/scores_big-*.jsonl.
#
# Fallback, best-effort: if those files are missing, the original selection is
# reproduced (same seed and sampling loop as build_colab_notebook.py) and
# candidates are regenerated with the same generator model and per-item seed
# hash. Item selection is exactly reproducible. Candidate SQL text is only as
# reproducible as do_sample=True plus torch.manual_seed is across library and
# GPU versions, so it is not guaranteed byte-identical across machines. If this
# path is used, treat comparability with conditions 1 and 2 as approximate.

items_path = SOURCE_DIR / "items.jsonl"
cand_path  = SOURCE_DIR / "candidates.jsonl"

if items_path.exists() and cand_path.exists():
    chosen  = [json.loads(l) for l in items_path.open()]
    records = [json.loads(l) for l in cand_path.open()]
    print(f"loaded from Drive: {len(chosen)} items, {len(records)} candidates")
else:
    print("items/candidates not found on Drive; reproducing (best-effort, see comment above)")
    import random, zlib
    from collections import defaultdict

    SEED, N_QUESTIONS, MAX_PER_DB, K_CANDIDATES = 0, 1200, 10, 5
    GENERATOR_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

    items_all = [it for s in ["dev", "train"] for it in load_parallel(s)]
    rng = random.Random(SEED)
    order = list(items_all); rng.shuffle(order)
    per_db, chosen, skipped = defaultdict(int), [], 0
    for it in order:
        if len(chosen) >= N_QUESTIONS: break
        if per_db[it["db_id"]] >= MAX_PER_DB: continue
        ok, rows, _ = execute(it["db_id"], it["gold_sql"])
        if not ok or not rows:
            skipped += 1; continue
        per_db[it["db_id"]] += 1; chosen.append(it)
    with open(SOURCE_DIR / "items.jsonl", "w") as f:
        for it in chosen: f.write(json.dumps(it, ensure_ascii=False) + "\\n")
    print(f"reproduced {len(chosen)} items across {len(per_db)} databases ({skipped} skipped)")

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
    records = []
    for it in chosen:
        schema = schema_ddl(it["db_id"])
        msgs = [{"role": "user", "content": GEN_PROMPT.format(schema=schema, question=it["questions"][GEN_PIVOT])}]
        prompt = gtok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = gtok(prompt, return_tensors="pt").to(gmodel.device)
        torch.manual_seed(zlib.crc32(it["idx"].encode()) % (2**31))
        with torch.inference_mode():
            out = gmodel.generate(**inputs, do_sample=True, temperature=0.8, top_p=0.95,
                                  num_return_sequences=K_CANDIDATES, max_new_tokens=256,
                                  pad_token_id=gtok.eos_token_id)
        texts = gtok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for j, t in enumerate(texts):
            sql = extract_sql(t)
            correct, status = result_match(it["db_id"], sql, it["gold_sql"])
            records.append(dict(candidate_id=f"{it['idx']}#{j}", item_idx=it["idx"], split=it["split"],
                                db_id=it["db_id"], sql=sql, correct=correct, exec_status=status))
    with open(SOURCE_DIR / "candidates.jsonl", "w") as f:
        for r in records: f.write(json.dumps(r, ensure_ascii=False) + "\\n")
    print(f"reproduced {len(records)} candidates")
    del gmodel; torch.cuda.empty_cache()

BY_IDX = {i["idx"]: i for i in chosen}
'''

PROMPTS = '''VER_SYSTEM = ("You are a meticulous database engineer auditing text-to-SQL output. "
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

def bilingual_question(item, lang):
    """Condition 3: native + English shown together. Needs no translator."""
    assert lang != "en", "bilingual is only a new condition for lang != en"
    return (f"[Native language question]: {item['questions'][lang]}\\n"
            f"[English question]: {item['questions']['en']}")

def pivot_question(item, lang, pivot_map):
    """Condition 4: machine translation of the native question. `pivot_map` is
    keyed (item_idx, lang) -> translated text, produced by the TRANSLATE cell.
    It never reads item["questions"]["en"], which would alias condition 4
    onto condition 1."""
    assert lang != "en", "pivot is only a new condition for lang != en"
    return pivot_map[(item["idx"], lang)]

# Print one bilingual prompt for inspection.
_demo_item = chosen[0]
_demo_lang = NEW_LANGS[0]
print("--- bilingual demo ---")
print(bilingual_question(_demo_item, _demo_lang)[:400])
'''

VERIFY_BILINGUAL = '''# Condition 3 (bilingual). Needs no translator, so it runs first.
#
# Same dedup and batch-1 discipline as the original VERIFY cell: identical
# (db_id, sql, prompt-text) triples score identically, so unique forward
# passes are computed once and fanned out. Batch size is 1 because batched
# bf16 forward passes are not numerically identical to unbatched ones on this
# workload (up to 6e-2 confidence drift, the same magnitude as the effects
# under study). Raw per-unique-key results are appended to Drive as they are
# computed and reloaded on restart.
BATCH = 1
SCHEMAS = {d: schema_ddl(d) for d in {r["db_id"] for r in records}}

def build_jobs_bilingual():
    jobs = {}
    for r in records:
        for lang in NEW_LANGS:
            it = BY_IDX[r["item_idx"]]
            q = bilingual_question(it, lang)
            jobs.setdefault((r["db_id"], r["sql"], "bilingual", lang, q), []).append(
                (r["candidate_id"], r["item_idx"], r["split"], lang, r["correct"]))
    return jobs

def run_verifier_condition(name, model_id, condition, jobs, out_name):
    raw_path = OUT / f"raw_{condition}_{name}.jsonl"
    conf = {}
    if raw_path.exists():
        for line in raw_path.open():
            rec = json.loads(line)
            conf[tuple(rec["key"])] = rec["confidence"]
        print(f"  {name}/{condition}: resuming, {len(conf)} unique passes already done")

    keys = [k for k in jobs if k not in conf]
    if keys:
        tok = AutoTokenizer.from_pretrained(model_id)
        tok.padding_side = "left"
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
        yes_ids, no_ids = first_ids(tok, YES_FORMS), first_ids(tok, NO_FORMS)
        assert not (set(yes_ids) & set(no_ids)), "Yes/No token ids collide"

        raw_f = raw_path.open("a")
        t0 = time.monotonic()
        for s, key in enumerate(keys, 1):
            d, sql, cond, lang, q = key
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": VER_SYSTEM},
                 {"role": "user", "content": VER_TEMPLATE.format(schema=SCHEMAS[d], question=q, sql=sql)}],
                tokenize=False, add_generation_prompt=True)
            enc = tok([prompt], return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                logits = model(**enc).logits[:, -1, :].float()
            p = torch.softmax(logits, dim=-1)
            yy, nn = p[:, yes_ids].sum().item(), p[:, no_ids].sum().item()
            c = (yy / (yy + nn)) if (yy + nn) > 0 else None
            conf[key] = c
            raw_f.write(json.dumps({"key": list(key), "confidence": c}, ensure_ascii=False) + "\\n")
            raw_f.flush()
            if s % 200 == 0 or s == len(keys):
                el = time.monotonic() - t0
                print(f"  {name}/{condition}: {s}/{len(keys)} new | ETA {(len(keys)-s)*el/max(s,1)/60:.0f}m", flush=True)
        raw_f.close()
        del model; torch.cuda.empty_cache()
    else:
        print(f"  {name}/{condition}: all {len(jobs)} unique passes already done")

    out = []
    for k, fanout in jobs.items():
        d, sql, cond, lang, q = k
        for cid, item_idx, split, l, correct in fanout:
            out.append(dict(candidate_id=cid, item_idx=item_idx, split=split, db_id=d,
                            lang=l, condition=cond, correct=correct, confidence=conf[k]))
    with open(OUT / out_name, "w") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\\n")
    miss = sum(1 for r in out if r["confidence"] is None)
    print(f"{name}/{condition}: wrote {len(out)} scores ({miss} without Yes/No mass)")

jobs_bilingual = build_jobs_bilingual()
print(f"{len(records)*len(NEW_LANGS)} bilingual scorings -> {len(jobs_bilingual)} unique forward passes")
for name, mid in VERIFIERS.items():
    run_verifier_condition(name, mid, "bilingual", jobs_bilingual, f"scores_mitigation_bilingual-{name}.jsonl")
'''

TRANSLATE = '''# Condition 4 prep: translate the native question into English with
# Qwen2.5-7B-Instruct (see the CONFIG cell). Appends to Drive per (item, lang)
# and skips done work on restart. This is plain generation, not the logit read
# the verifier does, so the batch-1 rule does not apply and batching is used
# for speed.
TRANSLATE_BATCH = 16
TRANSLATE_PROMPT = """Translate the following question into English. Respond with ONLY the English translation and nothing else -- no quotes, no explanation.

Question ({lang}): {question}"""

pivot_path = OUT / "pivot_translations.jsonl"
pivot_map = {}
if pivot_path.exists():
    for line in pivot_path.open():
        rec = json.loads(line)
        pivot_map[(rec["item_idx"], rec["lang"])] = rec["pivot_en_text"]
    print(f"resuming: {len(pivot_map)} translations already done")

todo = []
for it in chosen:
    for lang in NEW_LANGS:
        if (it["idx"], lang) not in pivot_map:
            # The source text must be the native-language question, never
            # questions["en"].
            src = it["questions"][lang]
            assert lang != "en"
            todo.append((it["idx"], lang, src))

if todo:
    ttok = AutoTokenizer.from_pretrained(TRANSLATOR_MODEL)
    ttok.padding_side = "left"
    if ttok.pad_token is None: ttok.pad_token = ttok.eos_token
    tmodel = AutoModelForCausalLM.from_pretrained(TRANSLATOR_MODEL, dtype=torch.bfloat16, device_map="cuda").eval()

    pivot_f = pivot_path.open("a")
    t0 = time.monotonic()
    for s in range(0, len(todo), TRANSLATE_BATCH):
        chunk = todo[s:s+TRANSLATE_BATCH]
        prompts = [ttok.apply_chat_template(
            [{"role": "user", "content": TRANSLATE_PROMPT.format(lang=lang, question=src)}],
            tokenize=False, add_generation_prompt=True) for (_, lang, src) in chunk]
        enc = ttok(prompts, return_tensors="pt", padding=True).to(tmodel.device)
        with torch.inference_mode():
            out = tmodel.generate(**enc, do_sample=False, max_new_tokens=96,
                                  pad_token_id=ttok.eos_token_id)
        texts = ttok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for (item_idx, lang, src), text in zip(chunk, texts):
            translated = text.strip().strip('"').strip()
            pivot_map[(item_idx, lang)] = translated
            pivot_f.write(json.dumps(dict(item_idx=item_idx, lang=lang, pivot_en_text=translated),
                                     ensure_ascii=False) + "\\n")
        pivot_f.flush()
        if s % (TRANSLATE_BATCH * 10) == 0 or s + TRANSLATE_BATCH >= len(todo):
            el = time.monotonic() - t0; done = s + len(chunk)
            print(f"  translate: {done}/{len(todo)} | ETA {(len(todo)-done)*el/max(done,1)/60:.0f}m", flush=True)
    pivot_f.close()
    del tmodel; torch.cuda.empty_cache()
else:
    print("all translations already done")

# Diagnostic: pivot text should rarely equal the original English question
# exactly; many matches would suggest condition 4 collapsed onto condition 1.
# A handful of matches on very short questions is plausible.
n_ident = sum(1 for it in chosen for lang in NEW_LANGS
              if pivot_map[(it["idx"], lang)].strip().lower() == it["questions"]["en"].strip().lower())
n_total = len(chosen) * len(NEW_LANGS)
frac = n_ident / n_total
print(f"pivot text == original English text for {n_ident}/{n_total} ({frac:.1%}): "
      f"{'OK' if frac < 0.2 else 'WARNING: suspiciously high, check for aliasing onto condition 1'}")
'''

VERIFY_PIVOT = '''# Condition 4 (pivot). Same dedup, batch-1, and resume discipline as bilingual above.
def build_jobs_pivot():
    jobs = {}
    for r in records:
        for lang in NEW_LANGS:
            q = pivot_question(BY_IDX[r["item_idx"]], lang, pivot_map)
            jobs.setdefault((r["db_id"], r["sql"], "pivot", lang, q), []).append(
                (r["candidate_id"], r["item_idx"], r["split"], lang, r["correct"]))
    return jobs

jobs_pivot = build_jobs_pivot()
print(f"{len(records)*len(NEW_LANGS)} pivot scorings -> {len(jobs_pivot)} unique forward passes")
for name, mid in VERIFIERS.items():
    run_verifier_condition(name, mid, "pivot", jobs_pivot, f"scores_mitigation_pivot-{name}.jsonl")
'''

SAVE = '''import shutil
shutil.make_archive("/content/xsql_mitigation_results", "zip", OUT)
print("results:", [p.name for p in OUT.iterdir()])

from google.colab import files
files.download("/content/xsql_mitigation_results.zip")
'''

MD_RUNTIME = """## Runtime estimate and caveats

**Volume.** The original run found ~48% of the 6000 candidates share SQL text,
so deduping by (db_id, sql, question) collapses ~6000 candidates x 1 language
down to ~3120 unique forward passes per language (this matched exactly: 6000 x
7 languages = 42000 scorings -> ~22k unique passes, quoted in
`build_colab_notebook.py`).

This notebook adds 2 new conditions x 6 non-English languages = 12
language-condition pairs (vs. 7 in the original run, which covered en+native
together in one dedup pass). At ~3120 unique passes each that is roughly
**~37k unique forward passes per verifier** for bilingual+pivot combined,
across 2 verifiers. That is more total verifier work than the original run,
because both new conditions apply to every non-English language.

A single batch-1 forward pass at this prompt length was tens of milliseconds
on an A100 in the original run; at that rate ~37k passes/verifier x 2
verifiers is on the order of an hour of pure compute, but wall-clock will run
higher due to tokenization/schema-lookup overhead per item, matching the
original run's observed ~2-3h for a comparable pass count.

**Translation adds a separate, likely dominant cost:** 1200 items x 6
languages = 7200 short generations with Qwen2.5-7B-Instruct. These are
batched (batch 16, greedy, ~96 new tokens each) since the batch-1 rule only
applies to the verifier's logit read, not to translation text generation, but
7200 generations is still likely **1-2h** on an A100.

**Total estimate: roughly 3-5h on an A100** (translation + bilingual verify +
pivot verify, across 2 verifiers), plausibly more depending on Colab's shared
A100 throughput. Run condition 3 (bilingual) first and confirm it produces
sensible output before starting condition 4's MT step. Under a compute or time
constraint, bilingual alone is the highest-value new result and can be
reported on its own.

**Known residual risk:** the fallback item/candidate regeneration path (used
only if items.jsonl/candidates.jsonl are not found on Drive) reproduces the
same *selection* exactly but cannot guarantee byte-identical candidate SQL
text across different library/GPU versions, a comparability risk against
conditions 1/2. Uploading the original `data/big/{items,candidates}.jsonl`
avoids it and is the recommended path.
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
        cell("markdown", "## 3. Data (schema DDL always needed; question files only for the fallback path)"),
        cell("code", DOWNLOAD),
        cell("code", DATA_HARNESS),
        cell("markdown", "## 4. Load existing items/candidates (reuse, do not regenerate)"),
        cell("code", LOAD),
        cell("markdown", "## 5. The four scoring conditions"),
        cell("code", PROMPTS),
        cell("markdown", "## 6. Condition 3 — bilingual (build and verify this first; needs no translator)"),
        cell("code", VERIFY_BILINGUAL),
        cell("markdown", "## 7. Translate native questions to English (for condition 4)"),
        cell("code", TRANSLATE),
        cell("markdown", "## 8. Condition 4 — pivot (machine-translated native question)"),
        cell("code", VERIFY_PIVOT),
        cell("markdown", "## 9. Export"),
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
