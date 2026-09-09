"""Local open-weights SQL generator (the reproducible half of the pipeline)."""

from __future__ import annotations

import re
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import GENERATOR_MODEL

PROMPT = """You are an expert at writing SQLite queries.

Database schema:
{schema}

Question: {question}

Write a single SQLite query that answers the question. Respond with only the \
query inside a ```sql code block."""


@lru_cache(maxsize=1)
def _load(model_name: str = GENERATOR_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="mps"
    )
    model.eval()
    return tokenizer, model


def extract_sql(text: str) -> str:
    """Pull the query out of the model's response.

    Falls back to the raw text so that malformed output still becomes an
    executable-or-not candidate rather than being silently dropped.
    """
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    sql = fenced.group(1) if fenced else text
    sql = sql.strip()
    # Keep only the first statement; trailing prose or extra queries confuse execution.
    if ";" in sql:
        sql = sql.split(";")[0]
    return " ".join(sql.split())


def generate_candidates(
    schema: str,
    question: str,
    k: int,
    temperature: float = 0.8,
    max_new_tokens: int = 256,
    seed: int | None = None,
) -> list[str]:
    """Sample k SQL candidates for one question."""
    tokenizer, model = _load()
    messages = [{"role": "user", "content": PROMPT.format(schema=schema, question=question)}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    if seed is not None:
        torch.manual_seed(seed)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            num_return_sequences=k,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    texts = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
    return [extract_sql(t) for t in texts]
