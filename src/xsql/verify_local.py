"""Local open-weights verifier scored by Yes/No token probability.

Instead of parsing a self-reported confidence string, this reads the model's
next-token distribution directly and returns

    P(Yes) / (P(Yes) + P(No))

which is continuous, needs no output parsing, and cannot be truncated away.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import LOCAL_VERIFIER_MODEL
from .verify import Verdict

SYSTEM = (
    "You are a meticulous database engineer auditing text-to-SQL output. "
    "You judge whether a candidate SQLite query correctly answers a user's "
    "question against a given schema. The question may be written in any "
    "language; judge the query on its merits, not on the question's language."
)

TEMPLATE = """Database schema:
{schema}

User question:
{question}

Candidate SQLite query:
{sql}

Does this query correctly answer the user's question? Answer with a single \
word, Yes or No."""

# Surface forms the model may pick for each verdict; probability mass is summed
# over each group so tokenizer quirks do not bias the score.
YES_FORMS = ["Yes", " Yes", "yes", " yes", "YES", "Y"]
NO_FORMS = ["No", " No", "no", " no", "NO", "N"]


@lru_cache(maxsize=2)
def _load(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="mps"
    )
    model.eval()
    return tokenizer, model


def _first_token_ids(tokenizer, forms: list[str]) -> list[int]:
    """Leading token id for each surface form, deduplicated."""
    ids = []
    for form in forms:
        encoded = tokenizer.encode(form, add_special_tokens=False)
        if encoded:
            ids.append(encoded[0])
    return sorted(set(ids))


class LocalVerifier:
    def __init__(self, model_name: str = LOCAL_VERIFIER_MODEL):
        self.model_name = model_name
        self.tokenizer, self.model = _load(model_name)
        self.yes_ids = _first_token_ids(self.tokenizer, YES_FORMS)
        self.no_ids = _first_token_ids(self.tokenizer, NO_FORMS)
        overlap = set(self.yes_ids) & set(self.no_ids)
        if overlap:
            raise ValueError(f"Yes/No token ids collide: {overlap}")

    def score(self, schema: str, question: str, sql: str) -> Verdict:
        messages = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": TEMPLATE.format(schema=schema, question=question, sql=sql),
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1, :].float()

        probs = torch.softmax(logits, dim=-1)
        yes = probs[self.yes_ids].sum().item()
        no = probs[self.no_ids].sum().item()
        total = yes + no
        if total <= 0:
            # The model put no mass on either verdict; report it rather than guess.
            return Verdict(confidence=None, raw="", stop_reason="no_yes_no_mass")

        return Verdict(
            confidence=yes / total,
            raw=f"P(yes)={yes:.4g} P(no)={no:.4g}",
            stop_reason="logprob",
        )
