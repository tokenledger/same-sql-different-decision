"""Remote verifier backed by NDIF (National Deep Inference Fabric).

Same Yes/No logprob scoring as the local backend, but executed on NDIF's hosted
weights via nnsight. This buys access to models too large to run locally.

Only the Yes/No token probabilities are pulled back rather than the full
vocabulary distribution, which is ~180KB per call at Llama's vocab size.

Note that NDIF hosts `meta-llama/Llama-3.1-8B` as a *base* model. The prompt
below assumes an instruct-tuned chat template, so use an -Instruct deployment.
"""

from __future__ import annotations

import os

import torch
from nnsight import CONFIG, LanguageModel

from .verify import Verdict
from .verify_local import NO_FORMS, SYSTEM, TEMPLATE, YES_FORMS, _first_token_ids

NDIF_VERIFIER_MODEL = "meta-llama/Llama-3.1-70B-Instruct"


class NDIFVerifier:
    def __init__(self, model_name: str = NDIF_VERIFIER_MODEL):
        key = os.environ.get("NDIF_API_KEY")
        if not key:
            raise RuntimeError("NDIF_API_KEY is not set")
        CONFIG.API.APIKEY = key

        self.model_name = model_name
        self.llm = LanguageModel(model_name)
        self.tokenizer = self.llm.tokenizer
        self.yes_ids = _first_token_ids(self.tokenizer, YES_FORMS)
        self.no_ids = _first_token_ids(self.tokenizer, NO_FORMS)

    def _prompt(self, schema: str, question: str, sql: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": TEMPLATE.format(schema=schema, question=question, sql=sql),
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def score(self, schema: str, question: str, sql: str) -> Verdict:
        return self.score_batch([(schema, question, sql)])[0]

    def score_batch(self, triples: list[tuple[str, str, str]]) -> list[Verdict]:
        """Score several candidates in one remote round trip.

        NDIF round-trip latency dominates compute for prompts this size, so
        batching is what makes a full-scale run practical.
        """
        prompts = [self._prompt(*t) for t in triples]

        with self.llm.session(remote=True):
            saved = []
            for prompt in prompts:
                with self.llm.trace(prompt):
                    probs = torch.softmax(self.llm.lm_head.output[0, -1].float(), dim=-1)
                    # Index remotely so only two scalars cross the network.
                    saved.append(
                        (probs[self.yes_ids].sum().save(), probs[self.no_ids].sum().save())
                    )

        verdicts = []
        for yes, no in saved:
            y, n = float(yes), float(no)
            total = y + n
            if total <= 0:
                verdicts.append(Verdict(None, "", "no_yes_no_mass"))
            else:
                verdicts.append(
                    Verdict(y / total, f"P(yes)={y:.4g} P(no)={n:.4g}", "logprob")
                )
        return verdicts
