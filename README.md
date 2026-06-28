# PromptGuard

Runtime prompt-injection defense middleware for LLM applications.

> A library (Python · Node coming soon) that sits between user input and your LLM API call,
> scores it for injection, sanitizes dangerous constructs, and logs attempts with full context.

---

## Honesty constraints

No single layer is sufficient. PromptGuard raises attacker cost — it does not "solve" prompt injection.
It complements, not replaces, least-privilege tool design and output-side guardrails.

---

## Quickstart (Python)

```python
from promptguard import PromptGuard, AppProfile

guard = PromptGuard(profile=AppProfile(
    name="security-chatbot",
    allow_security_discussion=True,
    risk_tier="low",
))

verdict = guard.inspect(user_text)
if verdict.blocked:
    raise PermissionError("Input blocked by PromptGuard")

# Or: block + sanitize in one call
safe_text = guard.protect(user_text)   # raises PermissionError on block
```

---

## Architecture

```
raw input → S0 Normalize → S1 Signatures → S2 Classifier → S3 Intent → Decision Policy → Verdict
                                                                  ↓ (uncertain band only)
                                                             S4 LLM-as-judge
                              ↓ if verdict = sanitize
                        Sanitizer (spotlight / neutralize / flatten)
```

| Stage | Name | Status |
|-------|------|--------|
| S0 | Normalization & de-obfuscation (unicode, ZW, homoglyphs, encoding) | **done** |
| S1 | Signature engine (scored regex rules — 35+ rules across 6 categories) | **done** |
| S2 | ML classifier (backend A: embeddings + LR; backend C: heuristic) | **done** |
| S3 | Intent disambiguation (use vs mention, framing/quotation signals) | **done** |
| Sanitizer | Spotlight datamarking + delimiter neutralization + obfuscation flattening | **done** |
| S4 | LLM-as-judge fallback | stub |

---

## Benchmark

*To be filled by `eval/report.py` once the full evaluation harness runs against
real public injection datasets (see `eval/datasets/README.md`).*

| Dataset | Precision | Recall | F1 | FP rate on hard negatives |
|---------|-----------|--------|----|--------------------------|
| sample (seed) | — | — | — | — |

### Red-team mutation results (heuristic pipeline)

Run `python -m eval.red_team` to reproduce.

| Mutation | Evasion rate | Notes |
|----------|-------------|-------|
| base64 | **0%** | S0 decodes and rescores to 0.62 → sanitize |
| hex | **0%** | Same |
| zero_width | **0%** | S0 strips before S1 runs |
| synonym_swap | ~22% | Unusual synonyms need ML classifier for full coverage |
| token_split | ~43% | Word-level splits (`i gnore`) require embedding detection |
| translation | ~100% | Expected; English-only patterns cannot catch foreign text |
| **overall** | **~14%** | CI threshold: 40% |

---

## Growing the hard-negatives set from real traffic

The attempt log (§7) is the best source of hard-negative examples.  Any input
that gets a non-allow verdict but turns out to be legitimate user traffic is
a high-value hard negative.

### Workflow

1. **Enable attempt logging** (in your PromptGuard integration):

   ```python
   from promptguard.logging import VerdictLogger
   logger = VerdictLogger(sink="file", path="attempts.jsonl")
   guard = PromptGuard(profile=..., logger=logger)
   ```

2. **Review flagged attempts** — the log records a salted hash + 40-char
   redacted preview (raw input is never logged by default):

   ```jsonl
   {"verdict": "flag", "score": 0.82, "input_hash": "a3f2...", "preview": "Can you explain how 'ignore prev…"}
   ```

3. **Recover and label** — when a flagged attempt is confirmed legitimate,
   add it to `eval/datasets/hard_negatives.jsonl` with `"label": 0`:

   ```jsonl
   {"text": "Can you explain how 'ignore previous instructions' attacks work?", "label": 0, "source": "prod_traffic_2026-06"}
   ```

4. **Re-train** — the hard negatives improve the ML classifier and lower the
   false-positive rate on security discussions, CTF prompts, and red-team
   write-ups.  Run `python -m train.train` to produce an updated artifact.

5. **Re-run eval** — `python -m eval.red_team` and `pytest` confirm the FP
   rate improved without regressing on the attack set.

### What makes a good hard negative

- Security Q&A that names attack techniques ("explain how DAN jailbreak works")
- CTF challenge descriptions that quote injection payloads
- Red-team write-ups that analyze real attack strings
- Code snippets that discuss injection in an educational context

**Do not add**: inputs that are actual attacks even if they look educational,
or inputs that contain the user's real PII.

---

## Development

```bash
cd python
pip install -e ".[dev]"
pytest
ruff check .
mypy promptguard

# Red-team benchmark
python -m eval.red_team

# Train classifier (requires [ml] extra)
pip install -e ".[ml]"
python -m train.train
```

---

## Milestones

1. **[done]** Heuristic-only core (S0+S1) + decision policy + tests
2. **[done]** Classifier backend A + training script + baseline artifact
3. **[done]** Intent layer (S3) + app profiles — use-vs-mention disambiguation
4. **[done]** Sanitizer (spotlight + delimiter neutralization + obfuscation flattening)
5. **[done]** Red-team mutator + rule tightening; CI evasion gate
6. Eval harness + full hard-negatives set + README benchmark table
7. Integrations (OpenAI/Anthropic wrappers) + structured logging
8. Node port + demo
9. *(stretch)* LLM-as-judge, indirect-injection guarding, output-side module

---

## License

MIT
