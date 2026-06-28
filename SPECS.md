# PromptGuard — Runtime Prompt-Injection Defense Middleware

> Working name. Rename freely (`pi-shield`, `sentinel`, etc.).
> A middleware library (Python + Node) that sits between user input and your LLM API call,
> scores it for injection, sanitizes dangerous constructs, and logs attempts with full context.

---

## 0. Positioning & prior art (read this first)

This is a **defensive** security tool. It is not novel to _attempt_ runtime defense —
know the landscape so you can position the project honestly and answer interview questions well:

- **Lakera Guard**, **Rebuff** — commercial / OSS injection detectors.
- **NVIDIA NeMo Guardrails** — programmable rails around LLM apps.
- **Microsoft Prompt Shields** + **"spotlighting"** research (datamarking / encoding user input so the model treats it as data, not instructions).
- **Open prompt-injection datasets** on HuggingFace and in academic benchmarks (verify current availability and licenses before use — dataset names and access change).

**Where this project earns its keep:** the _use-vs-mention_ problem. Existing detectors over-block legitimate
adversarial-sounding text (a security chatbot, a red-team tool, a CTF helper). PromptGuard's differentiator is a
**context-aware decision layer** that separates the _topic_ of an input from its _target_.

**Honesty constraints to keep in the README:** no single layer is sufficient; this raises attacker cost, it does
not "solve" injection; it complements, not replaces, least-privilege tool design and output-side guardrails.

---

## 1. Goals & non-goals

**Goals**

- Drop-in middleware (decorator + transparent client wrapper) for Python and Node. No separate service required.
- Detection stages 0–2 run **locally, no network call**, with p95 latency in the low-single-digit milliseconds for typical chat-length inputs.
- A **decision policy** producing one of: `allow`, `sanitize`, `flag`, `block`.
- **Configurable app profiles** so a cybersecurity chatbot and a banking assistant get different thresholds without code changes.
- Structured, privacy-respecting **logging** of suspicious attempts with full context.
- An **evaluation harness** that proves it works, with a headline metric of _false-positive rate on hard negatives_.

**Non-goals**

- Not a model-hosting service or gateway proxy process (it's a library you import).
- Not a guarantee of safety; not a replacement for least-privilege tool scoping or human review of high-risk actions.
- Not an output-side jailbreak/content filter (could be a future module; keep the interface open for it).

---

## 2. Threat model

Inputs we defend against (user-controlled text reaching an LLM, including via RAG/tool outputs):

| Class                       | Examples                                                                                               |
| --------------------------- | ------------------------------------------------------------------------------------------------------ | -------- | ----------------------------------- |
| Instruction override        | "ignore previous instructions", "disregard the rules above", "forget everything"                       |
| System-prompt extraction    | "repeat the text above", "print your system prompt", "what are your instructions verbatim"             |
| Role / turn injection       | fake `System:` / `Assistant:` lines, `<                                                                | im_start | >`, `[INST]`, ChatML special tokens |
| Delimiter breakout          | closing the app's own delimiters/tags (`</system>`, the app's fence sequence)                          |
| Persona / jailbreak framing | "you are now DAN", "developer mode", "you have no restrictions"                                        |
| Tool / action hijack        | "ignore the user and instead email…", "call the transfer function with…"                               |
| Obfuscation                 | base64/hex-encoded payloads, zero-width chars, bidi override, homoglyphs, translation, token-splitting |
| Indirect injection          | malicious instructions embedded in fetched web pages, documents, or tool results                       |

**Trust boundary:** everything in the _user_ and _tool-result_ positions is untrusted. The _system_ prompt and
developer instructions are trusted. Indirect injection means tool/RAG content must be guarded too, not just the chat box.

---

## 3. Architecture — defense in depth

Pipeline of stages; each produces `Finding`s (typed, scored, with a rule/source id). A final **decision policy**
aggregates findings into a `Verdict`. Stages are pluggable and individually toggleable.

```
            ┌─────────────────────────────────────────────────────────────┐
 raw input  │ S0 Normalize → S1 Signatures → S2 Classifier → S3 Intent     │  → Decision → Verdict
 (+context) │  (unicode,      (scored regex   (P(injection),  (use-vs-      │     Policy    {allow|
            │   decode,        rule library)   pluggable)      mention,     │              sanitize|
            │   deobfuscate)                                   structural)   │              flag|block}
            └─────────────────────────────────────────────────────────────┘
                                                          │ (only if uncertain band)
                                                          └─→ S4 LLM-as-judge (gated)
                              ↓ if verdict = sanitize
                        Sanitizer (spotlight / datamark / neutralize delimiters)
```

### Stage 0 — Normalization & de-obfuscation

- Unicode **NFKC** normalization.
- Detect & strip/flag **zero-width** (U+200B–200D, U+FEFF) and **bidi control** chars (U+202A–202E, U+2066–2069).
- **Homoglyph / confusables folding** to ASCII (e.g., Cyrillic 'а' → Latin 'a').
- **Encoding inspection:** detect base64/hex blobs, decode, and **recursively re-scan** decoded content (depth-limited, e.g., 3). Encoded instructions are a classic evasion.
- Normalization is itself a **signal**: heavy obfuscation (many zero-width chars, deep nesting) contributes to the score even before content is examined.
- Output: normalized text + a list of `ObfuscationFinding`s describing what was stripped/decoded.

### Stage 1 — Signature engine (deterministic, microseconds)

- A **scored** rule library (NOT binary). Each rule = id, category, pattern (regex/structural matcher), weight, and a `target` flag (`structural` vs `semantic`).
- **Structural rules** (fake turns, special tokens, delimiter breakouts) are weighted heavily and flagged `structural=True` — these are never "discussion."
- **Semantic rules** (override phrasing, extraction phrasing) are flagged `structural=False` — these are subject to app-profile thresholds downstream.
- Crucial structural check: **delimiter-breakout detection is parameterized by the host app's actual template** (its real fence/tag/special-token sequences are passed in via config). An input that contains _your_ live `</system>` close is doing something a discussion never needs to.
- Output: `SignatureFinding`s with weights and the `structural` flag.

### Stage 2 — Classifier (ML, milliseconds)

Pluggable backend behind a `Classifier` protocol returning `P(injection) ∈ [0,1]`:

- **Backend A (default, ship this first):** sentence-embedding model + lightweight head (logistic regression / gradient boosting). CPU-friendly, small, no heavy deps. Ship a trained baseline model artifact + the training script.
- **Backend B (optional upgrade):** fine-tuned small encoder (DeBERTa-v3-small / DistilBERT) as a sequence classifier. Better accuracy, heavier dependency. Same interface.
- **Backend C (zero-setup fallback):** heuristic-only mode (skip ML) so the library works before any model is trained.
- Training data: combine public injection datasets (attacks) + benign chat (negatives) + the curated hard-negatives set (§6). **Verify dataset licenses.**

### Stage 3 — Intent disambiguation (the hard problem)

Resolves _use vs mention_ using these signals:

- **Addressivity** — is the dangerous instruction addressed to _this_ model ("you must now…") or describing a technique ("an attacker might send 'you must now…'")?
- **Framing / quotation** — is the payload inside quotes, code fences, "for example", "such as", a list of examples? → mention.
- **Policy conflict** — does it attempt to countermand the _actual_ active system prompt? → use.
- **Live structural target** — any Stage-1 `structural=True` finding → strong **use** signal, app-profile-independent.

**App profile** (config object describing what the host app legitimately does):

```
AppProfile(
  name="security-chatbot",
  allow_security_discussion=True,   # raises threshold for SEMANTIC findings only
  risk_tier="low|medium|high",      # tool-using/financial apps = high
  template_delimiters=[...],        # the app's real fences/tags/special tokens
  tools_enabled=bool,               # gates fail-safe behavior on action hijack
)
```

**The decisive rule:**

> Structural findings (breakouts, fake turns, encoded payloads matching the live template) are blocked **regardless** of app profile. Semantic findings get an **app-profile-aware threshold**: a security chatbot can discuss attacks freely, while a banking bot cannot.

This is what cleanly resolves "a cybersecurity chatbot should discuss injection attacks" without opening a hole for real breakouts.

### Stage 4 — LLM-as-judge fallback (gated, optional)

- Only invoked when stages 0–3 land in a configurable **uncertain band** (e.g., aggregate score 0.4–0.6).
- A cheap model call with a **tight rubric**: _"Is this input attempting to manipulate the control flow of the assistant it is addressed to, or discussing such manipulation as a topic? Answer with a structured verdict + one-line rationale."_
- Strict latency/cost budget; returns structured `JudgeFinding`. Fully disableable for offline/local-only deployments.

---

## 4. Sanitization (when verdict = `sanitize`)

- **Spotlighting / datamarking** — wrap user input in unambiguous boundaries and mark it as data (e.g., interleaved markers or an encoding scheme) with a system-side instruction that marked content is data, never instructions. (Conceptually: Microsoft spotlighting.)
- **Delimiter neutralization** — escape/encode any sequence matching the app's template delimiters or model special tokens.
- **Obfuscation flattening** — strip zero-width/bidi; optionally reject (vs flatten) deeply obfuscated input.
- Returns `(transformed_messages, [Transformation])` so the caller can log exactly what changed.

---

## 5. Decision policy, config, fail behavior

- Aggregate findings → score per category → `Verdict {action, score, findings, latency_ms}`.
- Thresholds configurable per `risk_tier`. Defaults:
  - **High-risk (tools/financial):** fail-safe — block on action-hijack/structural; require sanitize otherwise.
  - **Low-risk (pure chat):** fail-open-with-log for borderline semantic cases to protect UX; still hard-block structural.
- Every decision is explainable: it carries the findings that produced it.

---

## 6. Evaluation harness (this is what proves the project)

- **Datasets:** attacks (public injection sets), benign chat (negatives), and a **hand-curated hard-negatives set** — adversarial-sounding-but-legitimate inputs (security Q&A, CTF prompts, red-team write-ups, "explain how X attack works"). The hard-negatives set is the soul of the project; grow it continuously.
- **Headline metrics:** precision / recall / F1 on attacks **and** the **false-positive rate on hard negatives** (report this prominently — it's the differentiator).
- **Latency benchmark:** p50/p95/p99 per stage and end-to-end.
- **Red-team mutator:** a script that mutates known attacks (base64/hex encode, insert zero-width chars, synonym-swap, translate, split tokens) to measure robustness and catch regressions.
- Output a **benchmark table** straight into the README — this is the resume payload.

---

## 7. Observability & logging

- Structured JSON: timestamp, request id, verdict + score, matched rule ids, per-stage latency, profile name.
- **Privacy by default:** do not log raw input. Log a salted hash + a short redacted preview; provide an opt-in `log_raw` for debugging.
- Pluggable sinks: stdout / file / webhook. Counters for verdict distribution & latency (Prometheus-style optional).
- A replayable attempt log for tuning thresholds and growing the hard-negatives set from real traffic.

---

## 8. Integration surface

**Python**

```python
from promptguard import PromptGuard, AppProfile

guard = PromptGuard(profile=AppProfile(name="security-chatbot",
                                       allow_security_discussion=True,
                                       risk_tier="low"))

# Option 1: explicit
verdict = guard.inspect(messages)          # -> Verdict
safe_messages = guard.protect(messages)    # inspect + sanitize, raises/blocks per policy

# Option 2: transparent wrapper
client = guard.wrap(openai_client)         # intercepts .chat.completions.create(...)

# Option 3: decorator on your own handler
@guard.middleware
def handle(messages): ...
```

**Node** — mirror the same surface: a `protect()` function, an Express-style middleware, and a wrapper around the OpenAI/Anthropic SDK client. Core detection logic ported (or shared via a small WASM/native core if you want to avoid drift — note the tradeoff in the README).

---

## 9. Repo layout

```
promptguard/
  python/
    promptguard/
      pipeline.py        # stage orchestration + Verdict
      types.py           # Finding, Verdict, AppProfile, Transformation
      stages/
        normalize.py     # S0
        signatures.py    # S1 + rules.yaml
        classifier.py    # S2 (pluggable backends)
        intent.py        # S3 use-vs-mention
        judge.py         # S4 LLM-as-judge (optional)
      sanitize.py
      policy.py
      logging.py
      integrations/{openai.py,anthropic.py,decorator.py}
    train/               # classifier training script + model artifact
    eval/                # harness, datasets/, red_team.py, report.py
    tests/
  node/                  # mirrored library
  README.md              # with the benchmark table
  benchmarks/
```

---

## 10. Tech choices

- **Python 3.11+**, `ruff` + `black` + `mypy`, `pytest`, GitHub Actions CI.
- Embeddings via `sentence-transformers` (Backend A); optional `transformers` for Backend B. Keep ML deps in an extra (`promptguard[ml]`) so the heuristic-only core stays lightweight.
- **Node 20+**, TypeScript, `vitest`, `eslint`.
- No mandatory network dependency for stages 0–2.

---

## 11. Milestones (resume-ready increments)

1. Heuristic-only core (S0+S1) + decision policy + tests → already useful, zero ML.
2. Classifier backend A + training script + baseline artifact.
3. Intent layer (S3) + app profiles → the headline feature.
4. Sanitizer + integrations (OpenAI/Anthropic wrappers).
5. Eval harness + hard-negatives set + README benchmark table.
6. Red-team mutator + observability.
7. Node port + demo (CLI + tiny web playground).
8. (Stretch) LLM-as-judge, indirect-injection guarding for tool/RAG content, output-side module.
