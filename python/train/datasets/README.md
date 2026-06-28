# PromptGuard Training Datasets

This directory holds the datasets used to train Classifier Backend A.

## Included

| File | Description | Examples |
|------|-------------|---------|
| `sample.jsonl` | Synthetic sample dataset (ships with the repo) | 64 |

The sample dataset is sufficient to demonstrate end-to-end training and verify
that the trained model beats the majority-class baseline.  It is **not**
sufficient for a production-quality classifier.

## Dataset format

Each line is a JSON object:

```json
{"text": "...", "label": 0, "split": "train"}
{"text": "...", "label": 1, "split": "test", "note": "optional free-text"}
```

| Field | Type | Meaning |
|-------|------|---------|
| `text` | string | The input message (after any pre-processing) |
| `label` | 0 or 1 | 0 = benign (including hard negatives), 1 = injection/attack |
| `split` | "train" \| "test" | Partition assignment |
| `note` | string (optional) | Human annotation, e.g. "hard-negative: security discussion" |

## Adding real datasets

Drop additional `.jsonl` files in this directory.  `train.py` loads every
`*.jsonl` file automatically.

Recommended public sources (verify current availability and licenses before use —
access terms and file formats change):

- **HuggingFace datasets tagged `prompt-injection`**: search hub.huggingface.co
- **Lakera PINT benchmark** (check for license/access changes)
- **Gandalf dataset** by Lakera (check current availability)
- **AIM (Adversarial Instruction-following Modification)** datasets
- **JailbreakBench** — https://jailbreakbench.github.io

Hard negatives (label=0 examples that discuss injection without being attacks)
are especially valuable.  Sources:
- Security research write-ups
- CTF challenge descriptions
- Red-team blog posts (with the payload removed or quoted)

## Privacy note

Do **not** commit datasets that contain real user messages, PII, or data under
restrictive licences.  The `train/datasets/` directory is gitignored except
for `sample.jsonl` and this README.
