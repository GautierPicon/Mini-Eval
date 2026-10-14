# mini-eval

Evaluation of [mini-llm](../mini-llm) models against reference models (Pythia, ...) with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). This repository is separate from the training repository on purpose: it only asks the models for probabilities, and `lm-eval` computes the scores.

The rules that make results comparable are in [PROTOCOL.md](PROTOCOL.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync
cp local.example.yaml local.yaml
```

Edit `local.yaml` so that `models_dir` points to the `models/` folder of the training repository (or set `MINI_EVAL_MODELS_DIR`). This file stays on your machine and is ignored by Git.

The training repository is a dependency, declared in `pyproject.toml`:

```
[tool.uv.sources]
mini-llm = { path = "../mini-llm", editable = true }
```

That is convenient while developing. For reported numbers, pin an exact tag instead (`mini-llm = { git = "<repo url>", tag = "..." }`) and run `uv lock`, so the code that produces the probabilities is identified by a commit.

## Commands

```
uv run mini-eval list
uv run mini-eval fingerprint mini-llm-34M-12000steps
uv run mini-eval check mini-llm-34M-12000steps
uv run mini-eval run smoke mini-llm-34M-12000steps pythia-70m-step128
uv run mini-eval run small_lm --all --require-clean
uv run mini-eval report small_lm
uv run pytest
```

| Command                      | What it does                                                   |
| ---------------------------- | -------------------------------------------------------------- |
| `list`                       | Shows the registered models and suites                         |
| `fingerprint MODEL`          | Prints the SHA-256 values to copy into `models.yaml`           |
| `check MODEL [--device mps]` | Batch invariance, device agreement with cpu, untrained control |
| `run SUITE MODEL... [--all]` | Evaluates models and saves each run in `results/`              |
| `report SUITE`               | Builds `reports/<suite>.md` and the two plots                  |

Useful options for `run`: `--device cpu|mps|cuda`, `--limit N` (smoke tests only), `--allow-unpinned` (models without a fingerprint, test runs only), `--require-clean` (refuses to run unless the working tree is committed).

## Adding a model

1. A reference model: add an entry with `kind: hf`, the Hub id and a `revision`, plus its parameter counts and the tokens it saw.
2. One of your models: copy the checkpoint into the training repository's `models/` folder (the `tok.json` next to it is used), add an entry with `kind: native` and `checkpoint: <file name>`, run `mini-eval fingerprint NAME`, and paste the two hashes into `models.yaml`.

## How native models are evaluated

`src/mini_eval/backend.py` implements the `lm-eval` model interface on top of your checkpoints: no export and no `config.json` are needed. Context and continuation are encoded with the same rules as the Hugging Face backend of `lm-eval` (trailing spaces move to the continuation, an empty context starts with the end-of-text token, long inputs are truncated on the left, long texts use disjoint rolling windows), so scores are computed the same way for references and for your models.

What it relies on from the training repository:

- `mini_llm.checkpoints.load_model(path, device)` returns `(model, cfg)`, and `load_tokenizer(path)` reads the `tok.json` next to the checkpoint.
- `model(x)` returns `(logits, loss)`, and `model.c` is the configuration with `ctx`.
- `tok.json` defines an `<eos>` token.

`tests/test_native.py` checks these assumptions against the real training package when it is installed.

## Results

```
results/<model>/<suite>/<timestamp>/
    results.json    scores, standard errors, versions, sample counts
    meta.json       model entry, suite, device, versions, git state, fingerprints
    samples/        one line per question (ignored by Git)
```

Runs are never overwritten. `report` uses the latest run of each model and suite.

## Tests

`uv run pytest` runs the adapter against hand computations, inside the real `lm-eval` harness on a tiny local task, and the registry, runner and report logic. These tests use fake models and need no network. `tests/test_native.py` is skipped unless PyTorch and the training package are installed.
