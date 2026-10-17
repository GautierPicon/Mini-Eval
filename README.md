# Mini-Eval

Evaluation of [Mini-LLM](../mini-llm) models against reference models (Pythia, ...)
with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
This repository is separate from the training repository on purpose: it only asks the
models for probabilities, and `lm-eval` computes the scores.

**Full documentation: [../documentation/Mini-Eval](../documentation/Mini-Eval/README.md)**

The rules that make every number comparable are in [PROTOCOL.md](PROTOCOL.md) (also
mirrored in the [documentation](../documentation/Mini-Eval/protocol.md)).

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync
cp local.example.yaml local.yaml
```

Edit `local.yaml` so that `models_dir` points to the `models/` folder of the training
repository (or set `MINI_EVAL_MODELS_DIR`). This file stays on your machine and is
ignored by Git.

## Typical run

```
uv run mini-eval list
uv run mini-eval run smoke mini-llm-34M-12000steps pythia-70m-step128
uv run mini-eval run small_lm --all --require-clean
uv run mini-eval report small_lm
uv run pytest
```

`run` always rebuilds `reports/<suite>.md` plus the two plots from the latest runs.
Tests use fake models and need no network. Everything else — command reference,
adding a model, the native backend, results layout — lives in the
[documentation](../documentation/Mini-Eval/README.md).
