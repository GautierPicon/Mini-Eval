import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "models.yaml"
LOCAL_PATH = ROOT / "local.yaml"
SUITES_DIR = ROOT / "suites"
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"


@dataclass
class Model:
    name: str
    kind: str
    family: str
    params: int
    non_embedding_params: int
    tokens_seen: int
    pretrained: str | None = None
    revision: str | None = None
    checkpoint: str | None = None
    sha256: str | None = None
    tokenizer_sha256: str | None = None


@dataclass
class Task:
    name: str
    primary: str
    chance: float | None = None


@dataclass
class Suite:
    name: str
    description: str
    num_fewshot: int
    limit: int | None
    batch_size: int
    tasks: list


def load_registry(path=REGISTRY_PATH):
    entries = yaml.safe_load(Path(path).read_text())["models"]
    models = {}
    for entry in entries:
        model = Model(**entry)
        if model.name in models:
            raise ValueError(f"Duplicate model name: {model.name}")
        if model.kind == "hf" and not (model.pretrained and model.revision):
            raise ValueError(f"{model.name}: an hf model needs 'pretrained' and 'revision'")
        if model.kind == "native" and not model.checkpoint:
            raise ValueError(f"{model.name}: a native model needs 'checkpoint'")
        if model.kind not in ("hf", "native"):
            raise ValueError(f"{model.name}: unknown kind {model.kind!r}")
        models[model.name] = model
    return models


def load_suite(name, directory=SUITES_DIR):
    path = Path(directory) / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in Path(directory).glob("*.yaml"))
        raise SystemExit(f"Unknown suite {name!r}. Available: {available}")
    data = yaml.safe_load(path.read_text())
    tasks = [Task(**t) for t in data["tasks"]]
    return Suite(
        name=data["name"],
        description=data.get("description", ""),
        num_fewshot=data["num_fewshot"],
        limit=data.get("limit"),
        batch_size=data["batch_size"],
        tasks=tasks,
    )


def models_dir():
    env = os.environ.get("MINI_EVAL_MODELS_DIR")
    if env:
        return Path(env).expanduser()
    if LOCAL_PATH.exists():
        local = yaml.safe_load(LOCAL_PATH.read_text()) or {}
        if "models_dir" in local:
            return Path(local["models_dir"]).expanduser()
    raise SystemExit("Set MINI_EVAL_MODELS_DIR or models_dir in local.yaml (see local.example.yaml)")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_paths(model, base=None):
    checkpoint = (base or models_dir()) / model.checkpoint
    return checkpoint, checkpoint.parent / "tok.json"


def fingerprint(model, base=None):
    checkpoint, tokenizer = native_paths(model, base)
    for path in (checkpoint, tokenizer):
        if not path.exists():
            raise SystemExit(f"File not found: {path}")
    return {"sha256": sha256_file(checkpoint), "tokenizer_sha256": sha256_file(tokenizer)}


def verify_native(model, allow_unpinned=False, base=None):
    checkpoint, _ = native_paths(model, base)
    actual = fingerprint(model, base)
    expected = {"sha256": model.sha256, "tokenizer_sha256": model.tokenizer_sha256}
    if None in expected.values():
        if not allow_unpinned:
            raise SystemExit(
                f"{model.name} has no fingerprint in models.yaml. "
                f"Run `mini-eval fingerprint {model.name}`, copy the values, or pass --allow-unpinned for a test run."
            )
    elif actual != expected:
        raise SystemExit(
            f"{model.name}: the files on disk do not match models.yaml.\n"
            f"  expected {expected}\n  found    {actual}"
        )
    return checkpoint, actual