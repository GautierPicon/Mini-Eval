import json
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from lm_eval import simple_evaluate
from lm_eval.utils import handle_non_serializable

from mini_eval import PROTOCOL_VERSION
from mini_eval.registry import RESULTS_DIR, ROOT, verify_native


def git_state(path):
    try:
        run = lambda *args: subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, check=True
        ).stdout.strip()
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except Exception:
        return {"commit": None, "dirty": None}


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def training_package_info():
    try:
        import mini_llm
    except ImportError:
        return {}
    return {"version": package_version("mini-llm"), **git_state(Path(mini_llm.__file__).resolve().parents[2])}


def build_target(model, device, batch_size, allow_unpinned):
    if model.kind == "hf":
        args = f"pretrained={model.pretrained},revision={model.revision},dtype=float32"
        return {"model": "hf", "model_args": args}, {}
    from mini_eval.backend import NativeLM

    checkpoint, fingerprints = verify_native(model, allow_unpinned)
    info = {**fingerprints, "checkpoint": model.checkpoint}
    lm = NativeLM.from_checkpoint(checkpoint, device, batch_size, name=model.name, info=info)
    return {"model": lm}, fingerprints


def save_run(directory, results, meta):
    directory.mkdir(parents=True, exist_ok=False)
    samples = results.pop("samples", {})
    (directory / "samples").mkdir()
    for task, items in samples.items():
        with open(directory / "samples" / f"{task}.jsonl", "w") as f:
            for item in items:
                f.write(json.dumps(item, default=handle_non_serializable) + "\n")
    (directory / "results.json").write_text(json.dumps(results, indent=2, default=handle_non_serializable))
    (directory / "meta.json").write_text(json.dumps(meta, indent=2))


def run_suite(
    model,
    suite,
    device="cpu",
    allow_unpinned=False,
    require_clean=False,
    lm=None,
    task_manager=None,
    results_dir=RESULTS_DIR,
):
    repo = git_state(ROOT)
    if require_clean and repo["dirty"] is not False:
        raise SystemExit("Refusing to run: the working tree must be clean and under git for reported numbers.")
    started = time.time()
    if lm is not None:
        target, fingerprints = {"model": lm}, {}
    else:
        target, fingerprints = build_target(model, device, suite.batch_size, allow_unpinned)
    kwargs = dict(
        tasks=[t.name for t in suite.tasks],
        num_fewshot=suite.num_fewshot,
        batch_size=suite.batch_size,
        limit=suite.limit,
        device=device,
        log_samples=True,
    )
    if task_manager is not None:
        kwargs["task_manager"] = task_manager
    results = simple_evaluate(**target, **kwargs)
    meta = {
        "protocol": PROTOCOL_VERSION,
        "model": asdict(model),
        "suite": asdict(suite),
        "device": device,
        "fingerprint": fingerprints,
        "lm_eval": package_version("lm_eval"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
        "mini_eval": repo,
        "mini_llm": training_package_info(),
        "date": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": round(time.time() - started, 1),
    }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = Path(results_dir) / model.name / suite.name / stamp
    save_run(directory, results, meta)
    return directory