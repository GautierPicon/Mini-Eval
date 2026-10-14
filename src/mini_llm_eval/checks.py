import math
from types import SimpleNamespace

from lm_eval import simple_evaluate

from mini_llm_eval.backend import NativeLM
from mini_llm_eval.registry import verify_native

SENTENCES = [
    "The sky is blue and the grass is green.",
    "A man mixes flour and water in a large bowl.",
    "She opened the jar by tapping the lid on the table.",
    "Plants need water, light and air to grow.",
    "He walked to the market and bought some bread.",
    "The children played in the garden until it got dark.",
    "To make tea, first boil some water in a kettle.",
    "The train arrived late because of the snow.",
]
CONTINUATIONS = [" and then everything changed.", " because of the weather.", " in the middle of the night."]
CONTROL_TASKS = {"arc_easy": 0.25, "hellaswag": 0.25, "piqa": 0.5}


def requests():
    return [SimpleNamespace(args=(s, c)) for s in SENTENCES for c in CONTINUATIONS]


def max_difference(a, b):
    return max(abs(x[0] - y[0]) for x, y in zip(a, b))


def check_batch_invariance(lm, limit=1e-3):
    original = lm.batch_size
    try:
        lm.batch_size = 1
        single = lm.loglikelihood(requests(), disable_tqdm=True)
        lm.batch_size = 8
        batched = lm.loglikelihood(requests(), disable_tqdm=True)
    finally:
        lm.batch_size = original
    diff = max_difference(single, batched)
    return diff < limit, f"largest difference between batch size 1 and 8: {diff:.2e} nats (limit {limit:.0e})"


def check_device_agreement(lm, checkpoint, device, limit=1e-2):
    if device == "cpu":
        return True, "skipped (already on cpu)"
    reference = NativeLM.from_checkpoint(checkpoint, "cpu", name="cpu-reference")
    on_device = lm.loglikelihood(requests(), disable_tqdm=True)
    on_cpu = reference.loglikelihood(requests(), disable_tqdm=True)
    diff = max_difference(on_device, on_cpu)
    return diff < limit, f"largest difference between {device} and cpu: {diff:.2e} nats (limit {limit:.0e})"


def check_untrained(lm, device, limit=300, margin=0.05):
    import torch
    from mini_llm.model import LLM

    torch.manual_seed(0)
    cfg = lm.model.c
    control = NativeLM(LLM(cfg), cfg, lm.tok, device, batch_size=lm.batch_size, name="untrained-control")
    results = simple_evaluate(
        model=control,
        tasks=list(CONTROL_TASKS),
        num_fewshot=0,
        limit=limit,
        batch_size=lm.batch_size,
        bootstrap_iters=0,
        log_samples=False,
    )
    ok, lines = True, []
    for task, chance in CONTROL_TASKS.items():
        acc = results["results"][task]["acc_norm,none"]
        n = results["n-samples"][task]["effective"]
        tolerance = 3 * math.sqrt(chance * (1 - chance) / n) + margin
        good = abs(acc - chance) <= tolerance
        ok &= good
        lines.append(f"{task}: {acc:.3f} vs chance {chance:.2f} (allowed +-{tolerance:.3f}, n={n}) {'ok' if good else 'TOO FAR'}")
    return ok, "; ".join(lines)


def run_checks(model, device="cpu", allow_unpinned=False):
    checkpoint, fingerprints = verify_native(model, allow_unpinned)
    lm = NativeLM.from_checkpoint(checkpoint, device, name=model.name, info=fingerprints)
    outcomes = [
        ("batch invariance", check_batch_invariance(lm)),
        ("device agreement", check_device_agreement(lm, checkpoint, device)),
        ("untrained control", check_untrained(lm, device)),
    ]
    passed = True
    for name, (ok, detail) in outcomes:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        passed &= ok
    return passed