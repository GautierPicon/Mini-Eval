"""Tests for the ScoringLM adapter and the registry/runner/report logic.

These tests use a fake logits function and a tiny session-scoped tokenizer
(see conftest.py): no network, no GPU, no training checkpoint needed.

The helpers PAIRS / encode_pair / request are also imported by
test_native.py, so their tokenization must match the real lm-eval harness:
trailing spaces move to the continuation, an empty context is replaced by
the end-of-text token (see TemplateLM._encode_pair / loglikelihood).
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from mini_eval.backend import ScoringLM, log_softmax


def request(*args):
    return SimpleNamespace(args=args)


PAIRS = [
    ("The sky is", " blue."),
    ("A man mixes flour", " and water."),
    ("", "Hello world."),
    ("Trailing space ", "here."),
    ("She opened the jar", " by tapping the lid."),
]


def encode_pair(tokenizer, context, continuation):
    if context == "":
        cont_enc = tokenizer.encode(continuation).ids
        eos = tokenizer.token_to_id("<eos>")
        if eos != cont_enc[0]:
            return [eos], cont_enc
        return cont_enc[:1], cont_enc[1:]
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole = tokenizer.encode(context + continuation).ids
    ctx_enc = tokenizer.encode(context).ids
    return ctx_enc, whole[len(ctx_enc) :]


@pytest.fixture
def lm(tokenizer):
    vocab = tokenizer.get_vocab_size()

    def logits_fn(seqs, tails):
        return [np.zeros((t, vocab)) for t in tails]

    return ScoringLM(tokenizer, logits_fn, max_length=64, batch_size=4, name="fake")


def test_loglikelihood_matches_hand_computation(lm, tokenizer):
    got = lm.loglikelihood([request(c, k) for c, k in PAIRS], disable_tqdm=True)
    vocab = tokenizer.get_vocab_size()
    for (context, continuation), (logprob, _) in zip(PAIRS, got):
        ctx_enc, cont_enc = encode_pair(tokenizer, context, continuation)
        assert len(ctx_enc) > 0 and len(continuation) > 0
        assert len(cont_enc) <= lm.max_length
        assert logprob == pytest.approx(-len(cont_enc) * math.log(vocab))


def test_log_softmax_is_normalized():
    x = np.array([[1.0, 2.0, 3.0], [-1000.0, 0.0, 1000.0]])
    logp = log_softmax(x)
    assert np.allclose(np.exp(logp).sum(axis=-1), 1.0)


def test_batching_does_not_change_scores(lm):
    reqs = [request(c, k) for c, k in PAIRS]
    lm.batch_size = 1
    single = lm.loglikelihood(reqs, disable_tqdm=True)
    lm.batch_size = 8
    batched = lm.loglikelihood(reqs, disable_tqdm=True)
    assert max(abs(a[0] - b[0]) for a, b in zip(single, batched)) < 1e-9


def test_long_inputs_are_truncated_on_the_left(lm, tokenizer):
    context = "The sky is blue. " * 40
    continuation = " And then everything changed."
    (logprob, _) = lm.loglikelihood([request(context, continuation)], disable_tqdm=True)[0]
    _, cont_enc = encode_pair(tokenizer, context, continuation)
    assert len(tokenizer.encode(context).ids + cont_enc) > lm.max_length + 1
    assert logprob == pytest.approx(-len(cont_enc) * math.log(tokenizer.get_vocab_size()))


def test_rolling_and_generation_run(lm):
    (score,) = lm.loglikelihood_rolling([request("The sky is blue.")], disable_tqdm=True)
    assert np.isfinite(score)
    out = lm.generate_until([request("The sky is", {"max_gen_toks": 5})], disable_tqdm=True)
    assert isinstance(out[0], str)


def test_registry_loads_expected_models():
    from mini_eval.registry import load_registry, load_suite

    registry = load_registry()
    assert "pythia-70m-step143000" in registry
    natives = [m for m in registry.values() if m.kind == "native"]
    assert natives, "no native models registered"
    assert all(m.sha256 and m.tokenizer_sha256 for m in natives), "native models must be fingerprinted"
    suite = load_suite("small_lm")
    assert {t.name for t in suite.tasks} >= {"hellaswag", "arc_easy", "piqa"}
    with pytest.raises(SystemExit):
        load_suite("does_not_exist")


def test_verify_native_refuses_unpinned(tmp_path):
    from mini_eval.registry import Model, fingerprint, verify_native

    model = Model(
        name="x",
        kind="native",
        family="f",
        params=1,
        non_embedding_params=1,
        tokens_seen=1,
        checkpoint="m.pt",
    )
    (tmp_path / "m.pt").write_bytes(b"ckpt")
    (tmp_path / "tok.json").write_bytes(b"tok")
    with pytest.raises(SystemExit):
        verify_native(model, base=tmp_path)
    fp = fingerprint(model, base=tmp_path)
    model.sha256 = fp["sha256"]
    model.tokenizer_sha256 = fp["tokenizer_sha256"]
    _, found = verify_native(model, base=tmp_path)
    assert found == fp


def test_save_run_collect_and_report(tmp_path):
    import json

    from mini_eval.registry import Model, load_suite
    from mini_eval.report import collect, consistency_warnings, markdown_table
    from mini_eval.runner import save_run

    model = Model(
        name="fake",
        kind="native",
        family="fake",
        params=100,
        non_embedding_params=80,
        tokens_seen=1000,
        checkpoint="m.pt",
    )
    suite = load_suite("smoke")
    results = {
        "results": {
            t.name: {f"{t.primary},none": 0.5, f"{t.primary}_stderr,none": 0.01} for t in suite.tasks
        },
        "n-samples": {},
        "samples": {t.name: [] for t in suite.tasks},
    }
    meta = {
        "protocol": 1,
        "model": {"name": "fake"},
        "suite": {"num_fewshot": 0, "limit": 50},
        "device": "cpu",
        "lm_eval": "0.4.13",
        "mini_eval": {"commit": "abc", "dirty": False},
    }
    run_dir = tmp_path / "fake" / "smoke" / "20240101-000000"
    save_run(run_dir, dict(results), dict(meta))
    rows = collect(suite, {"fake": model}, results_dir=tmp_path)
    assert len(rows) == 1
    table = markdown_table(suite, rows)
    assert "fake" in table and "0.500" in table
    warnings = consistency_warnings(rows)
    assert any("limit" in w for w in warnings), warnings
    got = json.loads((run_dir / "results.json").read_text())
    assert "arc_easy" in got["results"]


def test_consistency_warnings_accepts_legacy_meta_key():
    from mini_eval.report import consistency_warnings

    base = {
        "lm_eval": "0.4.13",
        "protocol": 1,
        "suite": {"num_fewshot": 0, "limit": None},
        "device": "cpu",
    }
    new = {"meta": {**base, "mini_eval": {"commit": "abc", "dirty": False}}}
    legacy = {"meta": {**base, "mini_llm_eval": {"commit": "abc", "dirty": True}}}
    assert consistency_warnings([new]) == []
    assert any("working tree" in w for w in consistency_warnings([legacy]))
