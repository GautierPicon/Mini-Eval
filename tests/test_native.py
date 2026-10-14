import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mini_llm")

from mini_llm.checkpoints import save_checkpoint
from mini_llm.config import Cfg
from mini_llm.model import LLM
from test_backend import PAIRS, encode_pair, request

from mini_llm_eval.backend import NativeLM, log_softmax
from mini_llm_eval.checks import check_batch_invariance


@pytest.fixture
def native(tmp_path, tokenizer):
    torch.manual_seed(0)
    cfg = Cfg(vocab=tokenizer.get_vocab_size(), dim=32, layers=2, heads=2, ffn=64, ctx=96)
    model = LLM(cfg)
    save_checkpoint(tmp_path / "m.pt", model, cfg, kind="base")
    tokenizer.save(str(tmp_path / "tok.json"))
    return NativeLM.from_checkpoint(tmp_path / "m.pt", "cpu", batch_size=4, name="tiny"), model


def test_scores_match_a_direct_forward_pass(native, tokenizer):
    lm, model = native
    model.eval()
    got = lm.loglikelihood([request(c, k) for c, k in PAIRS])
    for (context, continuation), (logprob, _) in zip(PAIRS, got):
        ctx_enc, cont_enc = encode_pair(tokenizer, context, continuation)
        seq = (ctx_enc + cont_enc)[-(lm.max_length + 1) :]
        with torch.no_grad():
            logits = model(torch.tensor([seq[:-1]]))[0][0].double().numpy()
        logp = log_softmax(logits)
        n = len(cont_enc)
        expected = sum(logp[len(seq) - 1 - n + i, token] for i, token in enumerate(cont_enc))
        assert logprob == pytest.approx(expected, abs=1e-4)


def test_batching_does_not_change_scores(native):
    lm, _ = native
    ok, detail = check_batch_invariance(lm)
    assert ok, detail


def test_generation_returns_text(native):
    lm, _ = native
    out = lm.generate_until([request("The sky is", {"max_gen_toks": 5})])
    assert isinstance(out[0], str)
    assert np.isfinite(lm.loglikelihood_rolling([request("The sky is blue.")])[0])


def test_model_info_is_recorded(native):
    lm, model = native
    info = lm.get_model_info()
    assert info["model_num_parameters"] == sum(p.numel() for p in model.parameters())
    assert info["model_dtype"] == "float32"