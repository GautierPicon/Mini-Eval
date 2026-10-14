from pathlib import Path
from types import SimpleNamespace

import numpy as np
from lm_eval.api.model import TemplateLM
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window
from tqdm import tqdm

CHAT_TEMPLATE_NAME = "mini-llm-user-assistant-v1"


def log_softmax(x):
    x = x.astype(np.float64)
    peak = x.max(axis=-1, keepdims=True)
    return x - (peak + np.log(np.exp(x - peak).sum(axis=-1, keepdims=True)))


class ScoringLM(TemplateLM):
    max_gen_toks = 256

    def __init__(self, tokenizer, logits_fn, max_length, batch_size=16, name="model", info=None):
        super().__init__()
        self.tok = tokenizer
        self.logits_fn = logits_fn
        self._max_length = max_length
        self.batch_size = batch_size
        self.config = SimpleNamespace(_name_or_path=name)
        self.info = dict(info or {})
        self._eot = tokenizer.token_to_id("<eos>")

    @property
    def eot_token_id(self):
        return self._eot

    @property
    def max_length(self):
        return self._max_length

    @property
    def tokenizer_name(self):
        return self.info.get("tokenizer_sha256", "mini-llm-tokenizer")

    def tok_encode(self, string, add_special_tokens=None, **kwargs):
        return self.tok.encode(string).ids

    def tok_decode(self, tokens):
        return self.tok.decode(list(tokens))

    def get_model_info(self):
        return dict(self.info)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
        batch_size = override_bs or self.batch_size
        order = sorted(range(len(requests)), key=lambda i: -(len(requests[i][1]) + len(requests[i][2])))
        results = [None] * len(requests)
        for start in tqdm(range(0, len(order), batch_size), disable=disable_tqdm, desc="loglikelihood"):
            idx = order[start : start + batch_size]
            inputs, conts = [], []
            for i in idx:
                _, context_enc, continuation_enc = requests[i]
                assert len(context_enc) > 0 and len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length
                inputs.append((context_enc + continuation_enc)[-(self.max_length + 1) :][:-1])
                conts.append(continuation_enc)
            logits = self.logits_fn(inputs, [len(c) for c in conts])
            for i, cont, rows in zip(idx, conts, logits):
                logp = log_softmax(rows)
                targets = np.asarray(cont)
                picked = logp[np.arange(len(cont)), targets]
                greedy = bool((logp.argmax(axis=-1) == targets).all())
                results[i] = (float(picked.sum()), greedy)
                key = requests[i][0]
                if key is not None:
                    self.cache_hook.add_partial("loglikelihood", key, results[i])
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        windows, counts = [], []
        for (string,) in (r.args for r in requests):
            made = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            windows.extend((None,) + w for w in made)
            counts.append(len(made))
        scores = self._loglikelihood_tokens(windows, disable_tqdm=disable_tqdm)
        out, k = [], 0
        for count in counts:
            out.append(sum(s[0] for s in scores[k : k + count]))
            k += count
        return out

    def generate_until(self, requests, disable_tqdm=False):
        outputs = []
        for context, gen_kwargs in tqdm([r.args for r in requests], disable=disable_tqdm, desc="generate"):
            until = gen_kwargs.get("until") or []
            if isinstance(until, str):
                until = [until]
            limit = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))
            ids = self.tok_encode(context) or [self.prefix_token_id]
            generated, text = [], ""
            for _ in range(limit):
                window = (ids + generated)[-self.max_length :]
                nxt = int(np.argmax(self.logits_fn([window], [1])[0][-1]))
                if nxt == self.eot_token_id:
                    break
                generated.append(nxt)
                text = self.tok_decode(generated)
                stops = [text.index(s) for s in until if s and s in text]
                if stops:
                    text = text[: min(stops)]
                    break
            outputs.append(text)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text)
        return outputs

    def chat_template(self, chat_template=False):
        return CHAT_TEMPLATE_NAME if chat_template else None

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        lines = []
        for message in chat_history:
            role, content = message["role"], message["content"].strip()
            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
            else:
                raise ValueError(f"Unsupported role {role!r}: the training format only has user and assistant")
        if add_generation_prompt:
            lines.append("Assistant:")
        return "\n".join(lines)


class NativeLM(ScoringLM):
    def __init__(self, model, cfg, tokenizer, device="cpu", batch_size=16, name="native", info=None):
        import torch

        self._torch = torch
        self.model = model.to(device).eval()
        self.dev = device
        super().__init__(
            tokenizer=tokenizer,
            logits_fn=self._forward,
            max_length=cfg.ctx,
            batch_size=batch_size,
            name=name,
            info=info,
        )
        self._device = device
        self.pad_id = tokenizer.token_to_id("<eos>")

    @classmethod
    def from_checkpoint(cls, path, device="cpu", batch_size=16, name=None, info=None):
        from mini_llm.checkpoints import load_model, load_tokenizer

        path = Path(path)
        model, cfg = load_model(path, device)
        n_params = sum(p.numel() for p in model.parameters())
        merged = {"model_num_parameters": n_params, "model_dtype": "float32", "device": device, **(info or {})}
        return cls(model, cfg, load_tokenizer(path), device, batch_size, name or path.stem, merged)

    def _forward(self, seqs, tails):
        torch = self._torch
        width = max(len(s) for s in seqs)
        x = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        with torch.no_grad():
            logits = self.model(x.to(self.dev))[0]
        return [
            logits[i, len(s) - t : len(s)].float().cpu().numpy()
            for i, (s, t) in enumerate(zip(seqs, tails))
        ]