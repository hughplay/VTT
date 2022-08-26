from asyncio.log import logger

import torch
import torch.nn.functional as F
from einops import rearrange

from src.dataset.text import SimpleTokenizer


def top_p(logits, p=0.0):
    if p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_rm = cum_probs > (1 - p)
        sorted_indices_rm[..., 1:] = sorted_indices_rm[..., :-1].clone()
        sorted_indices_rm[..., 0] = 0

        sorted_logits[sorted_indices_rm] = float("-inf")
        return sorted_logits.scatter(-1, sorted_indices, sorted_logits)
    return logits


def top_k(logits, k=0):
    if k > 0:
        k = min(k, logits.size(-1))
        val, ind = torch.topk(logits, k)
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(-1, ind, val)
        return logits
    return logits


class SimpleGenerationMixin:

    start_idx = SimpleTokenizer().start_idx
    first_hit = True

    def prepare_initial_input_ids(self, **model_kwargs):
        B, N = model_kwargs["trans_mask"].size()
        input_ids = (
            torch.empty((B, N, 1), device=model_kwargs["trans_mask"].device)
            .fill_(self.start_idx)
            .long()
        )
        return input_ids

    def prepare_model_inputs(
        self, input_ids: torch.Tensor = None, **model_kwargs
    ):
        label_mask = torch.ones_like(input_ids).bool()
        return {
            "label_ids": input_ids,
            "label_mask": label_mask,
            **model_kwargs,
        }

    def logits_process(self, logits: torch.Tensor):
        logits = top_k(logits, self.generate_cfg.get("top_k", 0))
        logits = top_p(logits, self.generate_cfg.get("top_p", 0.0))
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        **model_kwargs,
    ) -> torch.LongTensor:

        if self.first_hit:
            logger.info("Generating sequence from scratch...")
            logger.info(f"Generation configs: {self.generate_cfg}")
            self.first_hit = False

        if input_ids is None:
            input_ids = self.prepare_initial_input_ids(**model_kwargs)

        do_sample = self.generate_cfg.get("do_sample", False)
        temperature = self.generate_cfg.get("temperature", 1.0)
        max_words = self.generate_cfg.get("max_words", 24)

        while True:
            model_input = self.prepare_model_inputs(input_ids, **model_kwargs)
            outputs = self(**model_input, return_dict=True)
            next_token_logits = outputs["logits"][..., -1, :]

            if do_sample:
                next_token_logits = self.logits_process(next_token_logits)
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                B, N, W = probs.size()
                probs = rearrange(probs, "B N W -> (B N) W", B=B, N=N, W=W)
                next_tokens = torch.multinomial(probs, num_samples=1)
                next_tokens = rearrange(
                    next_tokens, "(B N) W -> B N W", B=B, N=N, W=1
                )
            else:
                next_tokens = torch.argmax(
                    next_token_logits, dim=-1, keepdim=True
                )

            input_ids = torch.cat([input_ids, next_tokens], dim=-1)

            if input_ids.size(2) >= max_words:
                break

        return {"sequence": input_ids}
