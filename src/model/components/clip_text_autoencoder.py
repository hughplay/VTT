import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from torch import nn

from src.dataset.text import SimpleTokenizer

from .image_encoder import _INPUT_OUTPUT_DIM
from .text_decoder import CLIPFusionDecoder
from .text_encoder import TextCLIP

logger = logging.getLogger(__name__)


class CLIPAutoEncoder(nn.Module):
    def __init__(
        self,
        from_decoder_ckpt: str = None,
        init_encoder: bool = True,
        encoder_name: str = "ViT-L/14",
        decoder_key_prefix: str = "model.text_decoder.",
        num_decoder_layers: int = 2,
        mode: str = "fusion",
        generate_cfg: Dict[str, Any] = {
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 100,
            "temperature": 1.0,
            "max_words": 77,
            "min_words": 1,
            "repetition_penalty": 1.0,
        },
    ):
        super().__init__()

        self.encoder_name = encoder_name
        self.decoder_ckpt = from_decoder_ckpt
        self.decoder_key_prefix = decoder_key_prefix
        self.init_encoder = init_encoder
        self.num_decoder_layers = num_decoder_layers
        self.mode = mode
        self.generate_cfg = generate_cfg

        self.tokenizer = SimpleTokenizer()

        state_dict = {}
        if from_decoder_ckpt is not None:
            if Path(from_decoder_ckpt).is_file():
                logger.info(f"Loading decoder from {from_decoder_ckpt}...")
            else:
                raise FileNotFoundError(
                    f"Decoder checkpoint {from_decoder_ckpt} not found"
                )
            ckpt = torch.load(from_decoder_ckpt, map_location="cpu")
            self.decoder_key_prefix = (
                decoder_key_prefix if decoder_key_prefix else ""
            )
            for k, v in ckpt["state_dict"].items():
                state_dict[k[len(decoder_key_prefix) :]] = v

            # load hyper parameters from ckpt
            params = ckpt["hyper_parameters"]["model"]
            self.encoder_name = params["encoder_name"]
            self.num_decoder_layers = params["num_decoder_layers"]
            self.mode = params["mode"]
            if self.generate_cfg is None:
                self.generate_cfg = params["generate_cfg"]

        if init_encoder:
            self.text_encoder = TextCLIP(name=self.encoder_name)
        embedding_dim = _INPUT_OUTPUT_DIM[self.encoder_name]["output"]

        if mode == "fusion":
            self.text_decoder = CLIPFusionDecoder(
                embedding_dim=embedding_dim,
                hidden_dim=embedding_dim,
                num_layers=self.num_decoder_layers,
                generate_cfg=self.generate_cfg,
            )
            if state_dict:
                self.text_decoder.load_state_dict(state_dict)
        else:
            raise ValueError(f"Mode {mode} is not supported")

    def encode(self, label_ids: torch.Tensor):
        self.eval()
        device = next(self.text_encoder.parameters()).device
        label_ids = label_ids.to(device)
        embedding = self.text_encoder(label_ids)
        return embedding

    def decode(self, embedding: torch.Tensor):
        self.eval()
        return self.text_decoder(embedding=embedding)

    def encode_raw(self, text: Union[str, List[str]]):
        if isinstance(text, str):
            text = [text]
        label_ids = self.tokenizer.tokenize(text)
        return self.encode(label_ids)

    def decode_raw(self, embedding: torch.Tensor):
        device = next(self.text_decoder.parameters()).device
        embedding = embedding.to(device)
        de_tokens = self.decode(embedding=embedding)["sequence"].cpu().tolist()
        de_text = [self.tokenizer.smart_decode(t) for t in de_tokens]
        if len(de_text) == 1:
            de_text = de_text[0]
        return de_text

    def forward(
        self,
        embedding: torch.Tensor,
        label_ids: torch.Tensor = None,
        label_mask: torch.Tensor = None,
    ):
        if label_ids is None:
            self.eval()
        else:
            self.train()
        return self.text_decoder(
            embedding=embedding, label_ids=label_ids, label_mask=label_mask
        )
