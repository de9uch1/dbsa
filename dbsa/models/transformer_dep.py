# Copyright (c) Hiroyuki Deguchi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional

from dbsa.models.transformer_dep_decoder import DependencyBasedTransformerDecoder
from dbsa.models.transformer_dep_encoder import DependencyBasedTransformerEncoder
from fairseq.models import register_model
from fairseq.models.transformer import TransformerConfig, TransformerModelBase
from fairseq.models.transformer.transformer_config import (
    DecoderConfig,
    EncDecBaseConfig,
)
from torch import Tensor


@dataclass
class DependencyBasedEncDecBaseConfig(EncDecBaseConfig):
    dependency_layer: int = field(
        default=0,
        metadata={
            "help": "Layer number which has to be supervised. 0 corresponding to the bottommost layer."
        },
    )


@dataclass
class DependencyBasedDecoderConfig(DependencyBasedEncDecBaseConfig, DecoderConfig):
    pass


@dataclass
class DependencyBasedTransformerConfig(TransformerConfig):
    dependency_heads: int = field(
        default=1,
        metadata={
            "help": "Number of cross attention heads per layer to supervised with dependency heads"
        },
    )
    full_context_dependency: bool = field(
        default=False,
        metadata={
            "help": "Whether or not dependency is supervised conditioned on the full target context."
        },
    )

    encoder: DependencyBasedEncDecBaseConfig = DependencyBasedEncDecBaseConfig()
    decoder: DependencyBasedDecoderConfig = DependencyBasedDecoderConfig()


@register_model("transformer_dep", dataclass=DependencyBasedTransformerConfig)
class DependencyBasedTransformerModel(TransformerModelBase):
    """
    See "Dependency-Based Self-Attention for Transformer
    NMT (Deguchi et al., 2019)" for more details.
    """

    def __init__(self, cfg, encoder, decoder):
        super().__init__(cfg, encoder, decoder)
        self.dependency_heads = cfg.dependency_heads
        self.encoder_dependency_layer = cfg.encoder.dependency_layer
        self.decoder_dependency_layer = cfg.decoder.dependency_layer
        self.full_context_dependency = cfg.full_context_dependency

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        encoder = DependencyBasedTransformerEncoder(cfg, src_dict, embed_tokens)
        if cfg.encoder.dependency_layer >= 0:
            encoder.layers[cfg.encoder.dependency_layer].self_attn.add_biaffine_layer()
        return encoder

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        decoder = DependencyBasedTransformerDecoder(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )
        if cfg.decoder.dependency_layer >= 0:
            decoder.layers[cfg.decoder.dependency_layer].self_attn.add_biaffine_layer()
        return decoder

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(
            src_tokens,
            src_lengths,
            dependency_layer=self.encoder_dependency_layer,
            dependency_heads=self.dependency_heads,
            src_deps=kwargs.get("src_deps", None),
        )
        decoder_out, decoder_attn = self.forward_decoder(
            prev_output_tokens,
            encoder_out,
        )
        attn = {"encoder_self_attn": encoder_out["encoder_attn"]}
        for k, v in decoder_attn.items():
            if k == "self_attn":
                attn["decoder_self_attn"] = v
            else:
                attn[k] = v
        return decoder_out, attn

    def forward_decoder(
        self,
        prev_output_tokens,
        encoder_out=None,
        incremental_state=None,
        features_only=False,
        **extra_args,
    ):
        attn_args = {
            "dependency_layer": self.decoder_dependency_layer,
            "dependency_heads": self.dependency_heads,
        }
        decoder_out = self.decoder(prev_output_tokens, encoder_out, **attn_args)

        if self.full_context_dependency:
            attn_args["full_context_alignment"] = self.full_context_dependency
            _, dependency_out = self.decoder(
                prev_output_tokens,
                encoder_out,
                features_only=True,
                **attn_args,
                **extra_args,
            )
            decoder_out[1]["self_attn"] = dependency_out["self_attn"]

        return decoder_out
