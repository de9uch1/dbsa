# Copyright (c) Hiroyuki Deguchi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from torch import Tensor

from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    base_architecture,
    transformer_wmt_en_de_big,
    TransformerModel,
    TransformerEncoder,
)

from dbsa.modules import DependencyBasedSelfAttention, TransformerDependencyEncoderLayer


@register_model("transformer_dep")
class TransformerDepModel(TransformerModel):
    """
    See "Dependency-Based Self-Attention for Transformer
    NMT (Deguchi et al., 2019)" for more details.
    """

    def __init__(self, encoder, decoder, args):
        super().__init__(args, encoder, decoder)
        self.dependency_heads = args.dependency_heads
        self.dependency_layer = args.dependency_layer

    @staticmethod
    def add_args(parser):
        # fmt: off
        super(TransformerDepModel, TransformerDepModel).add_args(parser)
        parser.add_argument('--dependency-heads', type=int, metavar='D',
                            help='Number of cross attention heads per layer to supervised with dependency heads')
        parser.add_argument('--dependency-layer', type=int, metavar='D',
                            help='Layer number which has to be supervised. 0 corresponding to the bottommost layer.')
        # fmt: on

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerDependencyEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_model(cls, args, task):
        # set any default arguments
        transformer_dep(args)

        transformer_model = TransformerModel.build_model(args, task)
        encoder, decoder = transformer_model.encoder, transformer_model.decoder
        encoder = cls.build_encoder(args, encoder.dictionary, encoder.embed_tokens)

        dep_layer = getattr(args, "dependency_layer", 0)
        if args.dependency_layer >= 0:
            encoder.layers[dep_layer] = TransformerDependencyEncoderLayer(args)
            decoder.layers[dep_layer].self_attn = DependencyBasedSelfAttention(
                decoder.layers[dep_layer].embed_dim,
                args.decoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=decoder.layers[dep_layer].quant_noise,
                qn_block_size=decoder.layers[dep_layer].quant_noise_block_size,
                dependency_heads=args.dependency_heads,
            )

        return TransformerDepModel(encoder, decoder, args)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(
            src_tokens,
            src_lengths,
            return_all_attn=True,
            src_deps=kwargs.get('src_deps', None),
        )
        decoder_out, decoder_attn = self.decoder(
            prev_output_tokens,
            encoder_out,
            return_all_self_attn=True,
        )
        encoder_self_attn = (
            encoder_out
            ['encoder_attn']
            [self.dependency_layer]
            [:self.dependency_heads]
            .mean(dim=0)
        )
        decoder_self_attn = (
            decoder_attn
            ['self_attn']
            [self.dependency_layer]
            [:self.dependency_heads]
            .mean(dim=0)
        )
        dependency_out = {
            'encoder_self_attn': encoder_self_attn,
            'decoder_self_attn': decoder_self_attn,
        }
        return decoder_out, dependency_out


class TransformerDependencyEncoder(TransformerEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def forward(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens: bool = False,
        return_all_attn: bool = False,
        token_embeddings: Optional[Tensor] = None,
        src_deps: Optional[Tensor] = None,
        dependency_layer: int = 0,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            return_all_attn (bool, optional): also return all of the
                intermediate layers' attention weights (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
                - **encoder_attn** (List[Tensor]): all intermediate
                  layers' attention weights of shape `(num_heads, batch, src_len, src_len)`.
                  Only populated if *return_all_attn* is True.
        """
        x, encoder_embedding = self.forward_embedding(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = []
        encoder_attn = []

        # encoder layers
        for i, layer in enumerate(self.layers):
            if i == dependency_layer and src_deps is not None:
                x, attn = layer(
                    x, encoder_padding_mask, need_head_weights=return_all_attn,
                    src_deps=src_deps,
                )
            else:
                x, attn = layer(x, encoder_padding_mask, need_head_weights=return_all_attn)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)
            if return_all_attn and attn is not None:
                assert encoder_attn is not None
                encoder_attn.append(attn)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "encoder_attn": encoder_attn,  # List[N x B x T x T]
            "src_tokens": [],
            "src_lengths": [],
        }


@register_model_architecture("transformer_dep", "transformer_dep")
def transformer_dep(args):
    args.dependency_heads = getattr(args, "dependency_heads", 1)
    args.dependency_layer = getattr(args, "dependency_layer", 0)
    base_architecture(args)


@register_model_architecture("transformer_dep", "transformer_wmt_en_de_big_dep")
def transformer_wmt_en_de_big_dep(args):
    args.dependency_heads = getattr(args, "dependency_heads", 1)
    args.dependency_layer = getattr(args, "dependency_layer", 0)
    transformer_wmt_en_de_big(args)
