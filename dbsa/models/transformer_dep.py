# Copyright (c) Hiroyuki Deguchi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    base_architecture,
    transformer_wmt_en_de_big,
    TransformerModel,
)

from dbsa.modules import DependencyBasedSelfAttention


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
    def build_model(cls, args, task):
        # set any default arguments
        transformer_dep(args)

        transformer_model = TransformerModel.build_model(args, task)
        encoder, decoder = transformer_model.encoder, transformer_model.decoder

        dep_layer = getattr(args, "dependency_layer", 0)
        if args.dependency_layer >= 0:
            encoder.layers[dep_layer].self_attn = DependencyBasedSelfAttention(
                encoder.layers[dep_layer].embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=encoder.layers[dep_layer].quant_noise,
                qn_block_size=encoder.layers[dep_layer].quant_noise_block_size,
                dependency_heads=args.dependency_heads,
            )
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

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_out = self.encoder(src_tokens, src_lengths, return_all_attn=True)
        decoder_out, decoder_attn = self.decoder(prev_output_tokens, encoder_out, return_all_self_attn=True)
        dep_layer, dep_heads = self.dependency_layer, self.dependency_heads
        encoder_self_attn = encoder_out.encoder_attn[dep_layer][:dep_heads].mean(dim=0)
        decoder_self_attn = decoder_attn['self_attn'][dep_layer][:dep_heads].mean(dim=0)
        dependency_out = {
            'encoder_self_attn': encoder_self_attn,
            'decoder_self_attn': decoder_self_attn,
        }
        return decoder_out, dependency_out


@register_model_architecture("transformer_dep", "transformer_dep")
def transformer_dep(args):
    args.dependency_heads = getattr(args, "dependency_heads", 1)
    args.dependency_layer = getattr(args, "dependency_layer", 3)
    base_architecture(args)


@register_model_architecture("transformer_dep", "transformer_wmt_en_de_big_dep")
def transformer_wmt_en_de_big_dep(args):
    args.dependency_heads = getattr(args, "dependency_heads", 1)
    args.dependency_layer = getattr(args, "dependency_layer", 3)
    transformer_wmt_en_de_big(args)
