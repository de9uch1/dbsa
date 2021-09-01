# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

import torch
from fairseq import utils
from fairseq.data import data_utils
from fairseq.sequence_generator import EnsembleModel, SequenceGenerator


class SequenceGeneratorWithAttention(SequenceGenerator):
    def __init__(
        self,
        models,
        tgt_dict,
        left_pad_target=False,
        print_alignment="hard",
        print_dependency="hard",
        **kwargs,
    ):
        """Generates translations of a given source sentence.

        Args:
            left_pad_target (bool, optional): Whether or not the
                hypothesis should be left padded or not when they are
                teacher forced for generating alignments.
        """
        super().__init__(EnsembleModelWithAttention(models), tgt_dict, **kwargs)
        self.left_pad_target = left_pad_target

        if print_alignment in {"hard", "hard_with_eos"}:
            self.extract_alignment = utils.extract_hard_alignment
        elif print_alignment == "soft":
            self.extract_alignment = utils.extract_soft_alignment

        if print_dependency in {"hard", "hard_with_eos"}:
            self.extract_dependency = utils.extract_hard_alignment
        elif print_dependency == "soft":
            self.extract_dependency = utils.extract_soft_alignment

        self.alignment_type = print_alignment
        self.dependency_type = print_dependency

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        finalized = super()._generate(sample, **kwargs)

        src_tokens = sample["net_input"]["src_tokens"]
        bsz = src_tokens.shape[0]
        beam_size = self.beam_size
        (
            src_tokens,
            src_lengths,
            prev_output_tokens,
            tgt_tokens,
        ) = self._prepare_batch_for_alignment(sample, finalized)
        attns = self.model.forward_attn(src_tokens, src_lengths, prev_output_tokens)

        if src_tokens.device != "cpu":
            src_tokens = src_tokens.to("cpu")
            tgt_tokens = tgt_tokens.to("cpu")
            attns_cpu = {}
            for k, attn in attns.items():
                attns_cpu[k] = [i.float().to("cpu") for i in attn]
            attns = attns_cpu

        # Process the attn matrix to extract hard alignments.
        def extract_alignments(extract, k, attn_q, attn_k, finalized_k, with_eos=False):
            eos = self.pad if with_eos else self.eos
            for i in range(bsz * beam_size):
                alignment = extract(attns[k][i], attn_k[i], attn_q[i], self.pad, eos)
                finalized[i // beam_size][i % beam_size][finalized_k] = alignment

        for k in attns:
            if k == "attn":
                extract_alignments(
                    self.extract_alignment,
                    k,
                    tgt_tokens,
                    src_tokens,
                    "alignment",
                    with_eos=(self.alignment_type == "hard_with_eos"),
                )
            elif k == "encoder_self_attn":
                extract_alignments(
                    self.extract_dependency,
                    k,
                    src_tokens,
                    src_tokens,
                    "source_dependency",
                    with_eos=(self.dependency_type == "hard_with_eos"),
                )
            elif k == "decoder_self_attn":
                extract_alignments(
                    self.extract_dependency,
                    k,
                    tgt_tokens,
                    tgt_tokens,
                    "target_dependency",
                    with_eos=(self.dependency_type == "hard_with_eos"),
                )

        return finalized

    def _prepare_batch_for_alignment(self, sample, hypothesis):
        src_tokens = sample["net_input"]["src_tokens"]
        bsz = src_tokens.shape[0]
        src_tokens = (
            src_tokens[:, None, :]
            .expand(-1, self.beam_size, -1)
            .contiguous()
            .view(bsz * self.beam_size, -1)
        )
        src_lengths = sample["net_input"]["src_lengths"]
        src_lengths = (
            src_lengths[:, None]
            .expand(-1, self.beam_size)
            .contiguous()
            .view(bsz * self.beam_size)
        )
        prev_output_tokens = data_utils.collate_tokens(
            [beam["tokens"] for example in hypothesis for beam in example],
            self.pad,
            self.eos,
            self.left_pad_target,
            move_eos_to_beginning=True,
        )
        tgt_tokens = data_utils.collate_tokens(
            [beam["tokens"] for example in hypothesis for beam in example],
            self.pad,
            self.eos,
            self.left_pad_target,
            move_eos_to_beginning=False,
        )
        return src_tokens, src_lengths, prev_output_tokens, tgt_tokens


class EnsembleModelWithAttention(EnsembleModel):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__(models)

    def forward_attn(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        avg_attns = defaultdict(lambda: None)
        for model in self.models:
            decoder_out = model(src_tokens, src_lengths, prev_output_tokens, **kwargs)
            for k, v in decoder_out[1].items():
                if "attn" in k and v is not None and len(v) > 0:
                    attn = v[0]
                    if avg_attns[k] is None:
                        avg_attns[k] = attn
                    else:
                        avg_attns[k].add_(attn)
        if len(self.models) > 1:
            for k, v in avg_attns.items():
                avg_attns[k].div_(len(self.models))
        return avg_attns
