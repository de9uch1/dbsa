# Copyright (c) Hiroyuki Deguchi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy_with_alignment import (
    LabelSmoothedCrossEntropyCriterionWithAlignment,
    LabelSmoothedCrossEntropyCriterionWithAlignmentConfig,
)


@dataclass
class LabelSmoothedCrossEntropyCriterionWithDependencyConfig(
    LabelSmoothedCrossEntropyCriterionWithAlignmentConfig
):
    source_dependency_lambda: float = field(
        default=0.5, metadata={"help": "weight for the source dependency loss"}
    )
    target_dependency_lambda: float = field(
        default=0.5, metadata={"help": "weight for the target dependency loss"}
    )


@register_criterion(
    "label_smoothed_cross_entropy_with_dependency",
    dataclass=LabelSmoothedCrossEntropyCriterionWithDependencyConfig,
)
class LabelSmoothedCrossEntropyCriterionWithDependency(
    LabelSmoothedCrossEntropyCriterionWithAlignment
):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        alignment_lambda,
        source_dependency_lambda,
        target_dependency_lambda,
    ):
        super().__init__(task, sentence_avg, label_smoothing, alignment_lambda)
        self.source_dependency_lambda = source_dependency_lambda
        self.target_dependency_lambda = target_dependency_lambda

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "nll_loss": utils.item(nll_loss.data) if reduce else nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        alignment_loss = None

        # Compute alignment loss only for training set and non dummy batches.
        if "alignments" in sample and sample["alignments"] is not None:
            alignment_loss = self.compute_alignment_loss(sample, net_output)

        if alignment_loss is not None:
            logging_output["alignment_loss"] = utils.item(alignment_loss.data)
            loss += self.alignment_lambda * alignment_loss

        source_dependency_loss = None
        target_dependency_loss = None

        # Compute dependency loss only for training set and non dummy batches.
        if (
            "src_dep" in sample
            and sample["src_dep"] is not None
            and self.source_dependency_lambda > 0.0
        ):
            source_dependency_loss = self.compute_dependency_loss(
                sample, net_output, target=False
            )
            source_dependency_loss *= sample_size / sample["src_dep"].size(0)
        if (
            "tgt_dep" in sample
            and sample["tgt_dep"] is not None
            and self.target_dependency_lambda > 0.0
        ):
            target_dependency_loss = self.compute_dependency_loss(
                sample,
                net_output,
                target=True,
                full_context=getattr(model, "full_context_dependency", False),
            )
            target_dependency_loss *= sample_size / sample["tgt_dep"].size(0)

        if source_dependency_loss is not None:
            logging_output["src_dep_loss"] = utils.item(source_dependency_loss.data)
            loss += self.source_dependency_lambda * source_dependency_loss
        if target_dependency_loss is not None:
            logging_output["tgt_dep_loss"] = utils.item(target_dependency_loss.data)
            loss += self.target_dependency_lambda * target_dependency_loss

        return loss, sample_size, logging_output

    def compute_dependency_loss(
        self, sample, net_output, target=False, full_context=False
    ):
        attn_probs = (
            net_output[1]["decoder_self_attn"][0]
            if target
            else net_output[1]["encoder_self_attn"][0]
        )
        bsz, seq_len, _ = attn_probs.shape
        attn = attn_probs.view(bsz * seq_len, seq_len).float()

        dep = sample["tgt_dep"] if target else sample["src_dep"]
        if target and not full_context:
            dep = dep[dep[:, 0].fmod(seq_len).type_as(dep) >= dep[:, 1], :]

        if len(dep) > 0:
            # Dependency loss computation.
            loss = -((attn[dep[:, 0][:, None], dep[:, 1][:, None]]).log()).sum()
        else:
            return None

        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss_sum = utils.item(
            sum(log.get("nll_loss", 0) for log in logging_outputs)
        )
        alignment_loss_sum = utils.item(
            sum(log.get("alignment_loss", 0) for log in logging_outputs)
        )
        src_dep_loss_sum = utils.item(
            sum(log.get("src_dep_loss", 0) for log in logging_outputs)
        )
        tgt_dep_loss_sum = utils.item(
            sum(log.get("tgt_dep_loss", 0) for log in logging_outputs)
        )
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "alignment_loss",
            alignment_loss_sum / sample_size / math.log(2),
            sample_size,
            round=3,
        )
        metrics.log_scalar(
            "src_dep_loss",
            src_dep_loss_sum / sample_size / math.log(2),
            sample_size,
            round=3,
        )
        metrics.log_scalar(
            "tgt_dep_loss",
            tgt_dep_loss_sum / sample_size / math.log(2),
            sample_size,
            round=3,
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
