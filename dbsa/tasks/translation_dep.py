# Copyright (c) Hiroyuki Deguchi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from fairseq import metrics, options, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    PrependTokenDataset,
    RawLabelDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
from fairseq.dataclass.constants import PRINT_ALIGNMENT_CHOICES
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask

from dbsa.data import LanguagePairDatasetWithDependency

logger = logging.getLogger(__name__)


def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    load_dependency=False,
    gold_dependency=False,
    dependency_with_input=False,
    truncate_source=False,
    remove_eos_from_source=True,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    prepend_bos_src=None,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    src_dep, tgt_dep = None, None
    if load_dependency:
        src_dep_path = os.path.join(data_path, "{}.dep.{}".format(split, src))
        tgt_dep_path = os.path.join(data_path, "{}.dep.{}".format(split, tgt))
        if os.path.exists(src_dep_path):
            src_deps = []
            with open(src_dep_path, "r") as src_dep_data:
                for h in src_dep_data:
                    src_deps.append(
                        torch.LongTensor(
                            [[i, int(x)] for i, x in enumerate(h.strip().split())]
                        )
                    )
            src_dep = RawLabelDataset(src_deps)
        if os.path.exists(tgt_dep_path):
            tgt_deps = []
            with open(tgt_dep_path, "r") as tgt_dep_data:
                for h in tgt_dep_data:
                    tgt_deps.append(
                        torch.LongTensor(
                            [[i, int(x)] for i, x in enumerate(h.strip().split())]
                        )
                    )
            tgt_dep = RawLabelDataset(tgt_deps)

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDatasetWithDependency(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        remove_eos_from_source=remove_eos_from_source,
        align_dataset=align_dataset,
        eos=eos,
        src_dep=src_dep,
        tgt_dep=tgt_dep,
        dependency_with_input=dependency_with_input,
        gold_dependency=gold_dependency,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


@dataclass
class TranslationDependencyConfig(TranslationConfig):
    remove_eos_from_source: bool = field(
        default=False,
        metadata={"help": "if set, remove eos from end of source if it's present"},
    )
    load_dependency: bool = field(
        default=False, metadata={"help": "load the dependency heads"}
    )
    dependency_with_input: bool = field(
        default=False,
        metadata={"help": "if set, target-side's dependencies are based on the inputs"},
    )
    use_gold_dependency: bool = field(
        default=False,
        metadata={"help": "use the source's gold dependency for inference"},
    )
    print_dependency: Optional[PRINT_ALIGNMENT_CHOICES] = field(
        default=None,
        metadata={
            "help": "if set, uses attention feedback to compute and print alignment to source tokens "
            "(valid options are: hard, soft, otherwise treated as hard alignment)",
            "argparse_const": "hard",
        },
    )


@register_task("translation_dep", dataclass=TranslationDependencyConfig)
class TranslationDependencyTask(TranslationTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: TranslationDependencyConfig

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if self.cfg.use_gold_dependency:
            self.cfg.load_dependency = True

        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            remove_eos_from_source=self.cfg.remove_eos_from_source,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            load_dependency=self.cfg.load_dependency,
            gold_dependency=self.cfg.use_gold_dependency,
            dependency_with_input=self.cfg.dependency_with_input,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDatasetWithDependency(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if self.cfg.print_dependency is not None:
            from dbsa.sequence_generator import SequenceGeneratorWithAttention
            seq_gen_cls = SequenceGeneratorWithAttention
            extra_gen_cls_kwargs["print_dependency"] = self.cfg.print_dependency
            if getattr(args, "print_alignment", False):
                extra_gen_cls_kwargs["print_alignment"] = args.print_alignment
        return super().build_generator(
            models,
            args,
            seq_gen_cls=seq_gen_cls,
            extra_gen_cls_kwargs=extra_gen_cls_kwargs,
        )
