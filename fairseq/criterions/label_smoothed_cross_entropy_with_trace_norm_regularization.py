# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import logging
import pdb
from fairseq import metrics, utils
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
from fairseq.criterions import register_criterion

from .label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)

from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@dataclass
class LabelSmoothedCrossEntropyCriterionWithTraceNormRegularizationConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    overall_trace_norm_weight: float = field(
        default=0.01,
        metadata={"help": "weight for the Trace-Norm loss"}
    )
    source_inter_trace_norm_weight: float = field(
        default=1,
        metadata={"help": "weight for the Inter-Language Trace-Norm of Source Embeddings Regularization loss"}
    )
    source_intra_trace_norm_weight: float = field(
        default=-1,
        metadata={"help": "weight for the Intra-Language Trace-Norm of Source Embeddings Regularization loss"}
    )
    target_inter_trace_norm_weight: float = field(
        default=1,
        metadata={"help": "weight for the Inter-Language Trace-Norm of Target Embeddings Regularization loss"}
    )
    target_intra_trace_norm_weight: float = field(
        default=-1,
        metadata={"help": "weight for the Intra-Language Trace-Norm of Target Embeddings Regularization loss"}
    )
    trace_norm_reg_interval: float = field(
        default=10,
        metadata={"help": "the interval steps for performing Trace-Norm Regularization."}
    )


def calculate_trace_norm(matrix):
    return torch.linalg.svdvals(matrix.float()).sum()


@register_criterion(
    "label_smoothed_cross_entropy_with_trace_norm_regularization",
    dataclass=LabelSmoothedCrossEntropyCriterionWithTraceNormRegularizationConfig,
)
class LabelSmoothedCrossEntropyCriterionWithTraceNormRegularization(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, sentence_avg, label_smoothing,
                 overall_trace_norm_weight,
                 source_inter_trace_norm_weight,
                 source_intra_trace_norm_weight,
                 target_inter_trace_norm_weight,
                 target_intra_trace_norm_weight,
                 trace_norm_reg_interval
                 ):
        super().__init__(task, sentence_avg, label_smoothing)
        self.w = overall_trace_norm_weight
        self.w1 = source_inter_trace_norm_weight
        self.w2 = source_intra_trace_norm_weight
        self.w3 = target_inter_trace_norm_weight
        self.w4 = target_intra_trace_norm_weight

        self.interval = trace_norm_reg_interval
        self.step_count = 0

        # 1. get the language-specific indices for each language
        if task.args.debug_mode:
            pdb.set_trace()
        self.source_langs = task.source_langs
        if len(task.source_langs) > 1:
            source_language_specific_words_indices = {}
            for lang in task.source_langs:
                # 1. loading language-specific dicts
                LS_dict = TranslationMultiSimpleEpochTask.load_dictionary(f"{task.args.data}/LS_dict.{lang}.txt")
                LS_dict.add_symbol(f"__{lang}__")

                # 2. constructing a mask vector to zero the probs in other languages.
                indices = []
                src_dict = task.dicts[lang]
                for i in range(len(src_dict)):
                    if src_dict[i] in LS_dict:
                        indices.append(i)
                indices = torch.tensor(indices, requires_grad=False)
                source_language_specific_words_indices[lang] = indices
            self.source_language_specific_words_indices = source_language_specific_words_indices
        if task.args.debug_mode:
            pdb.set_trace()

        self.target_langs = task.target_langs
        if len(task.target_langs) > 1:
            target_language_specific_words_indices = {}
            for lang in task.target_langs:
                # 1. loading language-specific dicts
                LS_dict = TranslationMultiSimpleEpochTask.load_dictionary(f"{task.args.data}/LS_dict.{lang}.txt")
                LS_dict.add_symbol(f"__{lang}__")

                # 2. constructing a mask vector to zero the probs in other languages.
                indices = []
                tgt_dict = task.dicts[lang]
                for i in range(len(tgt_dict)):
                    if tgt_dict[i] in LS_dict:
                        indices.append(i)
                indices = torch.tensor(indices, requires_grad=False)
                target_language_specific_words_indices[lang] = indices
            self.target_language_specific_words_indices = target_language_specific_words_indices
        if task.args.debug_mode:
            pdb.set_trace()

    def forward(self, model, sample, reduce=True):
        if self.task.args.debug_mode:
            pdb.set_trace()
        loss, sample_size, logging_output = super().forward(model, sample, reduce=True)

        self.step_count += 1
        if model.training and self.step_count % self.interval == 0:
            self.step_count = 0
            loss += self.calculate_trace_norm_loss(model) * self.w
        return loss, sample_size, logging_output

    def calculate_trace_norm_loss(self, student):
        if self.task.args.debug_mode:
            pdb.set_trace()
        source_embed = student.encoder.embed_tokens.weight
        target_embed = student.decoder.embed_tokens.weight

        # 1. source Inter-Language Trace Norm
        trace_norm_1 = 0
        if len(self.source_langs) > 1:
            trace_norm_1 += calculate_trace_norm(source_embed) * self.w1

        # 2. source Intra-Language Trace-Norm
        trace_norm_2 = 0
        for lang in self.source_langs:
            LS_source_embed = source_embed[self.source_language_specific_words_indices[lang]]
            trace_norm_2 += calculate_trace_norm(LS_source_embed) * self.w2

        # 3. target Inter-Language Trace Norm
        trace_norm_3 = 0
        if len(self.target_langs) > 1:
            trace_norm_3 += calculate_trace_norm(target_embed) * self.w3

        # 4. target Intra-Language Trace Norm
        trace_norm_4 = 0
        for lang in self.target_langs:
            LS_target_embed = target_embed[self.target_language_specific_words_indices[lang]]
            trace_norm_4 += calculate_trace_norm(LS_target_embed) * self.w4

        return trace_norm_1 + trace_norm_2 + trace_norm_3 + trace_norm_4
    # @staticmethod
    # def reduce_metrics(logging_outputs) -> None:
    #     """Aggregate logging outputs from data parallel training."""
    #     loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
    #     nll_loss_sum = utils.item(
    #         sum(log.get("nll_loss", 0) for log in logging_outputs)
    #     )
    #     alignment_loss_sum = utils.item(
    #         sum(log.get("alignment_loss", 0) for log in logging_outputs)
    #     )
    #     ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
    #     sample_size = utils.item(
    #         sum(log.get("sample_size", 0) for log in logging_outputs)
    #     )
    #
    #     metrics.log_scalar(
    #         "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
    #     )
    #     metrics.log_scalar(
    #         "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
    #     )
    #     metrics.log_scalar(
    #         "alignment_loss",
    #         alignment_loss_sum / sample_size / math.log(2),
    #         sample_size,
    #         round=3,
    #     )
    #     metrics.log_derived(
    #         "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
    #     )
    #
    # @staticmethod
    # def logging_outputs_can_be_summed() -> bool:
    #     """
    #     Whether the logging outputs returned by `forward` can be summed
    #     across workers prior to calling `reduce_metrics`. Setting this
    #     to True will improves distributed training speed.
    #     """
    #     return True
