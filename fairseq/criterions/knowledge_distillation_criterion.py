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
    label_smoothed_nll_loss
)

from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class KnowledgeDistillationConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    pass


@register_criterion(
    "knowledge_distillation_criterion",
    dataclass=KnowledgeDistillationConfig,
)
class KnowledgeDistillationCriterion(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, sentence_avg, label_smoothing,
                 ):
        super().__init__(task, sentence_avg, label_smoothing)

    def forward(self, student, sample, teacher=None, alpha=0, reduce=True):
        # 1. Computing Cross-Entropy Loss:
        student_logits = student(**sample["net_input"])[0]
        student_lprobs = utils.log_softmax(student_logits, dim=-1)  # shape: (Tdrop_num, B, L, d)
        target = sample["target"].view(-1, 1)

        loss, nll_loss = label_smoothed_nll_loss(
            student_lprobs.view(-1, student_lprobs.size(-1)),
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )

        # 2. Logging:
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        # 3. Computing Distillation Loss:
        if alpha > 0 \
                and sample != "DUMMY" \
                and student.training:
            with torch.no_grad():
                teacher_logits = teacher(**sample["net_input"])[0]
                teacher_probs = utils.softmax(teacher_logits, dim=-1)

            distillation_loss = self.compute_distillation_loss(student_lprobs,
                                                               teacher_probs,
                                                               sample,
                                                               reduce)

            loss = (1 - alpha) * loss + alpha * distillation_loss

        return loss, sample_size, logging_output

    def compute_distillation_loss(self, student_lprobs,
                                  teacher_probs, sample, reduce=True,
                                  ):
        """
        all the shapes of student_logits, student_lprobs, teacher_logits are:
                (Tdrop_num, B, L, d)
        """
        # 1. get lprobs of student_output and prob of teacher_output and target
        target = sample["target"].view(-1)

        loss = - teacher_probs * student_lprobs

        loss = loss.sum(dim=-1).view(-1)

        if self.padding_idx is not None:
            pad_mask = target.eq(self.padding_idx)
            loss.masked_fill_(pad_mask, 0.0)
        else:
            loss = loss.squeeze(-1)

        if reduce:
            loss = loss.sum()

        return loss

