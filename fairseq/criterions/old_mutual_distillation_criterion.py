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

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
#
#
# @dataclass
# class MutualDistillationConfig(
#     LabelSmoothedCrossEntropyCriterionConfig
# ):
#     distillation_weight: float = field(
#         default=1, metadata={"help": "weight for the mutual-distillation loss"}
#     )
#     distillation_start_epoch: int = field(
#         default=5, metadata={"help": "epoch of starting distillation"}
#     )
#     Tdropout_num: int = field(
#         default=1, metadata={"help": "If the Tdropout_num>0, we forward teacher models with dropout for "
#                                      "Tdropout_num times. Otherwise, we forward teacher model without dropout"}
#     )
#     # distillation_threshold: float = field(
#     #     default=10.0, metadata={"help": "threshold for perform distillation. "
#     #                                     "A large value means perform distillation without the consideration of "
#     #                                     "teacher performance."}
#     # )
#     teacher_update_mechanism: str = field(
#         default="asymmetric-updating", metadata={"help": "choice: ('frozen', 'symmetric-updating', 'asymmetric-updating')."}
#     )
#     distillation_loss_metric: str = field(
#         default="KL", metadata={"help": "choice: ('CE', 'KL')."}
#     )
#     loss_scale_factor: float = field(
#         default=5, metadata={"help": "the factor for up-scaling the loss weight divergence."}
#     )
#     loss_rescale_strategy: str = field(
#         default="val_loss", metadata={"help": "choice: ('val_loss', 'uncertainty')"}
#     )
#
#
# @register_criterion(
#     "mutual_distillation_criterion",
#     dataclass=MutualDistillationConfig,
# )
# class MutualDistillationCriterion(
#     LabelSmoothedCrossEntropyCriterion
# ):
#     def __init__(self, task, sentence_avg, label_smoothing,
#                  distillation_weight,
#                  distillation_start_epoch,
#                  Tdropout_num,
#                  # distillation_threshold,
#                  teacher_update_mechanism,
#                  distillation_loss_metric,
#                  loss_scale_factor,
#                  loss_rescale_strategy,
#                  ):
#         super().__init__(task, sentence_avg, label_smoothing)
#         self.distillation_weight = distillation_weight
#         self.distillation_start_epoch = distillation_start_epoch
#         self.Tdropout_num = Tdropout_num
#         self.loss_scale_factor = loss_scale_factor
#         self.loss_rescale_strategy = loss_rescale_strategy
#         # self.distillation_threshold = distillation_threshold
#
#         assert teacher_update_mechanism in ("frozen", "symmetric-updating", "asymmetric-updating"), \
#             "invalid teacher-update mechanism"
#         self.teacher_update_mechanism = teacher_update_mechanism
#
#         assert distillation_loss_metric in ('CE', 'KL'), \
#             "invalid distillation loss metric"
#         self.distillation_loss_metric = distillation_loss_metric
#
#         self.LS_valid_loss1 = {}
#         self.LS_valid_loss2 = {}
#
#     def forward(self, student, sample, teacher=None, reduce=True, epoch=-1, model2=False):
#         """
#         model2: indicating whether the student model is model2
#         """
#         # 1. Computing Cross-Entropy Loss:
#         # student_output = student(**sample["net_input"])
#         student_logits = self.repeat_forward_model(student, sample)
#
#         # lprobs, target = self.get_lprobs_and_target(student, student_output, sample)
#         student_lprobs = utils.log_softmax(student_logits, dim=-1)  # shape: (Tdrop_num, B, L, d)
#         target = sample["target"].view(-1)
#
#         # Note!!! We calculate cross-entropy loss only on the first output with dropout.
#         loss, nll_loss = label_smoothed_nll_loss(
#             student_lprobs[0].view(-1, student_lprobs.size(-1)),
#             target,
#             self.eps,
#             ignore_index=self.padding_idx,
#             reduce=reduce,
#         )
#
#         # 2. Logging:
#         sample_size = (
#             sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
#         )
#         logging_output = {
#             "loss": loss.data,
#             "nll_loss": nll_loss.data,
#             "ntokens": sample["ntokens"],
#             "nsentences": sample["target"].size(0),
#             "sample_size": sample_size,
#         }
#
#         # 3. Computing Distillation Loss:
#         if epoch >= self.distillation_start_epoch \
#                 and sample != "DUMMY" \
#                 and student.training:
#                 # and (epoch == 1 or self.get_switch_status(sample["language-pair"], model2)):
#
#             teacher_logits = self.get_teacher_output(teacher, sample)
#
#             distillation_loss = self.compute_distillation_loss(student, student_logits, student_lprobs,
#                                                                teacher, teacher_logits,
#                                                                sample,
#                                                                reduce,
#                                                                student_is_model2=model2)
#
#             loss = (loss + self.distillation_weight * distillation_loss) / 2
#
#         return loss, sample_size, logging_output
#
#     def get_teacher_output(self, teacher, sample):
#         """
#         return output_logits (Tdrop_num, B, L, d)
#         note that we don't return extra output
#         """
#         if self.teacher_update_mechanism == "frozen":
#             with torch.no_grad():
#                 return self.repeat_forward_model(teacher, sample)
#         else:
#             return self.repeat_forward_model(teacher, sample)
#
#     def repeat_forward_model(self, model, sample):
#         """
#         return output_logits (Tdrop_num, B, L, d)
#         """
#         if self.Tdropout_num > 0:  # forward model multiple times with dropout
#             sample_input = sample['net_input']
#             sample_concat_input = {
#                 'src_tokens': torch.cat([sample_input['src_tokens']] +
#                                          [sample_input['src_tokens'].clone() for i in range(self.Tdropout_num - 1)], 0),
#                 'src_lengths': torch.cat([sample_input['src_lengths']] +
#                                         [sample_input['src_lengths'].clone() for i in range(self.Tdropout_num - 1)], 0),
#                 'prev_output_tokens': torch.cat(
#                     [sample_input['prev_output_tokens']] +
#                     [sample_input['prev_output_tokens'].clone() for i in range(self.Tdropout_num - 1)], 0),
#             }
#             output = model(**sample_concat_input)[0]
#             # output = torch.split(output, output.size(0) // self.Tdropout_num, dim=0)
#             new_shape = [self.Tdropout_num, output.size(0) // self.Tdropout_num] + list(output.shape[1:])
#             output = output.view(new_shape)
#             return output
#         else:
#             model.eval()
#             output = model(**sample["net_input"])[0].unsqueeze(dim=0)
#             model.train()
#
#         return output
#
#     def compute_distillation_loss(self, student, student_logits, student_lprobs,
#                                   teacher, teacher_logits, sample, reduce=True,
#                                   student_is_model2=False):
#         """
#         all the shapes of student_logits, student_lprobs, teacher_logits are:
#                 (Tdrop_num, B, L, d)
#         """
#         # 1. get lprobs of student_output and prob of teacher_output and target
#         target = sample["target"].view(-1)
#         teacher_lprobs = utils.log_softmax(teacher_logits, dim=-1)
#         teacher_probs = utils.softmax(teacher_logits, dim=-1)
#         student_probs = utils.softmax(student_logits, dim=-1)
#
#         # 2. averaging probs and lprobs over Tdrop_num outputs.
#         avg_teacher_lprobs = teacher_lprobs.mean(dim=0).view(-1, teacher_lprobs.shape[-1])
#         avg_teacher_probs = teacher_probs.mean(dim=0).view(-1, teacher_probs.shape[-1])
#         avg_student_lprobs = student_lprobs.mean(dim=0).view(-1, student_lprobs.shape[-1])
#         avg_student_probs = student_probs.mean(dim=0).view(-1, student_probs.shape[-1])
#         # if self.Tdropout_num > 0:
#         #     teacher_probs = teacher_probs.view(self.Tdropout_num, teacher_probs.size(0) // self.Tdropout_num, -1)
#         #     teacher_probs = teacher_probs.mean(dim=0)
#         #     teacher_lprobs = teacher_lprobs.view(self.Tdropout_num, teacher_lprobs.size(0) // self.Tdropout_num, -1)
#         #     teacher_lprobs = teacher_lprobs.mean(dim=0)
#
#         if self.teacher_update_mechanism in ("frozen", "symmetric-updating"):
#             loss = self.calculate_distribution_divergence(avg_student_lprobs, avg_teacher_lprobs,
#                                                           avg_student_probs, avg_teacher_probs)
#         elif self.teacher_update_mechanism == "asymmetric-updating":
#             loss_student = self.calculate_distribution_divergence(avg_student_lprobs, avg_teacher_lprobs.detach(),
#                                                                   avg_student_probs, avg_teacher_probs.detach())
#             loss_teacher = self.calculate_distribution_divergence(avg_student_lprobs.detach(), avg_teacher_lprobs,
#                                                                   avg_student_probs.detach(), avg_teacher_probs)
#             student_loss_weight, teacher_loss_weight = \
#                 self.get_student_teacher_loss_weight(student_lprobs, teacher_lprobs,
#                                                      sample, student_is_model2=student_is_model2)
#
#             loss = loss_student * student_loss_weight + loss_teacher * teacher_loss_weight
#         else:
#             print("BUG is coming!!!!SB!!!")
#             exit(-1)
#
#         loss = loss.sum(dim=-1)
#
#         if self.padding_idx is not None:
#             pad_mask = target.eq(self.padding_idx)
#             loss.masked_fill_(pad_mask, 0.0)
#         else:
#             loss = loss.squeeze(-1)
#
#         if reduce:
#             loss = loss.sum()
#
#         return loss
#
#     def calculate_distribution_divergence(self, student_lprobs, teacher_lprobs, student_probs, teacher_probs):
#         if self.distillation_loss_metric == "CE":  # 'cross-entropy', 'KL-divergence'
#             loss_1 = -teacher_probs * student_lprobs
#             loss_2 = -student_probs * teacher_lprobs
#             loss = 0.5 * loss_1 + 0.5 * loss_2
#         elif self.distillation_loss_metric == "KL":
#             loss_1 = torch.nn.functional.kl_div(student_lprobs, teacher_probs, reduction='none')
#             loss_2 = torch.nn.functional.kl_div(teacher_lprobs, student_probs, reduction='none')
#             loss = 0.5 * loss_1 + 0.5 * loss_2
#         else:
#             raise Exception("Unknown distillation loss metric")
#
#         return loss
#
#     def get_student_teacher_loss_weight(self, student_lprobs, teacher_lprobs, sample, student_is_model2=False):
#         lang_pair = sample["language-pair"]
#
#         if self.loss_rescale_strategy == "val_loss":
#             if lang_pair not in self.LS_valid_loss1:
#                 return 0.5, 0.5
#             student_LS_valid_loss = self.LS_valid_loss2[lang_pair] if student_is_model2 else self.LS_valid_loss1[lang_pair]
#             teacher_LS_valid_loss = self.LS_valid_loss1[lang_pair] if student_is_model2 else self.LS_valid_loss2[lang_pair]
#             if student_LS_valid_loss - teacher_LS_valid_loss > 0.5:
#                 return 1, 0
#             elif teacher_LS_valid_loss - student_LS_valid_loss > 0.5:
#                 return 0, 1
#             else:
#                 with torch.no_grad():
#                     return (torch.tensor(
#                         [student_LS_valid_loss, teacher_LS_valid_loss]
#                     ) * self.loss_scale_factor).softmax(dim=-1)
#         elif self.loss_rescale_strategy == "uncertainty":
#             assert self.Tdropout_num > 1, \
#                 "make sure the value of Tdrop_num > 1 if using uncertainty-based loss rescale strategy."
#             target = sample["target"].unsqueeze(dim=0).unsqueeze(dim=-1).repeat(self.Tdropout_num, 1, 1, 1)
#             # 1. get student/teacher log likelihood probability: (Tdrop_num, B, L, 1)
#             student_llp = student_lprobs.gather(dim=-1, index=target)
#             teacher_llp = teacher_lprobs.gather(dim=-1, index=target)
#             # 2. get expectation of log likelihood probability: (B, L, 1)
#             # student_llp_exp = student_llp.mean(dim=0)
#             # teacher_llp_exp = teacher_llp.mean(dim=0)
#             # 3. get variance of log likelihood probability: (B, L, 1)
#             student_llp_var = student_llp.var(dim=0)
#             teacher_llp_var = teacher_llp.var(dim=0)
#             # 4. calculating uncertainty:
#             with torch.no_grad():
#                 # return a mutual learning ratio with shape (2, B, L)
#                 student_loss_scale, teacher_loss_scale = \
#                     (torch.stack([student_llp_var, teacher_llp_var]) * self.loss_scale_factor).softmax(dim=0)
#                 return student_loss_scale.view(-1, 1), teacher_loss_scale.view(-1, 1)
#         else:
#             raise Exception("Unknown loss rescale strategy.")
#
#     def get_probs(self, model, net_output):
#         probs = model.get_normalized_probs(net_output, log_probs=False)
#         if self.ignore_prefix_size > 0:
#             if getattr(probs, "batch_first", False):
#                 probs = probs[:, self.ignore_prefix_size :, :].contiguous()
#             else:
#                 probs = probs[self.ignore_prefix_size :, :, :].contiguous()
#         return probs.view(-1, probs.size(-1))
#
#     def update_LS_valid_loss(self, LS_valid_loss, is_model2):
#         if is_model2:
#             self.LS_valid_loss2 = LS_valid_loss
#         else:
#             self.LS_valid_loss1 = LS_valid_loss

    # def get_switch_status(self, lang_pair, student_is_model2):
    #     student_LS_valid_loss = self.LS_valid_loss2[lang_pair] if student_is_model2 else self.LS_valid_loss1[lang_pair]
    #     teacher_LS_valid_loss = self.LS_valid_loss1[lang_pair] if student_is_model2 else self.LS_valid_loss2[lang_pair]
    #     return (student_LS_valid_loss - teacher_LS_valid_loss) + self.distillation_threshold > 0
