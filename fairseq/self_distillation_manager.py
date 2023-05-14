"""
This file contains implementation of distillation methods for MNMT
"""
import json
import logging
import time
import pdb

import torch
import collections
from fairseq.file_io import PathManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SelfDistillationManager(object):
    """
    Parent class of LSSD_Manager and STSD_Manager.
    """

    def __init__(self, cfg, trainer, task, epoch_iter):
        self.cfg = cfg
        self.trainer = trainer
        self.task = task
        self.current_epoch = epoch_iter.epoch

        self.lang_pairs = cfg.model.lang_pairs.split(',')

        self.model_path = cfg.checkpoint.save_dir
        self.fixed_teacher = False
        if cfg.task.fixed_teacher_model_path is not None:
            self.fixed_teacher = True
            self.fixed_teacher_model_path = cfg.task.fixed_teacher_model_path

        # Hyper-parameter
        # only when the current epoch is `inspection_epoch_NUM` larger than the LS best epoch, distillation is allowed
        self.inspection_epoch_NUM = 1
        self.self_distillation_warmup_epoch = cfg.task.self_distillation_warmup_epoch

    def _averaging_multi_epoch_models(self, cluster_epochs):
        """
        Fusing (averaging) a cluster of teacher models。

        The difference with average_checkpoints.py is that:
            we avoid reloading the same checkpoints.
        """
        if cluster_epochs is None or len(cluster_epochs) == 0:
            return None

        start_time = time.time()

        # 1. loading and averaging model parameters
        params_dict = collections.OrderedDict()
        params_keys = None
        new_state = None

        for teacher_epoch, count in collections.Counter(cluster_epochs).items():
            fpath = f"{self.model_path}/checkpoint{teacher_epoch}.pt"
            model_weight = count / len(cluster_epochs)
            with PathManager.open(fpath, "rb") as f:
                state = torch.load(
                    f,
                    map_location=(
                        lambda s, _: torch.serialization.default_restore_location(s, "cpu")
                    ),
                )
            # Copies over the settings from the first checkpoint
            if new_state is None:
                new_state = state

            model_params = state["model"]

            model_params_keys = list(model_params.keys())
            if params_keys is None:
                params_keys = model_params_keys
            elif params_keys != model_params_keys:
                raise KeyError(
                    "For checkpoint {}, expected list of params: {}, "
                    "but found: {}".format(f, params_keys, model_params_keys)
                )

            for k in params_keys:
                p = model_params[k]
                if isinstance(p, torch.HalfTensor):
                    p = p.float()
                p = p * model_weight
                if k not in params_dict:
                    params_dict[k] = p.clone()
                    # NOTE: clone() is needed in case of p is a shared parameter
                else:
                    params_dict[k] += p
                p.requires_grad = False

        # 2. constructing the final model
        fused_model = self.task.build_model(self.cfg.model)
        fused_model.load_state_dict(params_dict, strict=True, model_cfg=self.cfg.model)
        del params_dict
        # Oh shit!!! bug
        # fused_model.half()
        # fused_model.to(device=self.trainer.device)
        # fused_model.eval()
        fused_model = fused_model.half()
        fused_model = fused_model.to(device=self.trainer.device)
        fused_model = fused_model.eval()

        end_time = time.time()
        if self.trainer.is_data_parallel_master:
            logger.info(f"model fusion operation takes {round(end_time - start_time, 3)} seconds")
        return fused_model

    def _load_model(self, checkpoint_path):
        """
        construct a model from a checkpoint
        """
        start_time = time.time()

        # 1. loading model parameters
        params_dict = collections.OrderedDict()
        params_keys = None
        new_state = None

        fpath = checkpoint_path
        with PathManager.open(fpath, "rb") as f:
            state = torch.load(
                f,
                map_location=(
                    lambda s, _: torch.serialization.default_restore_location(s, "cpu")
                ),
            )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state["model"]
        loss = state['extra_state']['val_loss']

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p
            p.requires_grad = False

        # 2. constructing the final model
        model = self.task.build_model(self.cfg.model)
        model.load_state_dict(params_dict, strict=True, model_cfg=self.cfg.model)
        del params_dict
        model.half()
        model.to(device=self.trainer.device)
        model.eval()

        end_time = time.time()
        if self.trainer.is_data_parallel_master:
            logger.info(f"model loading operation takes {round(end_time - start_time, 3)} seconds")
        return model, loss

    def update_teacher_models(self, LS_valid_loss, valid_loss, current_epoch):
        self.current_epoch = current_epoch

        logger.info(f"current epoch: {self.current_epoch}")
        # 1. update LS_best_models
        for lang_pair in self.lang_pairs:
            if self.LS_best_models[lang_pair]["valid_loss"] > LS_valid_loss[lang_pair]:
                self.LS_best_models[lang_pair]["valid_loss"] = LS_valid_loss[lang_pair]
                self.LS_best_models[lang_pair]["epoch"] = self.current_epoch

    def read_LS_best_models(self):
        # Read loss curve (history) of each language.
        LS_valid_loss_history = None
        if self.current_epoch > 1:
            f = open(self.model_path + '/LS_valid_loss_history.json')
            LS_valid_loss_history = json.load(f)
            f.close()

        LS_best_models = {}

        if LS_valid_loss_history is not None:
            for lang_pair in self.lang_pairs:
                LS_best_loss = min(LS_valid_loss_history[lang_pair])
                LS_best_epoch = 1 + LS_valid_loss_history[lang_pair].index(LS_best_loss)

                LS_best_models[lang_pair] = {
                    "epoch": LS_best_epoch,
                    "valid_loss": LS_best_loss
                }
        else:
            for lang_pair in self.lang_pairs:
                LS_best_models[lang_pair] = {
                    "epoch": -1,
                    "valid_loss": 10000
                }

        return LS_best_models


class LSSD_Manager(SelfDistillationManager):
    """
    Implementation of Language-Specific Self-Distillation (LSSD) for Multilingual Neural Machine Translation.
    Author: Yichong Huang
    """

    def __init__(self, cfg, trainer, task, epoch_iter):

        super().__init__(cfg, trainer, task, epoch_iter)

        # Hyper-parameter
        self.MAX_Teacher_NUM = cfg.task.MAX_Teacher_NUM  # The max teacher models number
        if self.MAX_Teacher_NUM == -1:
            self.MAX_Teacher_NUM = len(self.lang_pairs)

        # Key data structs
        """
        e.g.,
        {
            'bos-eng': 
                {
                    'epoch': 12, 
                    'valid_loss': 5.9
                }
        }
        """
        self.LS_best_models = None

        """
        e.g.,
        {
            'bos-eng': model
        }
        """
        self.LS_teacher_models = None

        """
        e.g., 
        [[14, 17, 24], [49, 51, 79], [198, 210]]
        """
        self.teacher_clusters = None

        """
        [model_1, model_2, model_3]
        """
        self.teacher_model_pool = None

        # initialize the above key data structs
        self._init_teacher_models()

    def _init_teacher_models(self):
        """
        Initialize language-specific teacher models.
        """
        # 1. init LS_best_models
        LS_best_models = self.read_LS_best_models()

        # 2. clustering teacher models
        all_LS_best_epochs = [LS_best_models[lang_pair]["epoch"] for lang_pair in self.lang_pairs]
        teacher_clusters = self._cluster_teacher_models(all_LS_best_epochs)
        if self.trainer.is_data_parallel_master:
            print("Teacher model clustering")
            print(f"LS_best_epochs:   | {all_LS_best_epochs}")
            print(f"teacher_cluster:  | {teacher_clusters}")

        # 3. creating cluster centers
        teacher_model_pool = []
        for cluster in teacher_clusters:
            teacher_model_pool.append(self._fuses_cluster_models(cluster))

        # 4. mapping LS-teacher to cluster center
        logger.debug(self.lang_pairs)
        LS_teacher_models = {}.fromkeys(self.lang_pairs)
        logger.debug(LS_teacher_models)
        for lang_pair in self.lang_pairs:
            LS_teacher_models[lang_pair] = None
            for cluster_id, cluster in enumerate(teacher_clusters):
                if LS_best_models[lang_pair]['epoch'] in cluster:  # Find the corresponding cluster
                    LS_teacher_models[lang_pair] = teacher_model_pool[cluster_id]
                    break

        self.LS_best_models = LS_best_models
        self.teacher_clusters = teacher_clusters
        self.teacher_model_pool = teacher_model_pool
        self.LS_teacher_models = LS_teacher_models

    def update_teacher_models(self, LS_valid_loss, valid_loss, current_epoch):
        super().update_teacher_models(LS_valid_loss, valid_loss, current_epoch)
        self._update_teacher_models(LS_valid_loss)

    def _update_teacher_models(self, LS_valid_loss):
        if self.fixed_teacher:
            return

        # 2. update teacher_clusters, e.g., re-clustering LS teacher epochs
        all_LS_best_epochs = [self.LS_best_models[lang_pair]["epoch"] for lang_pair in self.lang_pairs]
        new_teacher_clusters = self._cluster_teacher_models(all_LS_best_epochs)
        if self.trainer.is_data_parallel_master:
            print("Teacher model clustering")
            print(f"LS_best_epochs:       | {all_LS_best_epochs}")
            print(f"new_teacher_cluster:  | {new_teacher_clusters}")

        # 3. update teacher_model_pool. e.g.,  re-fusing cluster models if needed
        new_teacher_model_pool = []
        for new_cluster in new_teacher_clusters:
            if new_cluster in self.teacher_clusters:
                new_teacher_model_pool.append(self.teacher_model_pool[self.teacher_clusters.index(new_cluster)])
            else:
                new_teacher_model_pool.append(self._fuses_cluster_models(new_cluster))

        # 4. update LS_teacher_models
        LS_teacher_models = {}
        for lang_pair in self.lang_pairs:
            LS_teacher_models[lang_pair] = None
            for cluster_id, cluster in enumerate(new_teacher_clusters):
                if self.LS_best_models[lang_pair]['epoch'] in cluster:  # Find the corresponding cluster
                    LS_teacher_models[lang_pair] = new_teacher_model_pool[cluster_id]
                    break

        self.teacher_clusters = new_teacher_clusters
        self.teacher_model_pool = new_teacher_model_pool
        self.LS_teacher_models = LS_teacher_models

    def _cluster_teacher_models(self, LS_teacher_epochs):
        """
            The implementation of teacher epoch clustering
        """
        # filter valid teacher epochs
        LS_teacher_epochs = [i for i in LS_teacher_epochs if 0 < i]
        if not self.fixed_teacher and len(LS_teacher_epochs) > 0:
            assert max(LS_teacher_epochs) <= self.current_epoch, \
                "Language-Specific best epoch should not be greater than the current epoch."
            LS_teacher_epochs = [i for i in LS_teacher_epochs if i < self.current_epoch]
        LS_teacher_epochs = sorted(LS_teacher_epochs)

        if len(set(LS_teacher_epochs)) <= self.MAX_Teacher_NUM:
            # each unique LS teacher epoch is a cluster
            return [[i] for i in sorted(set(LS_teacher_epochs))]

        intervals = [LS_teacher_epochs[i+1] - LS_teacher_epochs[i]
                     for i in range(len(LS_teacher_epochs) - 1)]
        large_intervals = sorted(intervals, reverse=True)[:self.MAX_Teacher_NUM - 1]

        selected_intervals = [False] * len(intervals)
        for interval in large_intervals:
            for i, v in enumerate(intervals):
                if v == interval and not selected_intervals[i]:
                    selected_intervals[i] = True
                    break
        assert sum(selected_intervals) == self.MAX_Teacher_NUM - 1, "A Strange Problem"

        selected_interval_indices = [0] + [i+1 for i, v in enumerate(selected_intervals) if v] + [None]

        teacher_clusters = []
        for i in range(len(selected_interval_indices) - 1):
            begin_index = selected_interval_indices[i]
            end_index = selected_interval_indices[i + 1]
            teacher_clusters.append(LS_teacher_epochs[begin_index: end_index])
        return teacher_clusters

    def _fuses_cluster_models(self, cluster_epochs):
        return self._averaging_multi_epoch_models(cluster_epochs)

    def get_teacher_model(self, lang_pair=None):
        assert lang_pair is not None, "the language-pair argument shouldn't be None"
        return self.LS_teacher_models[lang_pair]

    def get_distillation_switch_status(self, lang_pair=None):
        """
            Determining whether to perform distillation for in the current epoch.
        """
        assert lang_pair is not None, "the language-pair argument shouldn't be None"
        cur_epoch = self.current_epoch
        LS_best_epoch = self.LS_best_models[lang_pair]["epoch"]
        if cur_epoch > self.self_distillation_warmup_epoch \
                and cur_epoch - LS_best_epoch > self.inspection_epoch_NUM:
            return True
        else:
            return False


class STSD_Manager(SelfDistillationManager):
    """
    The implementation of Single Teacher Self-Distillation
    """

    def __init__(self, cfg, trainer, task, epoch_iter):

        super().__init__(cfg, trainer, task, epoch_iter)

        self.overall_best_model = None
        self.teacher_model = None
        self._init_teacher_model()

    def _init_teacher_model(self):
        """
        Initialize the teacher model, i.e., the overall best checkpoint
        """
        # Read loss curve (history) of each language.
        LS_valid_loss_history = None
        if self.current_epoch > 1:
            f = open(self.checkpoint_path + '/LS_valid_loss_history.json')
            LS_valid_loss_history = json.load(f)
            f.close()

        # overall loss:
        overall_best_model = {
            "epoch": -1,
            "valid_loss": 10000
        }
        if LS_valid_loss_history is not None:
            overall_best_loss = min(LS_valid_loss_history["all"])
            overall_best_epoch = 1 + LS_valid_loss_history["all"].index(overall_best_loss)
            overall_best_model = {
                "epoch": overall_best_loss,
                "valid_loss": overall_best_epoch
            }
        if self.trainer.is_data_parallel_master:
            print("overall_best_model:")
            print(f"epoch      | {overall_best_model['epoch']}")
            print(f"valid_loss | {overall_best_model['valid_loss']}")

        teacher_model = self._averaging_multi_epoch_models([overall_best_model["epoch"]])

        self.overall_teacher_model = overall_best_model
        self.teacher_model = teacher_model

    def update_teacher_models(self, LS_valid_loss, valid_loss, current_epoch):
        super().update_teacher_models(LS_valid_loss, valid_loss, current_epoch)
        self._update_teacher_model(valid_loss)

    def _update_teacher_model(self, valid_loss):
        if self.fixed_teacher:
            return

        # 1. update overall_best_models
        if valid_loss < self.overall_best_model['valid_loss']:
            overall_best_model = {
                "epoch": self.current_epoch,
                "valid_loss": valid_loss
            }
            # 2. update teacher_model
            new_teacher_model = None
        else:
            if self.teacher_model is None:
                new_teacher_model = self._averaging_multi_epoch_models([self.overall_best_model["epoch"]])

        self.overall_best_model = overall_best_model
        self.teacher_model = new_teacher_model

    def get_teacher_model(self, lang_pair=None):
        return self.teacher_model

    def get_distillation_switch_status(self, lang_pair=None):
        """
            Determining whether to perform distillation for in the current epoch.
        """
        cur_epoch = self.current_epoch
        overall_best_epoch = self.overall_best_model["epoch"]
        if cur_epoch > self.self_distillation_warmup_epoch \
                and cur_epoch - overall_best_epoch > self.inspection_epoch_NUM:
            return True
        else:
            return False


class KD_Manager(SelfDistillationManager):
    """
    Re-implement ICLR2019: "Multilingual Neural Machine Translation with Knowledge Distillation."
    """

    def __init__(self, cfg, trainer, task, epoch_iter):
        super().__init__(cfg, trainer, task, epoch_iter)

        self.LS_last_models = None

        # initialize the above key data structs
        self._init_LS_last_models()

    def _init_LS_last_models(self):
        """
        Initialize language-specific teacher models.
        """

        # 1. init LS_last_models
        LS_valid_loss_history = None
        if self.current_epoch > 1:
            f = open(self.model_path + '/LS_valid_loss_history.json')
            LS_valid_loss_history = json.load(f)
            f.close()

        LS_last_models = {}

        if LS_valid_loss_history is not None:
            for lang_pair in self.lang_pairs:
                LS_last_loss = LS_valid_loss_history[lang_pair][-1]

                LS_last_models[lang_pair] = {
                    "valid_loss": LS_last_loss
                }
        else:
            for lang_pair in self.lang_pairs:
                LS_last_models[lang_pair] = {
                    "epoch": -1,
                    "valid_loss": 10000
                }

        if self.cfg.task.debug_mode:
            pdb.set_trace()

        # 2. read fixed teacher models
        LS_teacher_models = {}
        fixed_teacher_loss = {}
        for lang_pair in self.lang_pairs:
            LS_teacher_models[lang_pair], fixed_teacher_loss[lang_pair] = self._load_model(
                f"{self.fixed_teacher_model_path}/bilingual_{lang_pair}/checkpoint_best.pt"
            )

        self.LS_last_models = LS_last_models
        self.LS_teacher_models = LS_teacher_models
        self.fixed_teacher_loss = fixed_teacher_loss

    def update_teacher_models(self, LS_valid_loss, valid_loss, current_epoch):
        if self.cfg.task.debug_mode:
            pdb.set_trace()
        for lang_pair in self.lang_pairs:
            self.LS_last_models[lang_pair]["valid_loss"] = LS_valid_loss[lang_pair]

    def get_teacher_model(self, lang_pair=None):
        assert lang_pair is not None, "the language-pair argument shouldn't be None"
        return self.LS_teacher_models[lang_pair]

    def get_distillation_switch_status(self, lang_pair=None):
        """
            Determining whether to perform distillation for in the current epoch.
        """
        if self.cfg.task.debug_mode:
            pdb.set_trace()
        assert lang_pair is not None, "the language-pair argument shouldn't be None"
        return self.fixed_teacher_loss[lang_pair] <= self.LS_last_models[lang_pair]["valid_loss"]