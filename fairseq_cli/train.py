#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import sys
import copy
import pdb
import json
from typing import Dict, Optional, Any, List, Tuple, Callable
import collections

# We need to setup root logger before importing any fairseq libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "DEBUG").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")
logger.setLevel(logging.DEBUG)

import numpy as np
import torch
from fairseq import (
    checkpoint_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators, data_utils
from fairseq.data.plasma_utils import PlasmaStore
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap, utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
from omegaconf import DictConfig, OmegaConf


def main(cfg: FairseqConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    if distributed_utils.is_master(cfg.distributed_training) and "job_logging_cfg" in cfg:
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
            cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    if cfg.common.log_file is not None:
        handler = logging.FileHandler(filename=cfg.common.log_file)
        logger.addHandler(handler)

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)

    if cfg.checkpoint.write_checkpoints_asynchronously:
        try:
            import iopath  # noqa: F401
        except ImportError:
            logging.exception(
                "Asynchronous checkpoint writing is specified but iopath is "
                "not installed: `pip install iopath`"
            )
            return

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)

    cfg2 = copy.deepcopy(cfg)
    cfg2.task.sampling_temperature = cfg2.task.sampling_temperature_2
    assert cfg.task.sampling_temperature <= cfg2.task.sampling_temperature, "Please make sure the temperature of " \
                                                                            "model-2 is higher than that of model-1."
    cfg.task.pre_LS_distillation_weight = cfg.task.pre_LS_distillation_weight_model1
    cfg2.task.pre_LS_distillation_weight = cfg.task.pre_LS_distillation_weight_model2

    task2 = tasks.setup_task(cfg2.task)
    cfg.checkpoint.save_dir = cfg.checkpoint.save_dir + '/model1'
    cfg2.checkpoint.save_dir = cfg2.checkpoint.save_dir + '/model2'

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    if cfg.distributed_training.ddp_backend == "fully_sharded":  # False
        with fsdp_enable_wrap(cfg.distributed_training):
            model = fsdp_wrap(task.build_model(cfg.model))
    else:
        model = task.build_model(cfg.model)
        model2 = task2.build_model(cfg2.model)
    criterion = task.build_criterion(cfg.criterion)
    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False) and p.requires_grad)
        )
    )

    logger.info(
        "num. expert model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False) and p.requires_grad),
        )
    )

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    # We load the valid dataset AFTER building the model
    data_utils.raise_if_valid_subsets_unintentionally_ignored(cfg)
    if cfg.dataset.combine_valid_subsets:
        task.load_dataset("valid", combine=True, epoch=1)
        task2.load_dataset("valid", combine=True, epoch=1)
    else:
        for valid_sub_split in cfg.dataset.valid_subset.split(","):
            task.load_dataset(valid_sub_split, combine=False, epoch=1)
            task2.load_dataset(valid_sub_split, combine=False, epoch=1)

    # (optionally) Configure quantization
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion, quantizer)
        trainer2 = Trainer(cfg2, task2, model2, criterion, quantizer)
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)
        trainer2 = MegatronTrainer(cfg2, task2, model2, criterion)
    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )
    logger.info(
        "max tokens per device = {} and max sentences per device = {}".format(
            cfg.dataset.max_tokens,
            cfg.dataset.batch_size,
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )

    extra_state2, epoch_itr2 = checkpoint_utils.load_checkpoint(
        cfg2.checkpoint,
        trainer2,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )

    if cfg.common.tpu:
        import torch_xla.core.xla_model as xm
        xm.rendezvous("load_checkpoint")  # wait for all workers

    max_epoch = cfg.optimization.max_epoch or math.inf
    lr = trainer.get_lr()
    lr2 = trainer2.get_lr()

    train_meter = meters.StopwatchMeter()
    train_meter.start()

    # Added by Eachan: initializing the language-specific distillation weights.
    task.init_LS_distillation_weights()
    task2.init_LS_distillation_weights()

    while epoch_itr.next_epoch_idx <= max_epoch:

        if lr <= cfg.optimization.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr={cfg.optimization.stop_min_lr})"
            )
            break
        if lr2 <= cfg2.optimization.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr2}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr2={cfg2.optimization.stop_min_lr})"
            )
            break

        # *********************************************************************************
        # train for one epoch
        valid_losses, valid_losses2, should_stop = train(cfg, trainer, task, epoch_itr,
                                          cfg2, trainer2, task2, epoch_itr2)
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])
        lr2 = trainer2.lr_step(epoch_itr.epoch, valid_losses2[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )
        epoch_itr2 = trainer2.get_train_iterator(
            epoch_itr2.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task2.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task2.has_sharded_data("train"),
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))

    # ioPath implementation to wait for all asynchronous file writes to complete.
    if cfg.checkpoint.write_checkpoints_asynchronously:
        logger.info(
            "ioPath PathManager waiting for all asynchronous checkpoint "
            "writes to finish."
        )
        PathManager.async_close()
        logger.info("ioPath PathManager finished waiting.")


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(
        cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr,
        cfg2, trainer2, task2, epoch_itr2
) -> Tuple[List[Optional[float]], bool]:
    # Before training for an epoch, search better distillation weights
    update_distillation_weight(cfg, trainer, task, epoch_itr,
                               cfg2, trainer2, task2, epoch_itr2)

    logger.info(f"model-1 distillation weights: {task.LS_distillation_weights}")
    logger.info(f"model-2 distillation weights: {task2.LS_distillation_weights}")
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    # epoch_itr: iterators.EpochBatchIterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
    )
    itr2 = epoch_itr2.next_epoch_itr(
        fix_batches_to_gpus=cfg2.distributed_training.fix_batches_to_gpus,
        shuffle=(epoch_itr2.next_epoch_idx > cfg2.dataset.curriculum),
    )

    # 这个itr是个iterators.CountingIterator
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    # 这个GroupIterator的作用好像就是将原来的一次返回一个epoch变成一次返回一个list，
    # 这个list中包含update_freq个epoch的数据
    itr = iterators.GroupedIterator(itr, update_freq)
    itr2 = iterators.GroupedIterator(itr2, update_freq)

    if cfg.common.tpu:
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_file=cfg.common.log_file,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
        azureml_logging=(
            cfg.common.azureml_logging
            if distributed_utils.is_master(cfg.distributed_training)
            else False
        ),
    )
    progress.update_config(_flatten_config(cfg))

    trainer.begin_epoch(epoch_itr.epoch)
    trainer2.begin_epoch(epoch_itr2.epoch)

    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    logger.info("Start iterating over samples")

    for i, samples in enumerate(progress):

        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
                "train_step-%d" % i
        ):
            # =====================key code line================
            log_output = trainer.train_step(samples, epoch=epoch_itr.epoch, step=i,
                                            teacher_trainer=trainer2)

            if itr2.has_next():
                samples2 = next(itr2)
                log_output2 = trainer2.train_step(samples2, epoch=epoch_itr2.epoch, step=i,
                                                  teacher_trainer=trainer)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()

        while end_of_epoch and itr2.has_next():
            samples2 = next(itr2)
            log_output2 = trainer2.train_step(samples2, epoch=epoch_itr2.epoch, step=i, teacher_trainer=trainer)

        # test code: if you want to quickly jump into valid stage, delete the following annotation
        # if i > 1:
        #     end_of_epoch = True

        # valid
        valid_losses, should_stop = validate_and_save(
            cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )

        valid_losses2, should_stop2 = validate_and_save(
            cfg2, trainer2, task2, epoch_itr2, valid_subsets, end_of_epoch, model2=True
        )

        # added by Eachan: save the language-specific distillation weights of GPU-0
        if end_of_epoch:
            save_LS_distillation_weights(cfg, trainer)
            save_LS_distillation_weights(cfg2, trainer2)

        # Added by Eachan: Averaging model1 and model2:
        if cfg.task.model_fusion_interval > 0 and \
                end_of_epoch and \
                epoch_itr.epoch % cfg.task.model_fusion_interval == 0:
            avg_state = collections.OrderedDict()
            state_1 = trainer.model.state_dict()
            state_2 = trainer2.model.state_dict()
            for k in state_1.keys():
                if 'decoder' in k:
                    avg_state[k] = state_1[k] * (1 - cfg.task.recombination_beta) +\
                                   state_2[k] * cfg.task.recombination_beta
                else:
                    avg_state[k] = state_1[k] * cfg.task.recombination_beta +\
                                   state_2[k] * (1 - cfg.task.recombination_beta)
            trainer.model.load_state_dict(avg_state)
            trainer2.model.load_state_dict(avg_state)

        # test code: if you want to quickly jump into valid stage, removing the following annotation
        # if i > 1:
        #     break

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")

    return valid_losses, valid_losses2, should_stop


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def validate_and_save(
        cfg: DictConfig,
        trainer: Trainer,
        task: tasks.FairseqTask,
        epoch_itr,
        valid_subsets: List[str],
        end_of_epoch: bool,
        model2=False,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    # Stopping conditions (and an additional one based on validation loss later
    # on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if (
            cfg.optimization.stop_time_hours > 0
            and training_time_hours > cfg.optimization.stop_time_hours
    ):
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {cfg.optimization.stop_time_hours} hour(s)"
        )

    do_save = (
            (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
            or should_stop
            or (
                    cfg.checkpoint.save_interval_updates > 0
                    and num_updates > 0
                    and num_updates % cfg.checkpoint.save_interval_updates == 0
                    and num_updates >= cfg.dataset.validate_after_updates
            )
    )

    do_validate = (
                          (not end_of_epoch and do_save)  # validate during mid-epoch saves
                          or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
                          or should_stop
                          or (
                                  cfg.dataset.validate_interval_updates > 0
                                  and num_updates > 0
                                  and num_updates % cfg.dataset.validate_interval_updates == 0
                          )
                  ) and not cfg.dataset.disable_validation and num_updates >= cfg.dataset.validate_after_updates

    # Validate
    valid_losses = [None]
    if do_validate:
        if getattr(cfg.task, "LS_epoch", None):
            valid_losses, LS_valid_loss = validate(cfg, trainer, task, epoch_itr, valid_subsets)

        else:
            valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)

    should_stop |= should_stop_early(cfg, valid_losses[0])

    # Save checkpoint
    save_func = checkpoint_utils.save_checkpoint if not model2 else checkpoint_utils.save_checkpoint2
    if do_save or should_stop:
        if getattr(cfg.task, "LS_epoch", None):

            save_func(
                cfg.checkpoint, trainer, epoch_itr, valid_losses[0], LS_valid_loss=LS_valid_loss
            )
        else:
            save_func(
                cfg.checkpoint, trainer, epoch_itr, valid_losses[0]
            )

    return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
        cfg: DictConfig,
        trainer: Trainer,
        task: tasks.FairseqTask,
        epoch_itr,
        subsets: List[str],
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)

    # Added  by Eachan: create a dict (named 'LS_valid_losses') to record valid loss cumulative sum and cumulative size
    #     for each language pair.
    if getattr(cfg.task, "LS_epoch", None):
        LS_valid_cumulative_loss = {}
        for lang_pair in cfg.model.lang_pairs.split(','):
            LS_valid_cumulative_loss[lang_pair] = {
                'cumulative_loss': 0.0,
                'cumulative_size': 0
            }

    valid_losses = []
    for subset in subsets:  # ['valid']
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(
            shuffle=False, set_dataset_epoch=False  # use a fixed valid set
        )
        if cfg.common.tpu:
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for i, sample in enumerate(progress):
                if cfg.dataset.max_valid_steps is not None and i > cfg.dataset.max_valid_steps:
                    break

                # modified by eachan: receive three variables.
                if getattr(cfg.task, "LS_epoch", None) and (sample is not None and len(sample) > 0):
                    logging_output, loss_, sample_size = trainer.valid_step(sample)

                    # add loss and size into LS_valid_cumulative_loss:
                    for lang_pair in cfg.model.lang_pairs.split(','):
                        LS_valid_cumulative_loss[lang_pair]["cumulative_loss"] += round(
                            logging_output[lang_pair + '_loss'].item() / math.log(2), 3)
                        LS_valid_cumulative_loss[lang_pair]["cumulative_size"] += logging_output[
                            lang_pair + '_size']
                else:
                    trainer.valid_step(sample)

        # added by eachan: calculate the average valid loss for each language.
        if getattr(cfg.task, "LS_epoch", None):
            # print("eachan print:")
            # print(agg.get_smoothed_values())
            LS_valid_loss = {}
            for lang, item in LS_valid_cumulative_loss.items():
                LS_valid_loss[lang] = item['cumulative_loss'] / item['cumulative_size'] \
                    if item['cumulative_size'] > 0 else 10

        # log validation stats
        stats = get_valid_stats(cfg, trainer, agg.get_smoothed_values())

        if hasattr(task, "post_validate"):
            task.post_validate(trainer.get_model(), stats, agg)

        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])

    # Added by Eachan:
    if getattr(cfg.task, "LS_epoch", None):
        return valid_losses, LS_valid_loss
    else:
        return valid_losses


def get_valid_stats(
        cfg: DictConfig, trainer: Trainer, stats: Dict[str, Any]
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
        )
    return stats


def cli_main(
        modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    if cfg.common.use_plasma_view:
        server = PlasmaStore(path=cfg.common.plasma_path)
        logger.info(f"Started plasma server pid {server.server.pid} {cfg.common.plasma_path}")

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)

    # if cfg.common.use_plasma_view:
    #     server.server.kill()


def update_distillation_weight(cfg1, trainer1, task1, epoch_itr1,
                               cfg2, trainer2, task2, epoch_itr2):
    if epoch_itr1.epoch <= 1 or epoch_itr1.epoch < cfg1.task.distillation_start_epoch:
        return

    if cfg1.task.mutual_distillation_mode == "unidirectional" \
            and (epoch_itr1.epoch % cfg1.task.weight_update_interval == 0):
        # load language-specific valid loss of the last epoch
        f = open(f"{cfg1.checkpoint.save_dir}/LS_valid_loss_history.json")
        LS_valid_loss_history1 = json.load(f)
        f.close()
        LS_valid_loss_history1.pop("all")

        f = open(f"{cfg2.checkpoint.save_dir}/LS_valid_loss_history.json")
        LS_valid_loss_history2 = json.load(f)
        f.close()
        LS_valid_loss_history2.pop("all")

        tolerance = 0.05
        for lang_pair in LS_valid_loss_history1.keys():
            if LS_valid_loss_history1[lang_pair][-1] + tolerance < LS_valid_loss_history2[lang_pair][-1]:
                task1.LS_distillation_weights[lang_pair] = 0
            else:
                task1.LS_distillation_weights[lang_pair] = cfg1.task.distillation_weight

            if LS_valid_loss_history2[lang_pair][-1] + tolerance < LS_valid_loss_history1[lang_pair][-1]:
                task2.LS_distillation_weights[lang_pair] = 0
            else:
                task2.LS_distillation_weights[lang_pair] = cfg2.task.distillation_weight

    elif cfg1.task.mutual_distillation_mode == "automatic" and (epoch_itr1.epoch % cfg1.task.weight_update_interval == 0):
        search_better_distillation_weights(cfg1, trainer1, task1, epoch_itr1, trainer2, epoch_itr2=epoch_itr2)
        search_better_distillation_weights(cfg2, trainer2, task2, epoch_itr2, trainer1)


def search_better_distillation_weights(cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr,
                                       teacher_trainer, epoch_itr2=None):
    """
    trial different distillation weights on model.
    """
    # 1. Initialize some states
    # synchronize distillation weights across multiple GPUs.
    load_LS_distillation_weights(cfg, trainer)

    # save the model weights, optimizer states, update_nums
    model_copy = trainer.model.state_dict()
    optimizer_copy = trainer.optimizer.state_dict()
    num_updates_copy = trainer.get_num_updates()

    trial_records = {}
    for lang_pair in task.LS_distillation_weights.keys():
        trial_records[lang_pair] = []

    def step_size_schedule(strategy, epoch, start_distillation_epoch, max_epoch):
        pre_step_size = cfg.task.step_size
        if strategy == "None":
            return pre_step_size
        elif strategy == "inverse_sqrt_root":
            max_step_size = pre_step_size
            step_size = 0.2 * max_step_size + 0.8 * max_step_size * math.sqrt(
                (max_epoch - epoch) / (max_epoch - start_distillation_epoch))
            return step_size
        elif strategy == "inverse_sqrt_root2":
            max_step_size = pre_step_size
            step_size = 0.0 * max_step_size + 1.0 * max_step_size * math.sqrt(
                (max_epoch - epoch) / (max_epoch - start_distillation_epoch))
            return step_size
        elif strategy == "inverse_sqrt_root3":
            max_step_size = pre_step_size
            search_period = 0.8
            if epoch > max_epoch * search_period:
                return 0
            step_size = max_step_size * math.sqrt(
                (max_epoch * search_period - epoch) / (max_epoch * search_period - start_distillation_epoch))
            return step_size
        else:
            raise Exception(f"Unknown step size scheduler: {strategy}")

    step_size = step_size_schedule(cfg.task.step_size_scheduler, epoch_itr.epoch,
                                   cfg.task.distillation_start_epoch, cfg.optimization.max_epoch)
    search_space = [step_size, 0, step_size * (-1)]

    if step_size == 0:
        return

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def inverse_sigmoid(x):
        x = min(x, 0.98) # clip
        return -math.log(1 / x - 1)

    # print(f"我想知道此时每个GPU上的模型参数都是一样的嘛？于是我打印：")
    # print(f"mode param example: {trainer.model.encoder.embed_tokens.weight[0][0]}")
    # 2. Beginning trying
    for trial_step in search_space:
        logger.info(f"begin trial: distillation weight += {trial_step}")

        # change alpha
        for lang_pair, distillation_weight in task.LS_distillation_weights.items():
            ori_alpha = task.LS_distillation_weights[lang_pair]
            assert 0 < ori_alpha < 1, \
                "in automatic mutual distillation, alpha should be (0, 1) initially."
            # map alpha to (-inf, +inf), i.e., inverse sigmoid
            alpha = inverse_sigmoid(ori_alpha)
            # add alpha_base by trial_alpha
            alpha += trial_step
            # map alpha to (0, 1)
            alpha = sigmoid(alpha)

            task.LS_distillation_weights[lang_pair] = alpha

        # load trial dataset: in this work, we use the first `--trial-dataset-ratio` of training set as trial dataset
        # therefore, we just construct the same data iterator with which of training dataset.
        itr = epoch_itr._get_iterator_for_epoch(epoch_itr.epoch,
                                                shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
                                                fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus)
        update_freq = (
            cfg.optimization.update_freq[epoch_itr.epoch - 1]
            if epoch_itr.epoch <= len(cfg.optimization.update_freq)
            else cfg.optimization.update_freq[-1]
        )

        itr = iterators.GroupedIterator(itr, update_freq)

        logger.info("Start iterating over trial samples...")
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_file=cfg.common.log_file,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
            azureml_logging=(
                cfg.common.azureml_logging
                if distributed_utils.is_master(cfg.distributed_training)
                else False
            ),
        )
        progress.update_config(_flatten_config(cfg))

        if epoch_itr2 is not None:
            """
            For model-1 (cold-model), the temperature of which is 1, 
            we construct the trial set with temperature about (t1 + t2) / 2.
            
            For model-1 (cold-model), the temperature of which is 1, 
            we construct the trial set with temperature t2
            """
            itr2 = epoch_itr2._get_iterator_for_epoch(epoch_itr2.epoch,
                                                    shuffle=(epoch_itr2.next_epoch_idx > cfg.dataset.curriculum),
                                                    fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus)

            itr2 = iterators.GroupedIterator(itr2, update_freq)

        trainer.begin_epoch(epoch_itr.epoch)

        valid_subsets = cfg.dataset.valid_subset.split(",")
        should_stop = False
        num_updates = trainer.get_num_updates()
        logger.info("Start iterating over samples")

        for i, samples in enumerate(progress):

            # this line of code may be confusing......
            if epoch_itr2 is not None and i % 2 == 0:
                samples = next(itr2)

            with metrics.aggregate("trial_inner"), torch.autograd.profiler.record_function(
                    "trial_step-%d" % i
            ):
                # =====================key code line================
                log_output = trainer.train_step(samples, epoch=epoch_itr.epoch, step=i,
                                                teacher_trainer=teacher_trainer)

            if log_output is not None:  # not OOM, overflow, ...
                # log mid-epoch stats
                num_updates = trainer.get_num_updates()
                if num_updates % cfg.common.log_interval == 0:
                    stats = get_training_stats(metrics.get_smoothed_values("trial_inner"))
                    progress.log(stats, tag="trial_inner", step=num_updates)

                    # reset mid-epoch stats after each log interval
                    # the end-of-epoch stats will still be preserved
                    metrics.reset_meters("trial_inner")

            end_of_trial_epoch = (itr.n / len(itr) > cfg.task.trial_dataset_ratio)
            # valid
            if end_of_trial_epoch:
                valid_losses, LS_valid_loss = validate(cfg, trainer, task, epoch_itr, valid_subsets)
                if trainer.is_data_parallel_master:
                    logger.debug(f"LS_valid_loss after trial: {LS_valid_loss}")

                # record the trial results.
                for lang_pair, val_loss in LS_valid_loss.items():
                    if lang_pair in trial_records.keys():
                        trial_records[lang_pair].append(val_loss)

                should_stop = True

            if should_stop:
                break

        # log end-of-epoch stats
        stats = get_training_stats(metrics.get_smoothed_values("trial_inner"))
        progress.print(stats, tag="trial_inner", step=num_updates)

        # reset epoch-level meters
        metrics.reset_meters("trial")

        # roll back
        for lang_pair, distillation_weight in task.LS_distillation_weights.items():
            alpha = task.LS_distillation_weights[lang_pair]
            alpha = inverse_sigmoid(alpha)
            alpha -= trial_step
            alpha = sigmoid(alpha)
            task.LS_distillation_weights[lang_pair] = alpha
        logger.debug(f"eachan print lr| {trainer.get_lr()}")
        trainer.model.load_state_dict(model_copy)
        trainer.optimizer.load_state_dict(optimizer_copy)
        trainer.set_num_updates(num_updates_copy)

    # 3. Find the best distillation weight for each language pair
    """
    trial_records looks like:
        { 'bos-eng': [6.340326520140255, 6.535490635422902, 6.87558864277773], 
          'kor-eng': [6.397146437727406, 6.578106132553365, 6.900015743245029]
        }
    """
    print(f"trial records on {trainer.model.encoder.embed_tokens.weight[0][0].device}")
    print(trial_records)
    if cfg.task.uniform_distillation_weight.lower() == "true":
        """set a uniform distillation weight for all language pairs."""
        average_loss = [0] * len(search_space)
        for lang_pair in trial_records.keys():
            for i, v in enumerate(trial_records[lang_pair]):
                average_loss[i] += v
        optimal_trial = search_space[average_loss.index(min(average_loss))]
        # optimal_trial = search_space[
        #     trial_records["all"].index(min(trial_records["all"]))]
        for lang_pair, distillation_weight in task.LS_distillation_weights.items():
            alpha = task.LS_distillation_weights[lang_pair]
            alpha = inverse_sigmoid(alpha)
            alpha += optimal_trial
            alpha = sigmoid(alpha)
            task.LS_distillation_weights[lang_pair] = alpha
    else:
        for lang_pair, distillation_weight in task.LS_distillation_weights.items():
            optimal_trial = search_space[
                trial_records[lang_pair].index(min(trial_records[lang_pair]))]
            alpha = task.LS_distillation_weights[lang_pair]
            alpha = inverse_sigmoid(alpha)
            alpha += optimal_trial
            alpha = sigmoid(alpha)
            task.LS_distillation_weights[lang_pair] = alpha


def save_LS_distillation_weights(cfg, trainer):
    """
    This function writes the language-specific distillation-weights of GPU-0 into the save_dir/LS_distillation_weights.json

    """
    if trainer.is_data_parallel_master:
        data = ""
        if os.path.exists(f"{cfg.checkpoint.save_dir}/LS_distillation_weights.json"):
            f = open(f"{cfg.checkpoint.save_dir}/LS_distillation_weights.json")
            data = f.read()
            f.close()

        last_LS_distillation_weights = json.dumps(trainer.task.LS_distillation_weights)
        data += f"{last_LS_distillation_weights}\n"

        f = open(f"{cfg.checkpoint.save_dir}/LS_distillation_weights.json", 'w+')
        f.write(data)
        f.close()


def load_LS_distillation_weights(cfg, trainer):

    if os.path.exists(f"{cfg.checkpoint.save_dir}/LS_distillation_weights.json"):
        f = open(f"{cfg.checkpoint.save_dir}/LS_distillation_weights.json")
        data = f.read()
        f.close()

        last_LS_distillation_weights = data.strip().split('\n')[-1]
        last_LS_distillation_weights = json.loads(last_LS_distillation_weights)

        for lang_pair, weight in last_LS_distillation_weights.items():
            trainer.task.LS_distillation_weights[lang_pair] = weight


if __name__ == "__main__":
    cli_main()
