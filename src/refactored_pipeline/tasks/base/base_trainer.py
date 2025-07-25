import os
import time

import torch
from omegaconf import open_dict
from torch.utils.tensorboard import SummaryWriter

from refactored_pipeline.tasks.base.early_stopping import EarlyStopping
from refactored_pipeline.tasks.base.saver import ValidationSaver
from refactored_pipeline.utils.os_utils import makedir, remove_old_ckpt


class BaseTrainer:
    def __init__(
        self,
        model,
        loss,
        optimizer,
        config,
        train_loader,
        validation_loss_loader=None,
        validation_evaluator=None,
        test_loader=None,
        scheduler=None,
        regularizer=None,
    ):
        self.loss = loss
        self.optimizer = optimizer
        assert train_loader is not None, "provide at least train loader"
        self.train_loader = train_loader
        self.validation_loss_loader = validation_loss_loader
        self.validation_evaluator = validation_evaluator
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.regularizer = regularizer
        self.model = model
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        print(
            " --- total number parameters:",
            sum([p.nelement() for p in self.model.parameters()]),
        )
        self.model.train()

        self.checkpoint_dir = config["checkpoint_dir"]
        makedir(self.checkpoint_dir)
        makedir(os.path.join(self.checkpoint_dir, "model"))
        makedir(os.path.join(self.checkpoint_dir, "model_ckpt"))
        self.writer_dir = os.path.join(config["checkpoint_dir"], "tensorboard")
        self.writer = SummaryWriter(self.writer_dir)

        self.config = config
        print(" === trainer config === \n =========================", self.config)

        self.validation = (
            True
            if (
                self.validation_loss_loader is not None
                or self.validation_evaluator is not None
            )
            else False
        )

        if self.validation:
            # initialize early stopping or saver (if no early stopping):
            if "early_stopping" in self.config:
                self.saver = EarlyStopping(
                    self.config["patience"], self.config["early_stopping"]
                )
                # config["early_stopping"] either "loss" or any valid and defined metric
                self.val_decision = self.config[
                    "early_stopping"
                ]  # the validation perf (loss or metric) for
                # checkpointing decision
            else:
                assert "monitoring_ckpt" in self.config, (
                    "if no early stopping, provide monitoring for checkpointing on val"
                )
                self.saver = ValidationSaver(
                    loss=True if self.config["monitoring_ckpt"] == "loss" else False
                )
                self.val_decision = (
                    "loss"
                    if self.config["monitoring_ckpt"] == "loss"
                    else self.config["monitoring_ckpt"]
                )

        self.overwrite_final = (
            config["overwrite_final"] if "overwrite_final" in config else False
        )
        self.training_res_handler = open(
            os.path.join(self.checkpoint_dir, "training_perf.txt"), "a"
        )
        # => text file in which we record some training perf
        if self.validation:
            self.validation_res_handler = open(
                os.path.join(self.checkpoint_dir, "validation_perf.txt"), "a"
            )
        if self.test_loader is not None:
            self.test_res_handler = open(
                os.path.join(self.checkpoint_dir, "test_perf.txt"), "a"
            )
        self.fp16 = config["fp16"]

    def train(self):
        t0 = time.time()
        self.model.train()
        ###################################
        self.train_epochs()
        # this method defines how to train the model
        # (e.g. different behavior when we train for a given number of epochs vs a given number of iterations)
        ###################################
        self.writer.close()
        self.training_res_handler.close()
        if self.validation:
            self.validation_res_handler.close()
        if self.test_loader is not None:
            self.test_res_handler.close()
        print("======= TRAINING DONE =======")
        print("took about {} hours".format((time.time() - t0) / 3600))

    def save_checkpoint(self, step, perf, is_best, final_checkpoint=False):
        """logic to save checkpoints"""
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # when using DataParallel
        with open_dict(self.config):
            self.config["ckpt_step"] = step
        state = {
            "step": step,
            "perf": perf,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "regularizer": self.regularizer,
        }
        if self.scheduler is not None:
            scheduler_state_dict = self.scheduler.state_dict()
            state["scheduler_state_dict"] = scheduler_state_dict
        if not final_checkpoint:
            # rename last:
            if os.path.exists(
                os.path.join(self.checkpoint_dir, "model_ckpt/model_last.tar")
            ):
                last_config = torch.load(
                    os.path.join(self.checkpoint_dir, "model_ckpt/model_last.tar"),
                    weights_only=False,
                )
                step_last_config = last_config["step"]
                os.rename(
                    os.path.join(self.checkpoint_dir, "model_ckpt/model_last.tar"),
                    os.path.join(
                        self.checkpoint_dir,
                        "model_ckpt/model_ckpt_{}.tar".format(step_last_config),
                    ),
                )
            # save new last:
            torch.save(
                state, os.path.join(self.checkpoint_dir, "model_ckpt/model_last.tar")
            )
            if is_best:
                torch.save(state, os.path.join(self.checkpoint_dir, "model/model.tar"))
            # remove oldest checkpoint (by default only keep the last 3):
            remove_old_ckpt(os.path.join(self.checkpoint_dir, "model_ckpt"), k=3)
        else:
            torch.save(
                state,
                os.path.join(
                    self.checkpoint_dir, "model_ckpt/model_final_checkpoint.tar"
                ),
            )
            if self.overwrite_final:
                torch.save(state, os.path.join(self.checkpoint_dir, "model/model.tar"))

    def train_epochs(self):
        """
        full training logic
        """
        raise NotImplementedError
