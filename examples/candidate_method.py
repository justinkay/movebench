"""A short custom fit using real candidate batches and the existing grid helper."""

from itertools import islice

import torch
import torch.nn.functional as F

from movebench import ssf_grid
from movebench.datasets.chips import grid_dataset
from movebench.datasets.points import training_steps
from movebench.datasets.ssf import candidate_dataset, candidate_loader
from movebench.models.ssf import LinearStepSelect
from movebench.training import read_checkpoint, save_checkpoint


class CandidateMethod:
    def __init__(self, *, model_config, hparams, seed):
        self.model_config = dict(model_config)
        self.hparams = dict(hparams)
        self.seed = int(seed)

    def fit(self, train, validation, *, work_dir, restore_from=None):
        self.train_ = train
        self.sampler_state_ = None
        self.dataset_ = candidate_dataset(train, **self.model_config)
        self.model_ = LinearStepSelect(
            self.dataset_.c_env, include_time=self.dataset_.include_time
        )
        if restore_from is not None:
            checkpoint = read_checkpoint(restore_from)
            self.model_.load_state_dict(checkpoint["model_state"])
        else:
            loader = candidate_loader(
                self.dataset_, batch_size=self.hparams["batch_size"], shuffle=True,
                num_workers=0, device=torch.device("cpu"),
            )
            optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.hparams["lr"])
            # load_batch calls make_model_batch; observed endpoints are last.
            for step, batch in enumerate(islice(loader, self.hparams["max_steps"]), 1):
                logits = self.model_(batch)
                target = torch.full((len(logits),), logits.shape[1] - 1, dtype=torch.long)
                optimizer.zero_grad()
                F.cross_entropy(logits, target).backward()
                optimizer.step()
            validation_dataset = candidate_dataset(validation, **self.model_config)
            validation_batch = validation_dataset.load_batch(
                range(min(self.hparams["batch_size"], len(validation_dataset)))
            )
            with torch.no_grad():
                train_loss = F.cross_entropy(self.model_(batch), target).item()
                logits = self.model_(validation_batch)
                target = torch.full((len(logits),), logits.shape[1] - 1, dtype=torch.long)
                eval_loss = F.cross_entropy(logits, target).item()
            checkpoint = save_checkpoint(
                work_dir / "checkpoint_best.pt", model=self.model_,
                fit={
                    "best_step": step, "completed_steps": step,
                    "train_metrics": {"last_batch_nll": train_loss},
                    "eval_metrics": {"first_batch_nll": eval_loss},
                },
                schedule={}, model_metadata={},
            )
        self.training_summary_ = {
            "train_steps": checkpoint["completed_steps"],
            "best_step": checkpoint["best_step"],
            "train_metrics": checkpoint["train_metrics"],
            "eval_metrics": checkpoint["eval_metrics"],
        }
        return self

    @torch.no_grad()
    def predict(self, data):
        if self.sampler_state_ is None:
            _, steps = training_steps(self.train_)
            self.sampler_state_ = ssf_grid._replay_sampler_state(
                train_manifest=self.train_.inputs["point_stores"]["train"]["manifest"],
                train_steps=steps,
            )
        return ssf_grid.build_ssf_grid_artifact(
            grid_dataset=grid_dataset(data, plan=self.dataset_.plan, include_env=True),
            sampler_state=self.sampler_state_, batch_size=self.hparams["batch_size"],
            num_workers=0, device=torch.device("cpu"), model=self.model_,
            feature_adapter=self.dataset_,  # Grid cells also use make_model_batch.
        )
