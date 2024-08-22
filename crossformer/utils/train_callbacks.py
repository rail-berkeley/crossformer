from dataclasses import dataclass
from functools import partial
import logging
import os
from typing import Callable, Mapping, Optional

import flax
from flax.training import orbax_utils
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
import tensorflow as tf
import tqdm

from crossformer.data.dataset import make_single_dataset
from crossformer.data.utils.text_processing import TextProcessor
from crossformer.utils.train_utils import TrainState
from crossformer.utils.typing import Any, Data, Sequence


class Callback:
    def __call__(self, train_state: TrainState, step: int):
        raise NotImplementedError


def create_validation_dataset(
    dataset_kwargs: dict,
    traj_transform_kwargs: dict,
    frame_transform_kwargs: dict,
    train: bool = False,
):
    """Creates a dataset for validation and visualization purposes.

    Takes the training configuration and overwrites default parameters with more conservative
    options to ensure stable memory consumption.
    """
    return make_single_dataset(
        dataset_kwargs={
            **dataset_kwargs,
            "num_parallel_reads": 4,
            "num_parallel_calls": 4,
            "shuffle": False,
        },
        traj_transform_kwargs={
            **traj_transform_kwargs,
            "num_parallel_calls": 4,
        },
        frame_transform_kwargs={
            **frame_transform_kwargs,
            "num_parallel_calls": 16,
        },
        train=train,
    )


@dataclass
class SaveCallback(Callback):
    """Callback that saves checkpoints to `save_dir`. If `save_dir` is None, does nothing."""

    save_dir: Optional[str]

    def __post_init__(self):
        if self.save_dir is not None:
            if not self.save_dir.startswith("gs://"):
                self.save_dir = os.path.abspath(self.save_dir)
            if jax.process_index() == 0:
                tf.io.gfile.makedirs(self.save_dir)
                logging.info(f"Created {self.save_dir}")
            # make checkpointers
            # only keep latest full TrainState
            self.state_checkpointer = orbax.checkpoint.CheckpointManager(
                tf.io.gfile.join(self.save_dir, "state"),
                orbax.checkpoint.PyTreeCheckpointer(),
                options=orbax.checkpoint.CheckpointManagerOptions(
                    max_to_keep=1,
                ),
            )
            # keep every params checkpoint
            self.params_checkpointer = orbax.checkpoint.CheckpointManager(
                self.save_dir,
                orbax.checkpoint.PyTreeCheckpointer(),
            )

    def __call__(self, train_state: TrainState, step: int):
        if self.save_dir is not None:
            train_state.model.save_pretrained(
                step, checkpoint_manager=self.params_checkpointer
            )
            self.state_checkpointer.save(
                step,
                train_state,
                {"save_args": orbax_utils.save_args_from_target(train_state)},
            )


def remove_text(tasks: Data, zero_text_encoding: Optional[Data]):
    """Replaces language encoding inside task dict with that of the empty string.

    zero_text_encoding = jax.tree_map(lambda x: x[0], text_processor.encode([""]))
    """
    if zero_text_encoding is None:
        zero_text_encoding = jnp.zeros((1,))
    if "language_instruction" in tasks:
        new_language = jax.tree_map(
            lambda x, example: jnp.broadcast_to(example[None], x.shape),
            tasks["language_instruction"],
            zero_text_encoding,
        )
        new_pad_dict = flax.core.copy(
            tasks["pad_mask_dict"],
            {
                "language_instruction": jnp.zeros_like(
                    tasks["pad_mask_dict"]["language_instruction"]
                )
            },
        )
        tasks = flax.core.copy(
            tasks, {"language_instruction": new_language, "pad_mask_dict": new_pad_dict}
        )
    return tasks


def remove_images(tasks: Data):
    """Replaces images inside task dict with zero (black) images."""
    updates = {k: jnp.zeros_like(v) for k, v in tasks.items() if "image" in k}
    updates["pad_mask_dict"] = flax.core.copy(
        tasks["pad_mask_dict"],
        {
            k: jnp.zeros_like(v)
            for k, v in tasks["pad_mask_dict"].items()
            if "image" in k
        },
    )
    return flax.core.copy(tasks, updates)


@dataclass
class ValidationCallback(Callback):
    loss_fn: Callable
    process_batch_fn: Callable[[Data], Data]
    text_processor: Optional[TextProcessor]
    val_dataset_kwargs_list: Sequence[Mapping[str, Any]]
    dataset_kwargs: Mapping[str, Any]
    val_shuffle_buffer_size: int
    num_val_batches: int
    modes_to_evaluate: Sequence[str] = ("text_conditioned", "image_conditioned")
    train: bool = False

    def __post_init__(self):
        if self.text_processor is not None:
            self.zero_text = jax.tree_map(
                lambda x: x[0], self.text_processor.encode("")
            )
        else:
            self.zero_text = None
        self.val_iterators = {}
        for single_dataset_kwargs in self.val_dataset_kwargs_list:
            val_dataset = create_validation_dataset(
                single_dataset_kwargs,
                self.dataset_kwargs["traj_transform_kwargs"],
                self.dataset_kwargs["frame_transform_kwargs"],
                train=self.train,
            )
            val_iterator = (
                val_dataset.unbatch()
                .shuffle(self.val_shuffle_buffer_size)
                .repeat()
                .batch(self.dataset_kwargs["batch_size"])
                .iterator(prefetch=0)
            )
            val_iterator = map(self.process_batch_fn, val_iterator)
            self.val_iterators[single_dataset_kwargs["name"]] = val_iterator

        @partial(
            jax.jit,
            out_shardings=jax.sharding.PositionalSharding(jax.devices()).replicate(),
        )
        def eval_step(state: TrainState, batch: Data):
            loss_fn_partial = partial(
                self.loss_fn,
                params=state.model.params,
                rng=state.rng,
                train=False,
            )
            all_tasks = {}

            if "base" in self.modes_to_evaluate:
                all_tasks["base"] = batch["task"]
            if "image_conditioned" in self.modes_to_evaluate:
                all_tasks["image_conditioned"] = remove_text(
                    batch["task"], self.zero_text
                )
            if "text_conditioned" in self.modes_to_evaluate:
                all_tasks["text_conditioned"] = remove_images(batch["task"])

            if "unconditioned" in self.modes_to_evaluate:
                all_tasks["unconditioned"] = remove_text(
                    remove_images(batch["task"]), self.zero_text
                )
            return {
                k: loss_fn_partial(batch=flax.core.copy(batch, {"task": tasks}))[1]
                for k, tasks in all_tasks.items()
            }

        self.eval_step = eval_step

    def __call__(self, train_state: TrainState, step: int):
        wandb_metrics = {}
        for name, val_data_iter in self.val_iterators.items():
            metrics = []
            for _, batch in tqdm.tqdm(
                zip(range(self.num_val_batches), val_data_iter),
                total=self.num_val_batches,
                desc=name,
            ):
                metrics.append(self.eval_step(train_state, batch))
            metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
            wandb_metrics[f"validation_{name}"] = metrics
        return wandb_metrics
