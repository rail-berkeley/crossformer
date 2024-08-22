from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from einops import rearrange
import flax.linen as nn
import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike

from crossformer.model.components.base import TokenGroup
from crossformer.model.components.transformer import MAPHead
from crossformer.utils.typing import PRNGKey


class ActionHead(ABC):
    """Action prediction modules that take in the transformer token outputs and predict actions.

    Each action head here does chunked action prediction: i.e. at every timestep, it tries to predict the next
    `action_horizon` actions into the future from that timestep.  Setting `action_horizon=1` corresponds to
    the typical action prediction setup.
    """

    @abstractmethod
    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        actions: ArrayLike,
        timestep_pad_mask: ArrayLike,
        action_pad_mask: ArrayLike,
        train: bool = True,
    ) -> Tuple[Array, Dict[str, Array]]:
        raise NotImplementedError

    @abstractmethod
    def predict_action(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        argmax: bool = False,
        sample_shape: Tuple[int, ...] = (),
        rng: Optional[PRNGKey] = None,
        temperature: float = 1.0,
        train: bool = False,
        embodiment_action_dim: Optional[int] = None,
    ) -> Array:
        """Predict the action for the last timestep in the window. Returns shape
        (*sample_shape, batch_size, action_horizon, action_dim).
        """
        raise NotImplementedError


def masked_mean(x, mask):
    mask = jnp.broadcast_to(mask, x.shape)
    return jnp.mean(x * mask) / jnp.clip(jnp.mean(mask), a_min=1e-5, a_max=None)


def continuous_loss(
    pred_value: ArrayLike,
    ground_truth_value: ArrayLike,
    mask: ArrayLike,
    loss_type: str = "mse",
) -> Array:
    """
    Args:
        pred_value: shape (batch_dims...)
        ground_truth_value: continuous values w/ shape (batch_dims...)
        mask: broadcastable to ground_truth
    """
    if loss_type == "mse":
        loss = jnp.square(pred_value - ground_truth_value)
    elif loss_type == "l1":
        loss = jnp.abs(pred_value - ground_truth_value)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    loss = masked_mean(loss, mask)

    mse = jnp.square(pred_value - ground_truth_value)
    mse = masked_mean(mse, mask)

    sign_deltas = jnp.logical_or(
        jnp.logical_and(ground_truth_value > 0, pred_value <= 0),
        jnp.logical_and(ground_truth_value <= 0, pred_value > 0),
    )
    lsign = masked_mean(sign_deltas, mask)
    return loss, {
        "loss": loss,
        "mse": mse,
        "lsign": lsign,
    }


class ContinuousActionHead(nn.Module, ActionHead):
    """Predicts continuous actions (as opposed to discretized).

    Continuous actions are predicted by tanh squashing the model output to [-max_action, max_action], and then
    optimized using a standard regression loss.

    You may create an embedding by either mean-pooling across tokens (use_map=False) or using multi-head
    attention pooling (use_map=True). It is recommended to use MAP when decoding from the observation token
    stream.
    """

    readout_key: str
    pool_strategy: str = "mean"
    action_horizon: int = 1
    action_dim: int = 7
    clip_pred: bool = True
    max_action: float = 5.0
    loss_type: str = "mse"
    num_preds: int = 0
    loss_weight: float = 1.0
    constrain_loss_dims: bool = False

    def setup(self):
        if self.pool_strategy == "use_map":
            self.map_head = MAPHead()

        num_preds = (
            self.num_preds if self.num_preds else self.action_horizon * self.action_dim
        )
        self.mean_proj = nn.Dense(num_preds)

    def __call__(
        self, transformer_outputs: Dict[str, TokenGroup], train: bool = True
    ) -> jax.Array:
        """
        Returns:
            mean: Predicted actions w/ shape (batch_size, window_size, action_horizon, action_dim)
        """
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )
        if self.pool_strategy == "use_map":  # Multi-head attention pooling
            embeddings = self.map_head(token_group, train=train)[:, :, 0]
        elif self.pool_strategy == "mean":  # mean pooling
            embeddings = token_group.tokens.mean(axis=-2)
        elif self.pool_strategy == "pass":
            embeddings = token_group.tokens
        else:
            raise ValueError(f"{self.pool_strategy} not implemented!")

        if len(embeddings.shape) == 3:
            # Implies embeddings is (batch_size, window_size, embedding_size)
            mean = self.mean_proj(embeddings)
            mean = rearrange(
                mean, "b w (h a) -> b w h a", h=self.action_horizon, a=self.action_dim
            )
        else:
            # Assumes embeddings is (batch_size, window_size, H, embedding_size)
            assert embeddings.shape[-2] == self.action_horizon
            mean = self.mean_proj(embeddings)

        if self.clip_pred:
            mean = jnp.tanh(mean / self.max_action) * self.max_action

        return mean

    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        actions: ArrayLike,
        timestep_pad_mask: ArrayLike,
        action_pad_mask: ArrayLike,
        action_head_mask: Optional[ArrayLike] = None,
        train: bool = True,
    ) -> Tuple[Array, Dict[str, Array]]:
        """Computes the loss for the action regression objective.

        Args:
            transformer_ouputs: must contain self.readout_key with shape (batch_size, window_size, num_tokens,
                embedding_size)
            actions: shape (batch_size, window_size, action_horizon, action_dim)
            timestep_pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep
            action_pad_mask: boolean array (same shape as actions) which is True if the action dimension is not a padding dimension
            action_head_mask: boolean array (batch,) which is True if the action space corresponds to this specific head

        Returns:
            loss: float
            metrics: dict
        """
        if self.constrain_loss_dims:
            # when using separate heads we can constrain the loss to the action dimensions and action horizon specific to this head
            actions = actions[:, :, : self.action_horizon, : self.action_dim]
            action_pad_mask = action_pad_mask[
                :, :, : self.action_horizon, : self.action_dim
            ]

        # (batch, window_size, action_horizon, action_dim)
        mean = self(transformer_outputs, train=train)

        if action_head_mask is None:
            action_head_mask = jnp.ones(mean.shape[0], dtype=bool)

        # combine the timestep pad mask with the action pad mask and the action head mask
        mask = (
            timestep_pad_mask[:, :, None, None]
            & action_pad_mask
            & action_head_mask[:, None, None, None]
        )

        loss, metrics = continuous_loss(mean, actions, mask, loss_type=self.loss_type)
        return loss, metrics

    def predict_action(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        train: bool = True,
        *args,
        sample_shape: tuple = (),
        **kwargs,
    ) -> jax.Array:
        """Convenience methods for predicting actions for the final timestep in the window."""
        # only get the last timestep in the window
        mean = self(transformer_outputs, train=train)[:, -1]
        return jnp.broadcast_to(mean, sample_shape + mean.shape)


class L1ActionHead(ContinuousActionHead):
    loss_type: str = "l1"
