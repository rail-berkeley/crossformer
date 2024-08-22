"""
A server for hosting a CrossFormer model for inference.

On action server: pip install uvicorn fastapi json-numpy
On client: pip install requests json-numpy

On client:

import requests
import json_numpy
from json_numpy import loads
json_numpy.patch()

Reset and provide the task before starting the rollout:

requests.post("http://serverip:port/reset", json={"text": ...})

Sample an action:

action = loads(
    requests.post(
        "http://serverip:port/query",
        json={"observation": ...},
    ).json()
)
"""


import json_numpy

json_numpy.patch()
from collections import deque
import time
import traceback
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import jax
import numpy as np
import tensorflow as tf
import uvicorn

from crossformer.model.crossformer_model import CrossFormerModel


def json_response(obj):
    return JSONResponse(json_numpy.dumps(obj))


def resize(img, size=(224, 224)):
    img = tf.image.resize(img, size=size, method="lanczos3", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8).numpy()


def stack_and_pad(history: deque, num_obs: int):
    """
    Converts a list of observation dictionaries (`history`) into a single observation dictionary
    by stacking the values. Adds a padding mask to the observation that denotes which timesteps
    represent padding based on the number of observations seen so far (`num_obs`).
    """
    horizon = len(history)
    full_obs = {k: np.stack([dic[k] for dic in history]) for k in history[0]}
    pad_length = horizon - min(num_obs, horizon)
    timestep_pad_mask = np.ones(horizon)
    timestep_pad_mask[:pad_length] = 0
    full_obs["timestep_pad_mask"] = timestep_pad_mask
    return full_obs


class HttpServer:
    def __init__(self, paths):
        self.models = dict()
        for name, path, step in paths:
            self.models[name] = CrossFormerModel.load_pretrained(path, step=step)

        # settings for bimanual inference
        self.head_name = "bimanual"
        self.dataset_name = "aloha_pen_uncap_diverse_dataset"
        self.action_dim = 14
        self.pred_horizon = 100
        self.exp_weight = 0
        self.horizon = 5
        self.text = None
        self.task = None
        self.rng = jax.random.PRNGKey(0)

        self.reset_history()

        # trigger compilation
        for name in self.models.keys():
            payload = {
                "text": "",
                "model": name,
            }
            self.reset(payload)
            payload = {
                "observation": {
                    "proprio_bimanual": np.zeros((14,)),
                    "image_high": np.zeros((224, 224, 3)),
                    "image_left_wrist": np.zeros((224, 224, 3)),
                    "image_right_wrist": np.zeros((224, 224, 3)),
                },
                "modality": "l",
                "ensemble": True,
                "model": name,
                "dataset_name": self.dataset_name,
            }
            for _ in range(self.horizon):
                start = time.time()
                print(self.sample_actions(payload))
                print(time.time() - start)

        self.reset_history()

    def run(self, port=8000, host="0.0.0.0"):
        self.app = FastAPI()
        self.app.post("/query")(self.sample_actions)
        self.app.post("/reset")(self.reset)
        uvicorn.run(self.app, host=host, port=port)

    def reset_history(self):
        self.history = deque(maxlen=self.horizon)
        self.num_obs = 0
        self.act_history = deque(maxlen=self.pred_horizon)

    def reset(self, payload: Dict[Any, Any]):
        model_name = payload.get("model", "crossformer")
        if "goal" in payload:
            goal_img = resize(payload["goal"]["image_primary"])
            goal = {"image_primary": goal_img[None]}
            self.task = self.models[model_name].create_tasks(goals=goal)
        elif "text" in payload:
            text = payload["text"]
            self.text = text
            self.task = self.models[model_name].create_tasks(texts=[text])
        else:
            raise ValueError

        self.reset_history()

        return "reset"

    def sample_actions(self, payload: Dict[Any, Any]):
        try:

            model_name = payload.get("model", "crossformer")

            obs = payload["observation"]
            for key in obs:
                if "image" in key:
                    obs[key] = resize(obs[key])
                # normalize proprioception expect for bimanual proprioception
                if "proprio" in key and not key == "proprio_bimanual":
                    proprio_normalization_statistics = self.models[
                        model_name
                    ].dataset_statistics[self.dataset_name][key]
                    obs[key] = (obs[key] - proprio_normalization_statistics["mean"]) / (
                        proprio_normalization_statistics["std"]
                    )

            self.history.append(obs)
            self.num_obs += 1
            obs = stack_and_pad(self.history, self.num_obs)

            # add batch dim
            obs = jax.tree_map(lambda x: x[None], obs)

            unnormalization_statistics = self.models[model_name].dataset_statistics[
                self.dataset_name
            ]["action"]

            self.rng, key = jax.random.split(self.rng)
            actions = self.models[model_name].sample_actions(
                obs,
                self.task,
                unnormalization_statistics,
                head_name=self.head_name,
                rng=key,
            )[0, :, : self.action_dim]

            actions = np.array(actions)

            # whether to temporally ensemble the action predictions or return the full chunk
            if not payload.get("ensemble", True):
                print(actions)
                return json_response(actions)

            self.act_history.append(actions[: self.pred_horizon])
            num_actions = len(self.act_history)

            # select the predicted action for the current step from the history of action chunk predictions
            curr_act_preds = np.stack(
                [
                    pred_actions[i]
                    for (i, pred_actions) in zip(
                        range(num_actions - 1, -1, -1), self.act_history
                    )
                ]
            )

            # more recent predictions get exponentially *less* weight than older predictions
            weights = np.exp(-self.exp_weight * np.arange(num_actions))
            weights = weights / weights.sum()
            # compute the weighted average across all predictions for this timestep
            action = np.sum(weights[:, None] * curr_act_preds, axis=0)

            print(action)
            return json_response(action)
        except:
            print(traceback.format_exc())
            return "error"


def main():
    import argparse

    tf.config.set_visible_devices([], "GPU")

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host to run on", default="0.0.0.0", type=str)
    parser.add_argument("--port", help="Port to run on", default=8000, type=int)
    args = parser.parse_args()

    # name, path, step
    paths = [
        ("crossformer", "hf://rail-berkeley/crossformer", None),
    ]

    server = HttpServer(paths)
    server.run(args.port, args.host)


if __name__ == "__main__":
    main()
