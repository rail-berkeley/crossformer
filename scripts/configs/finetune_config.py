from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from crossformer.model.components.action_heads import L1ActionHead
from crossformer.model.components.tokenizers import ImageTokenizer
from crossformer.model.components.vit_encoders import ResNet26FILM
from crossformer.utils.spec import ModuleSpec


def get_config():
    # whether to finetune the entire model or just the action head
    mode = "full"

    # whether to finetune with image conditioning, language conditioning, or both
    task = "multimodal"

    # the name of the action head to finetune
    head_name = "single_arm"

    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only"]

    # fill this in to configure data loading for your dataset.
    FINETUNING_KWARGS = dict(
        name="bridge_dataset",
        data_dir="",
        image_obs_keys={"primary": "image_0"},
        proprio_obs_keys={},
        language_key="language_instruction",
        action_proprio_normalization_type="normal",
        # We want to avoid normalizing the gripper
        action_normalization_mask=[True, True, True, True, True, True, False],
        # standardize_fn is dynamically loaded from a file
        standardize_fn=ModuleSpec.create(
            "crossformer.data.oxe.oxe_standardization_transforms:bridge_dataset_transform",
        ),
        # If the default data loading speed is too slow, try these:
        # "num_parallel_reads": 8,  # for reading from disk / GCS
        # "num_parallel_calls": 16,  # for initial dataset construction
    )

    # an example of how to add a new observation tokenizer and action head
    UPDATE_CONFIG = dict(
        model=dict(
            observation_tokenizers=dict(
                new_primary=ModuleSpec.create(
                    ImageTokenizer,
                    obs_stack_keys=["image_primary"],
                    task_stack_keys=["image_primary"],
                    task_film_keys=["language_instruction"],
                    encoder=ModuleSpec.create(ResNet26FILM),
                )
            ),
            heads=dict(
                new_single_arm=ModuleSpec.create(
                    L1ActionHead,
                    action_horizon=4,
                    action_dim=7,
                    num_preds=7,
                    pool_strategy="pass",
                    readout_key="readout_new_single_arm",
                    clip_pred=False,
                    loss_weight=1.0,
                    constrain_loss_dims=True,
                ),
            ),
            readouts=dict(new_single_arm=4),
        )
    )

    if mode == "full":
        frozen_keys = None
    elif mode == "head_only":
        frozen_keys = ("crossformer_transformer.*",)
    else:
        raise ValueError("Invalid mode")

    max_steps = FieldReference(50000)
    window_size = FieldReference(default=1)

    config = dict(
        # update_config=UPDATE_CONFIG, # uncomment this line to add new observation tokenizer and action head
        pretrained_path="hf://rail-berkeley/crossformer",
        pretrained_step=placeholder(int),
        batch_size=256,
        shuffle_buffer_size=10000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=1000,
        save_interval=1000,
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(
            project="crossformer_finetune",
            group=placeholder(str),
            entity=placeholder(str),
        ),
        dataset_kwargs=FINETUNING_KWARGS,
        modality=task,
        finetuning_mode=mode,
        head_name=head_name,
        window_size=window_size,
        optimizer=dict(
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=3e-4,
                warmup_steps=2000,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=0.01,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
            grad_accumulation_steps=None,  # if you are using grad accumulation, you need to adjust max_steps accordingly
        ),
        val_kwargs=dict(
            val_shuffle_buffer_size=1000,
            num_val_batches=16,
        ),
    )

    if task == "image_conditioned":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 1.0
    elif task == "language_conditioned":
        goal_relabeling_strategy = None
        keep_image_prob = 0.0
    elif task == "multimodal":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 0.5
    else:
        raise ValueError("Invalid modality")

    traj_transform_kwargs = dict(
        window_size=window_size,
        action_horizon=4,
        goal_relabeling_strategy=goal_relabeling_strategy,
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=keep_image_prob,
        ),
        # If the default data loading speed is too slow, try these:
        # num_parallel_calls=16,  # for less CPU-intensive ops
    )
    workspace_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )

    frame_transform_kwargs = dict(
        resize_size={
            "primary": (224, 224),  # workspace (3rd person) camera is at 224x224
        },
        image_augment_kwargs=dict(
            primary=workspace_augment_kwargs,
        ),
    )
    # If the default data loading speed is too slow, try these:
    config[
        "frame_transform_threads"
    ] = 16  # for the most CPU-intensive ops (decoding, resizing, augmenting)

    config["traj_transform_kwargs"] = traj_transform_kwargs
    config["frame_transform_kwargs"] = frame_transform_kwargs
    return ConfigDict(config)
