from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from crossformer.data.oxe.oxe_dataset_mixes import OXE_NAMED_MIXES
from crossformer.data.utils.text_processing import UniversalSentenceEncoder
from crossformer.model.components.action_heads import L1ActionHead
from crossformer.model.components.tokenizers import ImageTokenizer, LowdimObsTokenizer
from crossformer.model.components.transformer import common_transformer_sizes
from crossformer.model.components.vit_encoders import ResNet26, ResNet26FILM
from crossformer.utils.spec import ModuleSpec
from crossformer.utils.train_utils import resnet_26_loader

BIMANUAL_ACTION_DIM = 14
SINGLE_ARM_ACTION_DIM = 7
NAV_ACTION_DIM = 2
QUADRUPED_ACTION_DIM = 12

HEAD_TO_DATASET = {
    "nav": ["omnimimic_gnm_dataset"],
    "single_arm": [
        "bridge_dataset",
        "fractal20220817_data",
        "kuka",
        "taco_play",
        "taco_extra",
        "jaco_play",
        "berkeley_cable_routing",
        "roboturk",
        "nyu_door_opening_surprising_effectiveness",
        "viola",
        "berkeley_autolab_ur5",
        "toto",
        "language_table",
        "stanford_hydra_dataset_converted_externally_to_rlds",
        "austin_buds_dataset_converted_externally_to_rlds",
        "nyu_franka_play_dataset_converted_externally_to_rlds",
        "furniture_bench_dataset_converted_externally_to_rlds",
        "austin_sailor_dataset_converted_externally_to_rlds",
        "austin_sirius_dataset_converted_externally_to_rlds",
        "bc_z",
        "dlr_edan_shared_control_converted_externally_to_rlds",
        "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
        "utaustin_mutex",
        "berkeley_fanuc_manipulation",
        "cmu_stretch",
        "droid",
        "droid_wipe",
        "droid_flip_pot_upright",
    ],
    "bimanual": [
        "aloha_pen_uncap_diverse_dataset",
        "aloha_new_sushi_dataset",
        "aloha_dough_cut_dataset",
        "aloha_lucy_dataset",
        "aloha_drawer_dataset",
        "aloha_pick_place_dataset",
        "aloha_static_dataset",
        "aloha_sushi_cut_full_dataset",
        "aloha_new_sushi_dataset,",
    ],
    "quadruped": ["go1_real_dataset", "a1", "go1"],
}


def get_config():
    window_size = FieldReference(default=5)

    return ConfigDict(
        dict(
            seed=42,
            num_steps=300000,
            save_dir="",
            model=get_model_config("detr"),
            window_size=window_size,
            dataset_kwargs=get_dataset_config("multi", window_size, 100),
            skip_norm_keys=["proprio_bimanual"],
            optimizer=dict(
                learning_rate=dict(
                    name="rsqrt",
                    init_value=0.0,
                    peak_value=3e-4,
                    warmup_steps=2000,
                    timescale=10000,
                ),
                weight_decay=0.1,
                clip_gradient=1.0,
                frozen_keys=tuple(),
            ),
            prefetch_num_batches=0,
            start_step=placeholder(int),
            log_interval=500,
            eval_interval=1,
            save_interval=1,
            val_kwargs=dict(
                val_shuffle_buffer_size=1000,
                num_val_batches=16,
            ),
            resume_path=placeholder(str),
            text_processor=ModuleSpec.create(UniversalSentenceEncoder),
            pretrained_loaders=(
                ModuleSpec.create(
                    resnet_26_loader,
                    restore_path="hf://rail-berkeley/ResNet-26-ImageNet",
                ),
            ),
            wandb=dict(
                project="crossformer",
                group=placeholder(str),
                entity=placeholder(str),
            ),
            wandb_resume_id=placeholder(str),
            eval_datasets=(),
        )
    )


def get_dataset_config(task_cond, window_size, action_horizon):
    traj_transform_kwargs, frame_transform_kwargs = get_augmentation_config(
        task_cond, window_size, action_horizon
    )

    mix = "cross_embodiment"
    assert all(
        [
            any([name in datasets for datasets in HEAD_TO_DATASET.values()])
            for name, weight in OXE_NAMED_MIXES[mix]
        ]
    ), "Dataset in mix doesn't have assigned head."

    return dict(
        oxe_kwargs=dict(
            data_mix=mix,
            data_dir="",
            load_camera_views=("primary", "high", "nav", "left_wrist", "right_wrist"),
            load_proprio=True,
            load_depth=False,
        ),
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=frame_transform_kwargs,
        batch_size=512,
        shuffle_buffer_size=50000,
        balance_weights=False,
        traj_transform_threads=48,
        traj_read_threads=48,
    )


def get_augmentation_config(task_cond, window_size, action_horizon):
    if task_cond == "image":
        keep_image_prob = 1.0
    elif task_cond == "lang":
        keep_image_prob = 0.0
    elif task_cond == "multi":
        keep_image_prob = 0.5
    else:
        raise ValueError("Invalid modality")

    traj_transform_kwargs = dict(
        window_size=window_size,
        action_horizon=action_horizon,
        max_action_dim=BIMANUAL_ACTION_DIM,
        head_to_dataset=HEAD_TO_DATASET,
        goal_relabeling_strategy="uniform",
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=keep_image_prob,
        ),
        subsample_length=100,
    )

    aloha_image_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.9, 1.0], ratio=[0.75, 4.0 / 3]),
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

    bridge_image_augment_kwargs = dict(
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
            "primary": (224, 224),
            "high": (224, 224),
            "nav": (224, 224),
            "left_wrist": (224, 224),
            "right_wrist": (224, 224),
        },
        image_augment_kwargs={
            "primary": bridge_image_augment_kwargs,
            "high": aloha_image_augment_kwargs,
            "nav": bridge_image_augment_kwargs,
            "left_wrist": aloha_image_augment_kwargs,
            "right_wrist": aloha_image_augment_kwargs,
        },
        num_parallel_calls=200,
    )
    return traj_transform_kwargs, frame_transform_kwargs


def get_model_config(transformer_size):
    token_embedding_size, transformer_kwargs = common_transformer_sizes(
        transformer_size
    )

    encoder = ModuleSpec.create(ResNet26FILM)
    return dict(
        observation_tokenizers=dict(
            primary=ModuleSpec.create(
                ImageTokenizer,
                obs_stack_keys=["image_primary"],
                task_stack_keys=["image_primary"],
                task_film_keys=["language_instruction"],
                encoder=encoder,
            ),
            high=ModuleSpec.create(
                ImageTokenizer,
                obs_stack_keys=["image_high"],
                task_stack_keys=["image_high"],
                task_film_keys=["language_instruction"],
                encoder=encoder,
            ),
            nav=ModuleSpec.create(
                ImageTokenizer,
                obs_stack_keys=["image_nav"],
                task_stack_keys=["image_nav"],
                task_film_keys=[],
                encoder=ModuleSpec.create(ResNet26),
            ),
            left=ModuleSpec.create(
                ImageTokenizer,
                obs_stack_keys=["image_left_wrist"],
                task_stack_keys=[],
                task_film_keys=["language_instruction"],
                encoder=encoder,
            ),
            right=ModuleSpec.create(
                ImageTokenizer,
                obs_stack_keys=["image_right_wrist"],
                task_stack_keys=[],
                task_film_keys=["language_instruction"],
                encoder=encoder,
            ),
            bimanual=ModuleSpec.create(
                LowdimObsTokenizer,
                obs_keys=["proprio_bimanual"],
                dropout_rate=0.2,
            ),
            quadruped=ModuleSpec.create(
                LowdimObsTokenizer,
                obs_keys=["proprio_quadruped"],
            ),
        ),
        task_tokenizers=dict(),
        heads=dict(
            bimanual=ModuleSpec.create(
                L1ActionHead,
                action_horizon=100,
                action_dim=BIMANUAL_ACTION_DIM,
                num_preds=BIMANUAL_ACTION_DIM,
                pool_strategy="pass",
                readout_key="readout_bimanual",
                clip_pred=False,
                loss_weight=1.0,
                constrain_loss_dims=True,
            ),
            single_arm=ModuleSpec.create(
                L1ActionHead,
                action_horizon=4,
                action_dim=SINGLE_ARM_ACTION_DIM,
                num_preds=SINGLE_ARM_ACTION_DIM,
                pool_strategy="pass",
                readout_key="readout_single_arm",
                clip_pred=False,
                loss_weight=1.0,
                constrain_loss_dims=True,
            ),
            nav=ModuleSpec.create(
                L1ActionHead,
                action_horizon=4,
                action_dim=NAV_ACTION_DIM,
                num_preds=NAV_ACTION_DIM,
                pool_strategy="pass",
                readout_key="readout_nav",
                clip_pred=False,
                loss_weight=1.0,
                constrain_loss_dims=True,
            ),
            quadruped=ModuleSpec.create(
                L1ActionHead,
                action_horizon=1,
                action_dim=QUADRUPED_ACTION_DIM,
                num_preds=QUADRUPED_ACTION_DIM,
                pool_strategy="pass",
                readout_key="readout_quadruped",
                clip_pred=False,
                loss_weight=1.0,
                constrain_loss_dims=True,
            ),
        ),
        readouts=dict(bimanual=100, single_arm=4, nav=4, quadruped=1),
        token_embedding_size=token_embedding_size,
        transformer_kwargs=transformer_kwargs,
        max_horizon=10,
    )
