import os
import numpy as np
import torch
from transformers import CLIPProcessor
import argparse
import wandb

from deep_learning.full_fine_tuner import FullFT
from deep_learning.frozen_fine_tuner import FrozenFT
from deep_learning.linear_tail_fine_tuner import LinearTailFT
from deep_learning.lora_fine_tuner import LoraFT
from deep_learning.bayes_tuner import BayesTune
from deep_learning.ln_fine_tuner import LNFT
from deep_learning.adapter_tuner import Adapter_FT
from deep_learning.vera_fine_tuner import VeraFT
from utils.config_parser import parse_cfg

from deepfake_detection.model.dfdet import DeepfakeDetectionModel
from deepfake_detection.config import Config

TUNER_CLASSES = {
    "FrozenFT": FrozenFT,
    "FullFT": FullFT,
    "LoraFT": LoraFT,
    "BayesFT": BayesTune,
    "LNFT": LNFT,
    'AdapterFT': Adapter_FT,
    "LinearTailFT": LinearTailFT,
    "VeraFT": VeraFT,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seed", default=42, type=int)
    cfg = parser.parse_args()
    cfg = parse_cfg(cfg, cfg.config)
    cfg.device = cfg.device if torch.cuda.is_available() else "cpu"

    # set seed
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True # Added this
    torch.backends.cudnn.benchmark = False  # Change this to False
    torch.use_deterministic_algorithms(True, warn_only=False)  # Added this
    torch.set_float32_matmul_precision("high")


    experiment_combinations = [
    # Stylegan
    {
        "dataset_type": "StyleGAN2",
        "real": ["Real_1_k_split"],
        "fake": ["StyleGAN2_split"],
        "real_test": ["Real_5_k_split"],
        "fake_test": ["StyleGAN2_split", "StableDiffusion_split",
                      "Midjourney_split", "firefly_split", "Dall-E_split"],
    },

    # Dall‑E
    {
        "dataset_type": "Dall‑E",
        "real": ["Real_2_k_split"],
        "fake": ["StyleGAN2_split", "Dall-E_split"],
        "real_test": ["Real_5_k_split"],
        "fake_test": ["StyleGAN2_split", "StableDiffusion_split",
                      "Midjourney_split", "firefly_split", "Dall-E_split"],
    },

    # Firefly
    {
        "dataset_type": "Firefly",
        "real": ["Real_3_k_split"],
        "fake": ["StyleGAN2_split", "Dall-E_split", "firefly_split"],
        "real_test": ["Real_5_k_split"],
        "fake_test": ["StyleGAN2_split", "StableDiffusion_split",
                      "Midjourney_split", "firefly_split", "Dall-E_split"],
    },

    # Midjourney
    {
        "dataset_type": "Midjourney",
        "real": ["Real_4_k_split"],
        "fake": ["StyleGAN2_split", "Dall-E_split", "firefly_split", "Midjourney_split"],
        "real_test": ["Real_5_k_split"],
        "fake_test": ["StyleGAN2_split", "StableDiffusion_split",
                      "Midjourney_split", "firefly_split", "Dall-E_split"],
    },

    # StableDiffusion
    {
        "dataset_type": "StableDiffusion",
        "real": ["Real_5_k_split"],
        "fake": ["StyleGAN2_split", "Dall-E_split", "firefly_split", "Midjourney_split", "StableDiffusion_split"],
        "real_test": ["Real_5_k_split"],
        "fake_test": ["StyleGAN2_split", "StableDiffusion_split",
                      "Midjourney_split", "firefly_split", "Dall-E_split"],
    },
]


    for combination in experiment_combinations:
        # sets model if necessary
        model = None
        preprocess = None
        if cfg.model_name == "deepfake_detection":
            model_path = "weights/model.ckpt"
            if not os.path.exists(model_path):
                print("Downloading model")
                os.makedirs("weights", exist_ok=True)
                os.system(f"wget https://huggingface.co/yermandy/deepfake-detection/resolve/main/model.ckpt -O {model_path}")
            ckpt = torch.load(model_path, map_location="cpu")

            model = DeepfakeDetectionModel(Config(**ckpt["hyper_parameters"]))
            model.load_state_dict(ckpt["state_dict"])

            # Get preprocessing function
            preprocess = model.get_preprocessing()

        # create config
        cfg.data_dir = os.path.join(os.getcwd(), "datasets")  # Ensure the data directory is correct -> datasets
        cfg.model = model if model is not None else None
        cfg.processor = preprocess if preprocess is not None else None

        # change the fake and real folder iteratively
        cfg.real_folder = combination['real']
        cfg.fake_folder = combination['fake']

        # initialize wandb
        project_name = "Fine-Tuning Experiment"
        mode = "online" if cfg.use_wandb else "disabled"
        group_name = cfg.tuning_type
        run_name = f"log_test_{cfg.model_name}_{cfg.tuning_type}_{combination['dataset_type']}"
        wandb.init(project=project_name, group=group_name, name=run_name, config=cfg, mode=mode)

        tuner_cls = TUNER_CLASSES.get(cfg.tuning_type)
        if tuner_cls is None:
            raise ValueError(f"Unsupported Tuning Type: {cfg.tuning_type}")
        tuner = tuner_cls(**vars(cfg))

        # set the leave-one-out test folder if necessary
        if len(combination['real']) == 1:
            tuner.set_TestFolder(combination['real_test'], combination['fake_test'])

        # let's go!
        tuned_model = tuner.Experiment()
