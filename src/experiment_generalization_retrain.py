import os
import numpy as np
import torch
from transformers import CLIPProcessor
import argparse
from collections import deque
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
    
    # --test_split Train Validation   # → ["Train", "Validation"]
    parser.add_argument(
        "--test_split",
        nargs="+",
        default=["Test"], 
        choices=["Train", "Validation", "Test"],
        help="Which dataset splits to evaluate on",
    )

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

    # Create Experiment Combinations

    real_test = ["Real_5_k_split"]
    fake_test =  ["StyleGAN2_split", "StableDiffusion_split", "Midjourney_split", "firefly_split", "Dall-E_split"]
    real_combination = ["Real_1-5_k_split", "Real_2-5_k_split", "Real_3-5_k_split", "Real_4-5_k_split", "Real_5-5_k_split"]
    base_combination = cfg.experiment_order

    experiment_queue = deque(base_combination)
    num_cycles  = len(base_combination)

    # Iterate through the combinations in cyclic order
    for cycle_idx in range(num_cycles):
        print(f"\n===== Cycle {cycle_idx+1}/{num_cycles} =====")
        dataset_type = ""

        # Load the initial model only once
        model, preprocess = None, None
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
        
        for order, combination in enumerate(experiment_queue):
            # create config
            cfg.data_dir = os.path.join(os.getcwd(), "datasets")
            cfg.model = model
            cfg.processor = preprocess

            # change the fake and real folder iteratively
            cfg.real_folder = [real_combination[order]]
            cfg.fake_folder = [combination]
            dataset_type += combination.replace("_split", "", 1)+"_"

            # initialize wandb
            project_name = "Fine-Tuning Experiment"
            mode = "online" if cfg.use_wandb else "disabled"
            group_name = cfg.tuning_type
            run_name = f"genralization_{cfg.model_name}_{cfg.tuning_type}_{dataset_type}"
            wandb.init(project=project_name, group=group_name, name=run_name, config=cfg, mode=mode)

            tuner_cls = TUNER_CLASSES.get(cfg.tuning_type)
            if tuner_cls is None:
                raise ValueError(f"Unsupported Tuning Type: {cfg.tuning_type}")
            tuner = tuner_cls(**vars(cfg))

            tuner.set_TestFolder(real_test, fake_test)

            # let's go!
            tuned_model = tuner.Experiment(dataset_type, cfg.experiment_name, cycle_idx, order, cfg.test_split)
            model = tuned_model
            ckpt_path = f"experiments/models/{cfg.experiment_name}_{cycle_idx}_{order}_{cfg.tuning_type}_{dataset_type}.pth"
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)

        # Rotate the queue to the left
        experiment_queue.rotate(-1)
# Example _______________________________________________________________________________________________________________
# python src/experiment_generalization_retrain.py --config=config/lora_full_tune.yaml --test_split Test