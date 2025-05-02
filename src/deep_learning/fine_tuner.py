import os
import numpy as np
import random
import pandas as pd
import wandb
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from IPython.display import display
import itertools

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, f1_score
from utils.data_loader import DeepfakeDataset


class FineTuner:
    """
    A Class of fine-tuning.
    """

    def __init__(
        self,
        model_name,
        data_dir,
        real_folder,
        fake_folder,
        num_epochs,
        batch_size,
        learning_rate,
        model=None,
        processor=None,
        device = "cpu"
    ):
        self.model_name = model_name

        # Load the model using timm if the model is None
        if model == None:
            self.model = timm.create_model(
                self.model_name, pretrained=True, num_classes=2
            )
            print("Loaded ", model_name, " for fine-tuning")
        else:
            self.model = model

        # Unfreeze all layers at the beginning
        self.unfreeze_all_layers()

        self.data_dir = data_dir
        self.real_folder = real_folder
        self.fake_folder = fake_folder
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.test_real_folder = None
        self.test_fake_folder = None

        self.processor = processor
        self.device = device
        self.seed =42

        # Move the model to the specified device
        self.model = self.model.to(self.device)

    def set_TestFolder(self, test_real_folder, test_fake_folder):
        self.test_fake_folder = test_fake_folder
        self.test_real_folder = test_real_folder

    def seed_worker(self, worker_id):
        """Set random seed for dataloader workers"""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_Train_Val_loader(self, custom_batch_size=None):
        # Config batch size if necessary
        if custom_batch_size is None:
            batch_size = self.batch_size
        else:
            batch_size = custom_batch_size

        if self.processor:
            print('Using the pre-loaded processor...')
            train_dataset = DeepfakeDataset(
                root_dir=self.data_dir,
                real_folder=[os.path.join(folder, "Train") for folder in self.real_folder], 
                fake_folder=[os.path.join(folder, "Train") for folder in self.fake_folder],
                processor=self.processor,
            )

            val_dataset = DeepfakeDataset(
                root_dir=self.data_dir,
                real_folder=[os.path.join(folder, "Validation") for folder in self.real_folder],
                fake_folder=[os.path.join(folder, "Validation") for folder in self.fake_folder],
                processor=self.processor,
            )
        else:
            print('No processor found, using timm transforms...')
            # Get the config file from timm
            config = resolve_data_config({}, model=self.model)
            base_transform = create_transform(**config)
            # Data augmentation and normalization for training
            train_transform_list = base_transform.transforms
            train_transform_list.append(transforms.RandomHorizontalFlip())
            train_transform_list.append(transforms.RandomRotation(10))

            brightness = np.random.uniform(
                0.05, 0.2
            )  # Random value between 0.05 and 0.2... you can change if you want
            contrast = np.random.uniform(0.05, 0.2)
            saturation = np.random.uniform(0.05, 0.2)
            hue = np.random.uniform(0, 0.1)  # Hue is typically smaller values
            train_transform_list.append(
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                )
            )
            train_transform = transforms.Compose(train_transform_list)
            val_transform = base_transform

            # Create datasets -> updated to take list of folders
            train_dataset = DeepfakeDataset(
                root_dir=self.data_dir,
                real_folder=[os.path.join(folder, "Train") for folder in self.real_folder], 
                fake_folder=[os.path.join(folder, "Train") for folder in self.fake_folder],
                transform=train_transform,
            )
            val_dataset = DeepfakeDataset(
                root_dir=self.data_dir,
                real_folder=[os.path.join(folder, "Validation") for folder in self.real_folder],
                fake_folder=[os.path.join(folder, "Validation") for folder in self.fake_folder],
                transform=val_transform,
            )

        
        # To control workers
        g = torch.Generator()
        g.manual_seed(self.seed)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            worker_init_fn=self.seed_worker,
            generator=g,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            worker_init_fn=self.seed_worker,
            generator=g,
            pin_memory=True,
        )
        return train_loader, val_loader
    
    def expand_split_folders(self, root: str, test_split=["Test"]):
        """
        >> expand_split_folders("StyleGAN2_split")
        ['StyleGAN2_split/Train', 'StyleGAN2_split/Validation', 'StyleGAN2_split/Test']
        """
        return [os.path.join(self.data_dir, root, s) for s in test_split]
    

    def get_Test_loader(self, real_folder_roots, fake_folder_roots, test_split=["Test"], batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        real_folders = list(
            itertools.chain.from_iterable(self.expand_split_folders(r, test_split) for r in real_folder_roots)
        )
        fake_folders = list(
            itertools.chain.from_iterable(self.expand_split_folders(f, test_split) for f in fake_folder_roots)
        )

        if self.processor:
            dataset = DeepfakeDataset(
                root_dir=self.data_dir,
                real_folder=real_folders,
                fake_folder=fake_folders,
                processor=self.processor,
            )
        else:
            config = resolve_data_config({}, model=self.model)
            transform = create_transform(**config)
            dataset = DeepfakeDataset(
                root_dir=self.data_dir,
                real_folder=real_folders,
                fake_folder=fake_folders,
                transform=transform,
            )

        g = torch.Generator(); g.manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            worker_init_fn=self.seed_worker,
            generator=g,
            pin_memory=True,)

    def freeze_all_layers(self):
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_all_layers(self):
        # Unfreeze all layers
        for param in self.model.parameters():
            param.requires_grad = True
    
    def count_trainable_params(self):
        # Count trainable parameters
        c = 0
        for param in self.model.parameters():  
            if param.requires_grad:
                c += param.numel()
        return c

    def Tune(self):
        # function to override
        pass

    def Evaluation(self, test_split=["Test"]):
        """
        Runs evaluation on each fake test folder inside self.test_fake_folder, using self.test_real_folder as the real side.

        Returns:
            dict[str, pd.DataFrame]: mapping { fake_folder → classification report }
        """
        results = {}
        global_preds, global_probs, global_labels = [], [], []

        real_roots = self.test_real_folder
        fake_domains = self.test_fake_folder

        for fake_root in fake_domains:
            print(f"\n\n----- Evaluating on real={real_roots}  |  fake={fake_root} -----")

            loader = self.get_Test_loader(real_roots, [fake_root], test_split=test_split)

            self.model.eval()
            all_preds, all_probs, all_labels, all_paths = [], [], [], []

            with torch.no_grad():
                for inputs, labels, paths in tqdm(loader, desc=f"Testing [{fake_root}]"):
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs).logits_labels if self.model_name == "deepfake_detection" else self.model(inputs)
                    probs = torch.softmax(outputs.float(), dim=1).cpu().numpy()
                    preds = (probs[:, 1] >= 0.5).astype(int)

                    all_preds.extend(preds)
                    all_probs.extend(probs)
                    all_labels.extend(labels.numpy())
                    all_paths.extend(paths)

            all_preds = np.array(all_preds)
            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels)

            # Global
            global_preds.extend(all_preds)
            global_probs.extend(all_probs)
            global_labels.extend(all_labels)

            # Save the results
            report_df = pd.DataFrame(
                classification_report(all_labels, all_preds, target_names=["Real", "Fake"], output_dict=True)
            )
            acc = (all_preds == all_labels).mean()
            tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
            tpr = tp / (tp + fn)
            tnr = tn / (tn + fp)
            fpr = fp / (tn + fp)
            fnr = fn / (tp + fn)
            f1_macro = f1_score(all_labels, all_preds, average="macro")

            prefix = f"{fake_root}."
            wandb.log({
                f"{prefix}dataset": fake_root,
                f"{prefix}accuracy": acc,
                f"{prefix}f1_macro"  : f1_macro,
                f"{prefix}confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_labels,
                    preds=all_preds,
                    class_names=["Real", "Fake"],
                    title=f"{fake_root} Confusion",
                ),
                f"{prefix}roc_curve": wandb.plot.roc_curve(
                    y_true=all_labels,
                    y_probas=all_probs,
                    labels=["Real", "Fake"],
                    title=f"{fake_root} ROC",
                ),
                f"{prefix}tpr": tpr,
                f"{prefix}tnr": tnr,
                f"{prefix}fpr": fpr,
                f"{prefix}fnr": fnr,
            })

            results[fake_root] = report_df

        # ─── Combined‑domains metrics ─────────────────────────────────────────
        global_preds  = np.array(global_preds)
        global_probs  = np.array(global_probs)
        global_labels = np.array(global_labels)

        acc_all       = (global_preds == global_labels).mean()
        f1_macro_all  = f1_score(global_labels, global_preds, average="macro")
        f1_micro_all  = f1_score(global_labels, global_preds, average="micro")
        f1_weight_all = f1_score(global_labels, global_preds, average="weighted")

        tn, fp, fn, tp = confusion_matrix(global_labels, global_preds).ravel()
        tpr_all = tp / (tp + fn)
        tnr_all = tn / (tn + fp)
        fpr_all = fp / (tn + fp)
        fnr_all = fn / (tp + fn)

        # Log to WandB under the prefix "All."
        wandb.log({
            "ALL.accuracy"     : acc_all,
            "ALL.f1_macro"     : f1_macro_all,
            "ALL.f1_micro"     : f1_micro_all,
            "ALL.f1_weighted"  : f1_weight_all,
            "ALL.confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=global_labels,
                preds=global_preds,
                class_names=["Real", "Fake"],
                title="ALL Domains ‑ Confusion",
            ),
            "ALL.roc_curve": wandb.plot.roc_curve(
                y_true=global_labels,
                y_probas=np.array(global_probs),
                labels=["Real", "Fake"],
                title="ALL Domains ‑ ROC",
            ),
            "ALL.tpr" : tpr_all,
            "ALL.tnr" : tnr_all,
            "ALL.fpr" : fpr_all,
            "ALL.fnr" : fnr_all,
        })
        results["ALL"] = pd.DataFrame(
            {
                "accuracy"    : [acc_all],
                "f1_macro"    : [f1_macro_all],
                "f1_micro"    : [f1_micro_all],
                "f1_weighted" : [f1_weight_all],
                "tpr"         : [tpr_all],
                "tnr"         : [tnr_all],
                "fpr"         : [fpr_all],
                "fnr"         : [fnr_all],
            }, index=["ALL_combined"]
        )
        return results


    def log_wandb_train(
        self,
        all_labels,
        all_preds,
        all_probs,
        epoch,
        train_loss,
        train_acc,
        val_loss,
        val_acc,
        optimizer,
        best_val_acc = None,
    ):
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        # Extra logging for the last epoch
        if epoch == self.num_epochs - 1:
            # Confusion Matrix
            cm = confusion_matrix(all_labels, all_preds)
            wandb.log(
                {
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=all_labels,
                        preds=all_preds,
                        class_names=["Real", "Fake"],
                        title="Training Confusion Matrix",
                    ),
                    "best_val_accuracy": best_val_acc,
                    
                }
            )

            # ROC curve - use probs instead of preds
            all_probs = np.array(all_probs)
            fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
            wandb.log(
                {
                    "roc_curve": wandb.plot.roc_curve(
                        y_true=all_labels,
                        y_probas=all_probs,
                        labels=["Real", "Fake"],
                        title="Training ROC Curve",
                    )
                }
            )

    def Experiment(self, dataset_type, test_split=["Test"]):
        # Fine-tune the model
        tuned_model = self.Tune()

        # Run evaluation on all test folders
        if self.test_fake_folder is not None:
            eval_results = self.Evaluation(test_split)
        else:
            raise ValueError("No test_fake_folder set. Please set it using set_TestFolder().")

        # Log model parameter summary to WandB
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = self.count_trainable_params()
        wandb.log({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "percent_trainable": 100 * trainable_params / total_params,
        })

        # Log the model artifact
        model_artifact = wandb.Artifact(name=f"{dataset_type}-{self.model_name}", type="model")
        wandb.log_artifact(model_artifact)

        # Combine all reports in a single CSV
        combined_report = pd.concat(
            [df.assign(domain=domain) for domain, df in eval_results.items()],
            axis=0,
            keys=eval_results.keys(),
            names=["Domain", "Metric"]
        )
        combined_csv_path = f"/home/ec2-user/madde/experiments/results/_{dataset_type}_{self.model_name}_combined_report.csv"
        combined_report.to_csv(combined_csv_path)
        print(f"Saved combined evaluation report to {combined_csv_path}")

        wandb.save(combined_csv_path)
        wandb.finish()

        return tuned_model

