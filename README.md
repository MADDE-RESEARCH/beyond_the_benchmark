# Deepfake Detection

This repository contains the code for the MADDE Capstone project.

## 1. Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/MADDE-RESEARCH/madde.git
cd madde
```

### Step 2: Set up the environment
Create a virtual env and install requirements;
```bash
python3 -m venv madde_env
source madde_env/bin/activate
pip install -r requirements.txt
```

### Step 3: Configure bash

```bash
echo 'export TF_CPP_MIN_LOG_LEVEL=2' >> ~/.bashrc # To turn off tensorflow's warning
echo 'export PATH=/opt/pytorch/bin:$PATH' >> ~/.bashrc # To download Kaggle dataset
echo 'export CUBLAS_WORKSPACE_CONFIG=:4096:8' >> ~/.bashrc # To make algorithm deterministic
source ~/.bashrc
```

### Step 4: Fetch dataset

```python
import kagglehub
import os

# Set target download directory
os.environ["KAGGLEHUB_CACHE_DIR"] = os.path.expanduser("~/madde/datasets")
# Download latest version
path = kagglehub.dataset_download("katsuyamucb/madde-dataset")

print("Path to dataset files:", path)
```

### Step 5: Login Wandb
Don't forget to log in to Wandb before running experiments;
```bash
wandb login <YOUR_API_KEY>
```

## 2. Running the One-shot fine-tuning


### 1. Classical machine learning approach (Linear/Non-Linear SVM)

Coming soon!

### 2.Deep Learning based approach
First, config your yaml file. Check real folders, fake folders, epochs and learning rate.
Then add '--config' to specify model you want to use!

```bash
python src/main.py --config=config/<filename>
```

Example:

```bash
source madde_env/bin/activate
python src/main.py --config=config/deepfake_full_tune.yaml
```

## 3. Batch Experiment

For batch-experiment, plesae run 'experiment.py'.
You need to pick a fine-tuning method by yaml file. and 
This time the real/fake folder in a yaml file will be ignored, and it will automatically be trained and tested on full dataset and all five leave-one-out combinations of our deepfake dataset.

```bash
tmux new -s experiment_fft
source madde_env/bin/activate
python src/experiment.py --config=config/deepfake_full_tune.yaml
```
```bash
Ctrl + b then d # For detaching the session
tmux a -t experiment_fft # For reopening the session
```
