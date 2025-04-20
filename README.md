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
Turning off tensorflow's warning;
```bash
echo 'export TF_CPP_MIN_LOG_LEVEL=2' >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Login Wandb
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
python src/main.py --config=config/deepfake_full_tune.yaml
```

## 3. Batch Experiment

For batch-experiment, plesae run 'experiment.py'.
You need to pick a fine-tuning method by yaml file. and 
This time the real/fake folder in a yaml file will be ignored, and it will automatically be trained and tested on full dataset and all five leave-one-out combinations of our deepfake dataset.

```bash
python src/experiment.py --config=config/deepfake_full_tune.yaml
```