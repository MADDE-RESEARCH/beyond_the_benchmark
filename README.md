# Deepfake Detection

This repository contains the code for the MADDE Capstone project.

## 1. Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/MADDE-RESEARCH/madde.git
cd madde_project
```

### Step 2: Set up the environment

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

## 2. Running the Code



### 1. Classical machine learning approach (K-Nearest, SVM, Random Forest)

Coming soon

### 2.Deep Learning based approach

```bash
python src/main.py --config=config/<filename>
```

See `config` folder for the model that you want to use.

Example:

```bash
python src/main.py --config config/deepfake_full_tune.yaml
```