# CSSL Repository Documentation

## Overview

This repository implements a **Continual Self-Supervised Learning (CSSL)** pipeline using PyTorch Lightning.

It supports multiple SSL frameworks and continual-learning plugins, and provides scripts for:

- continual training (`train.py`)
- pretraining/evaluation runs (`pretrain.py`)
- hyperparameter tuning with Optuna (`tune.py`)

## Supported Methods

Based on the configuration files in `config/model/`, the repository includes:

- Barlow Twins
- BYOL
- DINO
- MoCo v2+
- SimSiam
- SwAV
- VICReg

Plugin configurations in `config/plugin/` include:

- experience replay
- dark experience replay

## Repository Structure

Top-level structure (simplified):

- `cssl/`: main Python package
- `cssl/dataset/`: data handling and task/scenario construction
- `cssl/framework/`: framework-specific components (for example CaSSLe in `cassle.py`)
- `cssl/loss/`: SSL and auxiliary losses
- `cssl/metrics/`: logging and evaluation metrics
- `cssl/models/`: model definitions
- `cssl/plugins/`: plugin logic for continual learning behavior
- `cssl/utils/`: factories, data manager, callbacks, and helper utilities
- `config/model/`: model-specific YAML configs
- `config/plugin/`: plugin-specific YAML configs
- `train.py`: primary continual SSL training entry point
- `pretrain.py`: training/evaluation script variant
- `tune.py`: Optuna-based tuning workflow
- `requirements.txt`: Python dependencies
- `logs/`, `checkpoints/`, `lightning_logs/`, `wandb/`: experiment outputs

## Environment Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. (Optional) verify GPU setup

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Data

The repository currently contains datasets under `data/` (for example `cifar-100-python/` and `MNIST/`).

Training configuration chooses dataset and split behavior through model config fields such as:

- `dataset` (for example `cifar100`)
- `split_strategy` (for example `class`)
- `num_classes`
- `num_tasks`

## Running Experiments

## 1. Continual SSL training (recommended entry point)

Use a model config name (without extension) from `config/model/`.

```bash
python train.py --config simsiam_cifar_class
```

You can override any config key directly from CLI. Examples:

```bash
python train.py --config byol_cifar_class --num_tasks 10 --train_epochs 200
python train.py --config simsiam_cifar_class --plugin experience_replay
```

When `--plugin` is non-empty, `train.py` also loads plugin parameters from `config/plugin/<plugin>.yaml`.

## 2. Pretraining script variant

`pretrain.py` expects config files under `config/` and can be used for alternative runs:

```bash
python pretrain.py --config simclr_cifar_class
```

## 3. Hyperparameter tuning (Optuna)

```bash
python tune.py --config byol_cifar_class --num_trials 25
```

Tune results are stored in a SQLite study database under `logs/`.

## Configuration Guide

Model configs are YAML files in `config/model/` and typically define:

- experiment identity: `model_name`, `dataset`, `split_strategy`, `plugin`
- scenario setup: `num_classes`, `num_tasks`, `seeds`
- architecture sizes: `feature_dim`, projection/prediction dimensions
- training schedule: `train_batch_size`, `train_epochs`, `test_batch_size`, `test_epochs`
- optimizer settings: nested `optimizer` dict (learning rates, momentum, weight decay)
- logging/tracking: `wandb`, `wandb_project`
- augmentation controls: blur/solarization/color jitter parameters
- runtime: `accelerator`, `gpu_devices`, `precision`, `strategy`, `num_workers`

Plugin configs in `config/plugin/` define replay-specific parameters (for example `buffer_size`, `minibatch_size`, `alpha`).

## Outputs and Logging

### Local logs

Experiments create metric logs in `logs/`, with folders such as:

- `<Model>_linear_<dataset>_<num_tasks>`
- `<Model>_knn_<dataset>_<num_tasks>`
- `<Model>_ncm_<dataset>_<num_tasks>`

When plugins are enabled, plugin names are included in output folder naming.

### Checkpoints

Model checkpoints are stored under `checkpoints/` for compatible callbacks/configurations.

### PyTorch Lightning logs

Lightning outputs are written to `lightning_logs/`.

### Weights & Biases

If enabled in config (`wandb: True`), runs are tracked in W&B and local artifacts appear in `wandb/`.

## Python API Usage

The package exports `Trainer` from `cssl`:

```python
import cssl

trainer = cssl.Trainer("config/model/simsiam_cifar_class.yaml")
trainer.pretrain()
```

Note: the script entry points (`train.py`, `tune.py`) are currently the most complete and tested workflows.

## Troubleshooting

- If config loading fails, verify the config name/path and expected directory (`config/model/` vs `config/`).
- If plugin loading fails, ensure `--plugin` matches a YAML filename in `config/plugin/`.
- If training is slow or unstable, reduce `train_batch_size`, disable mixed precision, or lower learning rates.
- If GPU errors occur, set `accelerator: "cpu"` and `gpu_devices: []` for CPU-only execution.
- If W&B causes issues, set `wandb: False` in the model config.

## Reproducibility Tips

- Use explicit seeds in config (`seeds: [5]` or a list of seeds).
- Keep config files versioned with experiment runs.
- Record git commit hash alongside logs/checkpoints.

## Quick Start Checklist

1. Install dependencies from `requirements.txt`.
2. Pick a model config from `config/model/`.
3. Run `python train.py --config <config_name>`.
4. Inspect outputs in `logs/`, `checkpoints/`, and `lightning_logs/`.
