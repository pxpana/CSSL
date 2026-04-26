# Continual Self-Supervised Learning (CSSL)

This repository provides a PyTorch Lightning codebase for **Continual Self-Supervised Learning** with multiple SSL methods, class-incremental task splits, and optional replay-based plugins.

## Detailed Documentation

For a complete reference (setup, configs, scripts, outputs, troubleshooting), see [DOCUMENTATION.md](DOCUMENTATION.md).

## What This Repo Supports

### SSL methods

Configured under [config/model](config/model):

- Barlow Twins
- BYOL
- DINO
- MoCo v2+
- SimSiam
- SwAV
- VICReg

### Continual-learning plugins

Configured under [config/plugin](config/plugin):

- experience replay
- dark experience replay

## Project Layout

- [cssl](cssl): main package
- [cssl/dataset](cssl/dataset): datasets and continual scenario handling
- [cssl/framework](cssl/framework): framework components (for example CaSSLe)
- [cssl/loss](cssl/loss): loss functions
- [cssl/metrics](cssl/metrics): metrics and logging utilities
- [cssl/models](cssl/models): model modules
- [cssl/plugins](cssl/plugins): plugin implementations
- [cssl/utils](cssl/utils): data manager, factories, callbacks
- [train.py](train.py): main training entry point
- [pretrain.py](pretrain.py): pretraining/evaluation variant
- [tune.py](tune.py): Optuna hyperparameter tuning

## Quick Start

### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run training

```bash
python train.py --config simsiam_cifar_class
```

### 3. Common overrides

```bash
python train.py --config byol_cifar_class --num_tasks 10 --train_epochs 200
python train.py --config simsiam_cifar_class --plugin experience_replay
```

### 4. Tune hyperparameters

```bash
python tune.py --config byol_cifar_class --num_trials 25
```

## Outputs

Training artifacts are saved in:

- [logs](logs): linear, kNN, and NCM metrics
- [checkpoints](checkpoints): model checkpoints
- [lightning_logs](lightning_logs): PyTorch Lightning run logs
- [wandb](wandb): W&B local artifacts (when enabled)

## Config Notes

- Model configs are loaded from [config/model](config/model).
- Plugin configs are loaded from [config/plugin](config/plugin) when `--plugin` is set.
- You can override config keys from the command line.

## Presentation

Proposal defence slides: [CSSL Presentation](https://docs.google.com/presentation/d/1GZAxNqEZbV4wbk4tR_6SMvSf0MGnLgMEwihxRTXzaa8/edit?usp=sharing)
