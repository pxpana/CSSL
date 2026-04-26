Project Overview
================

CSSL is a PyTorch Lightning codebase for continual self-supervised learning.

Supported SSL methods
---------------------

The repository includes model configurations for:

- Barlow Twins
- BYOL
- DINO
- MoCo v2+
- SimSiam
- SwAV
- VICReg

Continual-learning plugins
--------------------------

Available plugin configurations include:

- experience replay
- dark experience replay

Repository layout
-----------------

Key paths in the repository:

- cssl/: main package
- cssl/dataset/: datasets and continual scenarios
- cssl/framework/: framework-specific modules
- cssl/loss/: loss functions
- cssl/metrics/: evaluation and logging
- cssl/models/: model components
- cssl/plugins/: continual-learning plugins
- cssl/utils/: utilities, factories, callbacks, data manager
- config/model/: model YAML configuration files
- config/plugin/: plugin YAML configuration files
- train.py: main training script
- pretrain.py: alternative pretraining/evaluation flow
- tune.py: Optuna hyperparameter tuning
