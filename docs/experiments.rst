Experiments and Outputs
=======================

Training entry points
---------------------

- train.py: primary continual SSL training flow
- pretrain.py: pretraining/evaluation script variant
- tune.py: Optuna-based hyperparameter search

Logs and artifacts
------------------

Typical output locations:

- logs/: linear, kNN, and NCM metrics
- checkpoints/: model checkpoints
- lightning_logs/: PyTorch Lightning artifacts
- wandb/: local Weights and Biases files

Common output naming pattern
----------------------------

Metrics folders often follow patterns such as:

- <Model>_linear_<dataset>_<num_tasks>
- <Model>_knn_<dataset>_<num_tasks>
- <Model>_ncm_<dataset>_<num_tasks>

When plugins are active, folder names can include plugin identifiers.
