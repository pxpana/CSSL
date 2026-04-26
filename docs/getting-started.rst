Getting Started
===============

Prerequisites
-------------

- Python 3.10+
- CUDA-enabled GPU (optional but recommended)

Installation
------------

Create and activate a virtual environment, then install dependencies:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt

Quick training run
------------------

Run a baseline continual SSL experiment:

.. code-block:: bash

   python train.py --config simsiam_cifar_class

Useful overrides
----------------

.. code-block:: bash

   python train.py --config byol_cifar_class --num_tasks 10 --train_epochs 200
   python train.py --config simsiam_cifar_class --plugin experience_replay

Hyperparameter tuning
---------------------

.. code-block:: bash

   python tune.py --config byol_cifar_class --num_trials 25

Python API note
---------------

The script entry points are currently the most complete workflow paths.
