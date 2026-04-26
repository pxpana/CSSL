Troubleshooting
===============

Configuration path issues
-------------------------

- Ensure train.py uses names from config/model/ without the .yaml extension.
- Ensure plugin names map to a file in config/plugin/.

Runtime issues
--------------

- If GPU setup is unavailable, switch to CPU in config:

.. code-block:: yaml

   accelerator: "cpu"
   gpu_devices: []

- If out-of-memory occurs, lower train_batch_size.
- If training is unstable, reduce learning rate or disable mixed precision.

Tracking issues
---------------

- Set wandb: False if Weights and Biases tracking is not desired.

Reproducibility suggestions
---------------------------

- Use fixed seeds.
- Keep config files versioned with experiment logs.
- Record the git commit hash with each run.
