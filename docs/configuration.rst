Configuration
=============

Model configuration files
-------------------------

Model YAML files are located in config/model/ and define parameters such as:

- model_name, dataset, split_strategy, plugin
- num_classes, num_tasks, seeds
- feature and projection dimensions
- optimizer settings
- train/test batch sizes and epochs
- runtime settings (accelerator, devices, precision, strategy)

Plugin configuration files
--------------------------

Plugin YAML files are located in config/plugin/ and are loaded when --plugin is set.

Examples of plugin parameters:

- buffer_size
- minibatch_size
- alpha

CLI overrides
-------------

Most YAML parameters can be overridden from the command line.

Example:

.. code-block:: bash

   python train.py --config simsiam_cifar_class --num_tasks 10 --train_epochs 200
