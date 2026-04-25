# Continual Self-Supervised Learning (CSSL)

Proposal Defence Presentation: [CSSL](https://docs.google.com/presentation/d/1GZAxNqEZbV4wbk4tR_6SMvSf0MGnLgMEwihxRTXzaa8/edit?usp=sharing)


import cssl
trainer = cssl.Trainer(backbone="resnet18", ssl='simsiam', plugins="ewc")
trainer.fit(dataset='cifar10')
trainer.evaluate()

import cssl
trainer = cssl.Trainer(config="config/barlowtwins_cifar_class.yaml")
trainer.fit()
trainer.evaluate()
