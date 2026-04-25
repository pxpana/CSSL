import cssl

trainer = cssl.Trainer(config_path="config/simclr_cifar_class.yaml")
trainer.pretrain()