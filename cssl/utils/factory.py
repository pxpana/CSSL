import torch
import lightly

def get_model(backbone, config, loggers):
    name = config.model_name.lower()
    if name == "simclr":
        from cssl.models import SimCLR
        Model = SimCLR
    elif name == "dclw":
        from cssl.models import DCLW
        Model = DCLW
    elif name == "mocov2":
        from cssl.models import MoCov2
        Model = MoCov2
    elif name == "mocov2plus":
        from cssl.models import MoCov2Plus
        Model = MoCov2Plus
    elif name == "byol":
        from cssl.models import BYOL
        Model = BYOL
    elif name == "barlowtwins":
        from cssl.models import BarlowTwins
        Model = BarlowTwins
    elif name == "simsiam":
        from cssl.models import SimSiam
        Model = SimSiam
    elif name == "vicreg":
        from cssl.models import VICReg
        Model = VICReg
    elif name == "swav":
        from cssl.models import SwAV
        Model = SwAV
    elif name == "dino":
        from cssl.models import DINO
        Model = DINO

    if isinstance(config.plugin, str):
        plugin_name = config.plugin.lower()
        if plugin_name == "experience_replay":
            from cssl.plugins import experience_replay
            Model = experience_replay(Model, config)

    model = Model(backbone=backbone, config=config, loggers=loggers)

    return model


def get_classifier(
    backbone, 
    num_classes, 
    logger,
    args
):
    from cssl.models import LinearClassifier
    linear_classifier = LinearClassifier(
        model=backbone,
        batch_size_per_device=args.test_batch_size,
        lr=args.optimizer["classifier_learning_rate"],
        feature_dim=args.feature_dim,
        num_classes=num_classes,
        logger=logger,
        num_tasks = args.num_tasks,
    )

    return linear_classifier
