import torch
import lightly

def get_model(backbone, config, loggers):
    name = config.model_name.lower()
    if name == "simclr":
        from cssl.models import SimCLR
        model = SimCLR(backbone=backbone, config=config, loggers=loggers)
    elif name == "mocov2":
        from cssl.models import MoCov2
        model = MoCov2(backbone=backbone, config=config, loggers=loggers)
    elif name == "mocov2plus":
        from cssl.models import MoCov2Plus
        model = MoCov2Plus(backbone=backbone, config=config, loggers=loggers)
    elif name == "byol":
        from cssl.models import BYOL
        model = BYOL(backbone=backbone, config=config, loggers=loggers)
    elif name == "barlowtwins":
        from cssl.models import BarlowTwins
        model = BarlowTwins(backbone=backbone, config=config, loggers=loggers)
    elif name == "simsiam":
        from cssl.models import SimSiam
        model = SimSiam(backbone=backbone, config=config, loggers=loggers)
    elif name == "vicreg":
        from cssl.models import VICReg
        model = VICReg(backbone=backbone, config=config, loggers=loggers)
    elif name == "swav":
        from cssl.models import SwAV
        model = SwAV(backbone=backbone, config=config, loggers=loggers)

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

def get_checkpoint(trainer, backbone, args):
    name = args.model_name.lower()
    if name == "simclr":
        from cssl.models import SimCLR
        Module = SimCLR
    elif name == "mocov2":
        from cssl.models import MoCov2
        Module = MoCov2
    elif name == "mocov2plus":
        from cssl.models import MoCov2Plus
        Module = MoCov2Plus
    elif name == "byol":
        from cssl.models import BYOL
        Module = BYOL
    elif name == "barlowtwins":
        from cssl.models import BarlowTwins
        Module = BarlowTwins
    elif name == "simsiam":
        from cssl.models import SimSiam
        Module = SimSiam
    elif name == "vicreg":
        from cssl.models import VICReg
        Module = VICReg
    elif name == "swav":
        from cssl.models import SwAV
        Module = SwAV

    model = Module.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        backbone=backbone,
        config=args
    )

    return model
