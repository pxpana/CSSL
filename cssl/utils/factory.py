import torch
import lightly

def get_model(backbone, args):
    name = args.model_name.lower()
    if name == "simclr":
        from cssl.models import SimCLR
        model = SimCLR(backbone=backbone, config=args)
    elif name == "mocov2":
        from cssl.models import MoCov2
        model = MoCov2(backbone=backbone, config=args)
    elif name == "mocov2plus":
        from cssl.models import MoCov2Plus
        model = MoCov2Plus(backbone=backbone, config=args)
    elif name == "byol":
        from cssl.models import BYOL
        model = BYOL(backbone=backbone, config=args)
    elif name == "barlowtwins":
        from cssl.models import BarlowTwins
        model = BarlowTwins(backbone=backbone, config=args)
    elif name == "simsiam":
        from cssl.models import SimSiam
        model = SimSiam(backbone=backbone, config=args)
    elif name == "vicreg":
        from cssl.models import VICReg
        model = VICReg(backbone=backbone, config=args)
    elif name == "swav":
        from cssl.models import SwAV
        model = SwAV(backbone=backbone, config=args)

    return model


def get_classifier(
    backbone, 
    num_classes, 
    loggers,
    classifier_type, 
    args
):
    if "linear" in classifier_type:
        from cssl.models import LinearClassifier
        linear_classifier = LinearClassifier(
            model=backbone,
            batch_size_per_device=args.test_batch_size,
            lr=args.optimizer["classifier_learning_rate"],
            feature_dim=args.feature_dim,
            num_classes=num_classes,
            logger=loggers["linear"],
        )
    if "knn" in classifier_type:
        from cssl.models import KNNClassifier
        knn_classifier = KNNClassifier(
            model=backbone,
            num_classes=num_classes,
            knn_k=200,
            knn_t=0.1,
            logger=loggers["knn"],
        )
    if "ncm" in classifier_type:
        from cssl.models import NCMClassifier
        ncm_classifier = NCMClassifier(
            model=backbone,
            num_classes=num_classes,
            knn_k=None,
            knn_t=None,
            logger=loggers["ncm"],
        )

    classifiers = {
        "linear": linear_classifier,
        "knn": knn_classifier,
        "ncm": ncm_classifier
    }
    return classifiers

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
