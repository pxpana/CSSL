import torch
import lightly

def get_model(backbone, args):
    name = args.model_name.lower()
    if name == "simclr":
        from models import SimCLR
        model = SimCLR(backbone=backbone, config=args)
    elif name == "mocov2":
        from models import MoCov2
        model = MoCov2(backbone=backbone, config=args)
    elif name == "mocov2plus":
        from models import MoCov2Plus
        model = MoCov2Plus(backbone=backbone, config=args)
    elif name == "byol":
        from models import BYOL
        model = BYOL(backbone=backbone, config=args)
    elif name == "barlowtwins":
        from models import BarlowTwins
        model = BarlowTwins(backbone=backbone, config=args)
    elif name == "simsiam":
        from models import SimSiam
        model = SimSiam(backbone=backbone, config=args)
    elif name == "vicreg":
        from models import VICReg
        model = VICReg(backbone=backbone, config=args)
    elif name == "swav":
        from models import SwAV
        model = SwAV(backbone=backbone, config=args)

    return model


def get_classifier(backbone, num_classes, logger, args):
    from models import Classifier

    classifier = Classifier(
        model=backbone,
        batch_size_per_device=args.test_batch_size,
        lr=args.optimizer["classifier_learning_rate"],
        feature_dim=args.feature_dim,
        num_classes=num_classes,
        logger=logger,
    )
    return classifier

def get_pretrain_transform(args):
    name = args.model_name.lower()
    pretrain_collate_function=None
    if name == "simclr":
        from lightly.transforms import SimCLRTransform
        transform = SimCLRTransform(input_size=args.image_dim)
    elif name == "mocov2plus":
        from lightly.transforms import MoCoV2Transform
        transform = MoCoV2Transform(input_size=args.image_dim, gaussian_blur=0.5)
    elif name in ["byol", "barlowtwins"]:
        from lightly.transforms import BYOLTransform
        transform = BYOLTransform(
            lightly.transforms.BYOLView1Transform(input_size=args.image_dim),
            lightly.transforms.BYOLView2Transform(input_size=args.image_dim)
            )
    elif name == "simsiam":
        from lightly.transforms import SimSiamTransform
        transform = SimSiamTransform(input_size=args.image_dim)
    elif name == "vicreg":
        from lightly.transforms import VICRegTransform
        transform = VICRegTransform(input_size=args.image_dim)
    elif name == "swav":
        from lightly.transforms import SwaVTransform
        transform = SwaVTransform(crop_sizes=args.crop_sizes)

    else:
        assert 0
    return transform

def get_checkpoint(trainer, backbone, args):
    name = args.model_name.lower()
    if name == "simclr":
        from models import SimCLR
        Module = SimCLR
    elif name == "mocov2":
        from models import MoCov2
        Module = MoCov2
    elif name == "mocov2plus":
        from models import MoCov2Plus
        Module = MoCov2Plus
    elif name == "byol":
        from models import BYOL
        Module = BYOL
    elif name == "barlowtwins":
        from models import BarlowTwins
        Module = BarlowTwins
    elif name == "simsiam":
        from models import SimSiam
        Module = SimSiam
    elif name == "vicreg":
        from models import VICReg
        Module = VICReg
    elif name == "swav":
        from models import SwAV
        Module = SwAV

    model = Module.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        backbone=backbone,
        config=args
    )

    return model
