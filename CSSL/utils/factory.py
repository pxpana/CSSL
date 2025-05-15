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
    elif name == "byol":
        from models import BYOL
        model = BYOL(backbone=backbone, config=args)
    elif name == "barlowtwins":
        from models import BarlowTwins
        model = BarlowTwins(backbone=backbone, config=args)
    elif name == "swav":
        from models import SwAV
        model = SwAV(backbone=backbone, config=args)

    return model


def get_classifier(backbone, num_classes, logger, args):
    from models import Classifier

    classifier = Classifier(
        model=backbone,
        batch_size_per_device=args.test_batch_size,
        lr=args.classifier_learning_rate,
        feature_dim=args.feature_dim,
        num_classes=num_classes,
        logger=logger,
    )
    return classifier

def get_pretrain_transform(args):
    name = args.model_name.lower()
    if name == "simclr":
        from lightly.transforms import SimCLRTransform
        pretrain_transform = SimCLRTransform(input_size=args.image_dim)
        transform = lambda x: torch.stack(pretrain_transform(x))
    elif name == "mocov2":
        from lightly.transforms import MoCoV2Transform
        pretrain_transform = MoCoV2Transform(input_size=args.image_dim)
        transform = lambda x: torch.stack(pretrain_transform(x))
    elif name in ["byol", "barlowtwins"]:
        from lightly.transforms import BYOLTransform
        pretrain_transform = BYOLTransform(
            lightly.transforms.BYOLView1Transform(input_size=args.image_dim),
            lightly.transforms.BYOLView2Transform(input_size=args.image_dim)
            )
        transform = lambda x: torch.stack(pretrain_transform(x))
    elif name == "swav":
        from lightly.transforms import SwaVTransform
        pretrain_transform = SwaVTransform(crop_sizes=args.crop_sizes)
        transform = lambda x: torch.stack(pretrain_transform(x))
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
    elif name == "byol":
        from models import BYOL
        Module = BYOL
    elif name == "barlowtwins":
        from models import BarlowTwins
        Module = BarlowTwins
    elif name == "swav":
        from models import SwAV
        Module = SwAV

    model = Module.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        backbone=backbone,
        config=args
    )

    return model