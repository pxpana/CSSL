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
    if name == "simclr":
        from lightly.transforms import SimCLRTransform
        pretrain_transform = SimCLRTransform(input_size=args.image_dim)
        transform = lambda x: torch.stack(pretrain_transform(x))
    elif name == "mocov2":
        from lightly.transforms import MoCoV2Transform
        pretrain_transform = MoCoV2Transform(input_size=args.image_dim)
        transform = lambda x: torch.stack(pretrain_transform(x))
    elif name == "mocov2plus":
        from lightly.transforms import MoCoV2Transform
        pretrain_transform = MoCoV2Transform(input_size=args.image_dim, gaussian_blur=0.5)
        transform = lambda x: torch.stack(pretrain_transform(x))
    elif name in ["byol", "barlowtwins"]:
        from lightly.transforms import BYOLTransform
        pretrain_transform = BYOLTransform(
            lightly.transforms.BYOLView1Transform(input_size=args.image_dim),
            lightly.transforms.BYOLView2Transform(input_size=args.image_dim)
            )
        transform = lambda x: torch.stack(pretrain_transform(x))
    elif name == "simsiam":
        from lightly.transforms import SimSiamTransform
        pretrain_transform = SimSiamTransform(input_size=args.image_dim)
        transform = lambda x: torch.stack(pretrain_transform(x))
    elif name == "vicreg":
        from lightly.transforms import VICRegTransform
        pretrain_transform = VICRegTransform(input_size=args.image_dim)
        transform = lambda x: torch.stack(pretrain_transform(x))
    elif name == "swav":
        from lightly.transforms import SwaVTransform
        pretrain_transform = SwaVTransform(crop_sizes=args.crop_sizes)
        def transform(x):
            transformed = pretrain_transform(x)
            num_crops = len(transformed)
            global_crops, mini_crops = [], []
            for i in range(num_crops):
                if transformed[0].shape[-1]==max(args.crop_sizes):
                    global_crops.append(transformed.pop(0))
                elif transformed[0].shape[-1]==min(args.crop_sizes):
                    mini_crops.append(transformed.pop(0))

            global_crops = torch.stack(global_crops)
            mini_crops = torch.stack(mini_crops)
            
            return [global_crops, mini_crops]

        #transform = lambda x: torch.stack(pretrain_transform(x))
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