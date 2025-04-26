import torch
from models import SimCLR, Classifier
from lightly.transforms import SimCLRTransform

def get_model(backbone, args):
    name = args.model_name.lower()
    if name == "simclr":
        model = SimCLR(backbone=backbone, config=args)
        pretrain_transform = SimCLRTransform()
        transform = lambda x: torch.stack(pretrain_transform(x))

        return model, transform
    else:
        assert 0

def get_classifier(backbone, num_classes, logger, args):
    classifier = Classifier(
        model=backbone,
        batch_size_per_device=args.test_batch_size,
        lr=args.test_learning_rate,
        feature_dim=args.feature_dim,
        num_classes=num_classes,
        logger=logger,
    )
    return classifier