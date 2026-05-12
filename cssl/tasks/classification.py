
from lightly.utils.benchmarking import LinearClassifier

from cssl.models import BaseClassifier

class Classification(
    LinearClassifier,
    BaseClassifier
):
    def __init__(self, backbone, config, loggers):
        self.classifier_name = "Linear"
        
        kwargs = {
            "model": backbone,
            "num_classes": config.num_classes,
            "feature_dim": config.feature_dim,
            "lr": config.optimizer["classifier_learning_rate"],
            "batch_size_per_device": config.test_batch_size,
        }
        
        self.config = config
        self.metrics_logger = loggers
        
        super().__init__(**kwargs)
        
        self.backbone = self.model
        
    
        
    