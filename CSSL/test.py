import torch
import numpy as np
from PIL import Image
from lightly.transforms import MoCoV2Transform

pretrain_transform = MoCoV2Transform(input_size=224)
image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)).convert("RGB")

out1, out2 = pretrain_transform(image)

print(out1.shape, out2.shape)