from torch.utils.data.dataset import Dataset
from PIL import Image

class PretrainDataset(Dataset):
    def __init__(self, data, transform):
        
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        image = self.data[idx][0]

        transformed = self.transform(image)

        return transformed

    def __len__(self):
        return len(self.data)