from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2 as transforms

class BufferDataset(Dataset):
    def __init__(self, data, transform, task_id):
        
        self.data = data
        self.transform = transform
        self.task_id = task_id

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        image = self.data[idx][0]
        transformed = self.transform(image)
        image = self.image_transform(image)

        return image, self.task_id, transformed

    def __len__(self):
        return len(self.data)