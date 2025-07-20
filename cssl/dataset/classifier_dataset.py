import torch
from collections import defaultdict
from torch.utils.data.dataset import Dataset
from PIL import Image

class ClassifierDataset(Dataset):
    def __init__(self, data, transform, tasks=None):
        
        self.data = data
        self.transform = transform

        self.label_to_tasks = None
        if tasks is not None:
            self.label_to_tasks = self.create_label_task_map(tasks)

        self.targets = [item[1] for item in data]

    def __getitem__(self, idx):
        image = self.data[idx][0]
        label = self.data[idx][1]

        transformed = self.transform(image)

        if self.label_to_tasks is not None:
            # If label_to_tasks is provided, return the tasks for the label
            task = self.label_to_tasks[label]

            return transformed, label, task
        
        return transformed, label

    def __len__(self):
        return len(self.data)

    def create_label_task_map(self, tasks):
        """
        Args:
            tasks: Tensor of shape [num_tasks, num_classes_per_task]
        Returns:
            Dictionary {label: [task_indices]} mapping each label to list of tasks containing it
        """
        label_to_tasks = {}
        
        for task in range(len(tasks)):
            for label in tasks[task]:
                label_to_tasks[label.item()] = task
        
        return label_to_tasks