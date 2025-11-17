import torch
from cssl.utils import Buffer

def experience_replay(Base, config):

    buffer = Buffer(
        buffer_size=config.buffer_size,
        device=None
    )

    class ExperienceReplay(Base):
        def __init__(self, backbone, config=None, loggers=None):
            super().__init__(backbone, config, loggers)

            self.buffer = buffer
            self.minibatch_size = config.minibatch_size

        def training_step(self, batch, batch_idx):
            images, tasks, transformed = batch[0], batch[1], batch[2]

            mini_batch = None
            if not self.buffer.is_empty() and tasks[0]>0:
                mini_batch = self.buffer.get_data(
                    self.minibatch_size, 
                    transform=self.trainer.train_dataloader.dataset.transform,
                    device=self.device,
                    current_task=tasks[0]
                )

            self.buffer.add_data(
                examples=images,
                task_labels=tasks,
                device=self.device
            )

            if tasks[0]>0:
                for i in range(len(transformed)):
                    transformed[i] = torch.cat((transformed[i], mini_batch[i]), dim=0)

            return super().training_step(transformed, batch_idx)
        
    return ExperienceReplay