import torch
from cssl.utils import Buffer

def experience_replay(Base):

    class ExperienceReplay(Base):
        def __init__(self, backbone, config=None, loggers=None):
            super().__init__(backbone, config, loggers)

            self.buffer = Buffer(
                buffer_size=config.buffer_size,
                device=self.device,
            )
            self.queue_size = config.queue_size

        def training_step(self, batch, batch_idx):
            batch_size = len(batch[0])
            for i in range(len(batch)):
                view = batch[i]

                if not self.buffer.is_empty():
                    buffer_view = self.buffer.get_data(
                        self.queue_size, 
                        transform=self.dataloader.dataset.transform
                    )

                self.buffer.add_data(
                    examples=view
                )

                if not self.buffer.is_empty():
                    batch[i] = torch.cat((view, buffer_view))

            return super().training_step(batch, batch_idx)
        
    return ExperienceReplay