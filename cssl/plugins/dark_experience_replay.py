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
            self.minibatch_size = config.minibatch_size

        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)



            buffer = None
            if not self.buffer.is_empty():
                buffer = self.buffer.get_data(
                    self.minibatch_size, 
                    transform=None,
                    device=self.device,
                )

            self.buffer.add_data(
                examples=batch
            )

            if buffer is not None:
                for i in range(len(batch)):
                    batch[i] = torch.cat((batch[i], buffer[i]), dim=0)

            return super().training_step(batch, batch_idx)
        
    return ExperienceReplay