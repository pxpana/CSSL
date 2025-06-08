import numpy as np
from avalanche.evaluation.metrics import TaskAwareAccuracy, Forgetting

class Logger():
    def __init__(self):
        self.logs = []

        self.task_id = None

        self.accuracy = TaskAwareAccuracy()
        self.forgetting = Forgetting()

    def log(self, data):
        self.accuracy.update(
            predicted_y=data["predicted"],
            true_y=data["targets"],
            task_labels=data["tasks"]
        )

    def results(self):
        accuracy = self.accuracy.result()

        self.forgetting.update(self.task_id, accuracy[self.task_id])
        

        accuracy = np.mean(list(accuracy.values()))

        return {
            "accuracy": accuracy
        } 

    def reset(self):
        self.accuracy.reset()
