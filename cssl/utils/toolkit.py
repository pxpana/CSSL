import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

from pytorch_lightning.loggers import WandbLogger

def get_callbacks_logger(args, training_type, task_id, scenario_id, project="CSSL"):
    task_id = task_id+1

    if training_type == "pretrain":
        monitor = "train_loss"
        mode = "min"
    elif training_type == "classifier":
        monitor = "Accuracy"
        mode = "max"

    callbacks = []
    # checkpoint_callback = ModelCheckpoint(
    #                         #monitor=monitor,
    #                         filename=f"{args.model_name}_{args.dataset}_{training_type}_scenario_{scenario_id}_task_{task_id}",           
    #                         #mode=mode,                    
    #                         #save_last=True,                 
    #                         verbose=True                  
    #                     )
    # callbacks.append(checkpoint_callback)
    #callbacks.append(PROGRESS_BAR)

    if args.wandb:
        wandb_logger = WandbLogger(
            name=f"{args.model_name}_{args.dataset}_{training_type}_scenario_{scenario_id}_task_{task_id}/{args.num_tasks}",
            #version=f"{args.model_name}_{args.dataset}_{training_type}2",
            group=f"scenario_{scenario_id}",
            config={"task_id": task_id, "scenario_id": scenario_id},
            log_model=False, 
            project=project
        )
    else:
        wandb_logger = None

    return callbacks, wandb_logger

PROGRESS_BAR = RichProgressBar(
    theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        progress_bar_pulse="#6206E0",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="#DB59A9",
        metrics_text_delimiter="\n",
        metrics_format=".3e",
    )
)

def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)
