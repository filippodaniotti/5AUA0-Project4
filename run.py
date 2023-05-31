import os
from config import Config

from data.data import get_data, get_collator
from wrapper import SleepStagingModel
from models.tiny_sleep_net import TinySleepNet

import lightning.pytorch as pl
import torch.nn as nn
import torch

def run():
    cfg = Config()
    train_loader, valid_loader, test_loader = get_data(
        root=cfg.data_dir,
        batch_size=cfg.batch_size,
        train_percentage=0.8,
        val_percentage=0.1,
        test_percentage=0.1,
        collate_fn=get_collator(low_resources=True)
    )
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1., 1.5, 1., 1., 1.]))
    model = SleepStagingModel(TinySleepNet(5), criterion, 5)
    logger = pl.loggers.TensorBoardLogger("lightning_logs", "baseline")
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        log_every_n_steps=cfg.log_iterations,
        logger=logger
    )
    
    trainer.test(model, test_loader)
    trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, test_loader)

if __name__ == "__main__":
    run()
    
   
   