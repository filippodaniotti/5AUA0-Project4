import os
import numpy as np
from config import configurations, Config
from argparse import ArgumentParser, Namespace

from data.data import get_data, get_collator
from model import SleepStagingModel
from models.tiny_sleep_net import TinySleepNet

import lightning.pytorch as pl
import torch.nn as nn
import torch

def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed)

def run(config_name: str):
    cfg: Config = configurations[config_name]
    seed_everything(cfg.seed)
    train_loader, valid_loader, test_loader = get_data(
        root=cfg.data_dir,
        dataset=cfg.dataset,
        batch_size=cfg.batch_size,
        train_percentage=0.9,
        val_percentage=0.0,
        test_percentage=0.1,
        collate_fn=get_collator(
            sampling_rate=cfg.sampling_rate,
            in_channels=cfg.in_channels,
            epoch_duration=cfg.epoch_duration,
            low_resources=cfg.low_resources)
    )
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1., 1.5, 1., 1., 1.]))
    model = SleepStagingModel(TinySleepNet(cfg), criterion, cfg)
    logger = pl.loggers.TensorBoardLogger(cfg.logs_dir, config_name)
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        log_every_n_steps=cfg.log_iterations,
        logger=logger
    )
    
    trainer.test(model, test_loader)
    trainer.fit(model, train_loader)
    trainer.test(model, test_loader)

if __name__ == "__main__":
    parser = ArgumentParser(description="Perform a full train and evaluation experiment.")
    parser.add_argument(
        "-c",
        "--config-name",
        dest="config_name",
        type=str,
        required=True,
        help="the configuration to be used in the experiment"
    )
    args = parser.parse_args()
    if args.config_name not in configurations.keys():
        raise ValueError(f"Configuration {args.config_name} not found. Check 'config.py'.")
    run(args.config_name)
    
   
   