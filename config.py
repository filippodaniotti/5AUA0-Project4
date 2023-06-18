from os.path import join
from dataclasses import dataclass

from typing import Dict

__all__ = ["configurations", "Config"]

@dataclass(frozen=True)
class Config:
    # model
    lr: float = 0.0001
    weight_decay: float = 1e-4
    num_classes: num_classes = 5
    sampling_rate: int = 100
    epoch_duration: int = 30
    in_channels: int = 1
    rnn_hidden_size: int = 128
    
    padding_conv1: tuple = (22, 22)
    padding_max_pool1: tuple = (2, 2)
    padding_conv2: tuple = (3, 4)
    padding_max_pool2: tuple = (0, 1)
    
    kernel_sizes_conv1: int = 50
    kernel_sizes_max_pool1: int = 8
    
    strides_conv1: int = 6
    strides_max_pool1: int = 8
    
    # dataset
    data_dir: str = join("dataset", "sleepedfx", "sleep-cassette", "eeg_fpz_cz")
    dataset: str = "sleepedfx"
    
    # training
    epochs: int = 50
    log_iterations: int = 1
    batch_size: int = 15
    test_batch_size: int = 1
    seq_len: int = 20
    
    # others
    low_resources: int = 0
    logs_dir: str = "experiments_logs"
    seed: int = 42

configurations: dict[str, Config] = {
    "baseline": Config(
        low_resources=128,
        epochs=10,
    ),
    "baseline_gpu": Config(
        low_resources=512,
        epochs=200,
    ),
    "baseline_hmc": Config(
        data_dir=join("dataset", "hmc"),
        sampling_rate=256,
        batch_size=15,
        epoch_duration=1,
        low_resources=128,
        dataset="hmc",
        in_channels=2,
        padding_conv1=(2, 2),
        padding_max_pool1=(4, 4),
        padding_conv2=(3, 4),
        padding_max_pool2=(0, 1),
        kernel_sizes_conv1=5,
        kernel_sizes_max_pool1=16,
        strides_conv1=1,
        strides_max_pool1=4,
    ),
    "baseline_hmc_gpu": Config(
        data_dir=join("dataset", "hmc"),
        sampling_rate=256,
        batch_size=15,
        epoch_duration=1,
        low_resources=2048,
        dataset="hmc",
        in_channels=2,
        padding_conv1=(2, 2),
        padding_max_pool1=(4, 4),
        padding_conv2=(3, 4),
        padding_max_pool2=(0, 1),
        kernel_sizes_conv1=5,
        kernel_sizes_max_pool1=16,
        strides_conv1=1,
        strides_max_pool1=4,
    )
}