from os.path import join
from dataclasses import dataclass

from typing import Dict

__all__ = ["configurations", "Config"]

@dataclass(frozen=True)
class Config:
    # model
    learning_rate: float = 0.0001
    weight_decay: float = 1e-4
    num_classes: num_classes = 5
    sampling_rate: int = 100
    epoch_duration: int = 30
    n_in_channels: int = 1
    in_channels: list[str] = None
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
    
    # available channels in HMC dataset:
    # ["EEG C4-M1", "EEG F4-M1", "EEG O2-M1", "EEG C3-M2", "ECG"]
    "baseline_hmc": Config(
        data_dir=join("dataset", "hmc"),
        sampling_rate=256,
        batch_size=15,
        epoch_duration=1,
        low_resources=128,
        dataset="hmc",
        n_in_channels=1,
        in_channels=["EEG C4-M1"],
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
        epochs=200,
        sampling_rate=256,
        batch_size=15,
        epoch_duration=1,
        low_resources=2048,
        dataset="hmc",
        n_in_channels=1,
        in_channels=["EEG C4-M1"],
        padding_conv1=(2, 2),
        padding_max_pool1=(4, 4),
        padding_conv2=(3, 4),
        padding_max_pool2=(0, 1),
        kernel_sizes_conv1=5,
        kernel_sizes_max_pool1=16,
        strides_conv1=1,
        strides_max_pool1=4,
    ),
    "hmc_gpu_c4m1": Config(
        data_dir=join("dataset", "hmc"),
        epochs=200,
        sampling_rate=256,
        batch_size=15,
        epoch_duration=1,
        low_resources=2048,
        dataset="hmc",
        n_in_channels=1,
        in_channels=["EEG C4-M1"],
        padding_conv1=(2, 2),
        padding_max_pool1=(4, 4),
        padding_conv2=(3, 4),
        padding_max_pool2=(0, 1),
        kernel_sizes_conv1=5,
        kernel_sizes_max_pool1=16,
        strides_conv1=1,
        strides_max_pool1=4,
    ),
    "hmc_gpu_o2m1": Config(
        data_dir=join("dataset", "hmc"),
        epochs=200,
        sampling_rate=256,
        batch_size=15,
        epoch_duration=1,
        low_resources=2048,
        dataset="hmc",
        n_in_channels=1,
        in_channels=["EEG O2-M1"],
        padding_conv1=(2, 2),
        padding_max_pool1=(4, 4),
        padding_conv2=(3, 4),
        padding_max_pool2=(0, 1),
        kernel_sizes_conv1=5,
        kernel_sizes_max_pool1=16,
        strides_conv1=1,
        strides_max_pool1=4,
    ),
    "hmc_gpu_f4m1": Config(
        data_dir=join("dataset", "hmc"),
        epochs=200,
        sampling_rate=256,
        batch_size=15,
        epoch_duration=1,
        low_resources=2048,
        dataset="hmc",
        n_in_channels=1,
        in_channels=["EEG F4-M1"],
        padding_conv1=(2, 2),
        padding_max_pool1=(4, 4),
        padding_conv2=(3, 4),
        padding_max_pool2=(0, 1),
        kernel_sizes_conv1=5,
        kernel_sizes_max_pool1=16,
        strides_conv1=1,
        strides_max_pool1=4,
    ),
    "hmc_gpu_c3m2": Config(
        data_dir=join("dataset", "hmc"),
        epochs=200,
        sampling_rate=256,
        batch_size=15,
        epoch_duration=1,
        low_resources=2048,
        dataset="hmc",
        n_in_channels=1,
        in_channels=["EEG C3-M2"],
        padding_conv1=(2, 2),
        padding_max_pool1=(4, 4),
        padding_conv2=(3, 4),
        padding_max_pool2=(0, 1),
        kernel_sizes_conv1=5,
        kernel_sizes_max_pool1=16,
        strides_conv1=1,
        strides_max_pool1=4,
    ),
    "hmc_gpu_ecg": Config(
        data_dir=join("dataset", "hmc"),
        epochs=200,
        sampling_rate=256,
        batch_size=15,
        epoch_duration=1,
        low_resources=2048,
        dataset="hmc",
        n_in_channels=1,
        in_channels=["ECG"],
        padding_conv1=(2, 2),
        padding_max_pool1=(4, 4),
        padding_conv2=(3, 4),
        padding_max_pool2=(0, 1),
        kernel_sizes_conv1=5,
        kernel_sizes_max_pool1=16,
        strides_conv1=1,
        strides_max_pool1=4,
    ),
    "hmc_gpu_c4m1_e12": Config(
        data_dir=join("dataset", "hmc"),
        epochs=200,
        sampling_rate=256,
        epoch_duration=12,
        batch_size=15,
        low_resources=2048,
        dataset="hmc",
        n_in_channels=1,
        in_channels=["EEG C4-M1"],
        padding_conv1=(22, 22),
        padding_max_pool1=(2, 2),
        padding_conv2=(3, 4),
        padding_max_pool2=(0, 0),
        kernel_sizes_conv1=50,
        kernel_sizes_max_pool1=8,
        strides_conv1=6,
        strides_max_pool1=8,
    ),
    "hmc_gpu_c4m1_e30": Config(
        data_dir=join("dataset", "hmc"),
        epochs=200,
        sampling_rate=256,
        epoch_duration=30,
        batch_size=15 ,
        low_resources=32,
        dataset="hmc",
        n_in_channels=1,
        in_channels=["EEG C4-M1"],
        padding_conv1=(72, 72),
        padding_max_pool1=(13, 13),
        padding_conv2=(3, 4),
        padding_max_pool2=(0, 1),
        kernel_sizes_conv1=128,
        kernel_sizes_max_pool1=8,
        strides_conv1=16,
        strides_max_pool1=8,
    ),  
}