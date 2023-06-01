from os.path import join
from dataclasses import dataclass

__all__ = ["configurations", "Config"]

@dataclass(frozen=True)
class Config:
    # model
    lr: float = 0.1
    lr_momentum: float = 0.9
    weight_decay: float = 1e-4
    num_classes: num_classes = 5
    optimizer: str = "sgd"
    sampling_rate: int = 100
    in_channels: int = 1
    rnn_hidden_size: int = 128
    
    # dataset
    data_dir: str = join("dataset", "sleepedfx", "sleep-cassette", "eeg_fpz_cz")
    
    # training
    epochs: int = 50
    log_iterations: int = 1
    batch_size: int = 15
    seq_len: int = 20
    
    # others
    low_resources: bool = True
    logs_dir: str = "experiments_logs"

configurations: dict[str, Config] = {
    "baseline": Config(
        low_resources=True
    ),
    "baseline_gpu": Config(
        low_resources=False
    ),
}