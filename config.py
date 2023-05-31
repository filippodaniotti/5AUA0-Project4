from os.path import join
from dataclasses import dataclass


@dataclass
class Config:
    batch_size = 15
    seq_len = 20
    # batch_size_test = 25
    lr = 0.1
    lr_momentum = 0.9
    weight_decay = 1e-4
    num_classes = 10
    data_dir = join("data", "sleepedfx", "sleep-cassette", "eeg_fpz_cz")
    epochs = 10
    log_iterations = 1
    enable_cuda = False
