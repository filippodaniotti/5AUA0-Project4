# Deep Automatic Sleep Staging

Sleep staging is important in the diagnosis of numerous
health-related problems. Manual sleep staging, which is
done by professionals reviewing polysomnography results,
is time-consuming and error-prone. This calls for the need
of process automation. Different models have already been
proposed using e.g. EEG data. This study consists of a com-
parison between different model adaptation strategies and
different types of input data when using the same model.
Used channels are four different EEG leads, ECG and heart
rate. The results demonstrate that the temporal resolution
of the data is crucial for learning the time-invariant rep-
resentation, that EEG channels are more suited than ECG
and heart rate for training the model and lastly, that com-
bining multiple EEG channels can further increase model
performance, achieving accuracy scores up to 77%.

## Requirements

This code has been tested with the following versions:

- `python` == 3.10
- `pytorch` == 2.0.1
- `torchmetrics` == 0.11.4
- `pytorch-lightning` == 2.0.2
- `numpy` == 1.24.3
- `pandas` == 2.0.1
- `matplotlib` == 3.7.1
- `tensorboard` == 2.0.1

The pre-processing utilities will require the following additional modules:

- `pyEDFlib` = 0.1.32
- `heartpy` = 1.2.7    

## Dataset

In order to prepare the datasets:

- Download the datasets
  - [Sleep-EDFx](https://physionet.org/content/sleep-edfx/1.0.0/)
    -   `wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/`
  - [Haaglanden Medisch Centrum](https://physionet.org/content/hmc-sleep-staging/1.1/)
    -  `wget -r -N -c -np https://physionet.org/files/hmc-sleep-staging/1.1/`
- Edit the pre-processing scripts to specify your local path to the datasets and destination folder, e.g. `./dataset/`
- Run the pre-processing scripts
  - Sleep-EDFx
    - `python3 pre-processing/prepare_sleepedf.py`
  - HMC
    -  `python3 pre-processing/prepare_hmc.py`
    -  `python3 pre-processing/extract_hr.py`

## Training and testing the network

To perform a full training and evaluation experiment, run:
```bash
python train.py -c baseline
```
This will train the network with the `baseline` configuration. The available configuratiosn are defined in the `confugurations` dictionary in the `src/config.py` files.

To visualize training curves and the test accuracy of the results, you can use `tensorboard`:
```bash
tensorboard --logdir=experiments_logs/
```
The results will be available at `http://localhost:6006/`.

In the `notebooks/results.ipynb` notebook, you can find the code to generate further metrics to evaluate the results of the experiments, e.g. per-class precision, recall, f1 and confusion matrices.

## Config and hyperparameters
The configuration is defined in the `src/config.py` file. The available configurations are defined in the `configurations` dictionary. The `baseline` configuration is defined as follows:

```python
class Config:
    # model
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_classes: int = 5
    sampling_rate: int = 100
    epoch_duration: int = 30
    n_in_channels: int = 1
    # available channels in HMC dataset:
    # ["EEG C4-M1", "EEG F4-M1", "EEG O2-M1", "EEG C3-M2", "ECG", "HR"]
    in_channels: list[str] = None
    rnn_hidden_size: int = 128
    
    # architecture
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
    ```