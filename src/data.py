import os
import re
import numpy as np
from math import ceil

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from torch import tensor

class SubjectDataset(Dataset):
    """Abstract Dataset class for an EDF dataset.
    This class should be subclassed when implementing a new dataset,
    as different dataset may require different data loading procedures.

    Args:
        root (str): The root directory of the dataset.
        subject_ids (set[str]): A set of subject IDs.

    Attributes:
        root (str): The root directory of the dataset.
        subject_ids (set[str]): A set of subject IDs.
        files_per_subject (dict): A dictionary mapping subject IDs to their corresponding files.
        files (list): A list of all files in the dataset.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(index): Retrieves an item from the dataset.
        _get_subject_files(subject_id, files): Retrieves the files associated with a subject ID.

    """
    def __init__(
            self,
            root: str,
            subject_ids: set[str],
        ):
        
        self.root = root
        self.subject_ids = subject_ids
        
        # get files per subject id
        files = [file for file in os.listdir(root) if file.endswith(".npz")]
        self.files_per_subject = {
            id_: self._get_subject_files(id_, files) for id_ in subject_ids
        }
        self.files = []
        for _, s_files in self.files_per_subject.items():
            self.files.extend(s_files)
    
    def __len__(self) -> int:
        return len(self.files)
        
    def __getitem__(self, index) -> tuple[tensor, tensor, int]:
        raise NotImplementedError("This method should be implemented in a subclass.")
    
    def _get_subject_files(self, subject_id: str, files: list[str]):
        raise NotImplementedError("This method should be implemented in a subclass.")

class SleepEDFxDataset(SubjectDataset):
    def __init__(
            self,
            root: str,
            subject_ids: set[str],
        ):
        
        super().__init__(root, subject_ids)
    
    def __len__(self) -> int:
        return super().__len__()
        
    def __getitem__(self, index) -> tuple[tensor, tensor, int]:
        file = self.files[index]
        with np.load(os.path.join(self.root, file)) as f:
            x = torch.tensor(f['x'], dtype=torch.float32)
            y = torch.tensor(f['y'], dtype=torch.int64)
            length = f['n_epochs'].item()
            
            return x.unsqueeze(1), y
    
    def _get_subject_files(self, subject_id: str, files: list[str]) -> list[str]:
        """Get a list of files storing each subject data."""
        # print(type(subject_id))

        # Pattern of the subject files from different datasets
        reg_exp = f"S[C|T][4|7]{str(subject_id).zfill(2)}[a-zA-Z0-9]+\.npz$"
        
        # Get the subject files based on ID
        subject_files = []
        for _, file in enumerate(files):
            pattern = re.compile(reg_exp)
            if pattern.search(file):
                subject_files.append(file)

        return subject_files
    
    
class HMCDataset(SubjectDataset):
    def __init__(
            self,
            root: str,
            subject_ids: set[str],
            selected_channels: list[str] = ["EEG C4-M1"],
            epoch_duration: int = 30,
            sampling_rate: int = 256,
        ):
        
        super().__init__(root, subject_ids)
        self.selected_channels = selected_channels
        self.epoch_duration = epoch_duration
        self.sampling_rate = sampling_rate
    
    def __len__(self) -> int:
        return super().__len__()
        
    def __getitem__(self, index) -> tuple[tensor, tensor, int]:
        file = self.files[index]
        
        with np.load(os.path.join(self.root, file)) as f:
            ch_labels = f['ch_label']
            channels: list[tensor] = []
            
            # prepare individual channels
            for channel in range(f['x'].shape[0]):
                if ch_labels[channel] in self.selected_channels:
                    x = torch.tensor(f['x'][channel], dtype=torch.float32)
                    
                    # strip channel length to multiple of epoch_duration
                    if x.shape[0] % self.epoch_duration != 0:
                        x = x[:x.shape[0] - x.shape[0] % self.epoch_duration]
                        
                    x = x.view(-1, self.sampling_rate * self.epoch_duration)
                    channels.append(x.unsqueeze(1))
                    
            if "HR" in self.selected_channels:
                hr = np.repeat(f['hr'], self.sampling_rate, axis=1)
                if hr.shape[0] % self.epoch_duration != 0:
                    hr = hr[:hr.shape[0] - hr.shape[0] % self.epoch_duration]
                hr = torch.tensor(hr, dtype=torch.float32) \
                    .view(-1, self.sampling_rate * self.epoch_duration) \
                    .unsqueeze(1)
                channels.append(hr)
            
            combined = torch.cat(channels, dim=1)
            
            y = torch.tensor(f['y'], dtype=torch.int64)
            if y.shape[0] % self.epoch_duration != 0:
                y = y[:y.shape[0] - y.shape[0] % self.epoch_duration]
                
            # aggregate labels at epoch level
            agg_labels: list[int] = []
            for idx in range(0, y.shape[0], self.epoch_duration):
                epoch_labels = y[idx : (idx+1)]
                agg_labels.append(epoch_labels.mode().values.item())
            labels = torch.tensor(agg_labels, dtype=torch.int64)
            
            return combined, labels
    
    def _get_subject_files(self, subject_id: str, files: list[str]) -> list[str]:
        """Get a list of files storing each subject data."""
        # Get the subject files based on ID
        subject_files = [f"SN{int(subject_id):03d}.npz"]

        return subject_files


    
def get_collator(
        seq_len: int = 20, 
        in_channels: int = 1,
        sampling_rate: int = 100,
        epoch_duration: int = 30,
        low_resources: int = 0,
        is_test_set: bool = False
    ) -> callable:
    """Closure to pass additional parameters to the collate function of the DataLoader.

    Args:
        seq_len (int, optional): Length of the sequence to feed to the RNN. Defaults to 20.
        in_channels (int, optional): Number of channels accepted by the model. Defaults to 1.
        sampling_rate (int, optional): Sampling rate of the data. Defaults to 100.
        epoch_duration (int, optional): Duration in seconds of a datapoint. Defaults to 30.
        low_resources (int, optional): Strip input tensor to given length. Defaults to 0 (no strip).
        is_test_set (bool, optional): Flag to indicate whether the collate function
            is to be used by a test dataloader. Defaults to False.

    Returns:
        callable: The collate function for the DataLoader.
    """
    
    def collate_fn(batch: list[tuple[tensor, tensor]]) -> tuple[tensor, tensor]:
        inputs = []
        targets = []
        
        for inp, tar in batch:
            # strip tensors to multiple of seq_len
            n_epochs = inp.shape[0]
            inp = inp[:n_epochs - n_epochs % seq_len, :]
            tar = tar[:n_epochs - n_epochs % seq_len]
            
            # reshape it to [seqs, seq_len, fs*epoch_duration]
            inp = inp.view(-1, seq_len, in_channels, sampling_rate * epoch_duration)
            tar = tar.view(-1, seq_len)
            
            if inp.shape[0] > 0: # throw away invalid samples    
                inp = [t.squeeze() for t in torch.chunk(inp, inp.shape[0], dim=0)]
                tar = [t.squeeze() for t in torch.chunk(tar, tar.shape[0], dim=0)]
                
                inputs.extend(inp)
                targets.extend(tar)
            
        if low_resources and len(inputs) > low_resources and not is_test_set:
            start = np.random.randint(0, len(inputs) - low_resources)
            inputs = inputs[start:start+low_resources]
            targets = targets[start:start+low_resources]      
        
        return torch.stack(inputs), torch.stack(targets)
    return collate_fn

def get_subject_ids(files: list[str], dataset: str) -> set[str]:
    ids_boundaries = {
        "sleepedfx": (3, 5),
        "hmc": (2, 5)
    }
    start, end = ids_boundaries[dataset]
    return {file[start:end] for file in files}

def get_data(
        root: str,
        dataset: str,
        epoch_duration: int = 30,
        selected_channels: list[str] = None,
        batch_size: int = 15,
        test_batch_size: int = 1,
        train_percentage: float = 0.8, 
        val_percentage: float = 0.1, 
        test_percentage: float = 0.1,
        train_collate_fn: callable = None, 
        test_collate_fn: callable = None, 
        seed: int = 42
    ) -> tuple[DataLoader, DataLoader, DataLoader]:        
    if dataset not in ["sleepedfx", "hmc"]:
        raise ValueError(f"Dataset {dataset} not found. Check 'config.py'.")
    
    # get subject ids 
    files = [file for file in os.listdir(root) if file.endswith(".npz")]
    subject_ids = get_subject_ids(files, dataset)
    
    # split on the subject level
    train_subjects, test_subjects = train_test_split(
        list(subject_ids),
        train_size=train_percentage + val_percentage,
        test_size=test_percentage,
        shuffle=True,
        random_state=seed
    )
    
    if val_percentage > 0:
        train_subjects, valid_subjects = train_test_split(
            list(train_subjects),
            train_size=ceil(train_percentage / (train_percentage + val_percentage)),
            test_size=ceil(val_percentage / (train_percentage + val_percentage)),
            shuffle=True,
            random_state=seed
        )
    
    dataset_classes = {'sleepedfx': SleepEDFxDataset, 'hmc': HMCDataset}
    
    if dataset == "hmc":
        train_dataset = dataset_classes[dataset](root, train_subjects, selected_channels, epoch_duration)
        test_dataset = dataset_classes[dataset](root, test_subjects, selected_channels, epoch_duration)
    else:
        train_dataset = dataset_classes[dataset](root, train_subjects)
        test_dataset = dataset_classes[dataset](root, test_subjects)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=train_collate_fn)
    test_loader = DataLoader(test_dataset, test_batch_size, shuffle=False, collate_fn=test_collate_fn)
    
    valid_loader = None
    if val_percentage > 0:
        valid_dataset = dataset_classes[dataset](root, valid_subjects)
        valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False, collate_fn=test_collate_fn)
    
    return train_loader, valid_loader, test_loader
