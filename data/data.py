import os
import re
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from torch import tensor

class SubjectDataset(Dataset):
    def __init__(
            self,
            root: str,
            subject_ids,
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
    
    def _get_subject_files(self, subject_id, files):
        raise NotImplementedError("This method should be implemented in a subclass.")

class SleepEDFxDataset(SubjectDataset):
    def __init__(
            self,
            root: str,
            subject_ids,
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
    
    def _get_subject_files(self, subject_id, files):
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
            subject_ids,
            ):
        super().__init__(root, subject_ids)
    
    def __len__(self) -> int:
        return super().__len__()
        
    def __getitem__(self, index) -> tuple[tensor, tensor, int]:
        file = self.files[index]
        with np.load(os.path.join(self.root, file)) as f:
            eeg = torch.tensor(f['x1'], dtype=torch.float32)
            ecg = torch.tensor(f['x2'], dtype=torch.float32)
            y = torch.tensor(f['y1'], dtype=torch.int64)
            
            sample = torch.cat((eeg.unsqueeze(1), ecg.unsqueeze(1)), dim=1)
            
            return sample, y
    
    def _get_subject_files(self, subject_id, files):
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
        is_test_set: bool = False):
    def collate_fn(batch: list[tuple[tensor, tensor]]):
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

def get_subject_ids(files, dataset):
    ids_boundaries = {
        "sleepedfx": (3, 5),
        "hmc": (2, 5)
    }
    start, end = ids_boundaries[dataset]
    return {file[start:end] for file in files}

def get_data(
        root: str,
        dataset: str,
        batch_size: int = 15,
        test_batch_size: int = 1,
        train_percentage: float = 0.8, 
        val_percentage: float = 0.1, 
        test_percentage: float = 0.1,
        train_collate_fn: callable = None, 
        test_collate_fn: callable = None, 
        seed: int = 42
        ):        
    if dataset not in ["sleepedfx", "hmc"]:
        raise ValueError(f"Dataset {dataset} not found. Check 'config.py'.")
    
    # get subject ids 
    files = [file for file in os.listdir(root) if file.endswith(".npz")]
    subject_ids = get_subject_ids(files, dataset)
    
    
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
            train_size=train_percentage / (train_percentage + val_percentage),
            test_size=val_percentage / (train_percentage + val_percentage),
            shuffle=True,
            random_state=seed
        )
    
    dataset_classes = {'sleepedfx': SleepEDFxDataset, 'hmc': HMCDataset}
    
    train_dataset = dataset_classes[dataset](root, train_subjects)
    test_dataset = dataset_classes[dataset](root, test_subjects)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=train_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=test_collate_fn)
    
    valid_loader = None
    if val_percentage > 0:
        valid_dataset = dataset_classes[dataset](root, valid_subjects)
        valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False, collate_fn=test_collate_fn)
    
    return train_loader, valid_loader, test_loader
    
if __name__ == "__main__":
    train, *_ = get_data(os.path.join("dataset", "hmc"), train_collate_fn=get_collator(sampling_rate=256, epoch_duration=1,   low_resources=128), test_collate_fn=get_collator(sampling_rate=256, epoch_duration=1,   low_resources=128), is_test_set=True)
    # inpsp = None 
    for idx, (inp, tar) in enumerate(train):
        print(type(inp))
        print(idx)
        print(inp.shape)
        print(tar.shape)
        print("")
        break
    #     if inp.shape != inpsp:
    #         inpsp = inp.shape
    #         print("shape: [bs, seq_len, sequences, fs*epoch_duration]")
    #         print(inp.shape, tar.shape)
    #     print("")