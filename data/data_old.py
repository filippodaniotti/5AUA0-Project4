import os
import re
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from torch import tensor

def get_data(
        root: str,
        batch_size: int = 15,
        train_percentage: float = 0.8, 
        val_percentage: float = 0.1, 
        test_percentage: float = 0.1,
        collate_fn: callable = None, 
        ):        
    # get subject ids 
    files = [file for file in os.listdir(root) if file.endswith(".npz")]
    subject_ids = {file[3:5] for file in files}
    
    
    train_subjects, test_subjects = train_test_split(
        list(subject_ids),
        train_size=train_percentage + val_percentage,
        test_size=test_percentage,
        shuffle=True
    )
    train_subjects, valid_subjects = train_test_split(
        list(train_subjects),
        train_size=train_percentage / (train_percentage + val_percentage),
        test_size=val_percentage / (train_percentage + val_percentage),
        shuffle=True
    )
    
    train_dataset = SubjectDataset(root, train_subjects)
    valid_dataset = SubjectDataset(root, valid_subjects)
    test_dataset = SubjectDataset(root, test_subjects)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, valid_loader, test_loader


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
            
        self.sequences = []
        self.targets = []
        # such a waste of memory but there is no other way
        for file in self.files:
            with np.load(os.path.join(self.root, file)) as f:
                x = torch.tensor(f['x'], dtype=torch.float32)
                y = torch.tensor(f['y'], dtype=torch.int64)
                self.sequences.append(x)
                self.targets.append(y)
        
        self.sequences = pad_sequence(self.sequences, batch_first = True)
        self.targets = pad_sequence(self.targets, batch_first = True)
        
        # strip epochs to multiple of seq_len
        self.sequences = self.sequences[:, :self.sequences.shape[1] - self.sequences.shape[1] % 20, :]
        self.targets = self.targets[:, :self.targets.shape[1] - self.targets.shape[1] % 20]
        
        self.sequences = self.sequences.view(self.sequences.shape[0], self.sequences.shape[1]//20, 20, 3000)
        self.targets = self.targets.view(self.targets.shape[0], self.targets.shape[1]//20, 20)
        self.sequences = self.sequences.reshape(self.sequences.shape[0]*self.sequences.shape[1], 20, 3000)
        self.targets = self.targets.reshape(self.targets.shape[0]*self.targets.shape[1], 20)
        # self.sequences = self.sequences.view(self.sequences.shape[0], 20, self.sequences.shape[1]//20, 3000)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, index) -> tuple[tensor, tensor]:
        return self.sequences[index], self.targets[index]
        
    # def __getitem__(self, index) -> tuple[tensor, tensor, int]:
    #     file = self.files[index]
    #     with np.load(os.path.join(self.root, file)) as f:
    #         x = torch.tensor(f['x'], dtype=torch.float32)
    #         y = torch.tensor(f['y'], dtype=torch.int32)
    #         fs = f['fs']
            
            
    #         # print(    f"input data shape: {x.shape}",)
    #         # print(    f"sampling rate: {f['fs']}",)
    #         # print(    f"file duration: {f['file_duration']}",)
    #         # print(    f"duration of a single epoch: {f['epoch_duration']}",)
    #         # print(    f"total number of epochs: {f['n_all_epochs']}",)
    #         # print(    f"actual number of epochs: {f['n_epochs']}",)
    #         return x, y, fs
    
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
    
def collate_fn(batch: list[tuple[tensor, tensor, int]]):
    inputs = []
    targets = []
    sampling_rates = []
    
    for inp, tar, fs in batch:
        inputs.append(inp)
        targets.append(tar)
        sampling_rates.append(fs)
        
    # pad epochs in batch to same length
    inputs = pad_sequence(inputs, batch_first = True)
    targets = pad_sequence(targets, batch_first = True)
    
    # strip epochs to multiple of seq_len
    # inputs = inputs[:, :inputs.shape[1] - inputs.shape[1] % 20, :]
    # targets = targets[:, :targets.shape[1] - targets.shape[1] % 20]
    
    # reshape into [bs, seq_len, sequences, fs*epoch_duration]
    # inputs = inputs.view(inputs.shape[0], 20, inputs.shape[1]//20, 3000)
    # targets = targets.view(targets.shape[0], 20, targets.shape[1]//20)
    
    return inputs, targets, sampling_rates

    
if __name__ == "__main__":
    train, *_ = get_data(os.path.join("data", "sleepedfx", "sleep-cassette", "eeg_fpz_cz"))
    # inpsp = None 
    for idx, (inp, tar) in enumerate(train):
        print(idx)
        print(inp.shape, tar.shape)
    #     if inp.shape != inpsp:
    #         inpsp = inp.shape
    #         print("shape: [bs, seq_len, sequences, fs*epoch_duration]")
    #         print(inp.shape, tar.shape)
    #     print("")