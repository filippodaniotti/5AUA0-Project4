import os
import re
import numpy as np

import torch
from torch.utils.data import Dataset

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
    
    
class HMCDataset(Dataset):
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
            
        print(len(self.files))
    
    def __len__(self) -> int:
        return len(self.files)
    
        
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
        # print(type(subject_id))
        
        # Get the subject files based on ID
        subject_files = [f"SN{int(subject_id):03d}.npz"]

        return subject_files