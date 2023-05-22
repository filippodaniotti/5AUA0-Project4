import os
import re
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from torch import tensor

def get_data(
        root: str,
        batch_size: int = 32,
        train_percentage: float = 0.8, 
        val_percentage: float = 0.1, 
        test_percentage: float = 0.1, 
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
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    
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
        
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index) -> tuple[tensor, tensor, int]:
        file = self.files[index]
        with np.load(os.path.join(self.root, file)) as f:
            x = torch.tensor(f['x'], dtype="float32")
            y = torch.tensor(f['y'], dtype="int32")
            fs = f['fs']
            
            return x, y, fs
    
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

    
if __name__ == "__main__":
    get_data(os.path.join("data", "sleep-edf-database-expanded-1.0.0", "sleep-cassette", "eeg_fpz_cz"))

