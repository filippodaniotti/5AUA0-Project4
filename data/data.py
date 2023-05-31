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
        file = self.files[index]
        with np.load(os.path.join(self.root, file)) as f:
            x = torch.tensor(f['x'], dtype=torch.float32)
            y = torch.tensor(f['y'], dtype=torch.int64)
            length = f['n_epochs'].item()
            
            # # random crop to seq_len
            # if x.shape[0] > 20:
            #     start = np.random.randint(0, x.shape[0] - 20)
            #     x = x[start:start+20, :]
            #     y = y[start:start+20]
            
            # print(    f"input data shape: {x.shape}",)
            # print(    f"sampling rate: {f['fs']}",)
            # print(    f"file duration: {f['file_duration']}",)
            # print(    f"duration of a single epoch: {f['epoch_duration']}",)
            # print(    f"total number of epochs: {f['n_all_epochs']}",)
            # print(    f"actual number of epochs: {f['n_epochs']}",)
            return x, y
    
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
    
def get_collator(low_resources: bool = False):
    def collate_fn(batch: list[tuple[tensor, tensor]]):
        inputs = []
        targets = []
        
        for inp, tar in batch:
            # strip tensors to multiple of 20
            n_epochs = inp.shape[0]
            inp = inp[:n_epochs - n_epochs % 20, :]
            tar = tar[:n_epochs - n_epochs % 20]
            
            # reshape it to [seqs, seq_len, fs*epoch_duration]
            inp = inp.view(-1, 20, 3000)
            tar = tar.view(-1, 20)
            
            inp = [t.squeeze() for t in torch.chunk(inp, inp.shape[0], dim=0)]
            tar = [t.squeeze() for t in torch.chunk(tar, tar.shape[0], dim=0)]
            
            inputs.extend(inp)
            targets.extend(tar)   
            
        
        if low_resources and len(inputs) > 128:
            start = np.random.randint(0, len(inputs) - 128)
            inputs = inputs[start:start+128]
            targets = targets[start:start+128]
        
        return torch.stack(inputs), torch.stack(targets)

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
    
if __name__ == "__main__":
    train, *_ = get_data(os.path.join("dataset", "sleepedfx", "sleep-cassette", "eeg_fpz_cz"), collate_fn=get_collator(low_resources=True))
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