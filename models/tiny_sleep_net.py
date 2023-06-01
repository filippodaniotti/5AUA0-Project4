import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch import tensor
from config import Config

class TinySleepNet(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config
        
        self.padding_edf = {  # same padding in tensorflow
            'conv1': (22, 22),
            'max_pool1': (2, 2),
            'conv2': (3, 4),
            'max_pool2': (0, 1),
        }
        
        self.representation = nn.Sequential(
            *self._get_conv_block(
                in_channels=self.cfg.in_channels,
                out_channels=128,
                kernel_size=int(self.cfg.sampling_rate / 2),
                stride=int(self.cfg.sampling_rate / 16),
                padding=self.padding_edf['conv1'],
                ),
            nn.ConstantPad1d(self.padding_edf['max_pool1'], 0),
            nn.MaxPool1d(8, stride=8),
            nn.Dropout(p=.5),
            *self._get_conv_block(128, 128, 8, 1, self.padding_edf['conv2']),
            *self._get_conv_block(128, 128, 8, 1, self.padding_edf['conv2']),
            *self._get_conv_block(128, 128, 8, 1, self.padding_edf['conv2']),
            nn.ConstantPad1d(self.padding_edf['max_pool2'], 0),
            nn.MaxPool1d(4, stride=4),
            nn.Flatten(),
            nn.Dropout(p=.5),
        )
        
        self.rnn = nn.LSTM(
                input_size=2048, 
                hidden_size=self.cfg.rnn_hidden_size, 
                num_layers=1, 
                batch_first=True,
        )
        self.rnn_dropout = nn.Dropout(p=.5)
        self.fc = nn.Linear(self.cfg.rnn_hidden_size, self.cfg.num_classes)
    
    def forward(self, x, state = None):
        batch_length = x.shape[0]
        x = x.view(batch_length * self.cfg.seq_len, -1).unsqueeze(1)
        x = self.representation(x)
        x = x.view(-1, self.cfg.seq_len, 2048)  # batch first == True
        assert x.shape[-1] == 2048
        # if state[0].shape[1] != bs:
        #     state = state[0][:, :bs, :], state[1][:, :bs, :]
        x, state = self.rnn(x, state)
        x = x.reshape(-1, self.cfg.rnn_hidden_size)
        # rnn output shape(seq_length, batch_size, hidden_size)
        x = self.rnn_dropout(x)
        x = self.fc(x)

        return x, state
    
    def _get_conv_block(
            self, 
            in_channels = 128, 
            out_channels = 128, 
            kernel_size = 8,
            stride = 1,
            padding = 0):
        return (
            nn.ConstantPad1d(padding, 0),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
            ),
            nn.BatchNorm1d(num_features=out_channels, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        
    def _init_hidden(self):
        state = (torch.zeros(size=(1, self.cfg.batch_size, self.cfg.rnn_hidden_size)),
                 torch.zeros(size=(1, self.cfg.batch_size, self.cfg.rnn_hidden_size)))
        return state

if __name__ == "__main__":
    # m = nn.Conv1d(16, 33, 3, stride=2)
    # input = torch.randn(20, 16, 50)
    # output = m(input)
    # print(output.shape)   
    a = TinySleepNet(num_classes=5)
    print(a)

    