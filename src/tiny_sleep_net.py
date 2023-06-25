import torch
import torch.nn as nn

from torch import tensor
from src.config import Config

class TinySleepNet(nn.Module):
    """Custom TinySleepNet implementation.
    Original paper: https://ieeexplore.ieee.org/document/9176741.

    Args:
        config (Config): The configuration object for the model.

    Attributes:
        cfg (Config): The configuration object for the model.
        kernel_sizes (dict): A dictionary of kernel sizes.
        strides (dict): A dictionary of strides.
        padding (dict): A dictionary of padding values.
        representation (nn.Sequential): The sequential module for the CNN encoder.
        rnn (nn.LSTM): The LSTM module for sequential learning (RNN encoder).
        rnn_dropout (nn.Dropout): The dropout layer for the LSTM outputs.
        fc (nn.Linear): The linear layer for the final classification.

    Methods:
        forward(x, state=None): Performs the forward pass of the model.
        _get_conv_block(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0): Returns a convolutional block.
        _init_hidden(): Initializes the hidden state for the LSTM.

    """
    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config
        
        self.kernel_sizes = {
            'conv1': self.cfg.kernel_sizes_conv1,
            'max_pool1': self.cfg.kernel_sizes_max_pool1,
        }
        
        self.strides = {
            'conv1': self.cfg.strides_conv1,
            'max_pool1': self.cfg.strides_max_pool1,
        }
        
        self.padding = {
            'conv1': self.cfg.padding_conv1,
            'max_pool1': self.cfg.padding_max_pool1,
            'conv2': self.cfg.padding_conv2,
            'max_pool2': self.cfg.padding_max_pool2
        }
        
        
        self.representation = nn.Sequential(
            *self._get_conv_block(
                in_channels=self.cfg.n_in_channels,
                out_channels=128,
                kernel_size=self.kernel_sizes['conv1'],
                stride=self.strides['conv1'],
                padding=self.padding['conv1'],
                ),
            nn.ConstantPad1d(self.padding['max_pool1'], 0),
            nn.MaxPool1d(self.kernel_sizes['max_pool1'], stride=self.strides['max_pool1']),
            nn.Dropout(p=.5),
            *self._get_conv_block(128, 128, 8, 1, self.padding['conv2']),
            *self._get_conv_block(128, 128, 8, 1, self.padding['conv2']),
            *self._get_conv_block(128, 128, 8, 1, self.padding['conv2']),
            nn.ConstantPad1d(self.padding['max_pool2'], 0),
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
    
    def forward(self, x: tensor, state = None) -> tuple[tensor, tuple[tensor, tensor]]:
        batch_length = x.shape[0]
        # reshape inputs as (batch_size * seq_len, n_in_channels, -1)
        x = x.view(batch_length * self.cfg.seq_len, self.cfg.n_in_channels, -1)
        x = self.representation(x)
        # reshape inputs as (batch_size, seq_len, 2048)
        x = x.view(-1, self.cfg.seq_len, 2048)  
        assert x.shape[-1] == 2048
        x, state = self.rnn(x, state)
        x = x.reshape(-1, self.cfg.rnn_hidden_size)
        x = self.rnn_dropout(x)
        x = self.fc(x)

        return x, state
    
    def _get_conv_block(
            self, 
            in_channels: int = 128, 
            out_channels: int = 128, 
            kernel_size: int = 8,
            stride: int = 1,
            padding: int = 0
        ) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
        
        return (
            nn.ConstantPad1d(padding, 0),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.BatchNorm1d(num_features=out_channels, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        
    def _init_hidden(self) -> tuple[tensor, tensor]:
        state = (torch.zeros(size=(1, self.cfg.batch_size, self.cfg.rnn_hidden_size)),
                 torch.zeros(size=(1, self.cfg.batch_size, self.cfg.rnn_hidden_size)))
        return state
