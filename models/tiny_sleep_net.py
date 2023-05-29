import torch
import torch.nn as nn
import torch.nn.functional as F

class TinySleepNet(nn.Module):
    def __init__(self, num_classes, in_channels=1, sampling_rate=100):
        super().__init__()
        
        self.padding_edf = {  # same padding in tensorflow
            'conv1': (22, 22),
            'max_pool1': (2, 2),
            'conv2': (3, 4),
            'max_pool2': (0, 1),
        }
        
        
        self.sampling_rate = sampling_rate
        self.representation = nn.Sequential(
            *self._get_conv_block(
                in_channels=in_channels,
                out_channels=128,
                kernel_size=int(sampling_rate / 2),
                stride=int(sampling_rate / 16),
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
                hidden_size=128, 
                num_layers=1, 
                batch_first=True,
        )
        self.rnn_dropout = nn.Dropout(p=.5)
        
        # in the paper they softmax but yeah
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x, state):
        print(x.shape)
        print(x.view(15*20, 1, 3000).shape)
        # x = self.representation(x)
        x = self.representation(x.view(15*20, 1, 3000))
        # input of LSTM must be shape(seq_len, batch, input_size)
        # x = x.view(self.config['seq_length'], self.config['batch_size'], -1)
        print(x.shape)
        x = x.view(-1, 20, 2048)  # batch first == True
        assert x.shape[-1] == 2048
        print(x.shape)
        x, state = self.rnn(x, state)
        # x = x.view(-1, self.config['n_rnn_units'])
        x = x.reshape(-1, 128)
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
        
    def _init_hidden(self, batch_size):
        state = (torch.zeros(size=(1, batch_size, 128)),
                 torch.zeros(size=(1, batch_size, 128)))
        # state = (state[0].to(self.device), state[1].to(self.device))
        return state

if __name__ == "__main__":
    # m = nn.Conv1d(16, 33, 3, stride=2)
    # input = torch.randn(20, 16, 50)
    # output = m(input)
    # print(output.shape)   
    a = TinySleepNet(num_classes=5)
    print(a)

    