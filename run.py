import os

from data import get_data
from models.tiny_sleep_net import TinySleepNet

def run():
    train_loader, valid_loader, test_loader = get_data(
        os.path.join("data", "sleepedfx", "sleep-cassette", "eeg_fpz_cz")
    )
    
    net = TinySleepNet(5)

    state = net._init_hidden(batch_size=15)

    for inputs, targets in train_loader:
        inputs = inputs.unsqueeze(1) # add 1 channel dimension
        outputs, state = net(inputs, state)
    
if __name__ == "__main__":
    run()
    
   
   