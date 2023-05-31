import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data.data import get_data

from models.tiny_sleep_net import TinySleepNet


def train():
    # Configuration settings
    cfg = Config()

    # Load dataset
    train_loader, valid_loader, test_loader = get_data(
        root=cfg.data_dir,
        batch_size=cfg.batch_size,
        train_percentage=0.8,
        val_percentage=0.1,
        test_percentage=0.1,
    )

    # Initialize network
    model = TinySleepNet(num_classes=5)
    model.train()

    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.lr_momentum, weight_decay=cfg.weight_decay)
    
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    accs = []
    print("Starting training...")
    pbar_epochs = tqdm(range(1, cfg.epochs + 1), position=0, leave=True)
    for epoch in pbar_epochs:
        cumulative_loss = 0.0
        cumulative_accuracy = 0.0
        state = model._init_hidden(batch_size=cfg.batch_size)
        pbar_batch = tqdm(range(len(train_loader)), position=0, leave=True)
        for _, (idx, (inputs, targets)) in zip(pbar_batch, enumerate(train_loader)):
            optimizer.zero_grad()
            batch_len = inputs.shape[0]
            outs, state = model(inputs, state)
            state = (state[0].detach(), state[1].detach())
            targets = targets.view(batch_len * cfg.seq_len)
            loss = criterion(outs, targets)

            cumulative_loss += loss.item()
            
            _, predictions = outs.max(1)
            cumulative_accuracy += predictions.eq(targets).sum() / (cfg.batch_size * cfg.seq_len) 

            # Apply back-propagation
            loss.backward()
            # Take one step with the optimizer
            optimizer.step()

            # if epoch % cfg.log_iterations == 0:
            #     if epoch == 0:
            #         loss_avg = running_loss
            #     else:
        loss_avg = cumulative_loss/cfg.batch_size
        acc_avg = cumulative_accuracy/cfg.batch_size * 100
        print("Epoch {} - Loss: {:.2f}".format(epoch, loss_avg))
        print("Epoch {} - Accuracy: {:.2f}".format(epoch, acc_avg))
        losses.append(loss_avg)
        accs.append(acc_avg)

    print("Finished training.")
    save_path = 'model.pth'
    torch.save(model.state_dict(), save_path)
    print("Saved trained model as {}.".format(save_path))


if __name__ == "__main__":
  train()
