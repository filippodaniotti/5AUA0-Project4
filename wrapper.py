import lightning.pytorch as pl
import torch
from torch import nn, tensor
from torchmetrics import Accuracy

from models.tiny_sleep_net import TinySleepNet
from sklearn.metrics import classification_report
from data.prepare_sleepedf import label2ann

class SleepStagingModel(pl.LightningModule):
    def __init__(
            self, 
            model: nn.Module, 
            cost_function: nn.Module,
            num_classes: int = 5):
        super().__init__()

        self.model = model    
        self.cost_fn = cost_function
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        
    def forward(self, x: tensor) -> tensor:
        return self.model(x)

    def training_step(self, batch: tuple[tensor, tensor], batch_idx: int):
        inputs, targets = batch
        loss, acc, _ = self._compute_loss_and_accuracy(inputs, targets)
        metrics = self._print_metrics(loss.item(), acc.item(), "Train")
        return loss

    def validation_step(self, batch: tuple[tensor, tensor], batch_idx: int):
        inputs, targets = batch
        loss, acc, _ = self._compute_loss_and_accuracy(inputs, targets)
        metrics = self._print_metrics(loss.item(), acc.item(), "Valid")
        return metrics

    def test_step(self, batch: tuple[tensor, tensor], batch_idx: int):
        inputs, targets = batch
        loss, acc, preds = self._compute_loss_and_accuracy(inputs, targets)
        metrics = self._print_metrics(loss.item(), acc.item(), "Test")
        self.print(classification_report(targets.view(-1), preds, labels=list(label2ann.keys()), target_names=list(label2ann.values())))
        return metrics  

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)

    def _compute_loss_and_accuracy(self, inputs: tensor, targets: tensor):
        outputs, _ = self(inputs)
        targets = targets.view(-1)
        loss = self.cost_fn(outputs, targets)
        _, predicted = outputs.max(1)
        acc = self.accuracy(predicted, targets) * 100
        return loss, acc, predicted

    def _print_metrics(self, loss: float, acc: float, stage: str):
        metrics = {f"Loss/{stage}": loss, f"Accuracy/{stage}": acc}
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)
        return metrics
    

if __name__ == "__main__":
    model = SleepStagingModel(TinySleepNet(5), nn.CrossEntropyLoss(), 5)