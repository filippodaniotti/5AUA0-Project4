import lightning.pytorch as pl
import torch
from torch import nn, tensor
from torchmetrics import Accuracy

from models.tiny_sleep_net import TinySleepNet
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from config import Config

class SleepStagingModel(pl.LightningModule):
    def __init__(
            self, 
            model: nn.Module, 
            cost_function: nn.Module,
            config: Config,
            evaluate: bool = False,
            learning_rate: float = None,
            weight_decay: float = None):
        super().__init__()
        self.cfg = config
        self.model = model    
        self.cost_fn = cost_function
        self.accuracy = Accuracy(task="multiclass", num_classes=self.cfg.num_classes)
        
        
        self.lr = learning_rate if learning_rate else self.cfg.learning_rate
        self.wd = weight_decay if weight_decay else self.cfg.weight_decay
        
        self.idx_to_class = {
            0: "Sleep stage W",
            1: "Sleep stage 1",
            2: "Sleep stage 2",
            3: "Sleep stage 3/4",
            4: "Sleep stage R"
        }
        
        self.evaluate = evaluate
        if self.evaluate:
            self.predictions = []
            self.targets = []
        
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
        if self.evaluate:
            self.predictions.append(preds)
            self.targets.append(targets)
        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

    def _compute_loss_and_accuracy(self, inputs: tensor, targets: tensor):
        # print(inputs.shape, targets.shape)
        outputs, _ = self(inputs)
        targets = targets.view(-1)
        # print(outputs.shape, targets.shape)
        loss = self.cost_fn(outputs, targets)
        _, predicted = outputs.max(1)
        acc = self.accuracy(predicted, targets) * 100
        return loss, acc, predicted

    def _print_metrics(self, loss: float, acc: float, stage: str):
        metrics = {f"Loss/{stage}": loss, f"Accuracy/{stage}": acc}
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)
        return metrics
    
    def compute_metrics(self):
        predicted = torch.cat(self.predictions, dim=0).detach().cpu().numpy()
        targets = torch.cat(self.targets, dim=0).detach().cpu().numpy()
        report = classification_report(
            targets, 
            predicted, 
            labels=list(self.class_to_idx.keys()), 
            target_names=list(self.class_to_idx.values()),
            output_dict=True
        )
        matrix = confusion_matrix(
            targets,
            predicted,
            labels=list(self.class_to_idx.keys())
        )
        kappa = cohen_kappa_score(targets, predicted, self.class_to_idx.keys())
        return {"report": report, "matrix": matrix, "kappa": kappa}
    

if __name__ == "__main__":
    from config import configurations
    from data.data import get_data, get_collator
    
    conf = configurations["baseline_hmc"]
    model = SleepStagingModel(TinySleepNet(conf), nn.CrossEntropyLoss(), conf)
    
    train_loader, _, test_loader = get_data(
        root=conf.data_dir,
        dataset=conf.dataset,
        batch_size=conf.batch_size,
        test_batch_size=conf.test_batch_size,
        train_percentage=0.9,
        val_percentage=0.0,
        test_percentage=0.1,
        train_collate_fn=get_collator(
            epoch_duration=conf.epoch_duration,
            in_channels=conf.in_channels,
            sampling_rate=conf.sampling_rate,
            low_resources=conf.low_resources),
        test_collate_fn=get_collator(
            epoch_duration=conf.epoch_duration,
            in_channels=conf.in_channels,
            sampling_rate=conf.sampling_rate,
            low_resources=conf.low_resources,
            is_test_set=True)
    )
    
    for t, s in test_loader:
        print(t.shape)
        print(s.shape)
        o = model(t)