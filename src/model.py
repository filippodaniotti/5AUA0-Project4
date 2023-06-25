import lightning.pytorch as pl
import torch
from torch import nn, tensor
from torchmetrics import Accuracy
from sklearn.metrics import classification_report, confusion_matrix

from src.config import Config

class SleepStagingModel(pl.LightningModule):
    """LightningModule for a sleep staging model.
    This class implements the LightningModule interface and defines
    the training and testing behaviour of the underlying model.

    Args:
        model (nn.Module): The underlying model.
        cost_function (nn.Module): The cost function for optimization.
        config (Config): The configuration object for the model.
        evaluate (bool): Flag indicating whether to perform evaluation. Defaults to False.
        learning_rate (float): The learning rate for optimization. 
            If None, uses the value from the config. Defaults to None.
        weight_decay (float): The weight decay for optimization. 
            If None, uses the value from the config. Defaults to None.

    Attributes:
        cfg (Config): The configuration object for the model.
        model (nn.Module): The underlying model.
        cost_fn (nn.Module): The cost function for optimization.
        accuracy (Accuracy): The accuracy metric for evaluation.
        lr (float): The learning rate for optimization.
        wd (float): The weight decay for optimization.
        idx_to_class (dict): A dictionary mapping class indices to class labels.
        evaluate (bool): Flag indicating whether to perform evaluation.
        predictions (list): A list to store the model predictions during evaluation.
        targets (list): A list to store the target values during evaluation.

    Methods:
        forward(x): Performs the forward pass of the model.
        training_step(batch, batch_idx): Defines a training step for the model.
        validation_step(batch, batch_idx): Defines a validation step for the model.
        test_step(batch, batch_idx): Defines a test step for the model.
        configure_optimizers(): Configures the optimizers for training.
        _compute_loss_and_accuracy(inputs, targets): Common step, cmputes the loss and accuracy 
            for a batch of inputs and targets.
        _print_metrics(loss, acc, stage): Prints and logs the metrics.
        compute_metrics(): Computes evaluation metrics.

    """
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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

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
    
    def compute_metrics(self):
        predicted = torch.cat(self.predictions, dim=0).detach().cpu().numpy()
        targets = torch.cat(self.targets, dim=0).detach().cpu().numpy()
        targets = targets.reshape(-1)
        report = classification_report(
            targets, 
            predicted, 
            labels=list(self.idx_to_class.keys()), 
            target_names=list(self.idx_to_class.values()),
            output_dict=True
        )
        matrix = confusion_matrix(
            targets,
            predicted,
            labels=list(self.idx_to_class.keys())
        )
        return {"report": report, "matrix": matrix, }
    
