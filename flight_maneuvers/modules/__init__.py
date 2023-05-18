import torch
import lightning as L

from .se3_resnet import SE3_ResNetBlock, SE3_PreActResNetBlock, SE3_ResNet
from .se2_resnet import SE2_ResNetBlock, SE2_PreActResNetBlock, SE2_ResNet
from .resnet import ResNetBlock, PreActResNetBlock, ResNet


class FlightManeuverModule(L.LightningModule):
    def __init__(self, model_name, model_hparams):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = model_name(**model_hparams)
        # Create loss module
        self.loss_module = torch.nn.CrossEntropyLoss()
        # This module is responsible for calling .backward(), .step(), .zero_grad().
        self.automatic_optimization = False

    def forward(self, trajectory):
        # Forward function that is run when visualizing the graph
        x = preprocess_trajectory(trajectory)
        return self.model(x)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = torch.optim.AdamW(
                self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f"Unknown optimizer: \"{self.hparams.optimizer_name}\""

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        x, labels = batch
        preds = self.model(x)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        preds = self.model(x).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log('test_acc', acc)
    
    def on_train_epoch_end(self):
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["loss"])
    
    def predict(self, trajectory_df):
        with torch.set_grad_enabled(False):            
            x, _ = preprocess_trajectory(trajectory_df)
            joint_dist = self.forward(x).softmax(-1)
            joint_dist = postprocess_joint(joint_dist)
            return joint_dist


__all__ = [
    # SE3 Block
    "SE3_ResNetBlock",
    "SE3_PreActResNetBlock",
    "SE3_ResNet",
    # SE2 Block
    "SE2_ResNetBlock",
    "SE2_PreActResNetBlock",
    "SE2_ResNet",
    # Resnet SE2 
    "ResNetBlock",
    "PreActResNetBlock",
    "ResNet"
]


from flight_maneuvers.utils import *