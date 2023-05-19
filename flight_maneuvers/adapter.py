import torch
import lightning as L
from flight_maneuvers.utils import *

class LigntingAdapter(L.LightningModule):
    def __init__(self, 
                 model_name, 
                 optimizer=torch.optim.Adam, 
                 lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau, 
                 **hparams
        ):
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
        self.model = model_name(**hparams)
        # Create loss module
        self.loss_module = torch.nn.CrossEntropyLoss()
        # This module is responsible for calling .backward(), .step(), .zero_grad().
        self.automatic_optimization = False
        self.gradient_ticker = 0

    def forward(self, trajectory):
        # Forward function that is run when visualizing the graph
        return self.model(trajectory)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer['opt'] == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.optimizer['lr'])
        elif self.hparams.optimizer['opt'] == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f"Unknown optimizer: \"{self.hparams.optimizer_name}\""

        if self.hparams.lr_scheduler['lrs'] == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.hparams.lr_scheduler['step_size'], gamma=0.1)
        if self.hparams.lr_scheduler['lrs'] == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, min_lr=self.hparams.lr_scheduler['min_lr'])
        else:
            # We will reduce the learning rate by 0.1 after 100 and 150 epochs
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        x, labels = preprocess_trajectory(batch[0], self.hparams['features'])
        preds = self.model(x)
        loss = self.loss_module(preds, labels) / len(labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc', acc, batch_size=1, on_step=False, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True)        

        self.manual_backward(loss)
        self.gradient_ticker += 1 
        if (self.gradient_ticker + 1) % 5 == 0:
            opt = self.optimizers()
            opt.step()
            opt.zero_grad()
            self.gradient_ticker = 0
        
    def validation_step(self, batch, batch_idx):
        x, labels = preprocess_trajectory(batch[0], self.hparams['features'])
        preds = self.model(x)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log('val_acc', acc, batch_size=1, prog_bar=True)
        self.log('val_loss', loss, batch_size=1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        imgs, labels = preprocess_trajectory(batch, self.hparams['features'])
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.save_hyperparameters({'test_acc': acc})
    
    def on_train_epoch_end(self):
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])
    
    def predict(self, trajectory_df):
        with torch.set_grad_enabled(False):            
            x, _ = preprocess_trajectory(trajectory_df, self.hparams['features'])
            joint_dist = self.forward(x).softmax(-1)
            joint_dist = postprocess_joint(joint_dist)
            return joint_dist
