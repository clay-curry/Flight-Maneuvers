import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping

from flight_maneuvers.utils import *
from flight_maneuvers.data_module import FlightTrajectoryDataModule

class TrainingModule(L.LightningModule):
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
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.optimizer['lr'])
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
        self.log('train_acc', acc, batch_size=1)
        self.log('train_loss', loss, prog_bar=True)        

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
    
        
    def validation_step(self, batch, batch_idx):
        x, labels = preprocess_trajectory(batch[0], self.hparams['features'])
        preds = self.model(x)
        loss = self.loss_module(preds, labels) / len(labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log('val_acc', acc, batch_size=1, prog_bar=True)
        self.log('val_loss', loss, batch_size=1, prog_bar=True)
        
        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(loss)


    def test_step(self, batch, batch_idx):
        x, labels = preprocess_trajectory(batch, self.hparams['features'])
        preds = self.model(x)
        loss = self.loss_module(preds, labels) / len(labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log('acc', acc, batch_size=1, prog_bar=True)
            
    def predict(self, trajectory_df):
        with torch.set_grad_enabled(False):            
            x, _ = preprocess_trajectory(trajectory_df, self.hparams['features'])
            joint_dist = self.forward(x).softmax(-1)
            joint_dist = postprocess_joint(joint_dist)
            return joint_dist


def train(model: torch.nn.Module, seed, num_train, num_valid, max_steps, hparams):
    
    LOG_DIR = 'logs'
    MAX_STEPS = 250 # * 5
    VALIDATE_EVERY_N_STEPS = 50

    L.seed_everything(seed)
    data_module = FlightTrajectoryDataModule(num_train=num_train, num_valid=num_valid, num_test=100)
    early_stopping = EarlyStopping(monitor='val_loss', mode='max', patience=15)
    litmodel = TrainingModule(model, **hparams)
    tb_logger = TensorBoardLogger(LOG_DIR, name='resnet-'+str(num_train)+'-train')
    trainer = L.Trainer(
        logger=tb_logger,
        enable_checkpointing=False,
        callbacks=[early_stopping], 
        max_steps=MAX_STEPS,
        check_val_every_n_epoch=None,
        val_check_interval=VALIDATE_EVERY_N_STEPS,
        fast_dev_run=True
    )
    trainer.fit(litmodel, data_module)
    return trainer.test(litmodel, data_module)[0]['acc']
        
