import torch
from abc import ABC, abstractmethod
from statistics import mean, stdev

from flight_maneuvers.nn.checkpoint import load_checkpoint
from flight_maneuvers.nn.log import TensorboardLogger, StandardOutLogger

class BaseTrainer(ABC):
     
    @abstractmethod
    def state_dict(self, model_type, **hparams):
        pass

    @abstractmethod
    def load_state_dict(self):
        pass
     
    @abstractmethod
    def fit(self):
        pass

class BatchTrainer(BaseTrainer):
    def __init__(self, model_list, optimizer_list, datamodule, hparam_list):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.console_logger = StandardOutLogger()
        self.tensorboard_logger = TensorboardLogger()
        self.models = [model.to(self.device) for model in model_list]
        self.optimizers = [optimizer(model.parameters(), **hparams) for model, optimizer in zip(models, optimizers)]
        self.schedulers = [scheduler(optimizer, **hparams) for optimizer in optimizers]

class DistributedTrainer(BaseTrainer):
    pass

class Trainer(BaseTrainer):
    def __init__(self, model, datamodule, **hparams):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.console_logger = StandardOutLogger()
        self.tensorboard_logger = TensorboardLogger()
        
    @classmethod
    def load_state_dict(cls):
        T = cls()
        def __init__(self, model_type, **hparams):
            pass

    def fit(self, model_type, early_stopping, max_steps=None, **hparams):
            
            # Loads the most recent checkpoint from indicated version of `model_type`, or creates 
            # a new model if no version is specified 
            self.model, self.optimizer, self.scheduler = load_checkpoint(model_type, **hparams)        
            self.model.to(self.device)

            # Loads the most recent checkpoint from indicated version of `model_type`, or creates
            train_loader = self.data_module.train_dataloader()
            val_loader = self.data_module.val_dataloader()
            best_val_loss = float('inf')

            for epoch in range(self.hparams['MAX_EPOCHS']):
                
                # Train
                loss_list = []
                accuracy_list = []
                for idx, batch in train_loader:
                    self.optimizer.zero_grad()
                    loss, accuracy = self.eval_step(self.model, train_loader, logger=self.log)
                    loss.backward()        
                    self.optimizer.step()
                    
                    self.tensorboard_logger.add_scalar('Loss/train', loss.item(), idx)
                    self.tensorboard_logger.add_scalar('Accuracy/train', accuracy.item().mean(), idx)
                    loss_list.append(loss.item())
                    accuracy_list.append(accuracy.item())
                
                
                # Validate
                loss_list = []
                accuracy_list = []
                for idx, batch in val_loader:                 
                    loss, accuracy = self.eval_step(self.model, val_loader, logger=self.log)
                    loss_list.append(loss.item())
                    accuracy_list.append(accuracy.item())
                
                self.tensorboard_logger.add_scalar('Loss/val/mean', mean(loss_list), idx)
                self.tensorboard_logger.add_scalar('Loss/val/stdev', stdev(loss_list), idx)
                self.tensorboard_logger.add_scalar('Accuracy/val/mean', mean(accuracy_list), idx)
                self.tensorboard_logger.add_scalar('Accuracy/val/stdev', stdev(accuracy_list), idx)
                if self.scheduler is torch.optim.lr_scheduler.ReduceLROnPlateau:
                    self.scheduler_step(sum(loss_list)/len(loss))


                # Checkpoint Model that minimizes validation error
                if best_val_loss > mean(loss_list):
                    best_val_loss = mean(loss_list)
                    self.save_checkpoint(self.model, self.optimizer, epoch, self.log, best_val_loss)
                
                # Early stopping
                if early_stopping == False:
                    break


            # Test the model
            loss_list = []
            accuracy_list = []
            test_loader = self.data_module.test_dataloader()
            for idx, batch in test_loader:
                loss, accuracy = self.eval_step(self.model, test_loader, logger=self.log)
                loss_list.append(loss.item())
                accuracy_list.append(accuracy.item())
                
            return sum(loss_list)/len(loss), sum(accuracy_list)/len(accuracy)


    def eval_step(self, batch):
            inputs, targets = batch
            logits = self.model(inputs)
            loss = self.hparams['LOSS'](logits, targets)
            accuracy = (logits.argmax(dim=-1) == targets).float()
            return loss, accuracy

    # If the selected scheduler is a ReduceLROnPlateau scheduler.
