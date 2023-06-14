import torch
from statistics import mean, stdev

from flight_maneuvers.nn.checkpoint import load_checkpoint
from flight_maneuvers.nn.log import TensorboardLogger, StandardOutLogger



class Trainer:
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

            for epoch in range(self.hparams['MAX_EPOCHS']):
                
                # train
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
                
                
                # Validate the model
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

                # Save the model
                self.save_checkpoint(self.model, self.optimizer, epoch, self.log)
            
            # Test the model
            loss_list = []
            accuracy_list = []
            test_loader = self.data_module.test_dataloader()
            for idx, batch in test_loader:
                loss, accuracy = self.eval_step(self.model, test_loader, logger=self.log)
                
            self.tb_log.add_scalar('Loss/test/mean', sum(loss_list)/len(loss), idx)
            self.tensorboard_logger.add_scalar('Accuracy/test', accuracy, idx)


    def eval_step(self, batch):
            inputs, targets = batch
            logits = self.model(inputs)
            loss = self.hparams['LOSS'](logits, targets)
            accuracy = (logits.argmax(dim=-1) == targets).float()
            return loss, accuracy

    # If the selected scheduler is a ReduceLROnPlateau scheduler.
