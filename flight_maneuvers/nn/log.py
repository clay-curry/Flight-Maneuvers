
import inspect
from time import sleep
from click import style
from columnar import columnar

from torch.utils.tensorboard import SummaryWriter

tb_layout = {
        'Loss': {
            'train': ['Multiline', ['model1/hparams']],
            'val': ['Multiline', ['model1/hparams']],
            'test': ['Margin', ['model1/hparams']]
        },
        'Accuracy': {
            'train': ['Multiline', ['model2/hparams','model1/hparams']],
            'val': ['Multiline', ['model1/hparams']],
            'test': ['Margin', ['model1/hparams']]
        },
}

class TensorboardLogger(SummaryWriter):
    def __init__(self, log_dir='runs'):
        super().__init__(log_dir)


    
class StandardOutLogger():
    def __init__(self):

        self.TAB_HEADER = ["NAME", "CURRENT", "PAST", "INCREASE", "INCREASE (%)"]
        
        self.curr_train_error = None
        self.past_train_error = None
        self.inc_train_error = None
        self.inc_percentage = None
        
        self.curr_train_error_variance = None
        self.past_train_error_variance = None
        self.inc_train_error_variance = None
        self.inc_percentage_train_error_variance = None
        
        self.curr_expected_train_error = None
        self.past_expected_train_error = None
        self.inc_expected_train_error = None
        self.inc_percentage_expected_train_error = None

        self.curr_val_error = None
        self.past_val_error = None
        self.inc_val_error = None
        self.inc_percentage_val_error = None
        
        self.curr_val_error_variance = None
        self.past_val_error_variance = None
        self.inc_val_error_variance = None
        self.inc_percentage_val_error_variance = None
        
        self.curr_expected_val_error = None
        self.past_expected_val_error = None
        self.inc_expected_val_error = None
        self.inc_percentage_expected_val_error = None

    def log(self, **hparams):
        for k in hparams:
            self.k = hparams[k]

        self.pre_update_data()
        data = [
            ['Training Error', '{.2f}'.format(self.curr_train_error), '{.2f}'.format(self.past_train_error), '{.2f}'.format(self.inc_train_error), '{.2f}'.format(self.inc_percentage)],
            ['Training Error Variance', '{.2f}'.format(self.curr_train_error_variance), '{.2f}'.format(self.past_train_error_variance), '{.2f}'.format(self.inc_train_error_variance), '{.2f}'.format(self.inc_percentage_train_error_variance)],
            ['Expected Training Error', '{.2f}'.format(self.curr_expected_train_error), '{.2f}'.format(self.past_expected_train_error), '{.2f}'.format(self.inc_expected_train_error), '{.2f}'.format(self.inc_percentage_expected_train_error)],
            ['Validation Error', '{.2f}'.format(self.curr_val_error), '{.2f}'.format(self.past_val_error), '{.2f}'.format(self.inc_val_error), '{.2f}'.format(self.inc_percentage_val_error)],
            ['Validation Error Variance',  '{.2f}'.format(self.curr_val_error_variance), '{.2f}'.format(self.past_val_error_variance), '{.2f}'.format(self.inc_val_error_variance), '{.2f}'.format(self.inc_percentage_val_error_variance)],
            ['Expected Validation Error',  '{.2f}'.format(self.curr_expected_val_error), '{.2f}'.format(self.past_expected_val_error), '{.2f}'.format(self.inc_expected_val_error), '{.2f}'.format(self.inc_percentage_expected_val_error)],
        ]      

        patterns = [
            ('\d+km', lambda text: style(text, fg='cyan')),
            ('Model Complexity', lambda text: style(text, fg='green')),
            ('Expected Training Error', lambda text: style(text, fg='red')),
            ('Expected Train. Err. Variance', lambda text: style(text, fg='blue')),
            ('Expected Validation Error', lambda text: style(text, fg='red')),
            ('Expected Val. Err. Variance', lambda text: style(text, fg='blue')),
            
        ]

        table = columnar(
                self.data, # elements to be printed in each table cell
                headers=self.TAB_HEADER, # headers in each column 
                justify=['r', 'c', 'l', 'c', 'c'], # justify each column right, center, left, center
                patterns=patterns # apply patterns to each element conditioned on its value
            )
        
        print(table)
        self.post_update_data()

    def pre_update_data(self):
        if self.past_expected_train_error is not None:
            self.inc_expected_train_error = self.curr_expected_train_error - self.past_expected_train_error
            self.inc_percentage_expected_train_error = self.inc_expected_train_error / self.past_expected_train_error * 100

        if self.past_expected_val_error is not None:
            self.inc_expected_val_error = self.curr_expected_val_error - self.past_expected_val_error
            self.inc_percentage_expected_val_error = self.inc_expected_val_error / self.past_expected_val_error * 100

        if self.past_train_error is not None:
            self.inc_train_error = self.curr_train_error - self.past_train_error
            self.inc_percentage_train_error = self.inc_train_error / self.past_train_error * 100
        
        if self.past_val_error is not None:
            self.inc_val_error = self.curr_val_error - self.past_val_error
            self.inc_percentage_val_error = self.inc_val_error / self.past_val_error * 100

        if self.past_train_error_variance is not None:
            self.inc_train_error_variance = self.curr_train_error_variance - self.past_train_error_variance
            self.inc_percentage_train_error_variance = self.inc_train_error_variance / self.past_train_error_variance * 100

    def post_update_data(self):
        
        if self.curr_train_error is not None:
            self.past_train_error = self.curr_train_error
        
        if self.curr_val_error is not None:
            self.past_val_error = self.curr_val_error

        if self.curr_train_error_variance is not None:
            self.past_train_error_variance = self.curr_train_error_variance
        
        if self.curr_val_error_variance is not None:
            self.past_val_error_variance = self.curr_val_error_variance
        
        if self.curr_expected_train_error is not None:
            self.past_expected_train_error = self.curr_expected_train_error

        if self.curr_expected_val_error is not None:
            self.past_expected_val_error = self.curr_expected_val_error
