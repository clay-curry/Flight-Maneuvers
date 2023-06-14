
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
        for i in range(100):
            sleep(0.1)
            TAB_HEADER = ["NAME", "CURRENT", "PAST", "INCREASE", "INCREASE (%)"]
            
            data = [
                ['Training Error', f'{i}', f'{i}', f'{i}', f'{i}'],
                ['Expected Training Error', f'{i}', f'{i}', f'{i}', f'{i}'],
                ['Expected Train. Err. Variance', f'{i}', f'{i}', f'{i}', f'{i}'],
                ['Expected Validation Error', f'{i}', f'{i}', f'{i}', f'{i}'],
                ['Expected Val. Err. Variance', f'{i}', f'{i}', f'{i}', f'{i}'],
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
                        data, # elements to be printed in each table cell
                        headers=TAB_HEADER, # headers in each column 
                        justify=['r', 'c', 'l', 'c', 'c'], # justify each column right, center, left, center
                        patterns=patterns # apply patterns to each element conditioned on its value
                    )
            
            print(table)