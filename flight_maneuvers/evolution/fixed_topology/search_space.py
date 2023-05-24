from flight_maneuvers.modules.resnet import ResNet

optimizer_space = {
    'optimizer': [ 
            {'opt': 'Adam', 'lr': 1e-3}, {'opt': 'Adam', 'lr': 1e-4}, {'opt': 'Adam', 'lr': 1e-5},
            {'opt': 'SGD' , 'lr': 1e-3}, {'opt': 'SGD' , 'lr': 1e-4}, {'opt': 'SGD' , 'lr': 1e-5},
        ],
    'lr_scheduler': [
            {'lrs': 'StepLR', 'step_size': 100 }, 
            {'lrs': 'StepLR', 'step_size': 200 },
            {'lrs': 'StepLR', 'step_size': 500 },
            {'lrs': 'ReduceLROnPlateau', 'min_lr': 1e-5}
        ],
}

model_space = {
    ResNet: {
        "features": [
                ['dpos'], ['vel', 'dpos'], 
                ['vel', 'dpos', 'dvel'],
                ['pos', 'vel', 'dpos', 'dvel']
            ],
        "c_hidden": [[16]*30, [16]*30 + [32]*30, [16]*30 + [32]*30 + [64]*30, [16]*30 + [32]*30 + [64]*30 + [128]*30],
        "kernel_size": [3, 5],
        "act_fn_name": ['relu', 'leakyrelu', 'tanh', 'gelu'],
        "block_name": ['PreActResNetBlock', 'ResNetBlock'],
    }
}


