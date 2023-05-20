import json, os
import random
from itertools import product
from typing import Any
from flight_maneuvers.train import train
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

EVAL_PATH = 'logs/eval'

def random_genotype(model):
    # Use itertools.product to generate combinations of values 
    hparams = { **model_space[model], **optimizer_space }
    keys = hparams.keys()
    values = hparams.values()
    cartesian_product = iter(dict(zip(keys, combination)) for combination in product(*values))
    return random.sample(list(cartesian_product), 1)[0]

class Search():
    """
    1. Decode the genotype
    extract weights and other features

    2. Evaluate each ANN
    store fitness of each individual

    3. Select the parents for reproduction
    based on the fitness or error

    4. Apply search operators
    mutation or crossover to the new individual

    5. Repeat if necessary
    if terminal criteria not reached, go to 1
    """

    def __init__(self, model):
        self.model = model
        self.trial_num = 0
        self.population = []
        self.fitness_table = {}
    
    def generate_population(self, n=10):
        while len(self.population) < 2:
            rand_genotype = random_genotype(self.model)
            self.population.append(rand_genotype)
            self.fitness_table[json.dumps(rand_genotype)] = -1    
        
        for _ in range(n):
            child = random.choice(self.population)
            # child = self.crossover(child)
            child = self.mutate(child)
            self.population.append(child)
            self.fitness_table[json.dumps(child)] = -1

    def evaluate(self):
        for hp in self.population:
            print(self.trial_num)
            if self.fitness_table[json.dumps(hp)] == -1:
                self.fitness_table[json.dumps(hp)] = (
                    train(ResNet, 0, 50, 20, 100, hp)
                )
        
    def select(self):
        self.population = 
        sorted(self.fitness_table.keys(), key=lambda x: - self.fitness_table[json.dumps(x)]).copy()
        self.population = self.population[0:2].copy()

    def crossover(self, child):
        individual1, individual2 = self.population[0:2] # select two best individuals
        channels1 = individual1['c_hidden']
        channels2 = individual2['c_hidden']
        max_len = max(len(channels1), len(channels2))

        split1 = random.randint(0, max_len-1)
        split2 = random.randint(0, max_len-split1)
        
        if split1 < split2:
            channels = channels1[:split1] + channels2[split1:split2] + channels1[split2:]
            child['c_hidden'] = channels
        return child
            

    def mutate(self, hp):
        
        if self.model == ResNet:
            MAX_DEPTH = 20
            MAX_CHANNELS = 265

            while json.dumps(hp) in self.fitness_table:
                # mutate features
                hp['features'] = random.choice(model_space[self.model]['features'])
                # remove a layer
                if len(hp['c_hidden']) > 1 and random.random() < 0.5:
                    hp['c_hidden'].pop(random.randint(0, len(hp['c_hidden']))-1) 
                # add a layer
                if len(hp['c_hidden']) < MAX_DEPTH and random.random() < 0.5:
                    hp['c_hidden'].insert(random.randint(0, len(hp['c_hidden'])-1), random.randint(1, MAX_CHANNELS))
                # change a layer
                if random.random() < 0.5:
                    hp['c_hidden'][random.randint(0, len(hp['c_hidden'])-1)] = random.randint(1, MAX_CHANNELS)
            return hp
                


    def evolve(self):
        self.generate_population(10)
        prev_fitness = 0
        curr_fitness = 0.1
        while True or prev_fitness < curr_fitness+1:
            self.generate_population()
            l = self.population[0].copy()
            self.evaluate()
            self.select()
            prev_fitness = curr_fitness
            sum_acc = sum([self.fitness_table[json.dumps(i)] for i in self.population])
            curr_fitness = sum_acc
            self.trial_num += 1
            print(f"Trial {self.trial_num}: {curr_fitness}")