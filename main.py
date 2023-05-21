import numpy as np
import flight_maneuvers.evolution as ne

xor_inputs, xor_outputs = ne.utils.make_xor_data(2)

xor_inputs = np.array(xor_inputs)
xor_outputs = np.array(xor_outputs)

# visualizing
for x, y in zip(xor_inputs, xor_outputs):
    print(f"{x} -> {y}")

def fitness_function(genome, log=False):
    """ Implementation of the fitness function we're going to use.

    It simply feeds the XOR inputs to the given genome and calculates how well
    it did (based on the squared error).
    """
    # Shuffling the input, in order to prevent our networks from memorizing the
    # sequence of the answers.
    idx = np.random.permutation(len(xor_inputs))
    
    error = 0
    for x, y in zip(xor_inputs[idx], xor_outputs[idx]):
        # Resetting the cached activations of the genome (optional).
        genome.reset()

        # Feeding the input to the genome. A numpy array with the value 
        # predicted by the neural network is returned.
        h = genome.process(x)[0]

        # Calculating the squared error.
        error += (y - h) ** 2

        if log:
            print(f"IN: {x}  |  OUT: {h:.4f}  |  TARGET: {y}")

    if log:
        print(f"\nError: {error}")

    return (1 / error) if error > 0 else 0


class RandomAgent:
    def reset(self):
        pass

    def process(self, x):
        y = np.random.randint(2)
        return np.array([y])

v = fitness_function(RandomAgent(), log=True)
print(f"Fitness: {v:2f}")

population = ne.neat.NeatPopulation(
    size=200,                       # number of genomes in the population
    num_inputs=len(xor_inputs[0]),  # number of input nodes in the genomes
    num_outputs=1,                  # number of output nodes in the genomes
)

history = population.evolve(generations=64,
                            fitness_function=fitness_function)

best_genome = population.fittest()
fitness_function(best_genome, log=True)

best_genome.visualize()

history.visualize()