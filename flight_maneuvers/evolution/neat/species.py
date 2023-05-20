
""" Implementation of the :class:`.NeatSpecies` class.
"""

from typing import List, Optional

import numpy as np

from nevopy.neat.genomes import NeatGenome


class NeatSpecies:
    """ Represents a species within NEAT's evolutionary environment.

    Args:
        species_id (int): Unique identifier of the species.
        generation (int): Current generation. The generation in which the
            species is born.

    Attributes:
        representative (Optional[NeatGenome]): Genome used to represent the
            species.
        members (List[NeatGenome]): List with the genomes that belong to the
            species.
        last_improvement (int): Generation in which the species last showed
            improvement of its fitness. The species fitness in a given
            generation is equal to the fitness of the species most fit genome on
            that generation.
        best_fitness (Optional[float]): The last calculated fitness of the
            species most fit genome.
    """

    def __init__(self, species_id: int, generation: int) -> None:
        self._id = species_id
        self.representative = None  # type: Optional[NeatGenome]
        self.members = []           # type: List[NeatGenome]

        self._creation_gen = generation
        self.last_improvement = generation
        self.best_fitness = None   # type: Optional[float]

    @property
    def id(self) -> int:
        """ Unique identifier of the species. """
        return self._id

    def random_representative(self) -> None:
        """ Randomly chooses a new representative for the species. """
        self.representative = np.random.choice(self.members)

    def avg_fitness(self) -> float:
        """ Returns the average fitness of the species genomes. """
        return float(np.mean([g.fitness for g in self.members]))

    def fittest(self) -> NeatGenome:
        """ Returns the fittest member of the species. """
        return self.members[int(np.argmax([g.fitness for g in self.members]))]
