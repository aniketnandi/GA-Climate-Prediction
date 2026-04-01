import copy
import random
from typing import Tuple

from .chromosome import Chromosome, SEARCH_SPACES


def uniform_crossover(parent1: Chromosome, parent2: Chromosome,
                      cx_prob: float = 0.5) -> Tuple[Chromosome, Chromosome]:
    assert parent1.model_type == parent2.model_type, (
        "uniform_crossover requires same model_type; use type_aware_crossover instead."
    )
    c1 = copy.deepcopy(parent1)
    c2 = copy.deepcopy(parent2)
    c1.fitness = c1.rmse = None
    c2.fitness = c2.rmse = None

    for key in SEARCH_SPACES[parent1.model_type]:
        if random.random() < cx_prob:
            c1.hparams[key], c2.hparams[key] = c2.hparams[key], c1.hparams[key]

    return c1, c2


def type_aware_crossover(parent1: Chromosome, parent2: Chromosome,
                         cx_prob: float = 0.5) -> Tuple[Chromosome, Chromosome]:
    if parent1.model_type == parent2.model_type:
        return uniform_crossover(parent1, parent2, cx_prob)

    c1 = copy.deepcopy(parent1)
    c2 = copy.deepcopy(parent2)
    c1.fitness = c1.rmse = None
    c2.fitness = c2.rmse = None

    if random.random() < cx_prob:
        from .chromosome import SEARCH_SPACES, MODEL_TYPES
        c1.model_type = parent2.model_type
        c1.hparams = {k: random.choice(v)
                      for k, v in SEARCH_SPACES[parent2.model_type].items()}
        c2.model_type = parent1.model_type
        c2.hparams = {k: random.choice(v)
                      for k, v in SEARCH_SPACES[parent1.model_type].items()}

    return c1, c2
