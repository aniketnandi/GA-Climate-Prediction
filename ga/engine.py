import random
import time
import sys
import numpy as np
from typing import List, Optional, Tuple, Dict, Any

from .chromosome import Chromosome, MODEL_TYPES
from .crossover import type_aware_crossover

# Fitness evaluation

def evaluate_chromosome(chrom: Chromosome, series: np.ndarray,
                         test_size: int) -> float:
    try:
        model = chrom.build_model()
        rmse = model.walk_forward_rmse(series, test_size)
        if not np.isfinite(rmse) or rmse <= 0:
            rmse = 1e6
    except Exception:
        rmse = 1e6
    return chrom.compute_fitness(rmse)

# Selection

def tournament_select(population: List[Chromosome], k: int = 3) -> Chromosome:
    contestants = random.sample(population, min(k, len(population)))
    return max(contestants, key = lambda c: c.fitness or 0.0)

# Genetic Algorithm

class GeneticAlgorithm:

    def __init__(
        self,
        series: np.ndarray,
        test_size: int = 24,
        pop_size: int = 30,
        n_generations: int = 50,
        cx_prob: float = 0.7,
        mut_rate: float = 0.2,
        structural_rate: float = 0.05,
        elitism_k: int = 2,
        tournament_k: int = 3,
        model_types: Optional[List[str]] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        self.series = np.array(series, dtype = float)
        self.test_size = test_size
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.cx_prob = cx_prob
        self.mut_rate = mut_rate
        self.structural_rate = structural_rate
        self.elitism_k = elitism_k
        self.tournament_k = tournament_k
        self.allowed_types = model_types or MODEL_TYPES
        self.verbose = verbose

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.population: List[Chromosome] = []
        self.log: List[Dict[str, Any]] = []

    # Initialisation

    def _init_population(self):
        self.population = [
            Chromosome.random(random.choice(self.allowed_types))
            for _ in range(self.pop_size)
        ]

    # Evaluation

    def _evaluate_all(self):
        unevaluated = [c for c in self.population if c.fitness is None]
        for c in unevaluated:
            evaluate_chromosome(c, self.series, self.test_size)

    # Logging

    def _log_generation(self, gen: int, elapsed: float):
        fitnesses = [c.fitness for c in self.population if c.fitness is not None]
        rmses = [c.rmse for c in self.population if c.rmse is not None]
        best = self._best()
        type_counts = {t: sum(1 for c in self.population if c.model_type == t) for t in MODEL_TYPES}
        entry = {
            "generation": gen,
            "best_fitness": best.fitness,
            "best_rmse": best.rmse,
            "best_type": best.model_type,
            "mean_fitness": float(np.mean(fitnesses)) if fitnesses else 0.0,
            "mean_rmse": float(np.mean(rmses)) if rmses else 0.0,
            "type_counts": type_counts,
            "elapsed_s": elapsed,
        }
        self.log.append(entry)

        if self.verbose:
            bar = "=" * min(gen + 1, 40)
            print(
                f"Gen {gen:3d}/{self.n_generations} | "
                f"Best RMSE: {best.rmse:.4f} ({best.model_type}) | "
                f"Mean RMSE: {entry['mean_rmse']:.4f} | "
                f"LR:{type_counts['LR']} AR:{type_counts['ARIMA']} LSTM:{type_counts['LSTM']}",
                flush = True
            )

    # Evolution

    def _best(self) -> Chromosome:
        return max(self.population, key = lambda c: c.fitness or 0.0)

    def _next_generation(self):
        sorted_pop = sorted(self.population,
                            key = lambda c: c.fitness or 0.0,
                            reverse = True)
        new_pop = sorted_pop[:self.elitism_k]

        while len(new_pop) < self.pop_size:
            p1 = tournament_select(self.population, self.tournament_k)
            p2 = tournament_select(self.population, self.tournament_k)

            if random.random() < self.cx_prob:
                c1, c2 = type_aware_crossover(p1, p2, self.cx_prob)
            else:
                import copy
                c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
                c1.fitness = c1.rmse = None
                c2.fitness = c2.rmse = None

            c1 = c1.mutate(self.mut_rate, self.structural_rate)
            c2 = c2.mutate(self.mut_rate, self.structural_rate)

            if c1.model_type not in self.allowed_types:
                c1 = Chromosome.random(random.choice(self.allowed_types))
            if c2.model_type not in self.allowed_types:
                c2 = Chromosome.random(random.choice(self.allowed_types))

            new_pop.extend([c1, c2])

        self.population = new_pop[:self.pop_size]

    # Main loop

    def run(self) -> Tuple[Chromosome, List[Dict]]:
        t0 = time.time()
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Genetic Algorithm - {self.pop_size} individuals × "
                  f"{self.n_generations} generations")
            print(f"Series length: {len(self.series)} | Test size: {self.test_size}")
            print(f"{'=' * 60}")

        self._init_population()
        self._evaluate_all()
        self._log_generation(0, time.time() - t0)

        for gen in range(1, self.n_generations + 1):
            self._next_generation()
            self._evaluate_all()
            self._log_generation(gen, time.time() - t0)

        best = self._best()
        if self.verbose:
            print(f"\nBest configuration found:")
            print(f"Type : {best.model_type}")
            print(f"RMSE : {best.rmse:.6f}")
            print(f"Fitness : {best.fitness:.6f}")
            print(f"Hparams : {best.hparams}")
            print(f"Total time : {time.time() - t0:.1f}s\n")

        return best, self.log
