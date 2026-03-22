import random
import time
import numpy as np
from typing import Dict, List, Optional, Tuple

from .chromosome import Chromosome, MODEL_TYPES
from .engine import evaluate_chromosome


class RandomSearch:

    def __init__(
        self,
        series: np.ndarray,
        test_size: int = 24,
        n_evals: int = 1500,
        model_types: Optional[List[str]] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        self.series = np.array(series, dtype = float)
        self.test_size = test_size
        self.n_evals = n_evals
        self.allowed_types = model_types or MODEL_TYPES
        self.verbose = verbose

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.log: List[Dict] = []

    def run(self) -> Tuple[Chromosome, List[Dict]]:
        t0 = time.time()
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"Random Search - {self.n_evals} evaluations")
            print(f"{'=' * 60}")

        best: Optional[Chromosome] = None
        for i in range(self.n_evals):
            mt = random.choice(self.allowed_types)
            chrom = Chromosome.random(mt)
            evaluate_chromosome(chrom, self.series, self.test_size)

            entry = {
                "eval": i + 1,
                "rmse": chrom.rmse,
                "fitness": chrom.fitness,
                "type": chrom.model_type,
            }
            self.log.append(entry)

            if best is None or chrom.fitness > best.fitness:
                best = chrom
                if self.verbose:
                    print(f"Eval {i+1:5d} | New best RMSE: {best.rmse:.4f} "
                          f"({best.model_type})", flush = True)

        elapsed = time.time() - t0
        if self.verbose:
            print(f"\nRandom search best:")
            print(f"Type : {best.model_type}")
            print(f"RMSE : {best.rmse:.6f}")
            print(f"Hparams : {best.hparams}")
            print(f"Time : {elapsed:.1f}s\n")

        return best, self.log
