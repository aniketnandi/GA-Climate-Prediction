import copy
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Hyperparameter search spaces

LR_SPACE = {
    "look_back": [6, 12, 18, 24, 36],
    "alpha": [0.01, 0.1, 1.0, 5.0, 10.0, 50.0],
    "diff_order": [0, 1, 2],
}

ARIMA_SPACE = {
    "p": [0, 1, 2, 3, 4],
    "d": [0, 1, 2],
    "q": [0, 1, 2, 3],
    "P": [0, 1, 2],
    "D": [0, 1],
    "Q": [0, 1, 2],
    "s": [12],
}

LSTM_SPACE = {
    "look_back": [6, 12, 18, 24, 36],
    "n_layers": [1, 2, 3],
    "units": [32, 64, 128, 256],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.4],
    "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3],
    "batch_size": [16, 32, 64],
    "epochs": [50, 100, 150],
    "patience": [5, 10, 15],
}

SEARCH_SPACES: Dict[str, Dict] = {
    "LR": LR_SPACE,
    "ARIMA": ARIMA_SPACE,
    "LSTM": LSTM_SPACE,
}

MODEL_TYPES = list(SEARCH_SPACES.keys())


# Complexity penalty weights  (larger -> heavier penalty on complex models)

COMPLEXITY_WEIGHT = {
    "LR": 0.001,
    "ARIMA": 0.002,
    "LSTM": 0.005,
}

def _complexity_score(model_type: str, hparams: Dict[str, Any]) -> float:
    if model_type == "LR":
        return hparams.get("look_back", 12) * (1 + hparams.get("diff_order", 0))
    if model_type == "ARIMA":
        return (hparams.get("p", 1) + hparams.get("q", 0) +
                hparams.get("P", 0) + hparams.get("Q", 0))
    if model_type == "LSTM":
        return (hparams.get("n_layers", 1) *
                hparams.get("units", 64) *
                hparams.get("look_back", 12))
    return 0.0


# Chromosome

@dataclass
class Chromosome:
    model_type: str
    hparams: Dict[str, Any]
    fitness: Optional[float] = field(default=None, compare=False)
    rmse: Optional[float] = field(default=None, compare=False)

    # Factory

    @classmethod
    def random(cls, model_type: Optional[str] = None) -> "Chromosome":
        mt = model_type or random.choice(MODEL_TYPES)
        space = SEARCH_SPACES[mt]
        hparams = {k: random.choice(v) for k, v in space.items()}
        return cls(model_type=mt, hparams=hparams)

    # Fitness

    def compute_fitness(self, rmse: float) -> float:
        penalty = (COMPLEXITY_WEIGHT[self.model_type] *
                   _complexity_score(self.model_type, self.hparams))
        self.rmse = rmse
        self.fitness = 1.0 / (rmse + penalty + 1e-9)
        return self.fitness

    # Mutation

    def mutate(self, rate: float = 0.2, structural_rate: float = 0.05) -> "Chromosome":
        child = copy.deepcopy(self)

        if random.random() < structural_rate:
            new_type = random.choice([t for t in MODEL_TYPES if t != child.model_type])
            child.model_type = new_type
            space = SEARCH_SPACES[new_type]
            child.hparams = {k: random.choice(v) for k, v in space.items()}
            child.fitness = None
            child.rmse = None
            return child

        space = SEARCH_SPACES[child.model_type]
        for key, choices in space.items():
            if random.random() < rate:
                child.hparams[key] = random.choice(choices)
        child.fitness = None
        child.rmse = None
        return child

    # Instantiation

    def build_model(self):
        from models.statistical import LinearRegressionModel, ARIMAModel
        from models.lstm_model import LSTMModel

        if self.model_type == "LR":
            return LinearRegressionModel(**self.hparams)
        if self.model_type == "ARIMA":
            return ARIMAModel(**self.hparams)
        if self.model_type == "LSTM":
            return LSTMModel(**self.hparams)
        raise ValueError(f"Unknown model type: {self.model_type}")

    def __repr__(self):
        fitness_str = f"{self.fitness:.6f}" if self.fitness is not None else "None"
        return (f"Chromosome(type = {self.model_type}, "
                f"fitness = {fitness_str}, hparams = {self.hparams})")
