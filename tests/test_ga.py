import random

import numpy as np
import pytest

from data.loaders import _synthetic_co2, _synthetic_sea_level, _synthetic_temperature
from ga.chromosome import MODEL_TYPES, SEARCH_SPACES, Chromosome
from ga.crossover import type_aware_crossover, uniform_crossover
from ga.engine import GeneticAlgorithm, tournament_select
from ga.random_search import RandomSearch
from models.statistical import ARIMAModel, LinearRegressionModel


def _assert_valid(chrom):
    assert chrom.model_type in MODEL_TYPES
    space = SEARCH_SPACES[chrom.model_type]
    assert set(chrom.hparams) == set(space)
    for key, value in chrom.hparams.items():
        assert value in space[key]


@pytest.fixture
def series():
    return _synthetic_temperature()["anomaly"].values[-240:]


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_random_chromosome_is_valid(model_type):
    random.seed(0)
    _assert_valid(Chromosome.random(model_type))


def test_mutation_keeps_genes_in_search_space():
    random.seed(0)
    chrom = Chromosome.random("ARIMA")
    for _ in range(200):
        chrom = chrom.mutate(rate=0.5, structural_rate=0.2)
        _assert_valid(chrom)


def test_uniform_crossover_only_swaps_existing_genes():
    random.seed(0)
    p1, p2 = Chromosome.random("LR"), Chromosome.random("LR")
    c1, c2 = uniform_crossover(p1, p2, cx_prob=0.5)
    for key in SEARCH_SPACES["LR"]:
        assert {c1.hparams[key], c2.hparams[key]} == {p1.hparams[key], p2.hparams[key]}


def test_type_aware_crossover_across_types_yields_valid_children():
    random.seed(0)
    for _ in range(50):
        c1, c2 = type_aware_crossover(
            Chromosome.random("LR"), Chromosome.random("ARIMA")
        )
        _assert_valid(c1)
        _assert_valid(c2)
        assert c1.fitness is None and c2.fitness is None


def test_complexity_penalty_lowers_fitness():
    simple = Chromosome("LR", {"look_back": 6, "alpha": 1.0, "diff_order": 0})
    complex_ = Chromosome("LR", {"look_back": 36, "alpha": 1.0, "diff_order": 2})
    assert simple.compute_fitness(0.1) > complex_.compute_fitness(0.1)


def test_tournament_select_returns_fittest_when_k_covers_population():
    pop = [Chromosome.random("LR") for _ in range(5)]
    for i, chrom in enumerate(pop):
        chrom.fitness = float(i)
    assert tournament_select(pop, k=len(pop)) is pop[-1]


@pytest.mark.parametrize(
    "model", [LinearRegressionModel(look_back=12), ARIMAModel(p=1, d=1, q=1)]
)
def test_models_produce_finite_rmse(model, series):
    rmse = model.walk_forward_rmse(series, test_size=24)
    assert np.isfinite(rmse) and rmse > 0


def test_ga_runs_and_logs_every_generation(series):
    ga = GeneticAlgorithm(
        series,
        test_size=24,
        pop_size=4,
        n_generations=3,
        model_types=["LR"],
        seed=1,
        verbose=False,
    )
    best, log = ga.run()
    _assert_valid(best)
    assert np.isfinite(best.rmse)
    assert len(log) >= 3
    # Elitism: the best RMSE never gets worse from one generation to the next
    best_rmses = [entry["best_rmse"] for entry in log]
    assert all(b <= a + 1e-12 for a, b in zip(best_rmses, best_rmses[1:]))


def test_random_search_respects_budget(series):
    rs = RandomSearch(
        series, test_size=24, n_evals=5, model_types=["LR"], seed=2, verbose=False
    )
    best, log = rs.run()
    _assert_valid(best)
    assert len(log) == 5


@pytest.mark.parametrize(
    "loader, column",
    [
        (_synthetic_temperature, "anomaly"),
        (_synthetic_co2, "co2_ppm"),
        (_synthetic_sea_level, "sea_level_mm"),
    ],
)
def test_synthetic_loaders_have_expected_shape(loader, column):
    df = loader()
    assert list(df.columns) == ["date", column]
    assert df["date"].is_monotonic_increasing
    assert df[column].notna().all()
