import argparse
import json
import os
import sys
import warnings
import logging
import numpy as np
import pandas as pd

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(__file__))

from data.loaders import (
load_temperature, load_co2, load_sea_level,
train_test_split_ts
)
from ga.engine import GeneticAlgorithm
from ga.random_search import RandomSearch
from models.statistical import LinearRegressionModel, ARIMAModel, evaluate_all
from models.lstm_model import LSTMModel
from utils.visualise import (
plot_fitness_curve, plot_type_diversity,
plot_forecast, plot_comparison, plot_projections
)

# Configuration

INDICATORS = {
    "temperature": {
        "loader": load_temperature,
        "value_col": "anomaly",
        "ylabel": "Temperature Anomaly (C)",
        "test_frac": 0.15,
    },
    "co2": {
        "loader": load_co2,
        "value_col": "co2_ppm",
        "ylabel": "CO2 Concentration (ppm)",
        "test_frac": 0.15,
    },
    "sea_level": {
        "loader": load_sea_level,
        "value_col": "sea_level_mm",
        "ylabel": "Sea Level Change (mm)",
        "test_frac": 0.15,
    },
}

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok = True)

# Default hyperparameter baselines

def default_models_rmse(series: np.ndarray, test_size: int) -> dict:
    results = {}

    lr = LinearRegressionModel(look_back =  12, alpha = 1.0)
    results["Default LR"] = lr.walk_forward_rmse(series, test_size)

    ar = ARIMAModel(p = 1, d = 1, q = 1)
    results["Default ARIMA"] = ar.walk_forward_rmse(series, test_size)

    lstm = LSTMModel(look_back = 12, n_layers = 1, units = 64, epochs = 50)
    results["Default LSTM"] = lstm.walk_forward_rmse(series, test_size)

    return results

# Projection with confidence intervals

def project_future(best_chrom, full_series: np.ndarray, n_steps: int = 300, n_bootstrap: int = 30):

    model = best_chrom.build_model()
    model.fit(full_series)
    point_forecast = model.predict(n_steps)

    boots = []
    residuals = []
    test_size = min(48, len(full_series) // 5)
    m_eval = best_chrom.build_model()
    m_eval.fit(full_series[:-test_size])
    preds = m_eval.predict(test_size)
    residuals = full_series[-test_size:] - preds
    res_std = np.std(residuals)

    for _ in range(n_bootstrap):
        noise = np.random.normal(0, res_std, n_steps)
        boots.append(point_forecast + noise)

    boots = np.array(boots)
    ci_lower = np.percentile(boots, 2.5, axis = 0)
    ci_upper = np.percentile(boots, 97.5, axis = 0)
    return point_forecast, ci_lower, ci_upper

# Single indicator pipeline

def run_indicator(indicator: str, pop_size: int, n_generations: int, seed: int, region: str = "global", verbose: bool = True):
    cfg = INDICATORS[indicator]
    print(f"\n{'#' * 60}")
    print(f"Running: {indicator.upper()}" + (f" ({region})" if indicator == "sea_level" else ""))
    print(f"{'#' * 60}")

    if indicator == "sea_level":
        df = cfg["loader"](region = region)
    else:
        df = cfg["loader"]()

    series = df[cfg["value_col"]].values.astype(float)
    dates = df["date"].values
    test_size = 24
    tag = f"{indicator}_{region}" if indicator == "sea_level" else indicator

    print(f"Series length: {len(series)}")
    print(f"Test size: {test_size}")

    print("\n[1/4] Evaluating default baselines...")
    defaults = default_models_rmse(series, test_size)
    for k, v in defaults.items():
        print(f"{k:<20} RMSE = {v:.4f}")

    print(F"\n[2/4] Running GA (pop = {pop_size}, gens = {n_generations})...")
    ga = GeneticAlgorithm(
        series, test_size = test_size,
        pop_size = pop_size, n_generations = n_generations,
        seed = seed, verbose = verbose
    )
    best_ga, ga_log = ga.run()

    n_evals = pop_size * n_generations
    print(f"\n[3/4] Running Random Search ({n_evals} evaluations)...")
    rs = RandomSearch(series, test_size = test_size, n_evals = n_evals, seed = seed + 1, verbose = verbose)
    best_rs, rs_log = rs.run()

    comparison = {**defaults, "GA": best_ga.rmse, "Random Search": best_rs.rmse}
    print("\nStrategy Comparison")
    for k, v in sorted(comparison.items(), key = lambda x: x[1]):
        flag = " <- best" if v == min(comparison.values()) else ""
        print(f"{k:<25} RMSE = {v:.4f}{flag}")

    print(f"\n[4/4] Generating 2025 - 2050 projections...")
    n_proj = 25 * 12
    proj, ci_lo, ci_hi = project_future(best_ga, series, n_proj)

    last_date = pd.Timestamp(dates[-1])
    future_dates = pd.date_range(
        last_date + pd.DateOffset(months = 1),
        periods = n_proj, freq = "MS"
    )

    plot_fitness_curve(ga_log, title = f"{indicator.capitalize()} - GA Convergence", save_path = f"{RESULTS_DIR}/{tag}_fitness.png")
    plot_type_diversity(ga_log, title = f"{indicator.capitalize()} - Population Diversity", save_path = f"{RESULTS_DIR}/{tag}_diversity.png")
    plot_comparison(comparison, title = f"{indicator.capitalize()} - Strategy Comparison", save_path = f"{RESULTS_DIR}/{tag}_comparison.png")
    plot_projections(
        historical_dates = dates,
        historical_values = series,
        future_dates = future_dates,
        future_values = proj,
        ci_lower = ci_lo, ci_upper = ci_hi,
        title = f"{indicator.capitalize()} Projection 2025 - 2050",
        ylabel = cfg["ylabel"],
        save_path = f"{RESULTS_DIR}/{tag}_projection.png"
    )

    summary = {
        "indicator": indicator,
        "region": region,
        "series_len": len(series),
        "test_size": test_size,
        "best_ga": {
            "model_type": best_ga.model_type,
            "rmse": best_ga.rmse,
            "hparams": best_ga.hparams,
        },
        "best_rs": {
            "model_type": best_rs.model_type,
            "rmse": best_rs.rmse,
            "hparams": best_rs.hparams,
        },
        "comparison": comparison,
    }
    with open(f"{RESULTS_DIR}/{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent = 2, default = str)

    print(f"\nResults saved to {RESULTS_DIR}/{tag}_*.png/json")
    return summary

# CLI

def parse_args():
    p = argparse.ArgumentParser(
        description= "Climate Forecasting with GA Hyperparameter Optimisation"
    )
    p.add_argument("--indicator", choices = list(INDICATORS.keys()) + ["all"],
                   default = "temperature",
                   help = "Which climate indicator to forcast (default: temperature)")
    p.add_argument("--all", action = "store_true",
                   help = "Run all three indicators")
    p.add_argument("--pop",  type=int, default=30,
                   help="GA population size (default: 30)")
    p.add_argument("--gens", type=int, default=50,
                   help="Number of GA generations (default: 50)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--region", type=str, default="global",
                   choices=["global", "indian_ocean", "bay_of_bengal", "arabian_sea"],
                   help="Sea-level region (only used when indicator=sea_level)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-generation output")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    verbose = not args.quiet

    if args.all or args.indicator == "all":
        indicators = ["temperature", "co2", "sea_level"]
        regions = ["global", "indian_ocean", "bay_of_bengal", "arabian_sea"]
        all_results = {}
        for ind in indicators:
            if ind == "sea_level":
                for reg in regions:
                    r = run_indicator(ind, args.pop, args.gens, args.seed, region = reg, verbose = verbose)
                    all_results[f"{ind}_{reg}"] = r
            else:
                r = run_indicator(ind, args.pop, args.gens, args.seed, verbose = verbose)
                all_results[ind] = r

        with open(f"{RESULTS_DIR}/all_summary.json", "w") as f:
            json.dump(all_results, f, indent = 2,  default = str)
        print("\nAll indicators complete. See results/all_summary.json")
    else:
        run_indicator(args.indicator, args.pop, args.gens, args.seed, region = args.region, verbose = verbose)