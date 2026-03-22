import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional

# Colors
COLORS = {
    "LR": "#4C72B0",
    "ARIMA": "#DD8452",
    "LSTM": "#55A868",
    "GA": "#C44E52",
    "RS": "#8172B2",
    "default": "#4C72B0",
}

# 1. Fitness / RMSE convergence

def plot_fitness_curve(log: List[Dict], title: str = "GA Convergence", save_path: Optional[str] = None):
    gens = [e["generation"] for e in log]
    best_rmse = [e["best_rmse"] for e in log]
    mean_rmse = [e["mean_rmse"] for e in log]

    fig, ax = plt.subplots(figsize = (9, 4))
    ax.plot(gens, best_rmse, color = COLORS["GA"], lw = 2, label = "Best RMSE")
    ax.plot(gens, mean_rmse, color = COLORS["GA"], lw = 1.2, linestyle = "--", alpha = 0.6, label = "Mean RMSE")
    ax.set_xlabel("Generation")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha = 0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi = 150)
    return fig

# 2. Type diversity

def plot_type_diversity(log: List[Dict], title: str = "Population Diversity", save_path: Optional[str] = None):
    gens = [e["generation"] for e in log]
    types = ["LR", "ARIMA", "LSTM"]
    pop_size = sum(log[0]["type_counts"].values()) if log else 1

    fig, ax = plt.subplots(figsize = (9, 4))
    bottom = np.zeros(len(gens))
    for t in types:
        counts = np.array([e["type_counts"].get(t, 0)/pop_size for e in log])
        ax.bar(gens, counts, bottom = bottom, label = t, color = COLORS[t], alpha = 0.8, width = 0.8)
        bottom += counts

    ax.set_xlabel("Generation")
    ax.set_ylabel("Population share")
    ax.set_title(title)
    ax.legend(loc = "upper right")
    ax.grid(True, alpha = 0.2, axis = "y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi = 150)
    return fig

# 3. Forecast vs actuals

def plot_forecast(dates, actuals: np.ndarray, predictions: np.ndarray,
                  title: str = "Forecast vs Actual",
                  ylabel: str = "Value",
                  ci_lower: Optional[np.ndarray] = None,
                  ci_upper: Optional[np.ndarray] = None,
                  save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize = (11, 4))
    ax.plot(dates[:len(actuals)], actuals, color = "#333333", lw = 1.5, label = "Actual")
    ax.plot(dates[len(actuals) - len(predictions):len(actuals)],
            predictions, color = COLORS["GA"], lw = 2, linestyle = "--", label = "Predicted")
    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(
            dates[len(actuals) - len(predictions):len(actuals)], ci_lower, ci_upper,
            color = COLORS["GA"], alpha = 0.15, label = "95% CI"
        )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha = 0.25)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi = 150)
    return fig

# 4. Strategy comparison bar chart

def plot_comparison(results: Dict[str, float],
                    title: str = "RMSE by Optimisation Strategy",
                    save_path: Optional[str] = None):
    names = list(results.keys())
    values = [results[n] for n in names]
    colors = [COLORS.get(n.split()[0], COLORS["default"]) for n in names]

    fig, ax = plt.subplots(figsize = (8, 4))
    bars = ax.bar(names, values, color = colors, edgecolor = "white", linewidth = 0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                f"{val:.4f}", ha = "center", va = "bottom", fontsize = 9)
    ax.set_ylabel("RMSE (test set)")
    ax.set_title(title)
    ax.grid(True, axis = "y", alpha = 0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi = 150)
    return fig

# 5. Future climate projections

def plot_projections(historical_dates, historical_values: np.ndarray,
                     future_dates, future_values: np.ndarray,
                     ci_lower: Optional[np.ndarray] = None,
                     ci_upper: Optional[np.ndarray] = None,
                     title: str = "Climate Projection 2025 - 2050",
                     ylabel: str = "Value",
                     save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize = (12, 5))
    ax.plot(historical_dates, historical_values, color = "#444444", lw = 1.2, label = "Historical", alpha = 0.85)
    ax.plot(future_dates, future_values, color = COLORS["GA"], lw = 2.5, label = "Projection", zorder = 5)
    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(future_dates, ci_lower, ci_upper, color = COLORS["GA"], alpha = 0.18, label = "95% CI")
    ax.axvline(historical_dates[-1], color = "gray", lw = 0.8, linestyle = ":")
    ax.set_title(title, fontsize = 13)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha = 0.22)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi = 150)
    return fig