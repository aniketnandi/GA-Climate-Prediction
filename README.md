# Multi-Model Climate Forecasting with Genetic Algorithm-Based Model Selection and Hyperparameter Optimization

**CS5100 - Foundations of Artificial Intelligence**
Aniket Nandi, Akshay Ashok Bannatti
---
## Overview
This project applies a Genetic Algorithm (GA) to jointly select and tune forecasting models for three climate indicators:
- **Global surface temperature anomaly** (NASA GISS)
- **Atmospheric CO2 concentration** (NOAA Mauna Loa)
- **Mean sea level change** (CSIRO/NOAA global + Indian Ocean sub-regions)

Each GA chromosome encodes a complete modelling pipeline (model type + hyperparameters).
The GA evolves optimal configurations through tournament selection, type-aware crossover, and parametric/structural mutation.
---
## Project Structure
```aiignore
GA Climate Prediction Project/
|--- data/
|     |--- loaders.py   # For data loading & preprocessing of all 3 datasets
|--- models/
|     |--- statistical.py   # Linear Regression + ARIMA wrappers
|     |--- lstm_model.py    # LSTM wrapper (TF/Keras, with Ridge fallback)
|--- ga/
|     |--- chromosome.py    # Chromosome encoding + mutation + complexity penalty
|     |--- crossover.py     # Type-aware uniform crossover
|     |--- engine.py    # Full GA engine (selection, evolution, logging)
|     |--- random_search.py     # Random search baseline (equivalent budged)
|--- utils/
|     |--- visualise.py     # Plotting helpers
|--- notebooks/
|     |--- climate_forecasting_ga.ipynb     # Main notebook
|--- results/   # Auto-created; stores PNGs and JSON summaries
|--- main.py    # CLI pipeline
|--- requirements.txt
```
---
## Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run via CLI (single indicator)
python main.py --indicator temperature --pop 10 --gens 10 --seed 42

# 3. Run all indicators
python main.py --all --pop 10 --gens 10

# 4. Fast run
python main.py --all --pop 2 -- gens 2

# 5. Or open the notebook
jupyter notebook notebooks/climate_forecasting_ga.ipynb
```

### CLI Options
Flag -> Default -> Description

`--indicator` -> `temperature` -> `temperature`, `co2`, `sea_level`, or `all`

`--pop` -> `30` -> GA population size

`--gens` -> `50` -> Number of generations

`--seed` -> `42` -> Random seed

`--region` -> `global` -> Sea-level region (`global`, `indian_ocean`, `bay_of_bengal`, `arabian_sea`)

`--quiet` -> off -> Suppress per-generation output

---
## GA Design
Component -> Detail

**Chromosome** -> model_type {LR, ARIMA, LSTM} + full hparam dict

**Fitness** -> `1 / RMSE + complexity_penalty)` - higher is better

**Selection** -> Tournament selection (k = 3)

**Crossover** -> Type-aware uniform crossover; cross-type parents preserve model integrity

**Mutation** -> Parametric (per-gene swap, rate = 0.2) + structural (model-type switch, rate = 0.05)

**Elitism** -> Top-2 individuals carried forward unchanged

---
## Outputs (in `results/`)
- `{indicator}_fitness.png` - GA convergence curve
- `{indicator}_diversity.png` - population model-type share over generations
- `{indicator}_comparison.png` - RMSE bar chart: defaults vs GA vs random search
- `{indicator}_forecast.png` - actual vs predicted on the test set
- `{indicator}_projection.png` - 2025 – 2050 forecast with 95% CI
- `{indicator}_summary.json` - best hyperparameters and all RMSE values
- `ablation.png` - sensitivity to pop size, mutation rate, crossover rate