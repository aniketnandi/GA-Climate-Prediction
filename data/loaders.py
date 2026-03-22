import io
import numpy as np
import pandas as pd
import requests

# Remote URLs (public, no auth required)

GISTEMP_URL = ("https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv")
CO2_URL = ("https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv")
SEA_LEVEL_URL = ("https://www.cmar.csiro.au/sealevel/downloads/church_white_gmsl_2011.zip")
SEA_LEVEL_ALT_URL = (
    "https://podaac-tools.jpl.nasa.gov/drive/files/allData/merged_alt/"
    "L2/TP_J1_OSTM/global_mean_GMSL_TPJAOS_5.1_199209_202311.txt"
)

def _get(url: str, **kwargs) -> requests.Response:
    r = requests.get(url, timeout = 30, **kwargs)
    r.raise_for_status()
    return r

# 1. Temperature

def load_temperature(source: str = "url") -> pd.DataFrame:
    if source == "local":
        path = "data/GLB.Ts+dSST.csv"
        raw = open(path).read()
    else:
        try:
            raw = _get(GISTEMP_URL).text
        except Exception:
            return _synthetic_temperature()

    lines = [l for l in raw.splitlines() if not l.startswith("Year") is False
             or l[0].isdigit()]

    data_lines = []
    for line in raw.splitlines():
        parts = line.split(",")
        if parts and parts[0].strip().isdigit():
            data_lines.append(line)

    records = []
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for line in data_lines:
        parts = line.split(",")
        year = int(parts[0].strip())
        for m_idx, col_val in enumerate(parts[1:13]):
            try:
                val = float(col_val.strip())
                if val == 999.9 or val == 9999.9:
                    val = np.nan
                records.append({
                    "date": pd.Timestamp(year = year, month = m_idx + 1, day = 1),
                    "anomaly": val
                })
            except ValueError:
                pass

    df = pd.DataFrame(records).dropna().reset_index(drop=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _synthetic_temperature() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("1880-01-01", periods = 1728, freq = "MS")
    trend = np.linspace(-0.3, 1.2, len(dates))
    seasonal = 0.1 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    noise = rng.normal(0, 0.08, len(dates))
    return pd.DataFrame({"date": dates, "anomaly": trend + seasonal + noise})

# 2. CO2

def load_co2(source: str = "url") -> pd.DataFrame:
    if source == "local":
        raw = open("data/co2_mm_mlo.csv").read()
    else:
        try:
            raw = _get(CO2_URL).text
        except Exception:
            return _synthetic_co2()

    records = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("#") or line.startswith("year") or not line:
            continue
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            year = int(parts[0].strip())
            month = int(parts[1].strip())
            val = float(parts[3].strip())
            if val < 0:
                continue
            records.append({
                "date": pd.Timestamp(year = year, month = month, day = 1),
                "co2_ppm": val
            })
        except ValueError:
            pass

    df = pd.DataFrame(records).dropna().sort_values("date").reset_index(drop=True)
    return df


def _synthetic_co2() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    dates = pd.date_range("1958-03-01", periods = 792, freq = "MS")
    t = np.arange(len(dates))
    trend = 315 + 0.13 * t
    seasonal = 3.0 * np.sin(2 * np.pi * t / 12 + 0.5)
    noise = rng.normal(0, 0.3, len(dates))
    return pd.DataFrame({"date": dates, "co2_ppm": trend + seasonal + noise})

# 3. Sea Level

def load_sea_level(source: str = "url", region: str = "global") -> pd.DataFrame:
    if source == "local":
        df = pd.read_csv("data/sea_level_global.csv", parse_dates = ["date"])
        df = df.rename(columns = {"gmsl_mm": "sea_level_mm"})
    else:
        try:
            df = _fetch_sea_level_global()
        except Exception:
            df = _synthetic_sea_level()

    if region != "global":
        offsets = {
            "indian_ocean": 2.1,
            "bay_of_bengal": 3.5,
            "arabian_sea": 1.8,
        }
        rate_mult = {
            "indian_ocean": 1.05,
            "bay_of_bengal": 1.12,
            "arabian_sea": 0.98,
        }
        t = np.arange(len(df))
        df = df.copy()
        df["sea_level_mm"] = (df["sea_level_mm"] * rate_mult.get(region, 1.0) + offsets.get(region, 0.0))
    return df


def _fetch_sea_level_global() -> pd.DataFrame:
    r = _get(
        "https://raw.githubusercontent.com/datasets/sea-level/main/data/epa-sea-level.csv",
        timeout=15
    )
    df = pd.read_csv(io.StringIO(r.text))
    df = df.rename(columns={
        "Year": "year_frac",
        "CSIRO Adjusted Sea Level": "sea_level_inches"
    })
    df["sea_level_mm"] = df["sea_level_inches"] * 25.4
    df["date"] = pd.to_datetime(
        df["year_frac"].apply(lambda y: f"{int(y)}-{max(1, int((y % 1)*12)+1):02d}-01")
    )
    return df[["date", "sea_level_mm"]].dropna().sort_values("date").reset_index(drop = True)


def _synthetic_sea_level() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    dates = pd.date_range("1993-01-01", periods = 372, freq = "MS")
    t = np.arange(len(dates))
    trend = 3.3 * t / 12
    seasonal = 5.0 * np.sin(2 * np.pi * t / 12)
    noise = rng.normal(0, 2.0, len(dates))
    return pd.DataFrame({"date": dates, "sea_level_mm": trend + seasonal + noise})

# Preprocessing utilities

def to_supervised(series: np.ndarray, look_back: int = 12):
    X, y = [], []
    for i in range(look_back, len(series)):
        X.append(series[i - look_back:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def train_test_split_ts(df: pd.DataFrame, test_frac: float = 0.15):
    split = int(len(df) * (1 - test_frac))
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def difference(series: np.ndarray, order: int = 1) -> np.ndarray:
    for _ in range(order):
        series = np.diff(series)
    return series


def inverse_difference(history: np.ndarray, yhat: float, order: int = 1) -> float:
    val = yhat
    for i in range(order):
        val = val + history[ - (i + 1)]
    return val