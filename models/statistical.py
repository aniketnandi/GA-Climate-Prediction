import warnings
import logging
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
logging.getLogger("statsmodels").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Shared utilities

def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))

def _mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_all(y_true, y_pred) -> dict:
    return {
        "rmse": _rmse(y_true, y_pred),
        "mae":  _mae(y_true, y_pred),
        "mape": _mape(y_true, y_pred),
    }

# Linear Regression (with lag features)

class LinearRegressionModel:

    def __init__(self, look_back: int = 12, alpha: float = 1.0,
                 diff_order: int = 0):
        self.look_back = look_back
        self.alpha = alpha
        self.diff_order = diff_order
        self._model = Ridge(alpha=self.alpha)
        self._scaler = StandardScaler()
        self._history = None
        self._fitted = False

    # Internal helpers

    def _make_features(self, series: np.ndarray):
        X, y = [], []
        for i in range(self.look_back, len(series)):
            X.append(series[i - self.look_back:i])
            y.append(series[i])
        return np.array(X), np.array(y)

    def _apply_diff(self, series: np.ndarray):
        orig = series.copy()
        for _ in range(self.diff_order):
            series = np.diff(series)
        return series, orig

    def _invert_diff(self, forecasts: np.ndarray, history: np.ndarray):
        result = forecasts.copy()
        for d in range(self.diff_order):
            last = history[-(d + 1)]
            result = np.cumsum(np.insert(result, 0, last))[1:]
        return result

    # Public API

    def fit(self, train_series: np.ndarray):
        series, self._history = self._apply_diff(train_series)
        X, y = self._make_features(series)
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self._fitted_series = series.copy()
        self._fitted = True

    def predict(self, steps: int) -> np.ndarray:
        assert self._fitted, "Call fit() first"
        history = list(self._fitted_series)
        preds = []
        for _ in range(steps):
            x = np.array(history[-self.look_back:]).reshape(1, -1)
            x_scaled = self._scaler.transform(x)
            yhat = self._model.predict(x_scaled)[0]
            preds.append(yhat)
            history.append(yhat)
        preds = np.array(preds)
        if self.diff_order > 0:
            preds = self._invert_diff(preds, self._history)
        return preds

    def walk_forward_rmse(self, series: np.ndarray, test_size: int) -> float:
        n_train = len(series) - test_size
        actuals, predictions = [], []
        self.fit(series[:n_train])
        preds = self.predict(test_size)
        return _rmse(series[n_train:], preds)

    @property
    def model_type(self):
        return "LR"

# ARIMA

class ARIMAModel:

    def __init__(self, p: int = 1, d: int = 1, q: int = 0,
                 P: int = 0, D: int = 0, Q: int = 0, s: int = 12):
        self.p = p; self.d = d; self.q = q
        self.P = P; self.D = D; self.Q = Q; self.s = s
        self._result = None

    def _is_seasonal(self):
        return (self.P + self.D + self.Q) > 0 and self.s > 1

    def fit(self, train_series: np.ndarray):
        try:
            if self._is_seasonal():
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                model = SARIMAX(
                    train_series,
                    order = (self.p, self.d, self.q),
                    seasonal_order = (self.P, self.D, self.Q, self.s),
                    enforce_stationarity = False,
                    enforce_invertibility = False,
                )
            else:
                from statsmodels.tsa.arima.model import ARIMA
                model = ARIMA(train_series, order = (self.p, self.d, self.q))
            self._result = model.fit()
        except Exception as e:
            from statsmodels.tsa.arima.model import ARIMA
            self._result = ARIMA(train_series, order = (1, 1, 0)).fit()
        self._train_len = len(train_series)

    def predict(self, steps: int) -> np.ndarray:
        assert self._result is not None, "Call fit() first"
        fc = self._result.forecast(steps = steps)
        return np.array(fc)

    def walk_forward_rmse(self, series: np.ndarray, test_size: int) -> float:
        n_train = len(series) - test_size
        self.fit(series[:n_train])
        preds = self.predict(test_size)
        return _rmse(series[n_train:], preds)

    @property
    def model_type(self):
        return "ARIMA"

# Auto-ARIMA baseline (uses pmdarima)

def auto_arima_rmse(series: np.ndarray, test_size: int, seasonal: bool = True, m: int = 12) -> dict:
    try:
        import pmdarima as pm
    except ImportError:
        raise ImportError("Install pmdarima: pip install pmdarima")

    n_train = len(series) - test_size
    model = pm.auto_arima(
        series[:n_train],
        seasonal = seasonal, m = m,
        stepwise = True, suppress_warnings = True, error_action = "ignore"
    )
    preds = model.predict(n_periods = test_size)
    metrics = evaluate_all(series[n_train:], preds)
    metrics["order"] = model.order
    metrics["seasonal_order"] = model.seasonal_order
    return metrics