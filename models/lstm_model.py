import numpy as np
import warnings
warnings.filterwarnings("ignore")

_TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    tf.get_logger().setLevel("ERROR")
    _TF_AVAILABLE = True
except ImportError:
    pass


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


class LSTMModel:

    def __init__(
        self,
        look_back: int = 12,
        n_layers: int = 1,
        units: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
    ):
        self.look_back = look_back
        self.n_layers = n_layers
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        self._model = None
        self._scaler_mean = None
        self._scaler_std = None
        self._history_seq = None

    # Helpers

    def _scale(self, arr):
        return (arr - self._scaler_mean) / (self._scaler_std + 1e-8)

    def _unscale(self, arr):
        return arr * (self._scaler_std + 1e-8) + self._scaler_mean

    def _make_sequences(self, series):
        X, y = [], []
        for i in range(self.look_back, len(series)):
            X.append(series[i - self.look_back:i])
            y.append(series[i])
        return np.array(X)[..., np.newaxis], np.array(y)

    def _build_model(self):
        if not _TF_AVAILABLE:
            return None
        inp = keras.Input(shape=(self.look_back, 1))
        x = inp
        for i in range(self.n_layers):
            return_seq = i < self.n_layers - 1
            x = layers.LSTM(self.units, return_sequences=return_seq)(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout)(x)
        out = layers.Dense(1)(x)
        model = keras.Model(inp, out)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse"
        )
        return model

    # Public API

    def fit(self, train_series: np.ndarray):
        self._scaler_mean = train_series.mean()
        self._scaler_std = train_series.std()
        scaled = self._scale(train_series)
        self._history_seq = scaled[-self.look_back:].copy()

        if not _TF_AVAILABLE:
            from sklearn.linear_model import Ridge
            X, y = [], []
            for i in range(self.look_back, len(scaled)):
                X.append(scaled[i - self.look_back:i])
                y.append(scaled[i])
            self._surrogate = Ridge(alpha=1.0).fit(np.array(X), np.array(y))
            return

        X, y = self._make_sequences(scaled)
        val_size = max(1, int(len(X) * 0.1))
        X_train, X_val = X[:-val_size], X[-val_size:]
        y_train, y_val = y[:-val_size], y[-val_size:]

        self._model = self._build_model()
        cb = keras.callbacks.EarlyStopping(
            monitor = "val_loss", patience = self.patience, restore_best_weights = True
        )
        self._model.fit(
            X_train, y_train,
            validation_data = (X_val, y_val),
            epochs = self.epochs,
            batch_size = self.batch_size,
            callbacks = [cb],
            verbose = 0,
        )

    def predict(self, steps: int) -> np.ndarray:
        assert self._history_seq is not None, "Call fit() first"
        history = list(self._history_seq)
        preds_scaled = []

        for _ in range(steps):
            seq = np.array(history[-self.look_back:])
            if _TF_AVAILABLE and self._model is not None:
                x = seq.reshape(1, self.look_back, 1)
                yhat = float(self._model.predict(x, verbose = 0)[0, 0])
            else:
                x = seq.reshape(1, -1)
                yhat = float(self._surrogate.predict(x)[0])
            preds_scaled.append(yhat)
            history.append(yhat)

        return self._unscale(np.array(preds_scaled))

    def walk_forward_rmse(self, series: np.ndarray, test_size: int) -> float:
        n_train = len(series) - test_size
        self.fit(series[:n_train])
        preds = self.predict(test_size)
        return _rmse(series[n_train:], preds)

    @property
    def model_type(self):
        return "LSTM"

    @property
    def tf_available(self):
        return _TF_AVAILABLE