import time

import numpy as np
from tensorflow import keras


def run_autoencoder(
    X,
    epochs: int = 30,
    batch_size: int = 32,
    threshold_percentile: float = 95.0,
    random_state: int = 42,
) -> dict:
    start = time.perf_counter()
    keras.utils.set_random_seed(random_state)
    feature_count = X.shape[1]
    hidden = max(2, feature_count // 2)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(feature_count,)),
            keras.layers.Dense(hidden, activation="relu"),
            keras.layers.Dense(hidden // 2 or 1, activation="relu"),
            keras.layers.Dense(hidden, activation="relu"),
            keras.layers.Dense(feature_count, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(
        X,
        X,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,
    )
    reconstructed = model.predict(X, verbose=0)
    scores = np.mean(np.square(X - reconstructed), axis=1)
    threshold = np.percentile(scores, threshold_percentile)
    predictions = scores >= threshold

    return {
        "name": "Autoencoder",
        "model": model,
        "predictions": np.asarray(predictions, dtype=bool),
        "scores": np.asarray(scores, dtype=float),
        "threshold": float(threshold),
        "history": history.history,
        "processing_time": time.perf_counter() - start,
    }

