"""
test_probability.py: checks for the kill-likelihood model (predict/probability.py).

Run from the project root:

    .venv/bin/python -m pytest tests/
"""

import numpy as np

from predict.probability import FEATURES, LogisticModel


def test_fit_learns_a_real_signal_and_stays_calibrated():
    """Made-up data where the true chance is 10% far apart and 60% close together."""
    rng = np.random.default_rng(0)
    n = 4000
    close = rng.random(n) < 0.3
    table = np.zeros((n, len(FEATURES)))
    table[:, 0] = np.where(close, 0.9, 0.1) + rng.normal(0, 0.02, n)   # "nearness"
    labels = (rng.random(n) < np.where(close, 0.6, 0.1)).astype(float)

    model = LogisticModel.fit(table, labels)
    p = model.predict(table)
    assert abs(p[close].mean() - 0.6) < 0.05
    assert abs(p[~close].mean() - 0.1) < 0.05


def test_save_and_load_give_the_same_predictions(tmp_path):
    rng = np.random.default_rng(1)
    table = rng.random((200, len(FEATURES)))
    labels = (rng.random(200) < table[:, 0]).astype(float)
    model = LogisticModel.fit(table, labels)
    path = tmp_path / "model.json"
    model.save(path, {"note": "test"})
    again = LogisticModel.load(path)
    assert np.allclose(model.predict(table), again.predict(table))
