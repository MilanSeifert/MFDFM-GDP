"""Comprehensive tests for the MFDFM implementation using unittest."""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mfdfm import MFDFM
from mfdfm.data import MFData
from mfdfm.kalman import kalman_filter, kalman_smoother
from mfdfm.estimation import estimate_pca_factors, estimate_loadings, estimate_var


def generate_synthetic_data(
    T=360, n_Q=5, n_M=20, r=2, seed=42, ragged_edge_months=2,
):
    """Generate synthetic mixed-frequency data from a known DGP."""
    rng = np.random.RandomState(seed)
    n = n_Q + n_M

    # Stable VAR(1)
    Phi = rng.randn(r, r) * 0.3
    max_eig = np.max(np.abs(np.linalg.eigvals(Phi)))
    if max_eig >= 0.95:
        Phi *= 0.9 / max_eig

    A = rng.randn(r, r) * 0.3
    Sigma_vv = A @ A.T + 0.1 * np.eye(r)

    factors = np.zeros((T, r))
    for t in range(1, T):
        v = rng.multivariate_normal(np.zeros(r), Sigma_vv)
        factors[t] = Phi @ factors[t - 1] + v

    Lambda = rng.randn(n, r) * 0.5 + 0.3
    Lambda[0, 0] = abs(Lambda[0, 0]) + 0.5
    Sigma_ww = np.abs(rng.randn(n)) * 0.3 + 0.2

    u = np.zeros((T, n))
    for i in range(n):
        u[:, i] = rng.normal(0, np.sqrt(Sigma_ww[i]), T)
    y_star = factors @ Lambda.T + u

    Y = np.full((T, n), np.nan)
    Y[:, n_Q:] = y_star[:, n_Q:]

    dates = pd.date_range("1990-01-01", periods=T, freq="MS")
    for t in range(2, T):
        if dates[t].month in (3, 6, 9, 12):
            for i in range(n_Q):
                Y[t, i] = (y_star[t, i] + y_star[t-1, i] + y_star[t-2, i]) / 3.0

    # Ragged edge
    if ragged_edge_months > 0:
        n_ragged = max(1, n_M // 3)
        ragged_vars = rng.choice(n_M, n_ragged, replace=False)
        for j in ragged_vars:
            Y[-ragged_edge_months:, n_Q + j] = np.nan

    # Ragged start
    n_late = max(1, n_M // 5)
    late_vars = rng.choice(n_M, n_late, replace=False)
    for j in late_vars:
        late_months = rng.randint(12, 60)
        Y[:late_months, n_Q + j] = np.nan

    q_names = [f"Q_{i}" for i in range(n_Q)]
    q_names[0] = "GDP"
    m_names = [f"M_{j}" for j in range(n_M)]
    cols = q_names + m_names
    df = pd.DataFrame(Y, index=dates, columns=cols)

    return {
        "data": df,
        "quarterly_vars": q_names,
        "gdp_var": "GDP",
        "true_factors": factors,
        "true_Lambda": Lambda,
        "true_Phi": Phi,
        "true_Sigma_vv": Sigma_vv,
        "true_Sigma_ww": Sigma_ww,
        "true_bci": factors @ Lambda[0],
    }


class TestMFData(unittest.TestCase):

    def setUp(self):
        self.d = generate_synthetic_data(T=240, n_Q=3, n_M=10, r=2, seed=123)
        self.mfd = MFData(
            self.d["data"], self.d["quarterly_vars"], self.d["gdp_var"]
        )

    def test_dimensions(self):
        self.assertEqual(self.mfd.n_Q, 3)
        self.assertEqual(self.mfd.n_M, 10)
        self.assertEqual(self.mfd.n, 13)
        self.assertEqual(self.mfd.T, 240)
        self.assertEqual(self.mfd.var_names[0], "GDP")

    def test_quarter_end_mask(self):
        for t in range(self.mfd.T):
            month = self.mfd.dates[t].month
            expected = month in (3, 6, 9, 12)
            self.assertEqual(self.mfd.is_quarter_end[t], expected)

    def test_quarterly_nan_at_non_quarter(self):
        for t in range(self.mfd.T):
            if not self.mfd.is_quarter_end[t]:
                self.assertTrue(np.all(np.isnan(self.mfd.Y[t, :self.mfd.n_Q])))

    def test_balanced_panel(self):
        panel, var_mask, t_start, t_end = self.mfd.get_balanced_monthly_panel()
        self.assertFalse(np.any(np.isnan(panel)))
        self.assertGreaterEqual(t_end - t_start, self.mfd.min_balanced_years * 12)

    def test_standardization(self):
        for i in range(self.mfd.n):
            col = self.mfd.Y[:, i]
            valid = col[~np.isnan(col)]
            if len(valid) > 10:
                self.assertAlmostEqual(np.mean(valid), 0.0, delta=0.15)
                self.assertAlmostEqual(np.std(valid), 1.0, delta=0.15)


class TestKalman(unittest.TestCase):

    def test_known_system(self):
        T, s, n = 100, 2, 1
        rng = np.random.RandomState(0)

        F = np.array([[0.9, 0.0], [0.0, 0.5]])
        Q = np.eye(s) * 0.1
        H = np.array([[1.0, 0.5]])
        R_diag = np.array([0.5])

        xi_true = np.zeros((T, s))
        y = np.zeros((T, n))
        for t in range(1, T):
            xi_true[t] = F @ xi_true[t-1] + rng.multivariate_normal(np.zeros(s), Q)
            y[t] = H @ xi_true[t] + rng.normal(0, np.sqrt(R_diag[0]))

        kf = kalman_filter(
            y, lambda t: H, lambda t: R_diag,
            F, Q, np.zeros(s), np.eye(s) * 10,
        )
        corr_filt = np.corrcoef(kf.xi_filt[:, 0], xi_true[:, 0])[0, 1]
        self.assertGreater(corr_filt, 0.5)

        kf = kalman_smoother(kf, F)
        corr_smooth = np.corrcoef(kf.xi_smooth[:, 0], xi_true[:, 0])[0, 1]
        self.assertGreaterEqual(corr_smooth, corr_filt - 0.05)

    def test_smoother_reduces_variance(self):
        T, s = 50, 2
        rng = np.random.RandomState(1)
        F = np.array([[0.8, 0.1], [0.0, 0.7]])
        Q = np.eye(s) * 0.2
        H = np.array([[1.0, 0.0], [0.0, 1.0]])
        R_diag = np.array([0.3, 0.3])

        y = np.zeros((T, 2))
        xi = np.zeros(s)
        for t in range(T):
            xi = F @ xi + rng.multivariate_normal(np.zeros(s), Q)
            y[t] = H @ xi + rng.normal(0, np.sqrt(R_diag))

        kf = kalman_filter(
            y, lambda t: H, lambda t: R_diag,
            F, Q, np.zeros(s), np.eye(s) * 10,
        )
        kf = kalman_smoother(kf, F)

        for t in range(5, T - 1):
            self.assertLessEqual(
                np.trace(kf.P_smooth[t]),
                np.trace(kf.P_filt[t]) + 1e-8,
            )

    def test_missing_observations(self):
        T, s = 100, 1
        rng = np.random.RandomState(2)
        F = np.array([[0.9]])
        Q = np.array([[0.1]])
        H = np.array([[1.0]])
        R_diag = np.array([0.5])

        xi_true = np.zeros(T)
        y = np.full((T, 1), np.nan)
        for t in range(1, T):
            xi_true[t] = 0.9 * xi_true[t-1] + rng.normal(0, np.sqrt(0.1))
        for t in range(0, T, 3):
            y[t, 0] = xi_true[t] + rng.normal(0, np.sqrt(0.5))

        kf = kalman_filter(
            y, lambda t: H, lambda t: R_diag,
            F, Q, np.zeros(1), np.eye(1) * 10,
        )
        kf = kalman_smoother(kf, F)
        corr = np.corrcoef(kf.xi_smooth[:, 0], xi_true)[0, 1]
        self.assertGreater(corr, 0.4)


class TestEstimation(unittest.TestCase):

    def setUp(self):
        self.d = generate_synthetic_data(T=240, n_Q=3, n_M=10, r=2, seed=123)
        self.mfd = MFData(
            self.d["data"], self.d["quarterly_vars"], self.d["gdp_var"]
        )

    def test_pca_factors(self):
        f_pc, eigenvalues, V, t_start, t_end = estimate_pca_factors(self.mfd, 2)
        self.assertEqual(f_pc.shape, (self.mfd.T, 2))
        self.assertGreaterEqual(eigenvalues[0], eigenvalues[1])
        self.assertFalse(np.any(np.isnan(f_pc[t_start:t_end])))

    def test_loadings(self):
        f_pc, _, _, _, _ = estimate_pca_factors(self.mfd, 2)
        Lambda, Sigma_ww = estimate_loadings(self.mfd, f_pc, 2)
        self.assertEqual(Lambda.shape, (self.mfd.n, 2))
        self.assertTrue(np.all(Sigma_ww > 0))

    def test_var(self):
        f_pc, _, _, _, _ = estimate_pca_factors(self.mfd, 2)
        Phi, Sigma_vv = estimate_var(f_pc, 1)
        self.assertEqual(Phi.shape, (2, 2))
        eigvals = np.abs(np.linalg.eigvals(Phi))
        self.assertTrue(np.all(eigvals < 1.5))  # roughly stable


class TestMFDFM(unittest.TestCase):

    def setUp(self):
        self.d = generate_synthetic_data()
        self.d_small = generate_synthetic_data(T=240, n_Q=3, n_M=10, r=2, seed=123)

    def test_fit_and_bci(self):
        model = MFDFM(n_factors=2, n_lags=1)
        model.fit(self.d["data"], self.d["quarterly_vars"], self.d["gdp_var"])

        self.assertTrue(model.is_fitted)
        bci = model.business_cycle_index
        self.assertIsInstance(bci, pd.Series)
        self.assertEqual(len(bci), len(self.d["data"]))
        self.assertFalse(np.any(np.isnan(bci)))

    def test_bci_correlates_with_true(self):
        model = MFDFM(n_factors=2, n_lags=1)
        model.fit(self.d["data"], self.d["quarterly_vars"], self.d["gdp_var"])

        bci = model.business_cycle_index.values
        true_bci = self.d["true_bci"]
        corr = np.abs(np.corrcoef(bci, true_bci)[0, 1])
        self.assertGreater(corr, 0.3, f"BCI correlation: {corr:.3f}")

    def test_smoothed_factors(self):
        model = MFDFM(n_factors=2, n_lags=1)
        model.fit(self.d["data"], self.d["quarterly_vars"], self.d["gdp_var"])

        factors = model.smoothed_factors
        self.assertEqual(factors.shape, (self.d["data"].shape[0], 2))
        self.assertFalse(np.any(np.isnan(factors)))

    def test_bci_variance(self):
        model = MFDFM(n_factors=2, n_lags=1)
        model.fit(self.d["data"], self.d["quarterly_vars"], self.d["gdp_var"])

        var = model.bci_variance
        self.assertTrue(np.all(var >= 0))

    def test_fitted_values(self):
        model = MFDFM(n_factors=2, n_lags=1)
        model.fit(self.d["data"], self.d["quarterly_vars"], self.d["gdp_var"])

        fitted = model.fitted_values()
        self.assertEqual(fitted.shape[1], model.data_.n)

        # Monthly fitted values should have no NaN
        for j in range(model.data_.n_Q, model.data_.n):
            self.assertFalse(
                np.any(np.isnan(fitted.iloc[:, j].values)),
                f"NaN in fitted monthly var {j}",
            )

    def test_predict(self):
        model = MFDFM(n_factors=2, n_lags=1)
        model.fit(self.d["data"], self.d["quarterly_vars"], self.d["gdp_var"])

        forecast = model.predict(h=6)
        self.assertEqual(forecast.shape, (6, model.data_.n))
        self.assertFalse(np.any(np.isnan(forecast)))

    def test_summary(self):
        model = MFDFM(n_factors=2, n_lags=1)
        model.fit(self.d["data"], self.d["quarterly_vars"], self.d["gdp_var"])
        summary = model.summary()
        self.assertIn("Mixed-Frequency Dynamic Factor Model", summary)

    def test_save_load(self):
        model = MFDFM(n_factors=2, n_lags=1)
        model.fit(self.d["data"], self.d["quarterly_vars"], self.d["gdp_var"])
        bci_before = model.business_cycle_index.values.copy()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = MFDFM.load(path)
            np.testing.assert_array_almost_equal(
                bci_before, loaded.business_cycle_index.values
            )
        finally:
            os.unlink(path)

    def test_not_fitted_raises(self):
        model = MFDFM()
        with self.assertRaises(RuntimeError):
            _ = model.business_cycle_index

    def test_single_factor(self):
        model = MFDFM(n_factors=1, n_lags=1)
        model.fit(
            self.d_small["data"],
            self.d_small["quarterly_vars"],
            self.d_small["gdp_var"],
        )
        bci = model.business_cycle_index
        self.assertEqual(len(bci), len(self.d_small["data"]))

    def test_multiple_lags(self):
        model = MFDFM(n_factors=2, n_lags=2)
        model.fit(
            self.d_small["data"],
            self.d_small["quarterly_vars"],
            self.d_small["gdp_var"],
        )
        bci = model.business_cycle_index
        self.assertEqual(len(bci), len(self.d_small["data"]))

    def test_observation_weights(self):
        model = MFDFM(n_factors=2, n_lags=1)
        model.fit(
            self.d_small["data"],
            self.d_small["quarterly_vars"],
            self.d_small["gdp_var"],
        )
        weights = model.observation_weights()
        self.assertIn("bci_weights", weights)
        self.assertEqual(
            weights["bci_weights"].shape,
            (model.data_.T, model.data_.n),
        )


class TestIntegration(unittest.TestCase):

    def test_gdp_only_quarterly(self):
        """Model with only GDP as quarterly variable."""
        rng = np.random.RandomState(99)
        T, n_M = 240, 15
        dates = pd.date_range("2000-01-01", periods=T, freq="MS")

        factors = np.zeros((T, 2))
        for t in range(1, T):
            factors[t] = 0.8 * factors[t-1] + rng.randn(2) * 0.3

        Lambda = rng.randn(1 + n_M, 2) * 0.5
        y_star = factors @ Lambda.T + rng.randn(T, 1 + n_M) * 0.3

        gdp = np.full(T, np.nan)
        for t in range(2, T):
            if dates[t].month in (3, 6, 9, 12):
                gdp[t] = (y_star[t, 0] + y_star[t-1, 0] + y_star[t-2, 0]) / 3

        data = {"GDP": gdp}
        for i in range(n_M):
            data[f"M_{i}"] = y_star[:, 1 + i]
        df = pd.DataFrame(data, index=dates)

        model = MFDFM(n_factors=2, n_lags=1)
        model.fit(df, quarterly_vars=["GDP"], gdp_var="GDP")

        bci = model.business_cycle_index
        self.assertEqual(len(bci), T)
        self.assertFalse(np.any(np.isnan(bci)))

    def test_heavy_missing_data(self):
        """Model with ragged start/end and publication lags."""
        rng = np.random.RandomState(77)
        T, n_M = 240, 15
        dates = pd.date_range("2000-01-01", periods=T, freq="MS")

        factors = np.zeros((T, 2))
        for t in range(1, T):
            factors[t] = 0.7 * factors[t-1] + rng.randn(2) * 0.3

        Lambda = rng.randn(3 + n_M, 2) * 0.4
        y_star = factors @ Lambda.T + rng.randn(T, 3 + n_M) * 0.3

        Y = np.full((T, 3 + n_M), np.nan)
        for t in range(2, T):
            if dates[t].month in (3, 6, 9, 12):
                for i in range(3):
                    Y[t, i] = (y_star[t, i] + y_star[t-1, i] + y_star[t-2, i]) / 3

        # Monthly vars: ragged starts and ragged ends
        for j in range(n_M):
            start = rng.randint(0, 36)  # start 0-3 years late
            end = T - rng.randint(0, 6)  # end 0-6 months early
            Y[start:end, 3 + j] = y_star[start:end, 3 + j]

        cols = ["GDP", "Q1", "Q2"] + [f"M_{i}" for i in range(n_M)]
        df = pd.DataFrame(Y, index=dates, columns=cols)

        model = MFDFM(n_factors=2, n_lags=1)
        model.fit(df, quarterly_vars=["GDP", "Q1", "Q2"], gdp_var="GDP")

        bci = model.business_cycle_index
        self.assertFalse(np.any(np.isnan(bci)))


if __name__ == "__main__":
    unittest.main()
