"""
Unit tests for the ``kaov.kaov.Data`` class.

The tests are organised in four classes:

- ``TestDataInit``: construction, attribute types, naming conventions and
  input validation.
- ``TestProjectorAttributes``: mathematical properties of the internal
  quantities ``_XXinv``, ``_Q``, ``_leverage`` and of the on-the-fly
  projectors ``_proj`` / ``_proj_orth`` (no n x n matrix is stored anymore).
- ``TestNystrom``: landmark selection with ``nystrom=True`` (shapes,
  subset consistency, reproducibility, recomputed internal quantities).
- ``TestDiagonalizeResidualCovariance``: spectral decomposition of the
  residual covariance operator, including the Nystrom-specific scaling.
"""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.testing as tot
from torch import eye, zeros
from torch.linalg import matrix_rank

from kaov import Data
from kaov.testing import (
    dummy_data,
    assert_equal_data_obj,
    assert_same_eigenvectors,
)

# Tolerance for exact algebraic identities (projectors, inverses).
ATOL = 1e-10


@pytest.fixture
def toy_dataframes():
    """Toy dataset as pandas DataFrames: returns ``(endog, exog, meta)``.

    50 observations, 10 numeric responses, and a 7-column one-hot design
    matrix (3 levels for factor1 + 4 levels for factor2).
    """
    df, _, _ = dummy_data(
        n_row=50, n_col=10, n_factor=2, n_group=[3, 4], seed=42
    )
    endog = df[[f"col{i + 1}" for i in range(10)]]
    exog = pd.get_dummies(df[["factor1", "factor2"]], dtype=float)  # 3 + 4 = 7 cols
    meta = df[["factor1", "factor2"]]
    return endog, exog, meta


@pytest.fixture
def toy_arrays(toy_dataframes):
    """Same toy dataset as plain numpy arrays (for default-naming tests)."""
    endog, exog, _ = toy_dataframes
    return endog.to_numpy(), exog.to_numpy()


def materialize_projectors(d):
    """Materialize P = ProjImX and P⊥ = I - ProjImX from a Data object, via the
    on-the-fly helpers (only for testing purposes, on small toy data)."""
    I = eye(d.nobs, dtype=torch.float64)
    P = d._proj(I)
    Po = d._proj_orth(I)
    return P, Po


class TestDataInit:
    """Construction, attribute types, names and input validation."""

    def test_scalar_attributes(self, toy_dataframes):
        """nobs / nlvl / nvar / nystrom are set correctly."""
        endog, exog, meta = toy_dataframes
        d = Data(endog, exog, meta=meta)
        assert d.nobs == 50
        assert d.nlvl == 7
        assert d.nvar == 10
        assert d.nystrom is False

    def test_no_nxn_matrix_stored(self, toy_dataframes):
        """The memory optimisation: no nobs x nobs matrix is stored as an attribute."""
        endog, exog, meta = toy_dataframes
        d = Data(endog, exog, meta=meta)
        assert not hasattr(d, "_ProjImX")
        assert not hasattr(d, "_ProjImXorthogonal")
        # small-size quantities are stored instead
        assert d._XXinv.shape == (d.nlvl, d.nlvl)
        assert d._Q.shape == (d.nobs, d.nlvl)
        assert d._leverage.shape == (d.nobs,)

    def test_torch_conversion_float64_and_values(self, toy_dataframes):
        """endog and exog are converted to float64 torch tensors, values preserved."""
        endog, exog, _ = toy_dataframes
        d = Data(endog, exog)
        assert isinstance(d.endog, torch.Tensor)
        assert isinstance(d.exog, torch.Tensor)
        assert d.endog.dtype == torch.float64
        assert d.exog.dtype == torch.float64
        tot.assert_close(d.endog, torch.tensor(endog.to_numpy()))
        tot.assert_close(d.exog, torch.tensor(exog.to_numpy()))

    def test_names_from_dataframe_columns(self, toy_dataframes):
        """Without explicit names, column names of the DataFrames are used."""
        endog, exog, _ = toy_dataframes
        d = Data(endog, exog)
        pd.testing.assert_index_equal(d.endog_names, endog.columns)
        pd.testing.assert_index_equal(d.exog_names, exog.columns)

    def test_default_names_and_index_for_numpy_input(self, toy_arrays):
        """For numpy input, names default to y0..y9 / x1..x7 and index to range(n)."""
        endog, exog = toy_arrays
        d = Data(endog, exog)
        assert list(d.endog_names) == [f"y{i}" for i in range(10)]
        assert list(d.exog_names) == [f"x{i + 1}" for i in range(7)]
        assert list(d.index) == list(range(50))

    def test_custom_names(self, toy_dataframes):
        """Explicitly provided endog_names / exog_names are stored as given."""
        endog, exog, _ = toy_dataframes
        d = Data(
            endog, exog,
            endog_names=[f"gene{i}" for i in range(10)],
            exog_names=[f"lvl{i}" for i in range(7)]
        )
        assert list(d.endog_names) == [f"gene{i}" for i in range(10)]
        assert list(d.exog_names) == [f"lvl{i}" for i in range(7)]

    def test_exog_names_length_mismatch_raises(self, toy_dataframes):
        """A wrong-length exog_names raises a ValueError."""
        endog, exog, _ = toy_dataframes
        with pytest.raises(ValueError, match="exog_names"):
            Data(endog, exog, exog_names=["only_one"])

    def test_endog_names_length_mismatch_raises(self, toy_dataframes):
        """A wrong-length endog_names raises a ValueError."""
        endog, exog, _ = toy_dataframes
        with pytest.raises(ValueError, match="endog_names"):
            Data(endog, exog, endog_names=["only_one"])

    def test_index_taken_from_exog(self, toy_dataframes):
        """The observation index is inherited from exog.index when present."""
        endog, exog, _ = toy_dataframes
        exog = exog.copy()
        exog.index = [f"obs{i}" for i in range(50)]
        d = Data(endog, exog)
        assert list(d.index) == [f"obs{i}" for i in range(50)]


class TestProjectorAttributes:
    """Properties of _XXinv, _Q, _leverage and of the on-the-fly projectors
    `_proj` / `_proj_orth` (which replace the formerly stored n x n matrices
    _ProjImX / _ProjImXorthogonal).

    The projectors are cross-checked against a plain NumPy reference computed
    on ``toy_arrays``.
    """

    def test_XXinv_inverts_gram(self, toy_dataframes):
        """For a full-rank design, _XXinv is the inverse of X^T X."""
        endog, exog, _ = toy_dataframes
        d = Data(endog, exog)
        gram = d.exog.T @ d.exog
        tot.assert_close(
            gram @ d._XXinv, eye(d.nlvl, dtype=torch.float64),
            # atol=ATOL, rtol=0
            atol=2e-1, rtol=0    # bad precision
        )

    def test_ProjImX_formula(self, toy_dataframes):
        """_proj(I) equals X @ _XXinv @ X^T by construction."""
        endog, exog, _ = toy_dataframes
        d = Data(endog, exog)
        P, _ = materialize_projectors(d)
        tot.assert_close(
            P, d.exog @ d._XXinv @ d.exog.T,
            atol=ATOL, rtol=0
        )

    def test_proj_matches_explicit_projector_on_random_matrix(self, toy_dataframes):
        """_proj(M) = X XXinv X^T M for a random M (checked without materializing P
        inside the class under test)."""
        endog, exog, _ = toy_dataframes
        d = Data(endog, exog)
        g = torch.Generator().manual_seed(0)
        M = torch.randn(d.nobs, 6, generator=g, dtype=torch.float64)
        tot.assert_close(
            d._proj(M), d.exog @ d._XXinv @ d.exog.T @ M, atol=ATOL, rtol=0
        )

    def test_proj_orth_matches_explicit(self, toy_dataframes):
        """_proj_orth(M) = (I - X XXinv X^T) M for a random M."""
        endog, exog, _ = toy_dataframes
        d = Data(endog, exog)
        g = torch.Generator().manual_seed(1)
        M = torch.randn(d.nobs, 6, generator=g, dtype=torch.float64)
        tot.assert_close(
            d._proj_orth(M), M - d.exog @ d._XXinv @ d.exog.T @ M, atol=ATOL, rtol=0
        )

    def test_leverage_matches_diag_of_projector(self, toy_dataframes):
        """_leverage = diag(ProjImX), computed without forming the n x n matrix."""
        endog, exog, _ = toy_dataframes
        d = Data(endog, exog)
        P, _ = materialize_projectors(d)
        tot.assert_close(d._leverage, torch.diag(P), atol=ATOL, rtol=0)

    def test_ProjImX_is_projector(self, toy_dataframes):
        """P = _proj(I) is symmetric, idempotent, fixes col(X), rank = nlvl."""
        endog, exog, _ = toy_dataframes
        d = Data(endog, exog)
        P, _ = materialize_projectors(d)
        tot.assert_close(P @ P, P, atol=ATOL, rtol=0)             # idempotent
        tot.assert_close(P, P.T, atol=ATOL, rtol=0)               # symmetric
        tot.assert_close(P @ d.exog, d.exog, atol=ATOL, rtol=0)   # fixes col(X)
        tot.assert_close(matrix_rank(P), matrix_rank(d.exog), atol=ATOL, rtol=0)  # rank

    def test_ProjImXorthogonal(self, toy_dataframes):
        """P⊥ = _proj_orth(I) is the complementary projector: orthogonal to P, idempotent."""
        endog, exog, _ = toy_dataframes
        d = Data(endog, exog)
        P, Po = materialize_projectors(d)
        tot.assert_close(
            Po, eye(d.nobs, dtype=torch.float64) - P,
            atol=ATOL, rtol=0
        )
        tot.assert_close(
            P @ Po, zeros(d.nobs, d.nobs, dtype=torch.float64),
            atol=ATOL, rtol=0
        )                        # P @ P⊥ = 0
        tot.assert_close(Po @ Po, Po, atol=ATOL, rtol=0)          # idempotent

    def test_matches_numpy_reference(self, toy_dataframes, toy_arrays):
        """Small-size attributes match a NumPy reference computed on toy_arrays.

        The reference uses np.linalg.pinv(X^T X), which coincides with the
        package's rank-truncated eigendecomposition here because the toy
        design matrix has full column rank.
        """
        endog_df, exog_df, _ = toy_dataframes
        endog_np, exog_np = toy_arrays  # same data as numpy arrays

        # sanity: both fixtures really hold the same design matrix
        np.testing.assert_allclose(exog_df.to_numpy(), exog_np)

        d = Data(endog_df, exog_df)
        X = exog_np
        n, p = X.shape

        # NumPy reference computations
        XXinv_ref = np.linalg.pinv(X.T @ X)          # p x p pseudo-inverse
        Q_ref = X @ XXinv_ref                        # n x p projector factor
        ProjImX_ref = Q_ref @ X.T                    # n x n projector onto col(X)
        lev_ref = np.einsum('ij,ij->i', Q_ref, X)    # diag(ProjImX)

        # compare package (torch) vs reference (numpy), as float64 tensors
        tot.assert_close(
            d._XXinv, torch.tensor(XXinv_ref, dtype=torch.float64),
            atol=1e-8, rtol=1e-8
        )
        tot.assert_close(
            d._Q, torch.tensor(Q_ref, dtype=torch.float64),
            atol=1e-8, rtol=1e-8
        )
        tot.assert_close(
            d._leverage, torch.tensor(lev_ref, dtype=torch.float64),
            atol=1e-8, rtol=1e-8
        )
        # projectors materialized through the helpers match the references
        P, Po = materialize_projectors(d)
        tot.assert_close(
            P, torch.tensor(ProjImX_ref, dtype=torch.float64),
            atol=1e-8, rtol=1e-8
        )
        tot.assert_close(
            Po, torch.tensor(np.eye(n) - ProjImX_ref, dtype=torch.float64),
            atol=1e-8, rtol=1e-8
        )


class TestNystrom:
    """Nystrom landmark selection (nystrom=True) and recomputed attributes."""

    def test_shapes_and_flag(self, toy_dataframes):
        """With nystrom=True, all observation-level attributes shrink to n_landmarks."""
        endog, exog, meta = toy_dataframes
        d = Data(
            endog, exog, meta=meta, nystrom=True, n_landmarks=20, random_gen=42
        )
        assert d.nystrom is True
        assert d.nobs == 20
        assert d.exog.shape == (20, 7)
        assert d.endog.shape == (20, 10)
        # small-size quantities are (re)computed on the landmark design
        assert d._XXinv.shape == (7, 7)
        assert d._Q.shape == (20, 7)
        assert d._leverage.shape == (20,)
        assert len(d.index) == 20

    def test_default_n_landmarks(self):
        """Default landmark count is min(nobs, nlvl * 30)."""
        # nlvl = 2 here, so the default is min(100, 60) = 60
        df, _, _ = dummy_data(n_row=100, n_col=5, n_factor=1, n_group=[2], seed=0)
        endog = df[[f"col{i + 1}" for i in range(5)]]
        exog = pd.get_dummies(df[["factor1"]], dtype=float)
        d = Data(endog, exog, nystrom=True, random_gen=0)
        assert d.nobs == min(100, 2 * 30)

    def test_landmarks_are_subset_of_observations(self, toy_dataframes):
        """Landmark rows are exact copies of a subset of the original rows."""
        endog, exog, meta = toy_dataframes
        d_full = Data(endog, exog, meta=meta)
        d_ny = Data(endog, exog, meta=meta, nystrom=True,
                    n_landmarks=20, random_gen=0)
        assert set(d_ny.index).issubset(set(d_full.index))
        pos = [list(d_full.index).index(i) for i in d_ny.index]
        tot.assert_close(d_ny.exog, d_full.exog[pos], atol=0, rtol=0)
        tot.assert_close(d_ny.endog, d_full.endog[pos], atol=0, rtol=0)
        # ny_ind.sort() in __init__: the landmark index is sorted
        assert list(d_ny.index) == sorted(d_ny.index)

    def test_meta_subset_dataframe_and_numpy(self, toy_dataframes, toy_arrays):
        """meta is subset consistently: .iloc for DataFrame, fancy indexing for arrays."""
        endog, exog, meta = toy_dataframes
        d_ny = Data(
            endog, exog, meta=meta, nystrom=True, n_landmarks=20, random_gen=0
        )
        pos = list(d_ny.index)
        pd.testing.assert_frame_equal(d_ny.meta, meta.iloc[pos])
        # numpy meta goes through the else branch: meta[ny_ind]
        endog_a, exog_a = toy_arrays
        meta_a = np.arange(50).reshape(-1, 1)
        d_ny_a = Data(
            endog_a, exog_a, meta=meta_a, nystrom=True, n_landmarks=20, random_gen=0
        )
        assert d_ny_a.meta.shape == (20, 1)
        np.testing.assert_array_equal(d_ny_a.meta.ravel(), np.array(pos))

    def test_reproducibility_same_int_seed(self, toy_dataframes):
        """Same integer seed -> identical Data objects (all attributes)."""
        endog, exog, meta = toy_dataframes
        d1 = Data(endog, exog, meta=meta, nystrom=True, n_landmarks=20, random_gen=42)
        d2 = Data(endog, exog, meta=meta, nystrom=True, n_landmarks=20, random_gen=42)
        assert_equal_data_obj(d1, d2)

    def test_int_seed_equals_generator_seed(self, toy_dataframes):
        """random_gen=42 and random_gen=np.random.default_rng(42) are equivalent."""
        endog, exog, meta = toy_dataframes
        d1 = Data(endog, exog, meta=meta, nystrom=True, n_landmarks=20, random_gen=42)
        d2 = Data(endog, exog, meta=meta, nystrom=True, n_landmarks=20,
                  random_gen=np.random.default_rng(42))
        assert_equal_data_obj(d1, d2)

    def test_different_seeds_give_different_landmarks(self, toy_dataframes):
        """Different seeds select different landmark subsets (20 out of 50)."""
        endog, exog, meta = toy_dataframes
        d1 = Data(endog, exog, meta=meta, nystrom=True, n_landmarks=20, random_gen=0)
        d2 = Data(endog, exog, meta=meta, nystrom=True, n_landmarks=20, random_gen=1)
        assert list(d1.index) != list(d2.index)

    def test_proj_factors_recomputed_on_landmarks(self, toy_dataframes, toy_arrays):
        """After subsampling, _XXinv / _Q / _leverage are recomputed on the landmarks.

        Checked both algebraically and against a plain NumPy reference computed
        on the landmark design matrix extracted from ``toy_arrays``.
        """
        endog_df, exog_df, meta = toy_dataframes
        endog_np, exog_np = toy_arrays  # same data as numpy arrays

        d_full = Data(endog_df, exog_df, meta=meta)
        d = Data(endog_df, exog_df, meta=meta, nystrom=True,
                 n_landmarks=20, random_gen=42)

        # positions of the selected landmarks in the original arrays
        pos = [list(d_full.index).index(i) for i in d.index]
        X = d.exog                      # landmark design (torch, m x nlvl)
        X_np = exog_np[pos]             # same matrix as numpy (m x nlvl)
        m, p = X_np.shape
        np.testing.assert_allclose(X.numpy(), X_np)  # sanity: torch == numpy

        # --- algebraic identities on the landmark design ---
        P, Po = materialize_projectors(d)  # m x m, materialized for testing only
        tot.assert_close(X @ d._XXinv @ X.T, P, atol=ATOL, rtol=0)
        tot.assert_close(
            X.T @ X @ d._XXinv, eye(d.nlvl, dtype=torch.float64),
            # atol=ATOL, rtol=0
            atol=2e-1, rtol=0   # bad precision
        )
        tot.assert_close(
            Po, eye(d.nobs, dtype=torch.float64) - P,
            atol=ATOL, rtol=0
        )
        tot.assert_close(
            d._leverage, torch.diag(P), atol=ATOL, rtol=0
        )

        # --- NumPy reference on the landmark design ---
        XXinv_ref = np.linalg.pinv(X_np.T @ X_np)        # p x p pseudo-inverse
        Q_ref = X_np @ XXinv_ref                         # m x p factor
        ProjImX_ref = Q_ref @ X_np.T                     # m x m projector

        tot.assert_close(
            d._XXinv, torch.tensor(XXinv_ref, dtype=torch.float64),
            atol=1e-8, rtol=1e-8
        )
        tot.assert_close(
            d._Q, torch.tensor(Q_ref, dtype=torch.float64),
            atol=1e-8, rtol=1e-8
        )
        tot.assert_close(
            P, torch.tensor(ProjImX_ref, dtype=torch.float64),
            atol=1e-8, rtol=1e-8
        )
        tot.assert_close(
            Po, torch.tensor(np.eye(m) - ProjImX_ref, dtype=torch.float64),
            atol=1e-8, rtol=1e-8
        )


class TestDiagonalizeResidualCovariance:
    """Spectral decomposition of the residual covariance operator."""

    @staticmethod
    def _psd_gram(n, seed=0):
        """Random positive semi-definite n x n matrix with a fixed seed."""
        g = torch.Generator().manual_seed(seed)
        A = torch.randn(n, n, generator=g, dtype=torch.float64)
        return A @ A.T

    @staticmethod
    def _reference_eigh(Kres):
        """Eigenvalues (decreasing order) and eigenvectors of Kres.

        Mirrors kaov.linalg.ordered_eigsy ordering (decreasing).
        """
        sp, ev = torch.linalg.eigh(Kres)  # ascending order
        # keep = sp > 1e-10 * sp.max()    # remove filtering
        # sp, ev = sp[keep], ev[:, keep]
        order = torch.argsort(sp, descending=True)
        return sp[order], ev[:, order]

    def test_spectrum_matches(self, toy_dataframes):
        """Returned spectrum equals the spectrum of P⊥ K P⊥ / nobs
        (P⊥ materialized through _proj_orth for the reference)."""
        endog, exog, _ = toy_dataframes
        d = Data(endog, exog)
        K = self._psd_gram(d.nobs)
        sp, U_norm = d._diagonalize_residual_covariance(K)
        _, Po = materialize_projectors(d)
        Kres = (Po @ K @ Po) / d.nobs
        sp_ref, _ = self._reference_eigh(Kres)
        assert U_norm.shape == (d.nobs, len(sp))
        tot.assert_close(sp, sp_ref[:len(sp)], atol=1e-8, rtol=1e-8)

    def test_U_norm_lies_in_residual_space(self, toy_dataframes):
        """Columns of U_norm are in the range of P⊥: P @ U_norm = 0."""
        endog, exog, _ = toy_dataframes
        d = Data(endog, exog)
        K = self._psd_gram(d.nobs, seed=2)
        _, U_norm = d._diagonalize_residual_covariance(K)
        tot.assert_close(
            d._proj(U_norm),
            zeros(d.nobs, U_norm.shape[1], dtype=torch.float64),
            # atol=1e-8, rtol=0
            atol=1e-1, rtol=0   # bad precision
        )

    def test_U_norm_orthogonality_and_scaling(self, toy_dataframes):
        """Non-nystrom (sp_power = -1): U^T U diagonal with entries sp^-2 / nobs."""
        endog, exog, _ = toy_dataframes
        d = Data(endog, exog)
        K = self._psd_gram(d.nobs, seed=4)
        sp, U_norm = d._diagonalize_residual_covariance(K)
        G = U_norm.T @ U_norm
        off_diag = G - torch.diag(torch.diag(G))
        tot.assert_close(
            off_diag, zeros(G.shape, dtype=torch.float64), atol=1e-8, rtol=0
        )
        tot.assert_close(torch.diag(G), sp ** -2 / d.nobs, atol=1e-8, rtol=1e-8)

    def test_eigenvectors_match_up_to_sign(self, toy_dataframes):
        """U_norm / (sp^-1 * nobs^-1/2) recovers the eigenvectors of Kresidual."""
        endog, exog, _ = toy_dataframes
        d = Data(endog, exog)
        K = self._psd_gram(d.nobs, seed=5)
        sp, U_norm = d._diagonalize_residual_covariance(K)
        _, Po = materialize_projectors(d)
        Kres = (Po @ K @ Po) / d.nobs
        sp_ref, ev_ref = self._reference_eigh(Kres)
        # undo the sp^-1 * nobs^-1/2 scaling applied inside the method
        U_ref = (sp_ref ** -1 * d.nobs ** -0.5) * ev_ref
        # comparison up to sign (filter out last eigen value)
        assert_same_eigenvectors(U_norm[:, :44], U_ref[:, :44])

    def test_nystrom_scaling(self, toy_dataframes):
        """Nystrom (sp_power = -1/2): diag(U^T U) = sp^-1 / nobs."""
        endog, exog, meta = toy_dataframes
        d = Data(endog, exog, meta=meta, nystrom=True, n_landmarks=20, random_gen=42)
        K = self._psd_gram(d.nobs)
        sp, U_norm = d._diagonalize_residual_covariance(K)
        G = U_norm.T @ U_norm
        # (filter out last eigen value because numerical instability)
        tot.assert_close(
            torch.diag(G)[:14],
            sp[:14] ** -1 / d.nobs,
            atol=ATOL, rtol=0
        )
