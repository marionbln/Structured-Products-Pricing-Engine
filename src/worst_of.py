"""
worst_of.py
============
Autocall "worst-of" sur un panier de sous-jacents corrélés.

Sur le marché, la plupart des autocalls et BRC portent sur plusieurs
sous-jacents (ex. 3 actions) : le produit regarde à chaque date la
performance du PLUS MAUVAIS sous-jacent. Le client prend plus de risque
qu'avec un seul sous-jacent, ce qui permet de lui payer un coupon plus élevé.

Simulation : GBM multi-dimensionnel, chocs corrélés via la décomposition
de Cholesky de la matrice de corrélation.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .monte_carlo import MCConfig
from .structured_products import NOMINAL, _observation_indices, _summary


def simulate_correlated_paths(sigmas, corr, T: float, r_d: float, r_f, config: MCConfig) -> np.ndarray:
    """
    Trajectoires de PERFORMANCE (niveau initial = 1) pour n sous-jacents corrélés.
    Retourne un array de forme (n_paths, n_steps + 1, n_assets).
    r_f : rendement (dividende) par sous-jacent, scalaire ou vecteur.
    """
    sigmas = np.asarray(sigmas, dtype=float)
    n_assets = len(sigmas)
    r_f = np.broadcast_to(np.asarray(r_f, dtype=float), (n_assets,))
    L = np.linalg.cholesky(np.asarray(corr, dtype=float))
    rng = np.random.default_rng(config.seed)
    dt = T / config.n_steps
    n_half = config.n_paths // 2 if config.antithetic else config.n_paths
    Z = rng.standard_normal((n_half, config.n_steps, n_assets)) @ L.T
    if config.antithetic:
        Z = np.concatenate([Z, -Z], axis=0)
    drift = (r_d - r_f - 0.5 * sigmas ** 2) * dt
    log_incr = drift + sigmas * np.sqrt(dt) * Z
    log_paths = np.cumsum(log_incr, axis=1)
    log_paths = np.concatenate([np.zeros((log_paths.shape[0], 1, n_assets)), log_paths], axis=1)
    return np.exp(log_paths)


@dataclass
class WorstOfAutocall:
    """
    Autocall worst-of (même mécanique que structured_products.Autocall,
    appliquée à la performance du plus mauvais sous-jacent).
    """
    sigmas: tuple
    corr: np.ndarray
    T: float
    r_d: float
    r_f: float = 0.0
    n_obs: int = 4
    autocall_level: float = 1.00
    coupon: float = 0.02
    protection: float = 0.60

    def price(self, config: MCConfig | None = None) -> dict:
        config = config or MCConfig(n_steps=self.n_obs * 21)
        perf = simulate_correlated_paths(self.sigmas, self.corr, self.T, self.r_d, self.r_f, config)
        worst = perf.min(axis=2)                                   # (n_paths, n_steps+1)
        n_steps = worst.shape[1] - 1
        idx = _observation_indices(n_steps, self.n_obs)
        t_obs = idx / n_steps * self.T
        obs = worst[:, idx]

        called = obs >= self.autocall_level
        any_call = called.any(axis=1)
        first = np.where(any_call, called.argmax(axis=1), self.n_obs)

        pv = np.zeros(len(obs))
        c = first[any_call]
        pv[any_call] = NOMINAL * (1 + (c + 1) * self.coupon) * np.exp(-self.r_d * t_obs[c])
        perf_T = obs[~any_call, -1]
        pv[~any_call] = np.where(perf_T >= self.protection, NOMINAL, NOMINAL * perf_T) * np.exp(-self.r_d * self.T)

        res = _summary(pv)
        res["autocall_probabilities"] = np.array([(first == i).mean() for i in range(self.n_obs)])
        res["prob_capital_loss"] = float(((first == self.n_obs) & (obs[:, -1] < self.protection)).mean())
        return res


def flat_correlation(n_assets: int, rho: float) -> np.ndarray:
    """Matrice de corrélation avec la même corrélation rho entre toutes les paires."""
    corr = np.full((n_assets, n_assets), rho)
    np.fill_diagonal(corr, 1.0)
    return corr
