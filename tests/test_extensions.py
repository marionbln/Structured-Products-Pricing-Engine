"""
Tests du worst-of, du backtest historique et du pricing en lot.
Lancer avec : python tests/test_extensions.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from src.monte_carlo import MCConfig
from src.structured_products import Autocall
from src.worst_of import WorstOfAutocall, flat_correlation, simulate_correlated_paths
from src.backtest import backtest_autocall
from src.pricing_requests import price_requests

CFG = MCConfig(n_paths=40_000, n_steps=84, seed=5)


def test_correlation_is_reproduced():
    """La corrélation des rendements simulés correspond à la corrélation demandée."""
    perf = simulate_correlated_paths((0.2, 0.3), flat_correlation(2, 0.7), 1.0, 0.03, 0.0,
                                     MCConfig(n_paths=20_000, n_steps=50, seed=0))
    r = np.diff(np.log(perf), axis=1).reshape(-1, 2)
    assert abs(np.corrcoef(r.T)[0, 1] - 0.7) < 0.01


def test_worst_of_single_asset_equals_autocall():
    """Avec un seul sous-jacent, le worst-of redonne exactement l'autocall simple."""
    wo = WorstOfAutocall((0.25,), np.eye(1), 1.0, 0.03, 0.01, n_obs=4, coupon=0.02).price(CFG)["price"]
    ac = Autocall(1.0, 1.0, 0.25, 0.03, 0.01, n_obs=4, coupon=0.02).price(CFG)["price"]
    assert abs(wo - ac) < 1e-9, f"{wo} vs {ac}"


def test_worst_of_cheaper_than_single():
    """Un panier worst-of est plus risqué pour le client : il vaut moins qu'un mono sous-jacent."""
    single = WorstOfAutocall((0.25,), np.eye(1), 2.0, 0.03, n_obs=8).price(MCConfig(n_paths=40_000, n_steps=168, seed=1))["price"]
    basket = WorstOfAutocall((0.25,) * 3, flat_correlation(3, 0.5), 2.0, 0.03, n_obs=8).price(MCConfig(n_paths=40_000, n_steps=168, seed=1))["price"]
    assert basket < single


def test_backtest_on_known_path():
    """Backtest sur une trajectoire construite à la main : chaque sortie est celle attendue."""
    prices = pd.Series([100, 100, 100, 110] + [110] * 30, index=pd.date_range("2000-01-01", periods=34, freq="MS"))
    bt = backtest_autocall(prices, maturity_months=24, obs_every=3, coupon=0.02)
    first = bt.iloc[0]
    assert first["sortie"] == "rappel anticipé" and first["duree_mois"] == 3 and abs(first["remboursement"] - 102) < 1e-9


def test_batch_pricing_runs_and_flags_errors():
    """Le pricing en lot traite chaque demande et signale les demandes invalides sans s'arrêter."""
    req = pd.DataFrame([
        {"id": "A", "produit": "BRC", "maturite_ans": 1, "vol": 0.25, "barriere": 0.65, "prix_cible": 98, "nb_sous_jacents": 1, "correlation": None},
        {"id": "B", "produit": "INCONNU", "maturite_ans": 1, "vol": 0.25, "barriere": 0.65, "prix_cible": 98, "nb_sous_jacents": 1, "correlation": None},
    ])
    out = price_requests(req)
    assert out.loc[0, "statut"] == "OK" and out.loc[0, "coupon_annuel_fair"] > 0
    assert out.loc[1, "statut"].startswith("ERREUR")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} tests passés avec succès.")
