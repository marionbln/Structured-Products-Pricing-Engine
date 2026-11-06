"""
backtest.py
============
Backtest historique d'un autocall : que se serait-il passé pour un client
qui aurait acheté le même autocall chaque mois, sur données réelles ?

Pour chaque date de lancement, on rejoue le produit sur la trajectoire
historique du sous-jacent (observations trimestrielles = tous les 3 mois
sur des données mensuelles) et on enregistre :
    - la date et le type de sortie (rappel anticipé, remboursement à 100%,
      perte en capital),
    - le montant remboursé et le rendement annualisé pour l'investisseur.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def backtest_autocall(prices: pd.Series, maturity_months: int = 24, obs_every: int = 3,
                      autocall_level: float = 1.00, coupon: float = 0.02,
                      protection: float = 0.60) -> pd.DataFrame:
    """
    prices : série de prix mensuels indexée par date.
    coupon : coupon par période d'observation (snowball, payé au rappel).
    Retourne un DataFrame avec une ligne par date de lancement.
    """
    prices = prices.dropna().astype(float)
    values, dates = prices.to_numpy(), prices.index
    n_obs = maturity_months // obs_every
    rows = []
    for i in range(len(values) - maturity_months):
        S0 = values[i]
        outcome, redemption, months = None, None, None
        for k in range(1, n_obs + 1):
            m = k * obs_every
            perf = values[i + m] / S0
            if k < n_obs and perf >= autocall_level:
                outcome, redemption, months = "rappel anticipé", 100 * (1 + k * coupon), m
                break
            if k == n_obs:
                months = m
                if perf >= autocall_level:
                    outcome, redemption = "rappel à maturité", 100 * (1 + k * coupon)
                elif perf >= protection:
                    outcome, redemption = "remboursé à 100%", 100.0
                else:
                    outcome, redemption = "perte en capital", 100 * perf
        years = months / 12
        rows.append({
            "date_lancement": dates[i],
            "sortie": outcome,
            "duree_mois": months,
            "remboursement": redemption,
            "rendement_annualise": (redemption / 100) ** (1 / years) - 1,
        })
    return pd.DataFrame(rows)


def summarize_backtest(bt: pd.DataFrame) -> pd.Series:
    """Indicateurs clés du backtest."""
    return pd.Series({
        "nombre de produits": len(bt),
        "% rappelés avant maturité": 100 * (bt["sortie"] == "rappel anticipé").mean(),
        "% perte en capital": 100 * (bt["sortie"] == "perte en capital").mean(),
        "durée de vie moyenne (mois)": bt["duree_mois"].mean(),
        "rendement annualisé moyen (%)": 100 * bt["rendement_annualise"].mean(),
        "pire rendement annualisé (%)": 100 * bt["rendement_annualise"].min(),
    }).round(2)
