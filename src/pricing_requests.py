"""
pricing_requests.py
====================
Traitement en lot de demandes de prix ("industrialisation" du pricing).

Un desk reçoit en continu des demandes de la vente : "quel coupon peut-on
offrir sur tel produit, avec telle barrière, telle maturité ?". Plutôt que
de pricer chaque demande à la main, on lit un fichier de demandes, on
résout le coupon fair de chacune, et on renvoie un fichier de réponses.

Format d'entrée (CSV) : une ligne par demande, colonnes
    id, produit (BRC | AUTOCALL | WORST_OF_AUTOCALL), maturite_ans, vol,
    barriere, prix_cible, nb_sous_jacents, correlation
"""

from __future__ import annotations
import time
import pandas as pd
from .monte_carlo import MCConfig
from .structured_products import BRC, Autocall, solve_fair_coupon
from .worst_of import WorstOfAutocall, flat_correlation

R_D, R_F = 0.03, 0.01


def build_product(req: pd.Series):
    T, vol, barrier = float(req["maturite_ans"]), float(req["vol"]), float(req["barriere"])
    kind = str(req["produit"]).upper()
    if kind == "BRC":
        return BRC(100.0, T, vol, R_D, R_F, barrier=barrier, coupon_freq=4), MCConfig(n_paths=40_000, n_steps=int(252 * T), seed=1)
    n_obs = int(round(4 * T))
    cfg = MCConfig(n_paths=40_000, n_steps=n_obs * 21, seed=1)
    if kind == "AUTOCALL":
        return Autocall(100.0, T, vol, R_D, R_F, n_obs=n_obs, protection=barrier), cfg
    if kind == "WORST_OF_AUTOCALL":
        n = int(req["nb_sous_jacents"])
        return WorstOfAutocall((vol,) * n, flat_correlation(n, float(req["correlation"])), T, R_D, R_F,
                               n_obs=n_obs, protection=barrier), cfg
    raise ValueError(f"Produit inconnu : {kind}")


def price_requests(requests: pd.DataFrame) -> pd.DataFrame:
    """Résout le coupon fair de chaque demande et renvoie le tableau de réponses."""
    out = []
    for _, req in requests.iterrows():
        t0 = time.perf_counter()
        try:
            product, cfg = build_product(req)
            cpn = solve_fair_coupon(product, float(req["prix_cible"]), cfg)
            status, coupon_pa = "OK", (cpn if req["produit"].upper() == "BRC" else cpn * 4)
        except ValueError as e:
            status, coupon_pa = f"ERREUR : {e}", float("nan")
        out.append({**req.to_dict(), "coupon_annuel_fair": coupon_pa, "statut": status,
                    "temps_calcul_s": round(time.perf_counter() - t0, 2)})
    return pd.DataFrame(out)
