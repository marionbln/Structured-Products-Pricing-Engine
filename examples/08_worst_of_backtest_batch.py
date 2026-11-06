"""
Exemple 8 : les trois outils d'un desk de pricing de produits structurés.
    1. Worst-of : coupon fair d'un autocall selon la corrélation du panier
    2. Backtest : autocall lancé chaque mois sur le S&P 500 depuis 1990
    3. Pricing en lot : traitement d'un fichier de demandes de prix
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.monte_carlo import MCConfig
from src.structured_products import solve_fair_coupon
from src.worst_of import WorstOfAutocall, flat_correlation
from src.backtest import backtest_autocall, summarize_backtest
from src.pricing_requests import price_requests

HERE = os.path.dirname(__file__)
TARGET = 98.0
cfg = MCConfig(n_paths=40_000, n_steps=168, seed=3)

# --- 1. Worst-of : coupon fair vs corrélation, pour 2 et 3 sous-jacents ---
rhos = np.linspace(0.2, 0.9, 8)
wo = {n: [4 * solve_fair_coupon(WorstOfAutocall((0.25,) * n, flat_correlation(n, r), 2.0, 0.03, 0.01, n_obs=8),
                                TARGET, cfg) for r in rhos] for n in (2, 3)}
mono = 4 * solve_fair_coupon(WorstOfAutocall((0.25,), np.eye(1), 2.0, 0.03, 0.01, n_obs=8), TARGET, cfg)

# --- 2. Backtest sur données réelles (S&P 500 mensuel, source : github.com/datasets/s-and-p-500) ---
px = pd.read_csv(os.path.join(HERE, "..", "data", "sp500_monthly.csv"), parse_dates=["Date"]).set_index("Date")["SP500"]
bt = backtest_autocall(px, maturity_months=24, obs_every=3, autocall_level=1.0, coupon=0.02, protection=0.60)
print("Backtest autocall 2 ans, coupon 2%/trimestre, protection 60% :")
print(summarize_backtest(bt).to_string(), "\n")

# --- 3. Pricing en lot ---
responses = price_requests(pd.read_csv(os.path.join(HERE, "pricing_requests.csv")))
responses.to_csv(os.path.join(HERE, "pricing_responses.csv"), index=False)
print(responses[["id", "produit", "barriere", "nb_sous_jacents", "correlation", "coupon_annuel_fair", "statut"]].to_string(index=False))

# --- Graphiques ---
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
ax = axes[0]
ax.axhline(mono * 100, color="gray", ls="--", label="1 sous-jacent")
for n, c in zip((2, 3), ("#1f77b4", "#d62728")):
    ax.plot(rhos, np.array(wo[n]) * 100, "o-", color=c, label=f"worst-of {n} sous-jacents")
ax.set_xlabel("Corrélation entre sous-jacents")
ax.set_ylabel("Coupon annuel fair (%)")
ax.set_title("Autocall 2 ans worst-of : coupon fair vs corrélation\n(vol 25%, protection 60%, prix cible 98%)")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
colors = {"rappel anticipé": "#2ca02c", "rappel à maturité": "#98df8a",
          "remboursé à 100%": "#ff7f0e", "perte en capital": "#d62728"}
for outcome, grp in bt.groupby("sortie"):
    ax.scatter(grp["date_lancement"], grp["rendement_annualise"] * 100, s=12, color=colors[outcome], label=outcome)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Rendement annualisé pour le client (%)")
ax.set_title("Backtest : autocall S&P 500 lancé chaque mois depuis 1990\n(coupon 2% / trimestre, protection 60%)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(HERE, "..", "outputs", "08_worst_of_and_backtest.png")
plt.savefig(out, dpi=130)
print(f"\nGraphique sauvegardé : {out}")
