"""Figures for the README. Usage: python scripts/09_figures.py [--label mlb_2026]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import expit  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abs_judgment.config import PROCESSED, ROOT  # noqa: E402

FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
INK, ACCENT, MUTED = "#1f2a44", "#c8553d", "#8a93a6"


def psychometric(label: str) -> None:
    c = pd.read_csv(PROCESSED / f"judgment_curve_{label}.csv")
    prior = json.load(open(PROCESSED / f"judgment_prior_{label}.json"))
    a, b = prior["mean"]
    off = prior.get("fixed_offset_mean", 0.0)  # average count/leverage shift, so curves are marginal
    J = pd.read_csv(PROCESSED / f"judgment_{label}.csv")
    J = J[(J.n_decisions >= 150) & (J.n_challenges >= 8)]
    x = np.linspace(-4, 4, 200)
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
    ax.axvspan(-4, 0, color="#eef1f6", lw=0)
    ax.scatter(c.edge_mid, c.challenge_rate, s=np.sqrt(c.n) * 1.6, color=INK, zorder=3, label="observed, by half-inch bin")
    ax.plot(x, expit(a + off + b * x), color=INK, lw=2, label=f"league curve, slope {b:.2f}/in")
    for name, col in ((J.nlargest(1, "slope"), ACCENT), (J.nsmallest(1, "slope"), MUTED)):
        r = name.iloc[0]
        ax.plot(x, expit(r.intercept + off + r.slope * x), color=col, lw=1.6, ls="--", label=f"{r.batter_name}, slope {r.slope:.2f}")
    ax.set_xlabel("true edge distance at mid-plate (inches; + = ball, called a strike)")
    ax.set_ylabel("P(hitter challenges)")
    ax.set_title("Challenge judgment on called strikes, 2026: perception stripped of intent", fontsize=11)
    ax.text(-3.9, 0.9, "inside ABS zone", color=MUTED, fontsize=9)
    ax.text(0.2, 0.9, "outside", color=MUTED, fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "psychometric_curve.png")
    plt.close(fig)


def token_value(label: str) -> None:
    tv = pd.read_csv(PROCESSED / f"token_value_{label}.csv")
    fig, ax = plt.subplots(figsize=(7, 4.0), dpi=160)
    for k, col in ((1, INK), (2, ACCENT)):
        d = tv[tv.tokens == k]
        ax.plot(d.half_inning, d.token_value, color=col, lw=2, label=f"token #{k} (marginal value)")
    ax.axhline(0.2, color=MUTED, ls=":", lw=1.5, label="Savant static price, 0.20 runs")
    ax.set_xticks(range(1, 19, 2))
    ax.set_xticklabels([f"{(h + 1) // 2 + 1}{'T' if h % 2 == 0 else 'B'}" for h in range(0, 18, 2)])
    ax.set_xlabel("half-inning")
    ax.set_ylabel("runs")
    ax.set_title("What a held challenge token is worth, by game state", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "token_value.png")
    plt.close(fig)


def judgment_vs_chase(label: str) -> None:
    p = PROCESSED / f"hitter_season_{label}.csv"
    if not p.exists():
        return
    hs = pd.read_csv(p)
    J = pd.read_csv(PROCESSED / f"judgment_{label}.csv")
    d = hs.merge(J, on="batter_id")
    d = d[(d.n_challenges >= 8) & (d.pa >= 150)]
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=160)
    ax.scatter(d.judgment_z, d.chase_rate, s=12, color=INK, alpha=0.6)
    r = np.corrcoef(d.judgment_z, d.chase_rate)[0, 1]
    m, b0 = np.polyfit(d.judgment_z, d.chase_rate, 1)
    xs = np.array([d.judgment_z.min(), d.judgment_z.max()])
    ax.plot(xs, m * xs + b0, color=ACCENT, lw=1.6)
    ax.set_xlabel("challenge judgment (z; higher = sharper perception on takes)")
    ax.set_ylabel("chase rate (swings at ABS balls)")
    ax.set_title(f"Eyes on takes vs chasing on swings, 2026 (r = {r:+.2f}, n = {len(d)})", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "judgment_vs_chase.png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="mlb_2026")
    a = ap.parse_args()
    psychometric(a.label)
    token_value(a.label)
    judgment_vs_chase(a.label)
    print("figures written to", FIG)
