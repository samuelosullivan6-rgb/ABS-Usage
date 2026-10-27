"""Chart and table PNGs for the postseason write-up (docs/figures/post_*.png).
Usage: python scripts/14_postseason_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abs_judgment.config import PROCESSED, ROOT  # noqa: E402

FIG = ROOT / "docs" / "figures"
def pc(v):
    return int(np.floor(v * 100 + 0.5))  # half-up, so 62.5% prints as 63%


INK, MUTED, GRID, S1, S2 = "#15202e", "#6b7686", "#e6eaf0", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10.5, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})


def chart() -> None:
    mon = pd.read_csv(PROCESSED / "monthly_trend_post.csv").set_index("month")
    cmp_ = pd.read_csv(PROCESSED / "postseason_comparison_post.csv").set_index("quantity")
    months = ["Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
    x = np.arange(len(months))
    xp = len(months) + 0.6
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=170)
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
    series = [("share_challenged", "se_share", "share of misses challenged", S1, "share of misses challenged"),
              ("overturn_rate", "se_overturn", "overturn rate, batter challenges", S2, "overturn rate when they did")]
    for col, se, key, color, label in series:
        y = mon[col].to_numpy() * 100; e = mon[se].to_numpy() * 100
        ax.plot(x, y, color=color, lw=2, solid_capstyle="round", zorder=3)
        ax.errorbar(x, y, yerr=e, fmt="o", ms=6, color=color, ecolor=color, elinewidth=1, capsize=0, zorder=4,
                    markeredgecolor="white", markeredgewidth=1.5)
        yp = cmp_.loc[key, "postseason_value"] * 100; ep = cmp_.loc[key, "postseason_se"] * 100
        ax.errorbar([xp], [yp], yerr=[ep], fmt="o", ms=8, color=color, ecolor=color, elinewidth=1.2, capsize=0, zorder=4,
                    markeredgecolor="white", markeredgewidth=1.5)
        ax.text(x[0] - 0.15, y[0] + (4 if col == "overturn_rate" else -5.5), label, color=INK, fontsize=10, fontweight="bold", ha="left")
        ax.text(xp + 0.25, yp, f"{pc(yp / 100)}%", color=INK, fontsize=10, fontweight="bold", va="center")
    ax.axvline(len(months) - 0.3, color=GRID, lw=1)
    ax.text(xp, 3, "October\n(37 games)", color=MUTED, fontsize=9, ha="center", va="bottom")
    ax.set_xticks(list(x) + [xp]); ax.set_xticklabels(months + [""])
    ax.set_ylim(0, 80); ax.set_yticks(range(0, 81, 20)); ax.set_yticklabels([f"{v}%" for v in range(0, 81, 20)])
    ax.set_xlim(-0.4, xp + 0.9)
    ax.set_title("Hitters' ABS challenges in 2026: how often, and how often right", loc="left", fontsize=12, color=INK, pad=30)
    ax.text(0, 1.03, "Umpire misses = called strikes that were balls by ABS, with a token in hand. Bars are one standard error.",
            transform=ax.transAxes, color=MUTED, fontsize=9, va="bottom")
    fig.tight_layout()
    fig.savefig(FIG / "post_monthly_rates.png"); plt.close(fig)


def table() -> None:
    c = pd.read_csv(PROCESSED / "postseason_comparison_post.csv").set_index("quantity")
    def pct(q):
        r = c.loc[q]
        return f"{pc(r.regular_value)}% ({int(r.regular_n):,})", f"{pc(r.postseason_value)}% ({int(r.postseason_n):,})"
    def num(q, f="{:.2f}"):
        r = c.loc[q]; return f.format(r.regular_value), f.format(r.postseason_value)
    rows = [
        ("Batter challenges per team-game", *num("batter challenges per team-game")),
        ("Umpire misses on hitters per team-game", *num("misses per team-game")),
        ("Share of misses challenged", *pct("share of misses challenged")),
        ("Overturn rate on batter challenges", *pct("overturn rate, batter challenges")),
        ("Median challenged pitch, inches from the edge (+ outside)", *num("median edge of batter challenges (in; + outside)")),
        ("Misses challenged, within 1 inch of the edge", *pct("share of misses challenged, 0-1 in outside")),
        ("Misses challenged, 1 to 2 inches", *pct("share of misses challenged, 1-2 in outside")),
        ("Misses challenged, beyond 2 inches", *pct("share of misses challenged, 2+ in outside")),
        ("High misses challenged (above the box)", *pct("share of misses challenged, above")),
        ("Low misses challenged (below the box)", *pct("share of misses challenged, below")),
        ("Share of challenges with two strikes", *pct("share of batter challenges with two strikes")),
        ("Share of challenges in the 7th inning or later", *pct("share of batter challenges in the 7th or later")),
        ("Misses challenged, close game and 7th or later", *pct("share of misses challenged, close and 7th or later")),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=170); ax.axis("off"); fig.patch.set_facecolor("white")
    t = ax.table(cellText=[[a, b, d] for a, b, d in rows], colLabels=["Batter challenges, 2026", "Regular season (n)", "Postseason (n)"],
                 loc="center", cellLoc="left", colLoc="left", colWidths=[0.56, 0.22, 0.22])
    t.auto_set_font_size(False); t.set_fontsize(9.6); t.scale(1, 1.55)
    for (r, cidx), cell in t.get_celld().items():
        cell.set_edgecolor(GRID); cell.set_linewidth(0.8)
        if r == 0:
            cell.set_text_props(color=MUTED, fontweight="bold"); cell.set_facecolor("#f3f5f8")
        else:
            cell.set_text_props(color=INK)
        if cidx > 0 and r > 0:
            cell.get_text().set_ha("right"); cell.PAD = 0.04
    ax.set_title("Regular season (2,429 games) against the postseason through World Series Game 2 (37 games)",
                 loc="left", fontsize=10.5, color=INK, pad=4)
    fig.tight_layout()
    fig.savefig(FIG / "post_regular_vs_postseason_table.png"); plt.close(fig)


if __name__ == "__main__":
    chart(); table(); print("wrote", FIG / "post_monthly_rates.png", "and", FIG / "post_regular_vs_postseason_table.png")
