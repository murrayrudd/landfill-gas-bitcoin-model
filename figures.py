"""Optional figure generation (manuscript Figures 4-9).

Requires matplotlib. Run:  python figures.py [output_dir]

Produces, as PNGs:
  - hashprice_distribution.png      (Fig. 4)
  - total_revenue_by_rig.png        (Fig. 5)
  - landfill_net_revenue.png        (Fig. 6)
  - miner_net_revenue.png           (Fig. 7)
  - combined_net_revenue.png        (Fig. 8)
  - miner_net_low_cost_high_hp.png  (Fig. 9)

Densities are kernel-smoothed for readability; following the manuscript, the
y-axis is unlabeled so curves can be compared across figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from landfill_btc import Scenario, sampling, simulate
from landfill_btc.model import FinancialParams

try:
    import matplotlib.pyplot as plt
    from scipy import stats
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib and scipy are required for figures.py") from exc

N = 10_000
SEED = 12345
COLORS = {"Antminer S19j XP": "#000000",
          "Antminer S21": "#d1495b",
          "Antminer S21 Hydro": "#2e7d32"}


def _density(ax, data, label, color, x_range=None):
    kde = stats.gaussian_kde(data)
    lo, hi = (x_range if x_range else (data.min(), data.max()))
    xs = np.linspace(lo, hi, 400)
    ax.plot(xs, kde(xs), label=label, color=color, linewidth=2)


def _style(ax, title, xlabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    ax.set_ylabel("Probability density")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(lambda v, _: f"${v/1e6:.0f}M" if abs(v) >= 1e6
                                 else f"${v/1e3:.0f}k")


def fig_hashprice(outdir: Path):
    rng = np.random.default_rng(SEED)
    hp = 0.075 * sampling.sample_lognormal_median(N, 1.0, 1.5, rng)
    fig, ax = plt.subplots(figsize=(7, 4))
    _density(ax, hp, "hashprice", "#1f4e79", x_range=(0, 0.4))
    ax.set_title("Figure 4 - Lognormal hashprice distribution (mean $0.075/TH/day)")
    ax.set_xlabel("Hashprice ($/TH/day)")
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "hashprice_distribution.png", dpi=150)
    plt.close(fig)


def _revenue_figure(outdir, fname, title, series_by_rig, x_range):
    fig, ax = plt.subplots(figsize=(7, 4))
    for rig, data in series_by_rig.items():
        _density(ax, data, rig, COLORS.get(rig, None), x_range=x_range)
    _style(ax, title, "$ / year")
    ax.set_xlim(x_range)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / fname, dpi=150)
    plt.close(fig)


def main(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    base = simulate(Scenario(name="Base case"), n_samples=N, seed=SEED)

    fig_hashprice(outdir)

    _revenue_figure(outdir, "total_revenue_by_rig.png",
                    "Figure 5 - Total (combined) revenue by rig",
                    base.combined_net_revenue, (-1e6, 8e6))

    _revenue_figure(outdir, "landfill_net_revenue.png",
                    "Figure 6 - Landfill operator net revenue",
                    base.landfill_net_revenue, (0, 2.5e6))

    _revenue_figure(outdir, "miner_net_revenue.png",
                    "Figure 7 - Bitcoin miner net revenue",
                    base.miner_net_revenue, (-1e6, 8e6))

    _revenue_figure(outdir, "combined_net_revenue.png",
                    "Figure 8 - Combined net revenue (landfill + miner)",
                    base.combined_net_revenue, (-1e6, 8e6))

    best = simulate(
        Scenario(name="Low c_be, high p_h",
                 breakeven_low=0.01, breakeven_high=0.07, hashprice_mean=0.150),
        n_samples=N, seed=SEED)
    _revenue_figure(outdir, "miner_net_low_cost_high_hp.png",
                    "Figure 9 - Miner net revenue (c_be $0.040, p_h $0.150)",
                    best.miner_net_revenue, (-1e6, 1.5e7))

    print(f"Figures written to {outdir.resolve()}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("figures"))
