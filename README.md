# landfill-gas-bitcoin-model

A Python reproduction of the Monte Carlo model published in:

> Rudd, M.A., M. Jones, D. Sechrest, D. Batten, and D. Porter. 2024. An
> integrated landfill
> gas-to-energy and Bitcoin mining framework. *Journal of Cleaner Production*
> 472: 143516. https://doi.org/10.1016/j.jclepro.2024.143516
>
> Open-access preprint: https://ssrn.com/abstract=4810964

The model was originally built in Lumina Analytica; this repository reproduces
its published results in open-source Python and then extends it. The code
regenerates the paper's landfill derivation, financial-performance tables, and
emissions-mitigation estimates from a single set of parameters, and the
reproduction is verified against the published tables by an included test
suite. The reproduction is the work of Murray A. Rudd; the co-authors of the
paper did not contribute to this software.

The extensions beyond the published model — a parasitic-split allocation, an
untruncated miner-to-landfill revenue ratio with loss probabilities, and a
current-generation ASIC catalog — are documented as such below and are not part
of the published results.

## What the model does

The model anchors on a midpoint assumption — the electricity made available for
Bitcoin mining (default 10.0 million kWh/year, ~1.14 MW) — and reasons in two
directions:

- backward, to the implied landfill characteristics and catchment population
  (~65,000 people, ~7.8 million m³/year of LFG, ~527 SCFM) that would supply
  that electricity; and
- forward, through a Monte Carlo simulation of revenue sharing between a
  landfill operator and a Bitcoin miner across three ASIC rig models, and to the
  social value of the methane emissions the operation mitigates.

Three quantities are treated as uncertain and drawn with median Latin hypercube
sampling (n = 10,000), matching Analytica's default: the breakeven electricity
production cost (truncated normal), the landfill revenue share (uniform), and a
lognormal hashprice multiplier with an extended upper tail.

## Installation

```bash
pip install -r requirements.txt
```

The core model depends only on `numpy` and `scipy`. `matplotlib` is needed only
for `figures.py`.

## Usage

Reproduce the published tables and compare against the manuscript values:

```bash
python reproduce_paper.py
```

Generate the distribution figures (manuscript Figures 4-9):

```bash
python figures.py figures/
```

Use the model directly:

```python
from landfill_btc import Scenario, simulate, mitigation_value, infer_catchment

# Backward step: what catchment supplies 10M kWh/year?
catchment = infer_catchment(10_000_000)
print(catchment["population"])          # ~65,116

# Forward step: simulate the base case.
result = simulate(Scenario(name="Base case"), n_samples=10_000, seed=12345)
import numpy as np
print(np.mean(result.miner_net_revenue["Antminer S21"]))   # ~$934,000

# Emissions value.
print(mitigation_value(Scenario())["net_mitigation_value_usd"])  # ~$7.63M
```

Define a custom scenario by overriding any parameter. The breakeven cost
bounds are free floats, and a parasitic split diverts part of generation away
from mining:

```python
sc = Scenario(
    name="Low cost, high hashprice, partial diversion",
    breakeven_low=0.01, breakeven_high=0.07,   # mean c_be = $0.040/kWh
    hashprice_mean=0.150,                       # $/TH/day
    parasitic_split=0.20,                       # 20% of generation sent elsewhere
)
result = simulate(sc)
print(result.electricity_for_mining_kwh)        # 8,000,000 (= potential x 0.8)
```

Run current-generation hardware instead of the paper's 2024 rigs:

```python
from landfill_btc import MODERN_RIGS
result = simulate(Scenario(rigs=list(MODERN_RIGS)))
```

Risk-sharing diagnostics:

```python
result = simulate(Scenario())
print(result.prob_miner_loss("Antminer S21"))        # P(miner net < 0)
print(np.median(result.miner_landfill_ratio["Antminer S21"]))  # median M:L
```

## Mapping to the manuscript

| Manuscript element | Code |
| --- | --- |
| Landfill assumptions, backward derivation | `model.infer_catchment`, `model.landfill_forward` |
| Table 1 (rig specifications) | `model.DEFAULT_RIGS` |
| Table 2 (parameters) | `LandfillParams`, `FinancialParams`, `MitigationParams`, `Scenario` |
| Table 3 (base-case S21 fleet) | `reproduce_paper.print_table3` |
| Table 4 (four scenarios) | `reproduce_paper.print_table4` |
| Figures 4-9 | `figures.py` |
| Emissions mitigation | `model.mitigation_value` |

Extensions beyond the published model:

- Parasitic split. `Scenario.parasitic_split` diverts a fraction of generated
  electricity away from mining; mining electricity is `potential x (1 - split)`.
  Methane mitigation is unchanged by the split (all captured LFG is still
  combusted), but the grid offset scales with the electricity actually mined.
- Modern rigs. `MODERN_RIGS` holds current-generation hardware (mid-2026 specs)
  alongside the frozen `DEFAULT_RIGS`. Hashrate and wall-power are sourced from
  manufacturer and review data; prices are time-stamped placeholders that track
  the BTC spot price and should be updated before use.
- Risk sharing. `SimulationResult.miner_landfill_ratio` exposes the raw,
  untruncated NETM/NETL ratio. Read its median, not its mean: the per-sample
  ratio is heavy-tailed because the landfill share floor (5%) puts the
  denominator near its minimum in many draws, so the mean mostly reports
  denominator behavior. The original Analytica file truncated the ratio to
  [0, 99], which biases the mean upward by folding loss cases into 0; that
  truncation is not reproduced here. The decision-relevant companion statistics
  are `prob_miner_loss()` and `prob_combined_loss()`, which report downside risk
  directly. The landfill's net revenue is always positive, so all loss risk
  falls on the miner.

## Reproducibility notes

Means, medians, standard deviations, and rig counts reproduce the published
tables to within rounding. Extreme minimum and maximum order statistics differ
slightly from the manuscript: these tails are determined by the specific Latin
hypercube permutation, which is generated differently by Analytica's and
NumPy's random number generators. Distributional shape and central tendency are
the robust comparisons and match. The deterministic stratum midpoints mean some
extremes (e.g., maximum annual hash revenue) reproduce exactly.

Two parameter labels in the original Analytica file were inconsistent with the
arithmetic that produced the published results; the code follows the arithmetic:

- The LFG generation parameter is labeled "150 kg/m³" but is applied as a yield
  of 0.15 m³ LFG per kg waste (`waste × 150 / 1000`), which is what produces the
  reported 7.8 million m³/year. The field is named `lfg_yield_m3_per_kg = 0.15`.
- The grid carbon intensity is described in text as "0.5 g/kWh" but applied as
  0.5 kg/kWh, which is what produces the reported $1.55M grid offset.

The final manuscript reports emissions value using point estimates — the social
cost of methane (SC-CH₄ = $4,200/mt CH₄) for gross value and SCC = $310/mt CO₂e
for the grid offset. The original Analytica mitigation module carried an earlier
stochastic CO₂e-based SCC; the code implements the published point-estimate
version.

## Citing this work

If you use this software, please cite the paper it reproduces:

> Rudd, M.A., M. Jones, D. Sechrest, D. Batten, and D. Porter. 2024. An
> integrated landfill
> gas-to-energy and Bitcoin mining framework. *Journal of Cleaner Production*
> 472: 143516. https://doi.org/10.1016/j.jclepro.2024.143516
>
> Open-access preprint: https://ssrn.com/abstract=4810964

To cite the software itself, see `CITATION.cff` (GitHub renders a "Cite this
repository" button from it). If a Zenodo archive of a release has been created,
cite that DOI for the exact version used.

## Code availability

This repository is the open-source reproduction referenced by the paper. A
suggested code-availability statement for the preprint or a revised manuscript:

> The model is reproduced in open-source Python at
> https://github.com/murrayrudd/landfill-gas-bitcoin-model (archived at
> Zenodo, DOI to be added).

## Author

Murray A. Rudd, Ph.D. — applied microeconomist and policy researcher.

- Blog: https://murrayrudd.pro/
- ORCID: https://orcid.org/0000-0001-9533-5070
- Google Scholar: https://scholar.google.co.uk/citations?hl=en&user=84qbofEAAAAJ

## License

Released under the MIT License, Copyright © 2024–2026 Murray A. Rudd. See
`LICENSE`.
