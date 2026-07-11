"""Landfill-gas-to-Bitcoin-mining model.

Python reproduction of the Monte Carlo model in:

    Rudd et al., "Bitcoin mining as a market-driven methane mitigation
    strategy at landfill gas-to-energy sites" (originally implemented in
    Lumina Analytica).

The model anchors on a midpoint assumption: the electricity the landfill can
generate (default 10.0 million kWh/year, ~1.14 MW). From that anchor it works:

  - backward, to the implied landfill characteristics and catchment population
    that would supply that much electricity; and
  - forward, to the financial performance of an integrated mining operation
    and the value of the methane emissions it mitigates.

A parasitic split may divert part of the generated electricity to other uses,
so the electricity available for mining is potential x (1 - split). All
monetary values are in USD. See ``reproduce_paper.py`` for the validation
harness against the published tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List

import numpy as np

from . import sampling


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LandfillParams:
    """Physical landfill / LFG-to-electricity conversion chain.

    The defaults reproduce the manuscript's base landfill: a catchment of
    ~65,000 people supplying ~7.8 million m3/yr of LFG and ~10.0 million
    kWh/yr (1.14 MW) of generated electricity. This chain is pure physics;
    the share of electricity sent to mining is a scenario-level decision
    (see ``Scenario.parasitic_split``), not a property of the landfill.

    Note on ``lfg_yield_m3_per_kg``: the original Analytica file labels the
    generation parameter "150 kg/m3" but computes LFG as waste * 150 / 1000,
    i.e. a yield of 0.15 m3 of LFG per kg of waste. The 0.15 figure is what
    produces the published 7.8 million m3/yr, so it is retained here.
    """

    per_capita_waste_kg_day: float = 2.2      # kg/person/day
    population: float = 65_116                 # persons in catchment
    lfg_yield_m3_per_kg: float = 0.15          # m3 LFG per kg waste
    collection_efficiency: float = 0.85        # fraction of LFG captured
    methane_fraction: float = 0.50             # CH4 share of LFG by volume
    energy_content_kwh_m3: float = 10.0        # kWh per m3 of methane
    conversion_efficiency: float = 0.30        # LFG energy -> electricity


@dataclass(frozen=True)
class RigSpec:
    """ASIC mining rig specification."""

    name: str
    hashrate_th_s: float       # TH/s
    power_w: float             # watts (wall)
    price_usd: float           # USD
    opex_fraction: float = 0.15  # non-electricity OPEX as share of power cost
    cooling: str = "Air"

    @property
    def efficiency_j_th(self) -> float:
        return self.power_w / self.hashrate_th_s


# Paper's rigs, frozen at the manuscript specs (Bitmain, 09 Feb 2024).
# Used for exact reproduction of the published results.
DEFAULT_RIGS: List[RigSpec] = [
    RigSpec("Antminer S19j XP", 151, 3_247, 3_473, 0.15, "Air"),
    RigSpec("Antminer S21", 200, 3_500, 5_000, 0.15, "Air"),
    RigSpec("Antminer S21 Hydro", 335, 5_360, 6_533, 0.15, "Liquid"),
]

# Current-generation rigs (specs current as of mid-2026; sources: Bitmain /
# MicroBT product pages and review aggregators). Hashrate and wall-power are
# well established; PRICES TRACK THE BTC SPOT PRICE AND VARY BY VENDOR -- the
# figures below are approximate mid-2026 placeholders and should be updated to
# current quotes before drawing financial conclusions.
MODERN_RIGS: List[RigSpec] = [
    RigSpec("Antminer S21 Pro", 234, 3_510, 3_800, 0.15, "Air"),
    RigSpec("Antminer S21 XP", 270, 3_645, 5_200, 0.15, "Air"),
    RigSpec("Antminer S21 XP Hyd", 473, 5_676, 6_800, 0.15, "Liquid"),
    RigSpec("Whatsminer M60S", 186, 3_441, 2_900, 0.15, "Air"),
]


@dataclass(frozen=True)
class FinancialParams:
    """Financial / revenue-sharing assumptions."""

    pool_fees: float = 0.02                 # mining pool fee
    landfill_share_min: float = 0.05        # min gross-revenue share to landfill
    landfill_share_max: float = 0.30        # max gross-revenue share to landfill
    non_rig_capex_fraction: float = 0.15    # non-rig CAPEX as share of rig CAPEX
    rig_depreciation_years: int = 5
    non_rig_depreciation_years: int = 10
    hashprice_gsdev: float = 1.5            # geometric SD of lognormal multiplier


@dataclass(frozen=True)
class MitigationParams:
    """Emissions / social-cost assumptions.

    The manuscript reports gross mitigation value using a point estimate of
    the social cost of methane (SC-CH4 = $4,200/mt CH4, 2% discount rate,
    horizon to 2050) applied to mitigated methane mass, and a grid offset
    using SCC = $310/mt CO2e applied to the CO2e of the grid power that would
    otherwise have run the rigs. (The original Analytica file carried a
    stochastic CO2e-based SCC; the point-estimate version below is the one
    used in the published results.)
    """

    methane_density_kg_m3: float = 0.656
    gwp100_methane: float = 28.0            # 100-yr GWP of CH4 vs CO2
    percent_vented: float = 1.0             # share vented absent development
    sc_ch4_per_mt: float = 4_200.0          # $/mt CH4
    scc_grid_per_mt: float = 310.0          # $/mt CO2e (grid offset)
    grid_carbon_intensity_kg_kwh: float = 0.5


@dataclass(frozen=True)
class Scenario:
    """A runnable configuration: electricity supply + cost / price levers.

    ``electricity_potential_kwh`` is the total electricity the landfill can
    generate. ``parasitic_split`` is the fraction diverted away from mining to
    other uses (grid sale, on-site load), so electricity available for mining
    is potential x (1 - split). The breakeven cost levers are free floats; set
    them to any plausible $/kWh bounds.
    """

    name: str = "Base case"
    electricity_potential_kwh: float = 10_000_000.0
    parasitic_split: float = 0.0            # fraction diverted from mining
    breakeven_low: float = 0.01             # $/kWh, lower truncation
    breakeven_high: float = 0.10            # $/kWh, upper truncation
    hashprice_mean: float = 0.075           # $/TH/day, mean of lognormal
    landfill: LandfillParams = field(default_factory=LandfillParams)
    financial: FinancialParams = field(default_factory=FinancialParams)
    mitigation: MitigationParams = field(default_factory=MitigationParams)
    rigs: List[RigSpec] = field(default_factory=lambda: list(DEFAULT_RIGS))

    @property
    def electricity_for_mining_kwh(self) -> float:
        return self.electricity_potential_kwh * (1.0 - self.parasitic_split)

    @property
    def unavailable_for_mining_kwh(self) -> float:
        return self.electricity_potential_kwh * self.parasitic_split

    @property
    def breakeven_mean(self) -> float:
        return (self.breakeven_low + self.breakeven_high) / 2.0

    @property
    def breakeven_sd(self) -> float:
        # Analytica: STDEV = (BEL + BEH) / 5
        return (self.breakeven_low + self.breakeven_high) / 5.0


# --------------------------------------------------------------------------- #
# Landfill chain: forward and inverse
# --------------------------------------------------------------------------- #
def _electricity_per_person_kwh(lf: LandfillParams) -> float:
    """kWh/yr of generated electricity per person in the catchment."""
    return (
        365.0
        * lf.per_capita_waste_kg_day
        * lf.lfg_yield_m3_per_kg
        * lf.collection_efficiency
        * lf.methane_fraction
        * lf.energy_content_kwh_m3
        * lf.conversion_efficiency
    )


def landfill_forward(lf: LandfillParams) -> Dict[str, float]:
    """Forward chain: population -> LFG -> methane -> generated electricity.

    Reproduces the manuscript landfill figures (LFG ~7.8M m3/yr, ~527 SCFM,
    ~10.0M kWh/yr generated, ~1.14 MW).
    """
    waste_kg_yr = 365.0 * lf.per_capita_waste_kg_day * lf.population
    lfg_generated_m3 = waste_kg_yr * lf.lfg_yield_m3_per_kg
    lfg_collected_m3 = lf.collection_efficiency * lfg_generated_m3
    methane_collected_m3 = lf.methane_fraction * lfg_collected_m3
    energy_available_kwh = methane_collected_m3 * lf.energy_content_kwh_m3
    electricity_potential_kwh = energy_available_kwh * lf.conversion_efficiency
    return {
        "waste_kg_yr": waste_kg_yr,
        "lfg_generated_m3_yr": lfg_generated_m3,
        "lfg_collected_m3_yr": lfg_collected_m3,
        "methane_collected_m3_yr": methane_collected_m3,
        "energy_available_kwh_yr": energy_available_kwh,
        "electricity_potential_kwh_yr": electricity_potential_kwh,
        "scfm": lfg_generated_m3 * 35.3147 / 525_600.0,
        "megawatt_capacity": electricity_potential_kwh / 8760.0 / 1000.0,
    }


def infer_catchment(
    target_electricity_kwh: float, lf: LandfillParams = LandfillParams()
) -> Dict[str, float]:
    """Inverse chain: given generated electricity, infer the catchment.

    Formalizes the manuscript's "work backward" step: choosing the midpoint
    electricity supply and solving for the landfill (and population) that
    would produce it under the stated conversion assumptions.
    """
    per_person = _electricity_per_person_kwh(lf)
    population = target_electricity_kwh / per_person
    derived = landfill_forward(replace(lf, population=population))
    derived["population"] = population
    return derived


# --------------------------------------------------------------------------- #
# Monte Carlo financial simulation
# --------------------------------------------------------------------------- #
@dataclass
class SimulationResult:
    scenario: Scenario
    electricity_potential_kwh: float
    electricity_for_mining_kwh: float
    unavailable_for_mining_kwh: float
    rig_names: List[str]
    rigs_deployed: Dict[str, int]
    # Each array has shape (n_samples,), one per rig name.
    annual_hash_revenue: Dict[str, np.ndarray]
    landfill_net_revenue: Dict[str, np.ndarray]
    miner_net_revenue: Dict[str, np.ndarray]
    combined_net_revenue: Dict[str, np.ndarray]
    miner_landfill_ratio: Dict[str, np.ndarray]   # raw NETM/NETL, untruncated
    electricity_cost_of_mining: np.ndarray
    annual_mining_depreciation: Dict[str, float]
    mining_capex: Dict[str, float]          # total: rig + non-rig
    rig_capex: Dict[str, float]             # rig only (matches manuscript)

    def summary(self, series: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        return {name: summary_stats(arr) for name, arr in series.items()}

    def prob_miner_loss(self, rig: str) -> float:
        """P(miner net revenue < 0) -- downside risk borne by the miner."""
        return float(np.mean(self.miner_net_revenue[rig] < 0))

    def prob_combined_loss(self, rig: str) -> float:
        return float(np.mean(self.combined_net_revenue[rig] < 0))


def summary_stats(arr: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr, ddof=1)),
    }


def rigs_deployed(electricity_kwh: float, rig: RigSpec) -> int:
    """Number of rigs that fully consume the available electricity."""
    annual_kwh_per_rig = (rig.power_w / 1000.0) * 24.0 * 365.0
    return int(round(electricity_kwh / annual_kwh_per_rig))


def simulate(
    scenario: Scenario,
    n_samples: int = 10_000,
    seed: int | None = 12345,
) -> SimulationResult:
    """Run the Monte Carlo simulation for every rig in the scenario."""
    rng = np.random.default_rng(seed)
    fin = scenario.financial

    potential = scenario.electricity_potential_kwh
    if potential is None:
        potential = landfill_forward(scenario.landfill)["electricity_potential_kwh_yr"]
    elec = potential * (1.0 - scenario.parasitic_split)   # electricity to mining

    # Three independent stochastic dimensions (shared across rigs).
    breakeven = sampling.sample_truncated_normal(
        n_samples,
        mean=scenario.breakeven_mean,
        sd=scenario.breakeven_sd,
        low=scenario.breakeven_low,
        high=scenario.breakeven_high,
        rng=rng,
    )
    landfill_share = sampling.sample_uniform(
        n_samples, fin.landfill_share_min, fin.landfill_share_max, rng=rng
    )
    miner_share = 1.0 - landfill_share
    hashprice = scenario.hashprice_mean * sampling.sample_lognormal_median(
        n_samples, median=1.0, gsdev=fin.hashprice_gsdev, rng=rng
    )

    # Electricity cost is independent of rig choice (fixed kWh, drawn $/kWh).
    electricity_cost = elec * breakeven

    deployed: Dict[str, int] = {}
    hash_rev: Dict[str, np.ndarray] = {}
    landfill_net: Dict[str, np.ndarray] = {}
    miner_net: Dict[str, np.ndarray] = {}
    combined_net: Dict[str, np.ndarray] = {}
    ratio: Dict[str, np.ndarray] = {}
    depreciation: Dict[str, float] = {}
    capex: Dict[str, float] = {}
    rig_capex_only: Dict[str, float] = {}

    for rig in scenario.rigs:
        n_rigs = rigs_deployed(elec, rig)
        deployed[rig.name] = n_rigs

        annual_hash = 365.0 * (n_rigs * rig.hashrate_th_s * hashprice)
        landfill_hash = annual_hash * landfill_share
        miner_hash = annual_hash * miner_share

        other_opex = rig.opex_fraction * electricity_cost
        net_landfill = landfill_hash - landfill_hash * fin.pool_fees
        net_miner = (
            miner_hash
            - electricity_cost
            - miner_hash * fin.pool_fees
            - other_opex
        )

        hash_rev[rig.name] = annual_hash
        landfill_net[rig.name] = net_landfill
        miner_net[rig.name] = net_miner
        combined_net[rig.name] = net_landfill + net_miner
        # Landfill net is always > 0 (share >= 5%, hash revenue > 0), so the
        # ratio is well defined; reported raw (untruncated). Use the median,
        # not the mean, and read alongside prob_miner_loss().
        ratio[rig.name] = net_miner / net_landfill

        rig_capex = n_rigs * rig.price_usd
        other_capex = fin.non_rig_capex_fraction * rig_capex
        rig_capex_only[rig.name] = rig_capex
        capex[rig.name] = rig_capex + other_capex
        depreciation[rig.name] = (
            rig_capex / fin.rig_depreciation_years
            + other_capex / fin.non_rig_depreciation_years
        )

    return SimulationResult(
        scenario=scenario,
        electricity_potential_kwh=potential,
        electricity_for_mining_kwh=elec,
        unavailable_for_mining_kwh=potential * scenario.parasitic_split,
        rig_names=[r.name for r in scenario.rigs],
        rigs_deployed=deployed,
        annual_hash_revenue=hash_rev,
        landfill_net_revenue=landfill_net,
        miner_net_revenue=miner_net,
        combined_net_revenue=combined_net,
        miner_landfill_ratio=ratio,
        electricity_cost_of_mining=electricity_cost,
        annual_mining_depreciation=depreciation,
        mining_capex=capex,
        rig_capex=rig_capex_only,
    )


# --------------------------------------------------------------------------- #
# Emissions mitigation (point-estimate, as published)
# --------------------------------------------------------------------------- #
def mitigation_value(
    scenario: Scenario, electricity_potential_kwh: float | None = None
) -> Dict[str, float]:
    """Value of methane mitigation for the modeled landfill.

    Methane is destroyed regardless of where the electricity is used, so gross
    value is keyed to total generated electricity (potential). The grid offset
    -- the counterfactual grid power that would have run the rigs -- is keyed to
    the electricity actually sent to mining, i.e. potential x (1 - split).

    With split = 0 this reproduces the published figures: ~2,187 mt CH4
    (~61,227 mt CO2e), gross ~$9.18M, grid offset ~$1.55M, net ~$7.63M.
    """
    mit = scenario.mitigation
    lf = scenario.landfill
    potential = (
        electricity_potential_kwh
        if electricity_potential_kwh is not None
        else scenario.electricity_potential_kwh
    )
    elec_mining = potential * (1.0 - scenario.parasitic_split)

    # Methane consistent with the generated-electricity anchor.
    methane_m3 = potential / lf.conversion_efficiency / lf.energy_content_kwh_m3
    ch4_mt = methane_m3 * mit.percent_vented * mit.methane_density_kg_m3 / 1000.0
    co2e_mt = ch4_mt * mit.gwp100_methane

    gross_value = ch4_mt * mit.sc_ch4_per_mt
    grid_co2e_mt = elec_mining * mit.grid_carbon_intensity_kg_kwh / 1000.0
    grid_offset = grid_co2e_mt * mit.scc_grid_per_mt
    net_value = gross_value - grid_offset

    return {
        "methane_m3_yr": methane_m3,
        "ch4_mitigated_mt_yr": ch4_mt,
        "co2e_mitigated_mt_yr": co2e_mt,
        "gross_mitigation_value_usd": gross_value,
        "grid_offset_usd": grid_offset,
        "net_mitigation_value_usd": net_value,
    }
