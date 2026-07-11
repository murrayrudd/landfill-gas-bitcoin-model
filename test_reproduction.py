"""Regression tests: the model must reproduce the published central values.

Run:  python -m pytest test_reproduction.py    (or)    python test_reproduction.py
"""

from __future__ import annotations

import numpy as np

from landfill_btc import (
    LandfillParams,
    Scenario,
    infer_catchment,
    landfill_forward,
    mitigation_value,
    simulate,
)

N = 10_000
SEED = 12345
RIG = "Antminer S21"


def _close(a, b, rel=0.01):
    return abs(a - b) <= rel * abs(b)


def test_landfill_chain():
    fwd = landfill_forward(LandfillParams())
    assert _close(fwd["lfg_generated_m3_yr"], 7_843_222, rel=0.001)
    assert round(fwd["scfm"]) == 527
    assert _close(fwd["electricity_potential_kwh_yr"], 10_000_000, rel=0.001)
    assert _close(fwd["megawatt_capacity"], 1.14, rel=0.01)


def test_inverse_catchment():
    inv = infer_catchment(10_000_000)
    assert _close(inv["population"], 65_116, rel=0.01)


def test_table3_base_case():
    res = simulate(Scenario(name="Base case"), n_samples=N, seed=SEED)
    assert res.rigs_deployed[RIG] == 326
    assert _close(res.rig_capex[RIG], 1_630_000, rel=0.001)
    assert _close(res.annual_mining_depreciation[RIG], 350_000, rel=0.01)

    means = {
        "H": (np.mean(res.annual_hash_revenue[RIG]), 1_937_735),
        "NETL": (np.mean(res.landfill_net_revenue[RIG]), 331_665),
        "NETM": (np.mean(res.miner_net_revenue[RIG]), 934_809),
        "REVtotal": (np.mean(res.combined_net_revenue[RIG]), 1_266_473),
        "ECM": (np.mean(res.electricity_cost_of_mining), 550_006),
    }
    for label, (got, expected) in means.items():
        assert _close(got, expected, rel=0.01), f"{label}: {got} vs {expected}"


def test_scenario_high_hashprice():
    res = simulate(
        Scenario(breakeven_low=0.01, breakeven_high=0.10, hashprice_mean=0.150),
        n_samples=N, seed=SEED,
    )
    assert _close(np.mean(res.miner_net_revenue[RIG]), 2_500_970, rel=0.01)


def test_mitigation():
    m = mitigation_value(Scenario())
    assert round(m["ch4_mitigated_mt_yr"]) == 2_187
    assert round(m["co2e_mitigated_mt_yr"]) == 61_227
    assert _close(m["gross_mitigation_value_usd"], 9_180_000, rel=0.01)
    assert _close(m["net_mitigation_value_usd"], 7_630_000, rel=0.01)


def test_parasitic_split():
    base = simulate(Scenario(parasitic_split=0.0), n_samples=N, seed=SEED)
    split = simulate(Scenario(parasitic_split=0.20), n_samples=N, seed=SEED)
    # 20% diverted -> 80% of mining electricity and roughly 80% of rigs.
    assert _close(split.electricity_for_mining_kwh,
                  0.80 * base.electricity_potential_kwh, rel=1e-9)
    assert split.rigs_deployed[RIG] < base.rigs_deployed[RIG]
    # Methane mitigated is unchanged (all LFG still combusted).
    m0 = mitigation_value(Scenario(parasitic_split=0.0))
    m2 = mitigation_value(Scenario(parasitic_split=0.20))
    assert _close(m0["ch4_mitigated_mt_yr"], m2["ch4_mitigated_mt_yr"], rel=1e-9)
    # Grid offset scales with mining electricity, so net value rises.
    assert m2["grid_offset_usd"] < m0["grid_offset_usd"]


def test_modern_rigs_run():
    from landfill_btc import MODERN_RIGS
    res = simulate(Scenario(rigs=list(MODERN_RIGS)), n_samples=N, seed=SEED)
    assert set(res.rig_names) == {r.name for r in MODERN_RIGS}
    for rig in MODERN_RIGS:
        assert res.rigs_deployed[rig.name] > 0


def test_risk_metrics():
    res = simulate(Scenario(), n_samples=N, seed=SEED)
    # Landfill net is always positive -> ratio is finite and loss prob > 0.
    assert np.all(res.landfill_net_revenue[RIG] > 0)
    assert 0.0 <= res.prob_miner_loss(RIG) <= 1.0
    assert np.median(res.miner_landfill_ratio[RIG]) > 1.0  # miner nets more


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("All reproduction tests passed.")
