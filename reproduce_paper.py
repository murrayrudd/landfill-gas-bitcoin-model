"""Reproduce the published results and compare against the manuscript.

Run:  python reproduce_paper.py

This regenerates:
  - the backward landfill / catchment derivation;
  - Table 3 (full S21 fleet, base case);
  - Table 4 (four scenarios x three rigs, landfill and miner net revenue);
  - the emissions-mitigation point estimates.

Published values from the manuscript are shown alongside for verification.
"""

from __future__ import annotations

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


def money(x: float) -> str:
    return f"(${abs(x):,.0f})" if x < 0 else f"${x:,.0f}"


def print_landfill() -> None:
    print("=" * 72)
    print("LANDFILL CHAIN  (anchor: 10.0M kWh/yr electricity for mining)")
    print("=" * 72)
    fwd = landfill_forward(LandfillParams())
    print(f"{'Quantity':<34}{'Model':>20}{'Published':>18}")
    rows = [
        ("Catchment population", f"{LandfillParams().population:,.0f}", "~65,000"),
        ("LFG generated (m3/yr)", f"{fwd['lfg_generated_m3_yr']:,.0f}", "~7,800,000"),
        ("LFG flow (SCFM)", f"{fwd['scfm']:,.0f}", "527"),
        ("Methane collected (m3/yr)", f"{fwd['methane_collected_m3_yr']:,.0f}", "~3,333,000"),
        ("Electricity generated (kWh/yr)", f"{fwd['electricity_potential_kwh_yr']:,.0f}", "10,000,000"),
        ("Capacity (MW)", f"{fwd['megawatt_capacity']:.2f}", "1.14"),
    ]
    for label, model, pub in rows:
        print(f"{label:<34}{model:>20}{pub:>18}")

    print("\nInverse check (work backward from a 10.0M kWh/yr anchor):")
    inv = infer_catchment(10_000_000.0)
    print(f"  implied population = {inv['population']:,.0f}  (manuscript ~65,000)")


def print_table3() -> None:
    print("\n" + "=" * 72)
    print("TABLE 3  Annual financial performance, full S21 (air-cooled) fleet")
    print("base case: mean c_be = $0.055/kWh, mean p_h = $0.075/TH/day")
    print("=" * 72)
    res = simulate(Scenario(name="Base case"), n_samples=N, seed=SEED)
    rig = "Antminer S21"
    print(f"  Rigs deployed (S21): {res.rigs_deployed[rig]}  (published 326)")
    print(f"  Rig CAPEX (S21):     {money(res.rig_capex[rig])}  (published ~$1.63M)")
    print(f"  Total CAPEX (+15%):  {money(res.mining_capex[rig])}")
    print(f"  Annual depreciation: {money(res.annual_mining_depreciation[rig])}  (published ~$350,000)\n")

    series = {
        "Annual hash revenue, H": res.annual_hash_revenue[rig],
        "Landfill net revenue, NETL": res.landfill_net_revenue[rig],
        "Miner net revenue, NETM": res.miner_net_revenue[rig],
        "Combined net revenue": res.combined_net_revenue[rig],
        "Electricity cost, ECM": res.electricity_cost_of_mining,
    }
    published = {
        "Annual hash revenue, H": (368_555, 1_784_850, 1_937_735, 8_643_725),
        "Landfill net revenue, NETL": (29_861, 288_327, 331_665, 2_178_809),
        "Miner net revenue, NETM": (-671_799, 820_687, 934_809, 5_867_640),
        "Combined net revenue": (-518_159, 1_129_990, 1_266_473, 8_046_449),
        "Electricity cost, ECM": (100_215, 550_006, 550_006, 999_797),
    }
    import numpy as np
    hdr = f"{'Metric':<28}{'Min':>14}{'Median':>14}{'Mean':>14}{'Max':>14}"
    print(hdr)
    for name, arr in series.items():
        vals = (np.min(arr), np.median(arr), np.mean(arr), np.max(arr))
        print(f"{name:<28}" + "".join(f"{money(v):>14}" for v in vals))
        pub = published[name]
        print(f"{'  (published)':<28}" + "".join(f"{money(v):>14}" for v in pub))


def print_table4() -> None:
    print("\n" + "=" * 72)
    print("TABLE 4  Net revenue across scenarios and rigs")
    print("=" * 72)
    scenarios = [
        Scenario(name="1. Original (c_be 0.055, p_h 0.075)",
                 breakeven_low=0.01, breakeven_high=0.10, hashprice_mean=0.075),
        Scenario(name="2. Low c_be (0.040), p_h 0.075",
                 breakeven_low=0.01, breakeven_high=0.07, hashprice_mean=0.075),
        Scenario(name="3. c_be 0.055, high p_h (0.150)",
                 breakeven_low=0.01, breakeven_high=0.10, hashprice_mean=0.150),
        Scenario(name="4. Low c_be (0.040), high p_h (0.150)",
                 breakeven_low=0.01, breakeven_high=0.07, hashprice_mean=0.150),
    ]
    import numpy as np
    for sc in scenarios:
        res = simulate(sc, n_samples=N, seed=SEED)
        print(f"\n{sc.name}")
        print(f"  {'Metric':<22}{'Rig':<20}{'Median':>14}{'Mean':>14}{'Max':>14}")
        for label, series in (("Landfill net", res.landfill_net_revenue),
                              ("Miner net", res.miner_net_revenue)):
            for rig in res.rig_names:
                arr = series[rig]
                print(f"  {label:<22}{rig:<20}"
                      f"{money(np.median(arr)):>14}{money(np.mean(arr)):>14}"
                      f"{money(np.max(arr)):>14}")


def print_risk_sharing() -> None:
    print("\n" + "=" * 72)
    print("RISK SHARING  (base case)  -- M:L ratio and loss probabilities")
    print("=" * 72)
    res = simulate(Scenario(name="Base case"), n_samples=N, seed=SEED)
    import numpy as np
    print(f"  {'Rig':<20}{'M:L (median)':>14}{'P(miner loss)':>16}{'P(combined loss)':>18}")
    for rig in res.rig_names:
        ml = np.median(res.miner_landfill_ratio[rig])
        print(f"  {rig:<20}{ml:>14.2f}{res.prob_miner_loss(rig):>16.1%}"
              f"{res.prob_combined_loss(rig):>18.1%}")
    print("  Note: the median miner nets ~2-3x the landfill, but bears all the")
    print("  downside (the landfill's net is always positive).")


def print_parasitic_split() -> None:
    print("\n" + "=" * 72)
    print("PARASITIC SPLIT  (base case, S21, 20% of electricity diverted)")
    print("=" * 72)
    import numpy as np
    for split in (0.0, 0.20):
        res = simulate(Scenario(name=f"split={split}", parasitic_split=split),
                       n_samples=N, seed=SEED)
        rig = "Antminer S21"
        print(f"  split {split:>4.0%}:  to mining {res.electricity_for_mining_kwh:>12,.0f} kWh"
              f" | rigs {res.rigs_deployed[rig]:>4d}"
              f" | mean miner net {money(np.mean(res.miner_net_revenue[rig]))}")


def print_modern_rigs() -> None:
    print("\n" + "=" * 72)
    print("MODERN RIGS  (base case levers; mid-2026 specs, placeholder prices)")
    print("=" * 72)
    import numpy as np
    from landfill_btc import MODERN_RIGS
    res = simulate(Scenario(name="Modern", rigs=list(MODERN_RIGS)),
                   n_samples=N, seed=SEED)
    print(f"  {'Rig':<22}{'J/TH':>7}{'Rigs':>7}{'Median miner net':>20}")
    for rig in res.scenario.rigs:
        arr = res.miner_net_revenue[rig.name]
        print(f"  {rig.name:<22}{rig.efficiency_j_th:>7.1f}"
              f"{res.rigs_deployed[rig.name]:>7d}{money(np.median(arr)):>20}")


def print_mitigation() -> None:
    print("\n" + "=" * 72)
    print("EMISSIONS MITIGATION  (point estimates, as published)")
    print("=" * 72)
    m = mitigation_value(Scenario())
    rows = [
        ("CH4 mitigated (mt/yr)", f"{m['ch4_mitigated_mt_yr']:,.0f}", "2,187"),
        ("CO2e mitigated (mt/yr)", f"{m['co2e_mitigated_mt_yr']:,.0f}", "61,227"),
        ("Gross value", money(m["gross_mitigation_value_usd"]), "~$9,180,000"),
        ("Grid offset", money(m["grid_offset_usd"]), "~$1,550,000"),
        ("Net value", money(m["net_mitigation_value_usd"]), "~$7,630,000"),
    ]
    print(f"{'Quantity':<28}{'Model':>20}{'Published':>18}")
    for label, model, pub in rows:
        print(f"{label:<28}{model:>20}{pub:>18}")


if __name__ == "__main__":
    print_landfill()
    print_table3()
    print_table4()
    print_risk_sharing()
    print_parasitic_split()
    print_modern_rigs()
    print_mitigation()
    print("\nDone. Compare 'Model' against 'Published' / '(published)' rows above.")
