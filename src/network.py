"""Network cost model: bits -> time, energy, and transfer-window compliance.

The federated job is treated as a tenant on an uplink with a stated capacity,
latency and transmit power. Every strategy is costed on the SAME profiles from
its measured per-round upload bits (`round_bits` in each run summary), so the
numbers are reproducible from the logs and the profile parameters are the
only assumptions. The profiles are deliberately simple (no scheduling,
fading or retransmission model); the paper states them as assumptions and
reports sensitivity to the uplink rate.

Profile parameters (edit `PROFILES` to change; cite the source in the paper):
  nr_cell_edge   5 Mbps uplink, 30 ms RTT, 0.2 W  -- 5G NR FR1 cell-edge class
  nr_typical     50 Mbps, 20 ms, 0.2 W            -- 5G NR FR1 mid-cell
  sixg_slice     1 Gbps, 5 ms, 0.5 W              -- 6G sub-THz edge slice target
  leo_sat        10 Mbps, 40 ms, 2.0 W            -- LEO satellite uplink
  hospital_wan   100 Mbps, 10 ms, 0.0 W (wired)   -- shared institutional WAN
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping


@dataclass(frozen=True)
class LinkProfile:
    name: str
    uplink_mbps: float
    rtt_ms: float
    tx_power_w: float
    window_s: float = 60.0        # scheduled transfer window per round


PROFILES: Dict[str, LinkProfile] = {
    "nr_cell_edge": LinkProfile("nr_cell_edge", 5.0, 30.0, 0.2),
    "nr_typical": LinkProfile("nr_typical", 50.0, 20.0, 0.2),
    "sixg_slice": LinkProfile("sixg_slice", 1000.0, 5.0, 0.5),
    "leo_sat": LinkProfile("leo_sat", 10.0, 40.0, 2.0),
    "hospital_wan": LinkProfile("hospital_wan", 100.0, 10.0, 0.0),
}


def cost_round(max_client_bits: float, upload_bits: float, n_senders: float,
               p: LinkProfile) -> Dict[str, float]:
    """Clients upload in parallel on independent links; the round waits for
    the slowest (largest) upload plus one RTT when anything is sent."""
    if upload_bits <= 0:
        return {"time_s": 0.0, "energy_j": 0.0, "fits_window": 1.0}
    t_max = max_client_bits / (p.uplink_mbps * 1e6) + p.rtt_ms / 1e3
    energy = (upload_bits / (p.uplink_mbps * 1e6)) * p.tx_power_w       # sum over clients
    return {"time_s": t_max, "energy_j": energy, "fits_window": float(t_max <= p.window_s)}


def cost_run(round_bits: List[Mapping[str, float]], profile: str | LinkProfile) -> Dict[str, float]:
    p = PROFILES[profile] if isinstance(profile, str) else profile
    tot_t = tot_e = fits = 0.0
    for rb in round_bits:
        c = cost_round(rb["max_client_bits"], rb["upload_bits"], rb["n_senders"], p)
        tot_t += c["time_s"]; tot_e += c["energy_j"]; fits += c["fits_window"]
    n = max(len(round_bits), 1)
    return {f"net_{p.name}_comm_time_s": tot_t,
            f"net_{p.name}_energy_j": tot_e,
            f"net_{p.name}_window_compliance": fits / n}


def cost_all_profiles(round_bits: List[Mapping[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name in PROFILES:
        out.update(cost_run(round_bits, name))
    return out
