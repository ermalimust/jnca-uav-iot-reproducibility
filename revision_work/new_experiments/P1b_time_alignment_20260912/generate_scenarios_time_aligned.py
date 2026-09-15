#!/usr/bin/env python3
"""Minimal controlled DES generator for Paper3.

This first executable version intentionally covers:

- normal factual traces,
- wifi_only factual traces,
- ble_only factual traces,
- mobility_only factual traces,
- video_only factual traces,
- wifi_video / wifi_mobility / all_mixed factual traces,
- remove_W / remove_B / remove_M / remove_V counterfactual traces,
- window-level feature export,
- label_W / label_B / label_M / label_V generation and basic diagnostics.

The implementation keeps retry/backoff as service-layer outcomes rather than
direct Wi-Fi labels, and aligns factual/counterfactual windows by wall-clock
time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CAUSE_W = "W"
CAUSE_B = "B"
CAUSE_M = "M"
CAUSE_V = "V"
RUN_FACTUAL = "factual"
RUN_REMOVE_W = "remove_W"
RUN_REMOVE_B = "remove_B"
RUN_REMOVE_M = "remove_M"
RUN_REMOVE_V = "remove_V"


def stable_seed(*parts: object) -> int:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def stable_rng(*parts: object) -> random.Random:
    return random.Random(stable_seed(*parts))


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def overlap_ms(start: float, end: float, intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    for a, b in intervals:
        if b <= start:
            continue
        if a >= end:
            break
        total += max(0.0, min(end, b) - max(start, a))
    return total


def overlaps(start: float, end: float, intervals: list[tuple[float, float]]) -> bool:
    return overlap_ms(start, end, intervals) > 0.0


def first_interval_end_after(t: float, intervals: list[tuple[float, float]]) -> float | None:
    for a, b in intervals:
        if a <= t < b:
            return b
        if a > t:
            return None
    return None


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    template_id: str
    seed: int
    family: str
    mission_type: str
    fleet_size: int
    area_type: str
    speed_mps: float
    distance_m: float
    rssi_mean_dbm: float
    rssi_sigma_db: float
    active_mechanisms: tuple[str, ...]
    wifi_regime: str
    ble_regime: str
    video_burst_regime: str


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def expand_scenarios(config: dict[str, Any], families: set[str]) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for template in config["scenarios"]:
        family = template["scenario_family"]
        if family not in families:
            continue
        for seed in template["seeds"]:
            scenarios.append(
                Scenario(
                    scenario_id=f"{template['scenario_template_id']}_seed{seed}",
                    template_id=template["scenario_template_id"],
                    seed=int(seed),
                    family=family,
                    mission_type=template["mission_type"],
                    fleet_size=int(template["fleet_size"]),
                    area_type=template["area_type"],
                    speed_mps=float(template["speed_mps"]),
                    distance_m=float(template["distance_m"]),
                    rssi_mean_dbm=float(template["rssi_mean_dbm"]),
                    rssi_sigma_db=float(template["rssi_sigma_db"]),
                    active_mechanisms=tuple(template["active_mechanisms"]),
                    wifi_regime=template["wifi_regime"],
                    ble_regime=template.get("ble_regime", "light"),
                    video_burst_regime=template["video_burst_regime"],
                )
            )
    return scenarios


def seed_bundle(seed: int) -> dict[str, int]:
    return {
        "traffic": stable_seed(seed, "traffic"),
        "wifi": stable_seed(seed, "wifi"),
        "ble": stable_seed(seed, "ble"),
        "video": stable_seed(seed, "video"),
        "mobility": stable_seed(seed, "mobility"),
        "service": stable_seed(seed, "service"),
    }


def range_draw(rng: random.Random, bounds: list[float]) -> float:
    return rng.uniform(float(bounds[0]), float(bounds[1]))


def generate_busy_intervals(
    config: dict[str, Any],
    scenario: Scenario,
    run_type: str,
    duration_ms: float,
) -> tuple[list[tuple[float, float]], dict[str, float]]:
    wifi_cfg = config["mechanisms"]["wifi_contention"]
    has_w = CAUSE_W in scenario.active_mechanisms
    removing_w = run_type == RUN_REMOVE_W
    regime = "low" if has_w and removing_w else scenario.wifi_regime
    timeline_kind = "baseline" if has_w and removing_w else "scenario"
    rng = stable_rng(seed_bundle(scenario.seed)["wifi"], timeline_kind, regime)
    regime_cfg = wifi_cfg[regime]
    busy_ratio = range_draw(rng, regime_cfg["busy_ratio"])
    burst_min, burst_max = regime_cfg["busy_burst_ms"]
    target_busy = duration_ms * busy_ratio
    intervals: list[tuple[float, float]] = []
    accumulated = 0.0
    guard = 0
    while accumulated < target_busy and guard < 100000:
        guard += 1
        length = rng.uniform(float(burst_min), float(burst_max))
        start = rng.uniform(0.0, max(0.0, duration_ms - length))
        intervals.append((start, start + length))
        accumulated += length
    merged = merge_intervals(intervals)
    actual_busy = sum(end - start for start, end in merged) / duration_ms
    meta = {
        "wifi_regime_effective": regime,
        "wifi_busy_ratio_target": busy_ratio,
        "wifi_busy_ratio_actual": actual_busy,
        "backoff_scale": float(regime_cfg["backoff_scale"]),
    }
    return merged, meta


def generate_ble_intervals(
    config: dict[str, Any],
    scenario: Scenario,
    run_type: str,
    duration_ms: float,
) -> tuple[list[tuple[float, float]], dict[str, float]]:
    has_b = CAUSE_B in scenario.active_mechanisms
    removing_b = run_type == RUN_REMOVE_B
    if not has_b or removing_b:
        return [], {
            "ble_regime_effective": "off",
            "ble_occupancy_ratio_actual": 0.0,
            "ble_interval_ms": 0.0,
        }
    ble_cfg = config["mechanisms"]["ble_rid"][scenario.ble_regime]
    rng = stable_rng(seed_bundle(scenario.seed)["ble"], "scenario", scenario.ble_regime)
    interval_ms = float(ble_cfg["advertising_interval_ms"])
    duration_min, duration_max = ble_cfg["occupancy_duration_ms"]
    phase = rng.uniform(0.0, interval_ms)
    intervals: list[tuple[float, float]] = []
    t = phase
    while t < duration_ms:
        length = rng.uniform(float(duration_min), float(duration_max))
        intervals.append((t, min(duration_ms, t + length)))
        t += interval_ms
    merged = merge_intervals(intervals)
    actual = sum(end - start for start, end in merged) / duration_ms
    return merged, {
        "ble_regime_effective": scenario.ble_regime,
        "ble_occupancy_ratio_actual": actual,
        "ble_interval_ms": interval_ms,
    }


def generate_c2_arrivals(config: dict[str, Any], scenario: Scenario, duration_ms: float) -> list[dict[str, float]]:
    c2_cfg = config["traffic"]["c2"]
    period_ms = float(c2_cfg["period_ms_by_family"].get(scenario.family, 50))
    jitter_min, jitter_max = c2_cfg["jitter_ms"]
    size_min, size_max = c2_cfg["packet_size_bytes"]
    rng = stable_rng(seed_bundle(scenario.seed)["traffic"], "c2")
    arrivals: list[dict[str, float]] = []
    t = 0.0
    packet_id = 0
    while t < duration_ms:
        jitter = rng.uniform(float(jitter_min), float(jitter_max))
        size = rng.uniform(float(size_min), float(size_max))
        arrivals.append({"packet_id": packet_id, "arrival_ms": t + jitter, "size_bytes": size})
        packet_id += 1
        t += period_ms
    return arrivals


def generate_video_windows(
    config: dict[str, Any],
    scenario: Scenario,
    run_type: str,
    duration_ms: float,
    window_ms: float,
) -> dict[int, dict[str, float]]:
    video_cfg = config["traffic"]["video"]
    normal_low, normal_high = video_cfg["normal_rate_mbps"]
    burst_low, burst_high = video_cfg["burst_rate_mbps"]
    has_v = CAUSE_V in scenario.active_mechanisms
    removing_v = run_type == RUN_REMOVE_V
    if has_v and removing_v:
        baseline_rate = (float(normal_low) + float(normal_high)) / 2.0
        return {
            window_id: {
                "video_offered_load_mbps": baseline_rate,
                "video_burst_ratio": 0.0,
            }
            for window_id in range(int(duration_ms // window_ms))
        }
    regime_cfg = video_cfg["burst_regimes"][scenario.video_burst_regime]
    rng = stable_rng(seed_bundle(scenario.seed)["video"], "scenario", scenario.video_burst_regime)
    state = "normal"
    windows: dict[int, dict[str, float]] = {}
    for window_id in range(int(duration_ms // window_ms)):
        if state == "normal" and rng.random() < float(regime_cfg["p_normal_to_burst"]):
            state = "burst"
        elif state == "burst" and rng.random() < float(regime_cfg["p_burst_to_normal"]):
            state = "normal"
        if state == "burst":
            offered = rng.uniform(float(burst_low), float(burst_high))
            burst_ratio = 1.0
        else:
            offered = rng.uniform(float(normal_low), float(normal_high))
            burst_ratio = 0.0
        windows[window_id] = {
            "video_offered_load_mbps": offered,
            "video_burst_ratio": burst_ratio,
        }
    return windows


def is_blocked(config: dict[str, Any], scenario: Scenario, run_type: str, t_ms: float) -> bool:
    if CAUSE_M not in scenario.active_mechanisms or run_type == RUN_REMOVE_M:
        return False
    mobility_cfg = config["mechanisms"]["mobility_fading"]
    rng = stable_rng(seed_bundle(scenario.seed)["mobility"], "blockage", int(t_ms // 100))
    return rng.random() < float(mobility_cfg["blockage_probability"])


def rssi_at(config: dict[str, Any], scenario: Scenario, run_type: str, t_ms: float) -> float:
    rng = stable_rng(seed_bundle(scenario.seed)["mobility"], "rssi", int(t_ms // 20))
    slow_wave = 1.5 * math.sin((t_ms / 1000.0) / 12.0)
    if CAUSE_M in scenario.active_mechanisms and run_type == RUN_REMOVE_M:
        sigma = float(config["mechanisms"]["mobility_fading"]["baseline_sigma_db"])
    else:
        sigma = scenario.rssi_sigma_db
    noise = rng.gauss(0.0, sigma)
    blockage_loss = 0.0
    if is_blocked(config, scenario, run_type, t_ms):
        blockage_loss = float(config["mechanisms"]["mobility_fading"]["blockage_loss_db"])
    return scenario.rssi_mean_dbm + slow_wave + noise - blockage_loss


def packet_error_rate(
    config: dict[str, Any],
    scenario: Scenario,
    run_type: str,
    t_ms: float,
    busy: bool,
    ble_blocked: bool,
    video_pressure: float,
) -> float:
    base_per = float(config["service"]["base_per"])
    rssi = rssi_at(config, scenario, run_type, t_ms)
    rssi_penalty = clamp((-66.0 - rssi) / 30.0, 0.0, 0.55)
    busy_penalty = 0.12 if busy else 0.0
    ble_penalty = 0.20 if ble_blocked else 0.0
    video_penalty = min(0.25, float(config["service"]["video_per_pressure_scale"]) * video_pressure)
    return clamp(base_per + rssi_penalty + busy_penalty + ble_penalty + video_penalty, 0.0, 0.95)


def simulate_packets(
    config: dict[str, Any],
    scenario: Scenario,
    run_type: str,
    arrivals: list[dict[str, float]],
    busy_intervals: list[tuple[float, float]],
    ble_intervals: list[tuple[float, float]],
    video_windows: dict[int, dict[str, float]],
    wifi_meta: dict[str, float],
    window_ms: float,
) -> list[dict[str, float | int | str]]:
    service_cfg = config["service"]
    service_min, service_max = service_cfg["base_service_ms"]
    backoff_min, backoff_max = service_cfg["base_backoff_ms"]
    max_attempts = int(service_cfg["max_attempts"])
    busy_collision_prob = float(service_cfg["busy_collision_prob"])
    ble_block_prob = float(service_cfg["ble_block_prob"])
    video_pressure_scale = float(service_cfg["video_service_pressure_scale"])
    target_video = float(config["global"]["target_video_mbps"])
    backoff_scale = float(wifi_meta["backoff_scale"])
    service_seed = seed_bundle(scenario.seed)["service"]
    server_free = 0.0
    packets: list[dict[str, float | int | str]] = []
    for packet in arrivals:
        packet_id = int(packet["packet_id"])
        arrival = float(packet["arrival_ms"])
        attempt_start = max(arrival, server_free)
        service_start = attempt_start
        retries = 0
        attempts = 0
        backoff_total = 0.0
        status = "delivered"
        while attempts < max_attempts:
            attempts += 1
            rng_service = stable_rng(service_seed, scenario.scenario_id, packet_id, attempts, "service")
            service_time = rng_service.uniform(float(service_min), float(service_max))
            video_window_id = int(attempt_start // window_ms)
            video_offered = float(video_windows.get(video_window_id, {"video_offered_load_mbps": target_video})["video_offered_load_mbps"])
            video_pressure = max(0.0, (video_offered - target_video) / max(target_video, 1e-9))
            service_time *= 1.0 + video_pressure_scale * video_pressure
            busy = overlaps(attempt_start, attempt_start + service_time, busy_intervals)
            ble_blocked = overlaps(attempt_start, attempt_start + service_time, ble_intervals)
            per = packet_error_rate(config, scenario, run_type, attempt_start, busy, ble_blocked, video_pressure)
            failed_by_collision = busy and rng_service.random() < busy_collision_prob
            failed_by_ble = ble_blocked and rng_service.random() < ble_block_prob
            failed_by_per = rng_service.random() < per
            if not failed_by_collision and not failed_by_ble and not failed_by_per:
                completion = attempt_start + service_time
                break
            retries += 1
            busy_end = first_interval_end_after(attempt_start, busy_intervals)
            ble_end = first_interval_end_after(attempt_start, ble_intervals)
            next_attempt = attempt_start + service_time
            if busy_end is not None:
                next_attempt = max(next_attempt, busy_end)
            if ble_end is not None:
                next_attempt = max(next_attempt, ble_end)
            attempt_start = next_attempt
            backoff_rng = stable_rng(service_seed, scenario.scenario_id, packet_id, attempts, "backoff")
            backoff = backoff_rng.uniform(float(backoff_min), float(backoff_max)) * backoff_scale
            backoff_total += backoff
            attempt_start += backoff
        else:
            status = "max_attempts"
            completion = attempt_start
        server_free = completion
        delay = completion - arrival
        period = float(config["traffic"]["c2"]["period_ms_by_family"].get(scenario.family, 50))
        packets.append(
            {
                "scenario_id": scenario.scenario_id,
                "run_type": run_type,
                "seed": scenario.seed,
                "packet_id": packet_id,
                "arrival_ms": round(arrival, 6),
                "service_start_ms": round(service_start, 6),
                "completion_ms": round(completion, 6),
                "delay_ms": round(delay, 6),
                "queue_wait_ms": round(max(0.0, service_start - arrival), 6),
                "attempts": attempts,
                "retries": retries,
                "backoff_total_ms": round(backoff_total, 6),
                "deadline_violation": int(delay > float(config["global"]["c2_delay_threshold_ms"])),
                "queue_len_proxy": max(0, int((service_start - arrival) // period)),
                "status": status,
            }
        )
    return packets


def summarize_window(
    config: dict[str, Any],
    scenario: Scenario,
    run_type: str,
    window_id: int,
    t_start: float,
    t_end: float,
    packets: list[dict[str, float | int | str]],
    busy_intervals: list[tuple[float, float]],
    ble_intervals: list[tuple[float, float]],
    video_windows: dict[int, dict[str, float]],
) -> dict[str, Any]:
    in_window = [p for p in packets if t_start <= float(p["arrival_ms"]) < t_end]
    delays = [float(p["delay_ms"]) for p in in_window]
    retries = sum(int(p["retries"]) for p in in_window)
    attempts = sum(int(p["attempts"]) for p in in_window)
    violations = sum(int(p["deadline_violation"]) for p in in_window)
    backoff_values = [float(p["backoff_total_ms"]) for p in in_window if int(p["retries"]) > 0]
    qlens = [float(p["queue_len_proxy"]) for p in in_window]
    completions = sorted(float(p["completion_ms"]) for p in in_window)
    intervals = [b - a for a, b in zip(completions, completions[1:])]
    busy_ratio = overlap_ms(t_start, t_end, busy_intervals) / (t_end - t_start)
    ble_occupancy_ratio = overlap_ms(t_start, t_end, ble_intervals) / (t_end - t_start)
    overlapping_ble = [(a, b) for a, b in ble_intervals if b > t_start and a < t_end]
    retry_rate = retries / attempts if attempts else 0.0
    backoff_proxy = statistics.mean(backoff_values) if backoff_values else 0.0
    video = video_windows.get(int(t_start // float(config["global"]["window_ms"])), {"video_offered_load_mbps": 1.0, "video_burst_ratio": 0.0})
    offered_video = float(video["video_offered_load_mbps"])
    video_throughput = max(0.0, offered_video * (1.0 - 0.60 * busy_ratio - 0.25 * retry_rate - 0.15 * ble_occupancy_ratio))
    violation_ratio = violations / len(in_window) if in_window else 0.0
    delay_mean = statistics.mean(delays) if delays else 0.0
    delay_p95 = pct(delays, 0.95)
    telemetry_period = float(config["traffic"]["telemetry"]["period_ms"])
    telemetry_count = int((t_end - t_start) / telemetry_period) if t_end - t_start >= telemetry_period else 0
    telemetry_loss = clamp(0.15 * busy_ratio + 0.10 * retry_rate + 0.10 * ble_occupancy_ratio + 0.05 * violation_ratio, 0.0, 1.0)
    jitter = statistics.pstdev(delays) if len(delays) > 1 else 0.0
    rssi_samples = [rssi_at(config, scenario, run_type, t_start + i * 20.0) for i in range(5)]
    rssi_mean = statistics.mean(rssi_samples)
    rssi_var = statistics.pvariance(rssi_samples) if len(rssi_samples) > 1 else 0.0
    service_rate_var = clamp(rssi_var / 20.0 + busy_ratio * 0.5 + ble_occupancy_ratio * 0.3, 0.0, 1.0)
    target_video = float(config["global"]["target_video_mbps"])
    delay_norm = clamp(delay_p95 / 20.0, 0.0, 1.0)
    video_inv = clamp(1.0 - video_throughput / target_video, 0.0, 1.0)
    score = 0.35 * delay_norm + 0.35 * violation_ratio + 0.15 * telemetry_loss + 0.15 * video_inv
    return {
        "scenario_id": scenario.scenario_id,
        "run_type": run_type,
        "seed": scenario.seed,
        "window_id": window_id,
        "t_start": round(t_start, 6),
        "t_end": round(t_end, 6),
        "mission_type": scenario.mission_type,
        "scenario_family": scenario.family,
        "c2_delay_mean_ms": round(delay_mean, 6),
        "c2_delay_p95_ms": round(delay_p95, 6),
        "c2_violation_ratio": round(violation_ratio, 6),
        "telemetry_loss_ratio": round(telemetry_loss, 6),
        "video_throughput_mbps": round(video_throughput, 6),
        "jitter_ms": round(jitter, 6),
        "channel_busy_ratio": round(busy_ratio, 6),
        "retry_rate": round(retry_rate, 6),
        "backoff_proxy_ms": round(backoff_proxy, 6),
        "ble_occupancy_ratio": round(ble_occupancy_ratio, 6),
        "ble_periodicity_score": round(1.0 if overlapping_ble else 0.0, 6),
        "short_interruption_count": len(overlapping_ble),
        "service_interruption_count": len(overlapping_ble),
        "rssi_mean_dbm": round(rssi_mean, 6),
        "rssi_var": round(rssi_var, 6),
        "distance_m": scenario.distance_m,
        "speed_mps": scenario.speed_mps,
        "service_rate_var": round(service_rate_var, 6),
        "c2_arrival_count": len(in_window),
        "telemetry_arrival_count": telemetry_count,
        "video_offered_load_mbps": round(offered_video, 6),
        "video_burst_ratio": round(float(video["video_burst_ratio"]), 6),
        "queue_len_mean": round(statistics.mean(qlens), 6) if qlens else 0.0,
        "queue_len_p95": round(pct(qlens, 0.95), 6) if qlens else 0.0,
        "service_interval_mean_ms": round(statistics.mean(intervals), 6) if intervals else 0.0,
        "S": round(score, 6),
    }


def run_trace(
    config: dict[str, Any],
    scenario: Scenario,
    run_type: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    duration_ms = float(config["global"]["duration_s"]) * 1000.0
    warmup_ms = float(config["global"]["warmup_s"]) * 1000.0
    window_ms = float(config["global"]["window_ms"])
    arrivals = generate_c2_arrivals(config, scenario, duration_ms)
    busy_intervals, wifi_meta = generate_busy_intervals(config, scenario, run_type, duration_ms)
    ble_intervals, ble_meta = generate_ble_intervals(config, scenario, run_type, duration_ms)
    video_windows = generate_video_windows(config, scenario, run_type, duration_ms, window_ms)
    packets = simulate_packets(config, scenario, run_type, arrivals, busy_intervals, ble_intervals, video_windows, wifi_meta, window_ms)
    windows: list[dict[str, Any]] = []
    first_window = int(warmup_ms // window_ms)
    total_windows = int(duration_ms // window_ms)
    for absolute_window_id in range(first_window, total_windows):
        t_start = absolute_window_id * window_ms
        t_end = t_start + window_ms
        window_id = absolute_window_id - first_window
        windows.append(
            summarize_window(
                config=config,
                scenario=scenario,
                run_type=run_type,
                window_id=window_id,
                t_start=t_start,
                t_end=t_end,
                packets=packets,
                busy_intervals=busy_intervals,
                ble_intervals=ble_intervals,
                video_windows=video_windows,
            )
        )
    return packets, windows, {**wifi_meta, **ble_meta}


def label_windows(
    config: dict[str, Any],
    factual: list[dict[str, Any]],
    removals: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    remove_w_by_window = {int(row["window_id"]): row for row in removals[RUN_REMOVE_W]}
    remove_b_by_window = {int(row["window_id"]): row for row in removals[RUN_REMOVE_B]}
    remove_m_by_window = {int(row["window_id"]): row for row in removals[RUN_REMOVE_M]}
    remove_v_by_window = {int(row["window_id"]): row for row in removals[RUN_REMOVE_V]}
    tau_deg = float(config["global"]["degradation_threshold"])
    tau_s = float(config["global"]["significance_threshold"])
    labeled: list[dict[str, Any]] = []
    cf_scores: list[dict[str, Any]] = []
    for row in factual:
        window_id = int(row["window_id"])
        counterpart_w = remove_w_by_window[window_id]
        counterpart_b = remove_b_by_window[window_id]
        counterpart_m = remove_m_by_window[window_id]
        counterpart_v = remove_v_by_window[window_id]
        s_fact = float(row["S"])
        s_remove_w = float(counterpart_w["S"])
        s_remove_b = float(counterpart_b["S"])
        s_remove_m = float(counterpart_m["S"])
        s_remove_v = float(counterpart_v["S"])
        delta_w = s_fact - s_remove_w
        delta_b = s_fact - s_remove_b
        delta_m = s_fact - s_remove_m
        delta_v = s_fact - s_remove_v
        is_degraded = int(s_fact >= tau_deg)
        label_w = int(is_degraded and delta_w >= tau_s)
        label_b = int(is_degraded and delta_b >= tau_s)
        label_m = int(is_degraded and delta_m >= tau_s)
        label_v = int(is_degraded and delta_v >= tau_s)
        label_count = label_w + label_b + label_m + label_v
        is_mixed = int(label_count >= 2)
        is_ambiguous = int(is_degraded and label_count == 0)
        if not is_degraded:
            window_class = "normal"
        elif label_count == 1:
            window_class = "clean_cause"
        elif label_count >= 2:
            window_class = "mixed_cause"
        else:
            window_class = "ambiguous_deg"
        enriched = dict(row)
        enriched.update(
            {
                "S_fact": round(s_fact, 6),
                "S_remove_W": round(s_remove_w, 6),
                "S_remove_B": round(s_remove_b, 6),
                "S_remove_M": round(s_remove_m, 6),
                "S_remove_V": round(s_remove_v, 6),
                "delta_W": round(delta_w, 6),
                "delta_B": round(delta_b, 6),
                "delta_M": round(delta_m, 6),
                "delta_V": round(delta_v, 6),
                "label_W": label_w,
                "label_B": label_b,
                "label_M": label_m,
                "label_V": label_v,
                "is_degraded": is_degraded,
                "is_mixed": is_mixed,
                "is_ambiguous": is_ambiguous,
                "window_class": window_class,
            }
        )
        labeled.append(enriched)
        cf_scores.append(
            {
                "scenario_id": row["scenario_id"],
                "seed": row["seed"],
                "window_id": window_id,
                "t_start": row["t_start"],
                "t_end": row["t_end"],
                "S_fact": round(s_fact, 6),
                "S_remove_W": round(s_remove_w, 6),
                "S_remove_B": round(s_remove_b, 6),
                "S_remove_M": round(s_remove_m, 6),
                "S_remove_V": round(s_remove_v, 6),
                "delta_W": round(delta_w, 6),
                "delta_B": round(delta_b, 6),
                "delta_M": round(delta_m, 6),
                "delta_V": round(delta_v, 6),
                "label_W": label_w,
                "label_B": label_b,
                "label_M": label_m,
                "label_V": label_v,
                "window_class": window_class,
            }
        )
    return labeled, cf_scores


def ensure_dirs(out_root: Path) -> dict[str, Path]:
    dirs = {
        "configs": out_root / "configs",
        "packet_logs": out_root / "packet_logs",
        "windows": out_root / "windows",
        "splits": out_root / "splits",
        "diagnostics": out_root / "diagnostics",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def split_name(seed: int) -> str:
    if 1000 <= seed < 2000:
        return "train"
    if 2000 <= seed < 3000:
        return "val"
    if 3000 <= seed < 4000:
        return "test_id"
    return "test_ood"


def diagnostics(labeled_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    total = len(labeled_rows)
    degraded = [r for r in labeled_rows if int(r["is_degraded"]) == 1]
    label_w = [r for r in labeled_rows if int(r["label_W"]) == 1]
    label_b = [r for r in labeled_rows if int(r["label_B"]) == 1]
    label_m = [r for r in labeled_rows if int(r["label_M"]) == 1]
    label_v = [r for r in labeled_rows if int(r["label_V"]) == 1]
    ambiguous = [r for r in labeled_rows if int(r["is_ambiguous"]) == 1]
    label_balance = [
        {
            "total_windows": total,
            "degraded_windows": len(degraded),
            "degraded_ratio": round(len(degraded) / total, 6) if total else 0.0,
            "label_W_positive": len(label_w),
            "label_W_positive_ratio": round(len(label_w) / total, 6) if total else 0.0,
            "label_B_positive": len(label_b),
            "label_B_positive_ratio": round(len(label_b) / total, 6) if total else 0.0,
            "label_M_positive": len(label_m),
            "label_M_positive_ratio": round(len(label_m) / total, 6) if total else 0.0,
            "label_V_positive": len(label_v),
            "label_V_positive_ratio": round(len(label_v) / total, 6) if total else 0.0,
            "ambiguous_deg": len(ambiguous),
            "ambiguous_deg_ratio_of_degraded": round(len(ambiguous) / len(degraded), 6) if degraded else 0.0,
        }
    ]
    score_delta_rows = []
    for family in sorted({str(r["scenario_family"]) for r in labeled_rows}):
        rows = [r for r in labeled_rows if str(r["scenario_family"]) == family]
        deltas_w = [float(r["delta_W"]) for r in rows]
        deltas_b = [float(r["delta_B"]) for r in rows]
        deltas_m = [float(r["delta_M"]) for r in rows]
        deltas_v = [float(r["delta_V"]) for r in rows]
        scores = [float(r["S_fact"]) for r in rows]
        score_delta_rows.append(
            {
                "scenario_family": family,
                "windows": len(rows),
                "S_fact_median": round(statistics.median(scores), 6) if scores else 0.0,
                "S_fact_p90": round(pct(scores, 0.90), 6) if scores else 0.0,
                "delta_W_mean": round(statistics.mean(deltas_w), 6) if deltas_w else 0.0,
                "delta_W_p90": round(pct(deltas_w, 0.90), 6) if deltas_w else 0.0,
                "label_W_ratio": round(sum(int(r["label_W"]) for r in rows) / len(rows), 6) if rows else 0.0,
                "delta_B_mean": round(statistics.mean(deltas_b), 6) if deltas_b else 0.0,
                "delta_B_p90": round(pct(deltas_b, 0.90), 6) if deltas_b else 0.0,
                "label_B_ratio": round(sum(int(r["label_B"]) for r in rows) / len(rows), 6) if rows else 0.0,
                "delta_M_mean": round(statistics.mean(deltas_m), 6) if deltas_m else 0.0,
                "delta_M_p90": round(pct(deltas_m, 0.90), 6) if deltas_m else 0.0,
                "label_M_ratio": round(sum(int(r["label_M"]) for r in rows) / len(rows), 6) if rows else 0.0,
                "delta_V_mean": round(statistics.mean(deltas_v), 6) if deltas_v else 0.0,
                "delta_V_p90": round(pct(deltas_v, 0.90), 6) if deltas_v else 0.0,
                "label_V_ratio": round(sum(int(r["label_V"]) for r in rows) / len(rows), 6) if rows else 0.0,
            }
        )
    feature_rows = []
    for cause, label_col in [("W", "label_W"), ("B", "label_B"), ("M", "label_M"), ("V", "label_V")]:
        for label in [0, 1]:
            rows = [r for r in labeled_rows if int(r[label_col]) == label]
            feature_rows.append(
                {
                    "cause": cause,
                    "label": label,
                    "windows": len(rows),
                    "channel_busy_ratio_mean": round(statistics.mean(float(r["channel_busy_ratio"]) for r in rows), 6) if rows else 0.0,
                    "retry_rate_mean": round(statistics.mean(float(r["retry_rate"]) for r in rows), 6) if rows else 0.0,
                    "backoff_proxy_ms_mean": round(statistics.mean(float(r["backoff_proxy_ms"]) for r in rows), 6) if rows else 0.0,
                    "ble_occupancy_ratio_mean": round(statistics.mean(float(r["ble_occupancy_ratio"]) for r in rows), 6) if rows else 0.0,
                    "short_interruption_count_mean": round(statistics.mean(float(r["short_interruption_count"]) for r in rows), 6) if rows else 0.0,
                    "rssi_var_mean": round(statistics.mean(float(r["rssi_var"]) for r in rows), 6) if rows else 0.0,
                    "service_rate_var_mean": round(statistics.mean(float(r["service_rate_var"]) for r in rows), 6) if rows else 0.0,
                    "rssi_mean_dbm_mean": round(statistics.mean(float(r["rssi_mean_dbm"]) for r in rows), 6) if rows else 0.0,
                    "video_offered_load_mbps_mean": round(statistics.mean(float(r["video_offered_load_mbps"]) for r in rows), 6) if rows else 0.0,
                    "video_burst_ratio_mean": round(statistics.mean(float(r["video_burst_ratio"]) for r in rows), 6) if rows else 0.0,
                    "queue_len_p95_mean": round(statistics.mean(float(r["queue_len_p95"]) for r in rows), 6) if rows else 0.0,
                    "c2_delay_p95_ms_mean": round(statistics.mean(float(r["c2_delay_p95_ms"]) for r in rows), 6) if rows else 0.0,
                }
            )
    return {
        "label_balance": label_balance,
        "score_delta_summary": score_delta_rows,
        "feature_signature_summary": feature_rows,
    }


def run(config: dict[str, Any], out_root: Path, families: set[str]) -> None:
    dirs = ensure_dirs(out_root)
    scenarios = expand_scenarios(config, families)
    if not scenarios:
        raise SystemExit(f"No scenarios selected for families: {sorted(families)}")
    all_factual_windows: list[dict[str, Any]] = []
    all_cf_scores: list[dict[str, Any]] = []
    all_labeled: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_config = {
            "scenario_id": scenario.scenario_id,
            "scenario_template_id": scenario.template_id,
            "seed": scenario.seed,
            "seed_streams": seed_bundle(scenario.seed),
            "scenario_family": scenario.family,
            "active_mechanisms": list(scenario.active_mechanisms),
            "wifi_regime": scenario.wifi_regime,
            "ble_regime": scenario.ble_regime,
            "duration_s": config["global"]["duration_s"],
            "warmup_s": config["global"]["warmup_s"],
        }
        write_json(dirs["configs"] / f"scenario_{scenario.scenario_id}.json", scenario_config)
        factual_packets, factual_windows, factual_meta = run_trace(config, scenario, RUN_FACTUAL)
        remove_w_packets, remove_w_windows, remove_w_meta = run_trace(config, scenario, RUN_REMOVE_W)
        remove_b_packets, remove_b_windows, remove_b_meta = run_trace(config, scenario, RUN_REMOVE_B)
        remove_m_packets, remove_m_windows, remove_m_meta = run_trace(config, scenario, RUN_REMOVE_M)
        remove_v_packets, remove_v_windows, remove_v_meta = run_trace(config, scenario, RUN_REMOVE_V)
        write_csv(dirs["packet_logs"] / f"factual_{scenario.scenario_id}.csv", factual_packets)
        write_csv(dirs["packet_logs"] / f"remove_W_{scenario.scenario_id}.csv", remove_w_packets)
        write_csv(dirs["packet_logs"] / f"remove_B_{scenario.scenario_id}.csv", remove_b_packets)
        write_csv(dirs["packet_logs"] / f"remove_M_{scenario.scenario_id}.csv", remove_m_packets)
        write_csv(dirs["packet_logs"] / f"remove_V_{scenario.scenario_id}.csv", remove_v_packets)
        labeled, cf_scores = label_windows(
            config,
            factual_windows,
            {
                RUN_REMOVE_W: remove_w_windows,
                RUN_REMOVE_B: remove_b_windows,
                RUN_REMOVE_M: remove_m_windows,
                RUN_REMOVE_V: remove_v_windows,
            },
        )
        for row in factual_windows:
            row["wifi_busy_ratio_actual"] = round(float(factual_meta["wifi_busy_ratio_actual"]), 6)
            row["ble_occupancy_ratio_actual"] = round(float(factual_meta["ble_occupancy_ratio_actual"]), 6)
        for row in labeled:
            row["split"] = split_name(scenario.seed)
        all_factual_windows.extend(factual_windows)
        all_cf_scores.extend(cf_scores)
        all_labeled.extend(labeled)
        split_rows.extend(
            {
                "scenario_id": scenario.scenario_id,
                "seed": scenario.seed,
                "split": split_name(scenario.seed),
                "window_id": row["window_id"],
            }
            for row in labeled
        )
        print(
            f"{scenario.scenario_id}: factual busy={factual_meta['wifi_busy_ratio_actual']:.3f}, "
            f"ble={factual_meta['ble_occupancy_ratio_actual']:.3f}, "
            f"remove_W busy={remove_w_meta['wifi_busy_ratio_actual']:.3f}, "
            f"remove_B ble={remove_b_meta['ble_occupancy_ratio_actual']:.3f}, "
            f"remove_M ready, remove_V ready, windows={len(labeled)}"
        )
    write_csv(dirs["windows"] / "factual_windows.csv", all_factual_windows)
    write_csv(dirs["windows"] / "counterfactual_scores.csv", all_cf_scores)
    write_csv(dirs["windows"] / "labeled_windows.csv", all_labeled)
    for split in ["train", "val", "test_id", "test_ood"]:
        rows = [r for r in split_rows if r["split"] == split]
        write_csv(dirs["splits"] / f"{split}_ids.csv", rows)
    diag = diagnostics(all_labeled)
    for name, rows in diag.items():
        write_csv(dirs["diagnostics"] / f"{name}.csv", rows)
    print(f"Wrote outputs under {out_root}")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_config = script_dir.parent / "configs" / "mvp_scenarios_min.json"
    default_out = script_dir.parent.parent / "data"
    parser = argparse.ArgumentParser(description="Generate Paper3 MVP DES traces.")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--out-root", type=Path, default=default_out)
    parser.add_argument(
        "--families",
        default="normal,wifi_only,ble_only,mobility_only,video_only,wifi_video,wifi_mobility,all_mixed",
    )
    parser.add_argument("--duration-s", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.duration_s is not None:
        config["global"]["duration_s"] = args.duration_s
    families = {item.strip() for item in args.families.split(",") if item.strip()}
    run(config=config, out_root=args.out_root, families=families)


if __name__ == "__main__":
    main()
