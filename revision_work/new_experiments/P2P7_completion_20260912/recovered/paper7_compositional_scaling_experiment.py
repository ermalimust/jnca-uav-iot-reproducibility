"""Compositional action-library scaling probe for Paper 7.

The main evaluation uses six supported intervention families.  This probe tests
whether the candidate-generation claim survives when each family is expanded
into parameterized, still-supported variants.  It is not a live actuation test.

The key control is a shared deterministic variant-expansion rule.  The LLM-based
method and the non-LLM posterior-family control use the same rule; they differ
only in which action families are exposed before expansion.  This keeps the
evidence focused on candidate-family selection rather than on a hidden advantage
inside the deterministic expansion mechanism.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


BASE = Path(__file__).resolve().parent
LLM_DIR = BASE / "results" / "llm_necessity"
MISSION_FILE = LLM_DIR / "challenge_mission_intents.jsonl"
POLICY_FILE = LLM_DIR / "challenge_qwen_plus_policies.jsonl"
RESULT_DIR = BASE / "results" / "compositional_scaling"
SCALING_SIZES = [6, 18, 36, 60]
NEAR_ORACLE_EPSILON = 0.15
SEEDS = list(range(20))

ARCHETYPES = [
    "low_confidence",
    "wifi_dominant",
    "ble_rid_dominant",
    "mobility_dominant",
    "video_dominant",
    "mixed_high_risk",
]

CAUSES = ["W", "B", "M", "V"]


@dataclass(frozen=True)
class Variant:
    name: str
    family: str
    targets: frozenset[str]
    tags: frozenset[str]
    overhead: float
    video_loss: float
    safety_gain: float
    rid_gain: float
    intensity: float


def variant(
    name: str,
    family: str,
    targets: list[str],
    tags: list[str],
    overhead: float,
    video_loss: float,
    safety_gain: float = 0.0,
    rid_gain: float = 0.0,
    intensity: float = 0.5,
) -> Variant:
    return Variant(
        name=name,
        family=family,
        targets=frozenset(targets),
        tags=frozenset(tags),
        overhead=overhead,
        video_loss=video_loss,
        safety_gain=safety_gain,
        rid_gain=rid_gain,
        intensity=intensity,
    )


VARIANTS_BY_FAMILY: dict[str, list[Variant]] = {
    "Observe": [
        variant("Observe.generic", "Observe", [], ["neutral"], 0.01, 0.00, 0.0, 0.0, 0.1),
        variant("Observe.low_confidence", "Observe", [], ["confidence", "audit"], 0.01, 0.00, 0.0, 0.0, 0.1),
        variant("Observe.short_window", "Observe", [], ["short", "audit"], 0.01, 0.00, 0.0, 0.0, 0.1),
        variant("Observe.video_preserving", "Observe", [], ["video", "low_overhead"], 0.01, 0.00, 0.0, 0.0, 0.1),
        variant("Observe.energy_saving", "Observe", [], ["energy", "low_overhead"], 0.00, 0.00, 0.0, 0.0, 0.1),
        variant("Observe.rid_monitor", "Observe", [], ["rid", "audit"], 0.01, 0.00, 0.0, 0.1, 0.1),
        variant("Observe.local_only", "Observe", [], ["local", "privacy"], 0.01, 0.00, 0.0, 0.0, 0.1),
        variant("Observe.audit_only", "Observe", [], ["audit", "policy"], 0.01, 0.00, 0.0, 0.0, 0.1),
        variant("Observe.until_confident", "Observe", [], ["confidence", "short"], 0.01, 0.00, 0.0, 0.0, 0.1),
        variant("Observe.local_audit_hold", "Observe", [], ["local", "audit", "privacy"], 0.01, 0.00, 0.0, 0.0, 0.1),
    ],
    "WiFiRelief": [
        variant("WiFiRelief.generic", "WiFiRelief", ["W"], ["wifi"], 0.10, 0.05, 0.1, 0.0, 0.5),
        variant("WiFiRelief.local_medium", "WiFiRelief", ["W"], ["wifi", "local"], 0.08, 0.04, 0.1, 0.0, 0.5),
        variant("WiFiRelief.fleet_medium", "WiFiRelief", ["W"], ["wifi", "fleet"], 0.14, 0.05, 0.1, 0.0, 0.6),
        variant("WiFiRelief.video_coexist", "WiFiRelief", ["W", "V"], ["wifi", "video"], 0.12, 0.03, 0.1, 0.0, 0.6),
        variant("WiFiRelief.c2_priority", "WiFiRelief", ["W"], ["wifi", "c2", "safety"], 0.13, 0.07, 0.3, 0.0, 0.7),
        variant("WiFiRelief.energy_light", "WiFiRelief", ["W"], ["wifi", "energy", "low_overhead"], 0.04, 0.04, 0.0, 0.0, 0.4),
        variant("WiFiRelief.high_contention", "WiFiRelief", ["W"], ["wifi", "high"], 0.18, 0.09, 0.1, 0.0, 0.9),
        variant("WiFiRelief.rid_coexist", "WiFiRelief", ["W", "B"], ["wifi", "rid"], 0.12, 0.05, 0.1, 0.2, 0.6),
        variant("WiFiRelief.short_window", "WiFiRelief", ["W"], ["wifi", "short"], 0.08, 0.04, 0.1, 0.0, 0.5),
        variant("WiFiRelief.mixed_relay", "WiFiRelief", ["W", "M"], ["wifi", "mobility", "mixed"], 0.15, 0.06, 0.2, 0.0, 0.7),
    ],
    "BLEAvoid": [
        variant("BLEAvoid.generic", "BLEAvoid", ["B"], ["rid"], 0.08, 0.03, 0.0, 0.4, 0.5),
        variant("BLEAvoid.rid_window_shift", "BLEAvoid", ["B"], ["rid", "short"], 0.07, 0.03, 0.0, 0.5, 0.5),
        variant("BLEAvoid.c2_timing", "BLEAvoid", ["B"], ["rid", "c2", "safety"], 0.09, 0.04, 0.2, 0.4, 0.6),
        variant("BLEAvoid.low_overhead", "BLEAvoid", ["B"], ["rid", "energy", "low_overhead"], 0.04, 0.02, 0.0, 0.3, 0.4),
        variant("BLEAvoid.compliance_strict", "BLEAvoid", ["B"], ["rid", "policy"], 0.10, 0.04, 0.0, 0.7, 0.7),
        variant("BLEAvoid.video_balance", "BLEAvoid", ["B", "V"], ["rid", "video"], 0.09, 0.02, 0.0, 0.5, 0.6),
        variant("BLEAvoid.mixed_guard", "BLEAvoid", ["B", "W"], ["rid", "mixed"], 0.12, 0.04, 0.1, 0.5, 0.7),
        variant("BLEAvoid.local_guard", "BLEAvoid", ["B"], ["rid", "local", "privacy"], 0.07, 0.03, 0.0, 0.5, 0.5),
        variant("BLEAvoid.timed_avoidance", "BLEAvoid", ["B"], ["rid", "short", "confidence"], 0.07, 0.03, 0.0, 0.5, 0.5),
        variant("BLEAvoid.fleet_stagger", "BLEAvoid", ["B"], ["rid", "fleet"], 0.13, 0.04, 0.0, 0.5, 0.6),
    ],
    "LinkAdapt": [
        variant("LinkAdapt.generic", "LinkAdapt", ["M"], ["mobility"], 0.12, 0.04, 0.1, 0.0, 0.5),
        variant("LinkAdapt.mobility_short", "LinkAdapt", ["M"], ["mobility", "short"], 0.11, 0.04, 0.1, 0.0, 0.5),
        variant("LinkAdapt.path_relay", "LinkAdapt", ["M"], ["mobility", "relay"], 0.16, 0.05, 0.2, 0.0, 0.7),
        variant("LinkAdapt.energy_light", "LinkAdapt", ["M"], ["mobility", "energy", "low_overhead"], 0.06, 0.03, 0.0, 0.0, 0.4),
        variant("LinkAdapt.c2_protect", "LinkAdapt", ["M"], ["mobility", "c2", "safety"], 0.15, 0.06, 0.3, 0.0, 0.7),
        variant("LinkAdapt.video_smooth", "LinkAdapt", ["M", "V"], ["mobility", "video"], 0.14, 0.03, 0.1, 0.0, 0.6),
        variant("LinkAdapt.high_mobility", "LinkAdapt", ["M"], ["mobility", "high"], 0.19, 0.07, 0.2, 0.0, 0.9),
        variant("LinkAdapt.local_relay", "LinkAdapt", ["M"], ["mobility", "local", "relay"], 0.13, 0.04, 0.1, 0.0, 0.6),
        variant("LinkAdapt.handover_guard", "LinkAdapt", ["M"], ["mobility", "policy"], 0.14, 0.05, 0.2, 0.0, 0.6),
        variant("LinkAdapt.mixed_path", "LinkAdapt", ["M", "W"], ["mobility", "wifi", "mixed"], 0.17, 0.06, 0.2, 0.0, 0.7),
    ],
    "VideoShape": [
        variant("VideoShape.generic", "VideoShape", ["V"], ["video"], 0.06, 0.10, 0.0, 0.0, 0.5),
        variant("VideoShape.mild_pacing", "VideoShape", ["V"], ["video", "low_overhead"], 0.04, 0.05, 0.0, 0.0, 0.4),
        variant("VideoShape.strong_shape", "VideoShape", ["V"], ["video", "high"], 0.08, 0.18, 0.0, 0.0, 0.8),
        variant("VideoShape.c2_video_trade", "VideoShape", ["V", "W"], ["video", "c2", "safety"], 0.07, 0.12, 0.2, 0.0, 0.6),
        variant("VideoShape.rid_balance", "VideoShape", ["V", "B"], ["video", "rid"], 0.07, 0.08, 0.0, 0.3, 0.6),
        variant("VideoShape.energy_low", "VideoShape", ["V"], ["video", "energy", "low_overhead"], 0.03, 0.08, 0.0, 0.0, 0.4),
        variant("VideoShape.wifi_video_shape", "VideoShape", ["V", "W"], ["video", "wifi"], 0.07, 0.08, 0.0, 0.0, 0.6),
        variant("VideoShape.burst_payload", "VideoShape", ["V"], ["video", "short"], 0.06, 0.09, 0.0, 0.0, 0.6),
        variant("VideoShape.keep_awareness", "VideoShape", ["V"], ["video", "c2"], 0.05, 0.06, 0.1, 0.0, 0.5),
        variant("VideoShape.local_shape", "VideoShape", ["V"], ["video", "local", "privacy"], 0.05, 0.07, 0.0, 0.0, 0.5),
    ],
    "FallbackProtect": [
        variant("FallbackProtect.generic", "FallbackProtect", ["W", "B", "M"], ["fallback", "mixed"], 0.22, 0.25, 0.5, 0.3, 0.8),
        variant("FallbackProtect.c2", "FallbackProtect", ["W", "B", "M"], ["fallback", "c2", "safety"], 0.24, 0.28, 0.8, 0.3, 0.9),
        variant("FallbackProtect.rid_c2", "FallbackProtect", ["W", "B", "M"], ["fallback", "rid", "c2", "safety"], 0.25, 0.28, 0.8, 0.6, 0.9),
        variant("FallbackProtect.mixed_high_risk", "FallbackProtect", ["W", "B", "M"], ["fallback", "mixed", "high"], 0.26, 0.30, 0.8, 0.5, 0.9),
        variant("FallbackProtect.energy_aware", "FallbackProtect", ["W", "B", "M"], ["fallback", "energy"], 0.16, 0.24, 0.5, 0.3, 0.7),
        variant("FallbackProtect.video_preserving", "FallbackProtect", ["W", "B", "M", "V"], ["fallback", "video"], 0.23, 0.14, 0.5, 0.3, 0.8),
        variant("FallbackProtect.c2_mobility", "FallbackProtect", ["W", "B", "M"], ["fallback", "c2", "mobility", "safety"], 0.25, 0.27, 0.8, 0.4, 0.9),
        variant("FallbackProtect.local", "FallbackProtect", ["W", "B", "M"], ["fallback", "local", "privacy"], 0.20, 0.22, 0.5, 0.3, 0.7),
        variant("FallbackProtect.audit_escalation", "FallbackProtect", ["W", "B", "M"], ["fallback", "audit", "policy"], 0.22, 0.24, 0.5, 0.4, 0.8),
        variant("FallbackProtect.certified_confidence", "FallbackProtect", ["W", "B", "M"], ["fallback", "confidence", "safety"], 0.21, 0.25, 0.7, 0.3, 0.8),
    ],
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_policies(path: Path) -> dict[str, dict[str, list[str]]]:
    policies: dict[str, dict[str, list[str]]] = {}
    for row in load_jsonl(path):
        policy = row.get("policy", {})
        mission_id = str(row.get("mission_id") or policy.get("mission_id"))
        actions = policy.get("archetype_actions", {})
        policies[mission_id] = {key: list(value) for key, value in actions.items()}
    return policies


def library_for_scale(size: int) -> list[Variant]:
    if size % len(VARIANTS_BY_FAMILY) != 0:
        raise ValueError("scale must be divisible by the number of action families")
    per_family = size // len(VARIANTS_BY_FAMILY)
    variants: list[Variant] = []
    for family in VARIANTS_BY_FAMILY:
        variants.extend(VARIANTS_BY_FAMILY[family][:per_family])
    return variants


def posterior(archetype: str, seed: int) -> dict[str, float]:
    base = {
        "low_confidence": {"W": 0.25, "B": 0.25, "M": 0.25, "V": 0.25},
        "wifi_dominant": {"W": 0.70, "B": 0.08, "M": 0.10, "V": 0.12},
        "ble_rid_dominant": {"W": 0.08, "B": 0.72, "M": 0.10, "V": 0.10},
        "mobility_dominant": {"W": 0.10, "B": 0.08, "M": 0.72, "V": 0.10},
        "video_dominant": {"W": 0.10, "B": 0.08, "M": 0.10, "V": 0.72},
        "mixed_high_risk": {"W": 0.31, "B": 0.27, "M": 0.29, "V": 0.13},
    }[archetype].copy()
    drift = (seed - 2) * 0.01
    if archetype != "low_confidence":
        dom = max(base, key=base.get)
        base[dom] = min(0.78, base[dom] + drift)
        others = [cause for cause in CAUSES if cause != dom]
        for cause in others:
            base[cause] = max(0.04, base[cause] - drift / len(others))
    total = sum(base.values())
    return {cause: value / total for cause, value in base.items()}


def mission_tags(mission: dict[str, Any]) -> set[str]:
    guards = mission.get("guards", {})
    intent = str(mission.get("intent", "")).lower()
    tags: set[str] = set()
    if guards.get("safety") or "c2" in intent or "command" in intent:
        tags.update(["c2", "safety"])
    if guards.get("rid") or "rid" in intent or "remote-id" in intent or "regulated" in intent:
        tags.add("rid")
    if guards.get("video") or "video" in intent or "stream" in intent or "inspection" in intent:
        tags.add("video")
    if guards.get("energy") or "battery" in intent or "endurance" in intent:
        tags.update(["energy", "low_overhead"])
    if "privacy" in intent or "local" in intent or "approved" in intent:
        tags.update(["local", "privacy"])
    if "mobility" in intent or "route" in intent or "blockage" in intent:
        tags.add("mobility")
    if "wi-fi" in intent or "wifi" in intent or "contention" in intent:
        tags.add("wifi")
    if "mixed" in intent or "combined" in intent:
        tags.add("mixed")
    return tags


def structured_context_tags(mission: dict[str, Any]) -> set[str]:
    """Tags available to deterministic expansion without reading free text."""
    guards = mission.get("guards", {})
    weights = mission.get("cost_weights", {})
    tags: set[str] = set()
    if guards.get("safety") or float(weights.get("safety", 0.0)) >= 0.50:
        tags.update(["c2", "safety"])
    if guards.get("rid") or float(weights.get("mismatch", 0.0)) >= 0.22:
        tags.add("rid")
    if guards.get("video") or float(weights.get("throughput", 0.0)) >= 0.35:
        tags.add("video")
    if guards.get("energy") or float(weights.get("overhead", 0.0)) >= 0.25:
        tags.update(["energy", "low_overhead"])
    return tags


def family_targets_for_archetype(archetype: str) -> list[str]:
    return {
        "low_confidence": ["Observe", "FallbackProtect"],
        "wifi_dominant": ["WiFiRelief", "VideoShape", "FallbackProtect"],
        "ble_rid_dominant": ["BLEAvoid", "FallbackProtect", "Observe"],
        "mobility_dominant": ["LinkAdapt", "FallbackProtect", "WiFiRelief"],
        "video_dominant": ["VideoShape", "WiFiRelief", "FallbackProtect"],
        "mixed_high_risk": ["FallbackProtect", "WiFiRelief", "BLEAvoid", "LinkAdapt", "VideoShape"],
    }[archetype]


def verifier_accepts(variant: Variant, mission: dict[str, Any], archetype: str) -> bool:
    guards = mission.get("guards", {})
    if variant.family == "Observe" and guards.get("safety") and archetype in {"wifi_dominant", "mobility_dominant", "mixed_high_risk"}:
        return False
    if guards.get("rid") and archetype in {"ble_rid_dominant", "mixed_high_risk"}:
        if variant.rid_gain < 0.25 and "rid" not in variant.tags and variant.family != "FallbackProtect":
            return False
    if guards.get("energy") and variant.overhead > 0.22 and archetype not in {"mobility_dominant", "mixed_high_risk"}:
        return False
    if "privacy" in mission_tags(mission) and "fleet" in variant.tags:
        return False
    return True


def action_loss(variant: Variant, mission: dict[str, Any], archetype: str, seed: int) -> float:
    q = posterior(archetype, seed)
    tags = mission_tags(mission)
    guards = mission.get("guards", {})
    weights = mission.get("cost_weights", {})

    residual = sum(prob * (0.25 if cause in variant.targets else 1.0) for cause, prob in q.items())
    if variant.family == "Observe" and archetype == "low_confidence":
        residual *= 0.65
    if variant.family == "FallbackProtect" and archetype == "video_dominant" and not guards.get("safety"):
        residual *= 1.20

    safety_term = 0.0
    if guards.get("safety"):
        mixed_pressure = q["W"] + q["B"] + q["M"]
        safety_term = max(0.0, mixed_pressure * (1.0 - variant.safety_gain))
        if "c2" in variant.tags:
            safety_term *= 0.75

    rid_term = 0.0
    if guards.get("rid"):
        rid_term = max(0.0, q["B"] * (1.0 - variant.rid_gain))

    video_term = variant.video_loss
    if guards.get("video") and "video" in variant.tags:
        video_term *= 0.65
    if guards.get("video") and variant.family == "FallbackProtect" and "video" not in variant.tags:
        video_term *= 1.30

    overhead_term = variant.overhead
    if guards.get("energy") and "low_overhead" not in variant.tags:
        overhead_term *= 1.65
    if guards.get("energy") and "energy" in variant.tags:
        overhead_term *= 0.65

    semantic_overlap = len(tags & variant.tags)
    semantic_bonus = min(0.18, 0.045 * semantic_overlap)
    if archetype == "mixed_high_risk" and "fallback" in variant.tags:
        semantic_bonus += 0.08
    if archetype == "video_dominant" and "video" in variant.tags:
        semantic_bonus += 0.05
    if archetype == "ble_rid_dominant" and "rid" in variant.tags:
        semantic_bonus += 0.05
    if archetype == "mobility_dominant" and "mobility" in variant.tags:
        semantic_bonus += 0.05
    if archetype == "wifi_dominant" and "wifi" in variant.tags:
        semantic_bonus += 0.05

    loss = (
        1.15 * residual
        + float(weights.get("safety", 0.4)) * safety_term
        + float(weights.get("mismatch", 0.15)) * rid_term
        + float(weights.get("throughput", 0.25)) * video_term
        + float(weights.get("overhead", 0.1)) * overhead_term
        - semantic_bonus
    )
    return max(0.0, loss * 2.2)


def by_name(library: list[Variant]) -> dict[str, Variant]:
    return {variant.name: variant for variant in library}


def variants_for_family(library: list[Variant], family: str) -> list[Variant]:
    return [variant for variant in library if variant.family == family]


def compact_template(library: list[Variant], mission: dict[str, Any], archetype: str) -> list[Variant]:
    out: list[Variant] = []
    for family in family_targets_for_archetype(archetype)[:2]:
        candidates = variants_for_family(library, family)
        if candidates:
            ranked = sorted(
                candidates,
                key=lambda item: (structured_variant_score(item, mission, archetype), -item.overhead),
                reverse=True,
            )
            out.append(ranked[0])
    return unique_variants(out)


def mission_blind_broad(library: list[Variant], mission: dict[str, Any], archetype: str) -> list[Variant]:
    per_family_available = len(library) // len(VARIANTS_BY_FAMILY)
    expand = 1 if per_family_available == 1 else 2 if per_family_available <= 3 else 3 if per_family_available <= 6 else 4
    out: list[Variant] = []
    for family in family_targets_for_archetype(archetype):
        candidates = sorted(
            variants_for_family(library, family),
            key=lambda item: (structured_variant_score(item, mission, archetype), -item.overhead),
            reverse=True,
        )
        out.extend(candidates[:expand])
    return unique_variants(out)


def structured_variant_score(variant: Variant, mission: dict[str, Any], archetype: str) -> float:
    tags = structured_context_tags(mission)
    score = 0.0
    score += 1.0 * len(tags & variant.tags)
    if archetype == "wifi_dominant" and "wifi" in variant.tags:
        score += 1.2
    if archetype == "ble_rid_dominant" and "rid" in variant.tags:
        score += 1.2
    if archetype == "mobility_dominant" and "mobility" in variant.tags:
        score += 1.2
    if archetype == "video_dominant" and "video" in variant.tags:
        score += 1.2
    if archetype == "mixed_high_risk" and "fallback" in variant.tags:
        score += 1.2
    if archetype == "low_confidence" and ("confidence" in variant.tags or variant.family == "Observe"):
        score += 0.8
    if mission.get("guards", {}).get("energy") and variant.overhead > 0.16:
        score -= 0.8
    if "privacy" in tags and "fleet" in variant.tags:
        score -= 1.0
    return score


def unique_families(families: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for family in families:
        if family in VARIANTS_BY_FAMILY and family not in seen:
            out.append(family)
            seen.add(family)
    return out


def expand_selected_families(
    library: list[Variant],
    mission: dict[str, Any],
    archetype: str,
    families: list[str],
) -> list[Variant]:
    """Shared deterministic variant expansion used by both family selectors."""
    out: list[Variant] = []
    per_family_available = len(library) // len(VARIANTS_BY_FAMILY)
    for rank, family in enumerate(unique_families(families)):
        candidates = variants_for_family(library, family)
        if not candidates:
            continue
        if per_family_available == 1:
            take = 1
        else:
            take = 2 if rank < 2 else 1
        ranked = sorted(
            candidates,
            key=lambda item: (structured_variant_score(item, mission, archetype), -item.overhead, -item.video_loss),
            reverse=True,
        )
        out.extend(ranked[:take])
    return unique_variants(out)


def llm_families_for(
    mission: dict[str, Any],
    archetype: str,
    policies: dict[str, dict[str, list[str]]],
) -> list[str]:
    mission_id = str(mission["mission_id"])
    families = policies.get(mission_id, {}).get(archetype)
    if not families:
        families = family_targets_for_archetype(archetype)[:3]
    return unique_families(families)


def llm_semantic_expansion(
    library: list[Variant],
    mission: dict[str, Any],
    archetype: str,
    policies: dict[str, dict[str, list[str]]],
) -> list[Variant]:
    return expand_selected_families(library, mission, archetype, llm_families_for(mission, archetype, policies))


def posterior_family_expansion(
    library: list[Variant],
    mission: dict[str, Any],
    archetype: str,
    policies: dict[str, dict[str, list[str]]],
) -> list[Variant]:
    family_budget = len(llm_families_for(mission, archetype, policies))
    families = family_targets_for_archetype(archetype)[:family_budget]
    return expand_selected_families(library, mission, archetype, families)


def verified_full_library(library: list[Variant], _mission: dict[str, Any], _archetype: str) -> list[Variant]:
    return list(library)


def unique_variants(items: list[Variant]) -> list[Variant]:
    seen: set[str] = set()
    out: list[Variant] = []
    for item in items:
        if item.name not in seen:
            out.append(item)
            seen.add(item.name)
    return out


def best_verified(
    candidates: list[Variant],
    mission: dict[str, Any],
    archetype: str,
    seed: int,
) -> tuple[Variant | None, float]:
    verified = [item for item in candidates if verifier_accepts(item, mission, archetype)]
    if not verified:
        return None, 4.0
    best = min(verified, key=lambda item: action_loss(item, mission, archetype, seed))
    return best, action_loss(best, mission, archetype, seed)


def se(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return pstdev(values) / math.sqrt(len(values))


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)

    summary: list[dict[str, Any]] = []
    metrics = [
        "candidate_slots",
        "candidate_fraction",
        "oracle_coverage",
        "near_oracle_coverage",
        "best_verified_regret",
        "selected_guarded_regret",
        "selected_guarded_loss",
        "verified_empty",
    ]
    for group_key, group_rows in sorted(grouped.items()):
        item = {key: value for key, value in zip(keys, group_key)}
        for metric in metrics:
            values = [float(row[metric]) for row in group_rows]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_se"] = se(values)
        item["runs"] = len(group_rows)
        summary.append(item)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format4(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_report(by_scale: list[dict[str, Any]]) -> None:
    report = RESULT_DIR / "compositional_scaling_report.md"
    columns = [
        "library_size",
        "method",
        "candidate_slots_mean",
        "candidate_fraction_mean",
        "oracle_coverage_mean",
        "near_oracle_coverage_mean",
        "best_verified_regret_mean",
        "selected_guarded_regret_mean",
    ]
    with report.open("w", encoding="utf-8") as handle:
        handle.write("# Compositional Action-Library Scaling Probe\n\n")
        handle.write(
            "This probe expands the six supported intervention families into "
            "parameterized variants and replays the saved family-level LLM "
            "policy as a semantic variant selector. It is an offline "
            "candidate-exposure experiment, not a live actuation test.\n\n"
        )
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("|" + "|".join(["---"] * len(columns)) + "|\n")
        for row in by_scale:
            handle.write("| " + " | ".join(format4(row[col]) for col in columns) + " |\n")
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "- The compact template keeps using generic family-level actions as the library grows.\n"
        )
        handle.write(
            "- The LLM-semantic generator reuses the saved Qwen family-level policy and expands "
            "selected families with the same deterministic rule used by the posterior-family control.\n"
        )
        handle.write(
            "- The posterior-family control uses no free-text mission intent for family selection; "
            "it uses posterior archetypes and the same expansion rule.\n"
        )
        handle.write(
            "- The full-library reference remains the upper bound, but its candidate fraction is 1.0 by definition.\n"
        )
        handle.write(
            "- The key evidence is not that language beats exhaustive search; it is that semantic "
            "candidate generation preserves coverage with a much smaller candidate fraction as K grows.\n"
        )


def run() -> None:
    missions = load_jsonl(MISSION_FILE)
    policies = load_policies(POLICY_FILE)
    methods = {
        "compact_template": lambda lib, mission, arch: compact_template(lib, mission, arch),
        "posterior_family_shared_expansion": lambda lib, mission, arch: posterior_family_expansion(lib, mission, arch, policies),
        "mission_blind_broad": lambda lib, mission, arch: mission_blind_broad(lib, mission, arch),
        "llm_semantic_expansion": lambda lib, mission, arch: llm_semantic_expansion(lib, mission, arch, policies),
        "verified_full_library": lambda lib, mission, arch: verified_full_library(lib, mission, arch),
    }

    rows: list[dict[str, Any]] = []
    for library_size in SCALING_SIZES:
        library = library_for_scale(library_size)
        for mission in missions:
            for archetype in ARCHETYPES:
                for seed in SEEDS:
                    oracle_variant, oracle_loss = best_verified(library, mission, archetype, seed)
                    if oracle_variant is None:
                        continue
                    for method_name, generator in methods.items():
                        candidates = generator(library, mission, archetype)
                        best_variant, best_loss = best_verified(candidates, mission, archetype, seed)
                        verified_candidates = [
                            item for item in candidates if verifier_accepts(item, mission, archetype)
                        ]
                        candidate_names = {item.name for item in verified_candidates}
                        rows.append(
                            {
                                "library_size": library_size,
                                "mission_id": mission["mission_id"],
                                "archetype": archetype,
                                "seed": seed,
                                "method": method_name,
                                "candidate_slots": len(candidates),
                                "candidate_fraction": len(candidates) / library_size,
                                "verified_slots": len(verified_candidates),
                                "oracle_action": oracle_variant.name,
                                "best_action": best_variant.name if best_variant else "ESCALATE",
                                "oracle_coverage": 1.0 if oracle_variant.name in candidate_names else 0.0,
                                "near_oracle_coverage": 1.0 if best_loss <= oracle_loss + NEAR_ORACLE_EPSILON else 0.0,
                                "best_verified_regret": best_loss - oracle_loss,
                                "selected_guarded_regret": best_loss - oracle_loss,
                                "selected_guarded_loss": best_loss,
                                "oracle_loss": oracle_loss,
                                "verified_empty": 1.0 if not verified_candidates else 0.0,
                            }
                        )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    by_scale = summarize(rows, ["library_size", "method"])
    overall = summarize(rows, ["method"])
    write_csv(RESULT_DIR / "compositional_scaling_raw.csv", rows)
    write_csv(RESULT_DIR / "compositional_scaling_by_scale.csv", by_scale)
    write_csv(RESULT_DIR / "compositional_scaling_overall.csv", overall)
    write_report(by_scale)

    print(f"wrote {len(rows)} rows to {RESULT_DIR}")


if __name__ == "__main__":
    run()
