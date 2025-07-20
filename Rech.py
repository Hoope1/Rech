#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TFT Set 14 Optimizer – Ein-Datei Implementierung (Features A–E integriert)

Funktionen:
-----------
1. Dynamisches Laden & Normalisierung von Champion- und Traitdaten (Data Dragon).
2. Variant / Duplikat Handling (Base-Mask).
3. Parametrierbare Score-Funktion mit leicht anpassbaren Gewichtungen.
4. Vorberechnung von Trait-Level-Werten (LEVEL_LOOKUP / LEVEL_VALUE).
5. Numba-optimierte Merge-&Score Routine (Fallback auf Python).
6. Meet-in-the-Middle (optionale Teil-Enumeration) mit Basic-Dominance-Pruning.
7. Score-Decomposition, Re-Scoring vorhandener JSON/CSV Resultate.
8. Parameter-Sets (P1, P2) und automatisierter Re-Score.
9. Fortschritt via tqdm (immer aktiv, Fallback auf rudimentären Balken falls tqdm fehlt).

Verwendung:
-----------
Einfach ausführen. Optional vorhandene Datei "tft_full_bruteforce_results.json" / ".csv"
im selben Verzeichnis wird automatisch erkannt und neu bewertet.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from itertools import count

import hashlib
import subprocess

import requests

# -----------------------------
# TQDM / NUMBA / OPTIONAL LIBS
# -----------------------------
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _fallback_tqdm(iterable, total=None, desc="", unit="it", **kw):
    """Sehr einfacher Fallback wenn tqdm fehlt."""
    if total is None:
        try:
            total = len(iterable)  # kann fehlschlagen
        except Exception:
            total = None
    start = time.time()
    for i, x in enumerate(iterable, 1):
        if total and i % max(1, total // 50) == 0:
            pct = i * 100.0 / total
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0
            sys.stderr.write(f"\r{desc} {pct:5.1f}% ({i}/{total}) {rate:,.0f} {unit}/s")
            sys.stderr.flush()
        yield x
    sys.stderr.write("\n")


if tqdm is None:

    def prog(iterable, total=None, desc="", unit="it", **kw):
        return _fallback_tqdm(iterable, total=total, desc=desc, unit=unit, **kw)

else:

    def prog(iterable, total=None, desc="", unit="it", **kw):
        return tqdm(iterable, total=total, desc=desc, unit=unit, **kw)


# Numba optional
NUMBA_AVAILABLE = False
try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:
    # Dummy decorator
    def njit(*a, **k):
        def wrap(f):
            return f

        return wrap


# -----------------------------
# GLOBAL CONSTANTS
# -----------------------------
HTTP_TIMEOUT = 8  # seconds for all network requests
SET14_CACHE_HASH_FILE = "tft_set14_cache.json.gz.sha256"
SNAPSHOT_FILE = "tft_set14_snapshot.json"


def compute_sha256(data: Any) -> str:
    """Return SHA256 hex digest for the given JSON-serialisable data."""
    blob = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def get_json_with_retries(
    url: str, retries: int = 3, timeout: int = HTTP_TIMEOUT
) -> Optional[Dict]:
    delay = 0.5
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code in (403, 404):
                return None
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(delay)
            delay *= 2
    return None


def log_diff(old_data: Dict, new_data: Dict) -> None:
    lines = []
    old_map = {c.get("apiName"): c for c in old_data.get("champions", [])}
    new_map = {c.get("apiName"): c for c in new_data.get("champions", [])}
    for cid in sorted(set(old_map) | set(new_map)):
        oc = old_map.get(cid)
        nc = new_map.get(cid)
        if oc and not nc:
            lines.append(f"Removed champion {cid}")
        elif nc and not oc:
            lines.append(f"Added champion {cid}")
        elif oc and nc:
            if oc.get("cost") != nc.get("cost") or sorted(
                oc.get("traits", [])
            ) != sorted(nc.get("traits", [])):
                lines.append(
                    "Changed %s: cost %s -> %s traits %s -> %s"
                    % (
                        cid,
                        oc.get("cost"),
                        nc.get("cost"),
                        oc.get("traits"),
                        nc.get("traits"),
                    )
                )
    if lines:
        with open("tft_set14_diff.log", "a", encoding="utf-8") as f:
            for ln in lines:
                f.write(ln + "\n")


def write_run_meta(param_set: str) -> None:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = "unknown"
    meta = {
        "commit": commit,
        "timestamp": time.time(),
        "param_set": param_set,
    }
    with open("run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# -----------------------------
# CONFIG & PARAMETER SETS
# -----------------------------
# Aktive Parameter-Sets – Du kannst leicht umschalten.
PARAM_SETS = {
    "P1": {
        "LEVEL_WEIGHT": {
            2: 20.0,
            3: 30.0,
            4: 45.0,
            5: 62.5,
            6: 85.0,
            7: 100.0,
            8: 120.0,
        },
        "HIGH_COST_UNIT_BONUS": 7.0,
        "GOLD_UTIL_LINEAR": 0.55,
        "LEFTOVER_PENALTY_FACTOR": 0.50,
        "HIGH_COST_THRESHOLD": 4,
        "TEAM_SIZE": 8,
        "AVG_TARGET_COST": 4.0,
    },
    "P2": {
        "LEVEL_WEIGHT": {
            2: 18.0,
            3: 28.0,
            4: 42.0,
            5: 60.0,
            6: 90.0,
            7: 115.0,
            8: 140.0,
        },
        "HIGH_COST_UNIT_BONUS": 8.5,
        "GOLD_UTIL_LINEAR": 0.60,
        "LEFTOVER_PENALTY_FACTOR": 0.70,
        "HIGH_COST_THRESHOLD": 4,
        "TEAM_SIZE": 8,
        "AVG_TARGET_COST": 4.0,
    },
}

ACTIVE_PARAM_SET = "P1"  # <- Hier umstellen auf "P2" falls aggressiver gewünscht.

# -----------------------------
# DATA LOADING
# -----------------------------

DD_VERSION_CACHE_FILE = "dd_version_cache.json"
SET14_CACHE_FILE = "tft_set14_cache.json.gz"

EMBEDDED_SET14_SNAPSHOT = {
    "snapshot_version": 1,
    "snapshot_date": "2024-01-01",
    "champions": [
        {
            "apiName": "TFT_TestCh1",
            "name": "TestCh1",
            "cost": 1,
            "traits": ["Set14_TraitA"],
        }
    ],
    "traits": [
        {
            "apiName": "Set14_TraitA",
            "name": "Trait A",
            "effects": [{"minUnits": 1}, {"minUnits": 2}],
        }
    ],
}


def get_latest_ddragon_version(timeout=8) -> Optional[str]:
    try:
        r = requests.get(
            "https://ddragon.leagueoflegends.com/api/versions.json", timeout=timeout
        )
        r.raise_for_status()
        versions = r.json()
        if versions:
            with open(DD_VERSION_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump({"version": versions[0], "ts": time.time()}, f)
            return versions[0]
    except Exception:
        if os.path.isfile(DD_VERSION_CACHE_FILE):
            try:
                with open(DD_VERSION_CACHE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("version")
            except Exception:
                pass
    return None


def load_set14_data(
    force_refresh: bool = False, timeout: int = HTTP_TIMEOUT
) -> Tuple[List[Dict], Dict[str, List[int]]]:
    """
    Lädt Champions + Traits (Set 14) aus Data Dragon.
    Rückgabe:
        champions: List[dict] – Felder: id, name, cost, traits (Tuple[str,...])
        trait_breakpoints: Dict[str, List[int]] – Liste der minUnits Breakpoints
    """
    version = get_latest_ddragon_version(timeout=timeout) or "latest"
    urls = [
        f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/tft/set14.json",
        f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/tft-set14.json",
        f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/tft-set14/tft-set14.json",
        f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/tft-set13.json",
    ]
    data = None
    if not force_refresh and os.path.isfile(SET14_CACHE_FILE):
        try:
            with open(SET14_CACHE_HASH_FILE, "r", encoding="utf-8") as hf:
                expected = hf.read().strip()
            with gzip.open(SET14_CACHE_FILE, "rt", encoding="utf-8") as gz:
                cached = json.load(gz)
            if compute_sha256(cached) == expected:
                data = cached
        except Exception:
            data = None
    if data is None:
        for url in urls:
            data = get_json_with_retries(url, timeout=timeout)
            if data:
                log_diff(EMBEDDED_SET14_SNAPSHOT, data)
                with gzip.open(SET14_CACHE_FILE, "wt", encoding="utf-8") as gz:
                    json.dump(data, gz)
                with open(SET14_CACHE_HASH_FILE, "w", encoding="utf-8") as hf:
                    hf.write(compute_sha256(data))
                with open(SNAPSHOT_FILE, "w", encoding="utf-8") as sf:
                    json.dump(data, sf, indent=2)
                break
    if data is None:
        cd_urls = [
            "https://raw.communitydragon.org/latest/cdragon/tft/en_us.json",
            "https://raw.communitydragon.org/pbe/cdragon/tft/en_us.json",
        ]
        for cu in cd_urls:
            cd_json = get_json_with_retries(cu, timeout=timeout)
            if not cd_json:
                continue
            cd = cd_json.get("sets", {}).get("14")
            if not cd:
                logging.warning(
                    "[WARN_DATA] CommunityDragon sets->14 missing for %s", cu
                )
                continue
            data = {
                "champions": cd.get("champions", []),
                "traits": cd.get("traits", []),
            }
            log_diff(EMBEDDED_SET14_SNAPSHOT, data)
            with open(SNAPSHOT_FILE, "w", encoding="utf-8") as sf:
                json.dump(data, sf, indent=2)
            break
    if data is None:
        if not force_refresh and os.path.isfile(SET14_CACHE_FILE):
            try:
                with gzip.open(SET14_CACHE_FILE, "rt", encoding="utf-8") as gz:
                    data = json.load(gz)
            except Exception:
                data = None
        if data is None:
            data = EMBEDDED_SET14_SNAPSHOT
    champs_raw = data.get("champions", [])
    traits_raw = data.get("traits", [])
    champions: List[Dict] = []
    for ch in champs_raw:
        name = ch.get("name") or ch.get("character_id") or ch.get("apiName")
        cid = ch.get("character_id") or ch.get("apiName") or name
        cost = ch.get("cost", 1)
        traits = tuple([t for t in ch.get("traits", []) if t])
        champions.append({"id": cid, "name": name, "cost": int(cost), "traits": traits})
    trait_breakpoints: Dict[str, List[int]] = {}
    for tr in traits_raw:
        tname = tr.get("name") or tr.get("apiName")
        tiers = tr.get("tiers") or tr.get("effects") or []
        bps = set()
        for tier in tiers:
            mu = tier.get("minUnits")
            if isinstance(mu, int) and mu > 0:
                bps.add(mu)
        trait_breakpoints[tname] = sorted(bps)
    return champions, trait_breakpoints


# -----------------------------
# DATA NORMALIZATION & VARIANT HANDLING
# -----------------------------
def normalize_and_tag_variants(
    champions: List[Dict], high_cost_threshold: int
) -> List[Dict]:
    """
    Mark all variants and set helper fields:
    ``base`` name, ``base_index`` and ``high`` flag.
    Variants are detected via '+' suffixes or duplicate IDs.
    """
    # Basisname: Entferne abschließende '+' (mehrfach) und whitespace.
    for c in champions:
        base = re.sub(r"\++$", "", c["name"].strip())
        c["base"] = base
        c["high"] = c["cost"] >= high_cost_threshold

    # Base-Indizes vergeben
    bases = sorted({c["base"] for c in champions})
    base_index = {b: i for i, b in enumerate(bases)}
    for c in champions:
        c["base_index"] = base_index[c["base"]]
    return champions


def validate_champions(champions: List[Dict]) -> List[Dict]:
    """Filter or warn about champions without traits."""
    valid = []
    for ch in champions:
        if not ch["traits"]:
            logging.warning("[WARN_DATA] %s has no traits", ch.get("name"))
            continue
        valid.append(ch)
    return valid


# -----------------------------
# TRAIT INDEXING & LOOKUPS
# -----------------------------
def build_trait_indices(trait_breakpoints: Dict[str, List[int]]) -> Dict[str, int]:
    return {t: i for i, t in enumerate(sorted(trait_breakpoints.keys()))}


def compute_level_tables(
    champions: List[Dict],
    trait_breakpoints: Dict[str, List[int]],
    trait_index: Dict[str, int],
    level_weight_map: Dict[int, float],
    max_cap: int = 16,
):
    """Build lookup tables for trait levels.

    ``LEVEL_LOOKUP[trait][count]`` stores the best breakpoint count.
    ``LEVEL_VALUE[trait][count]`` stores its weighted score according to
    ``level_weight_map``. If no exact weight exists the ordinal level is used.
    """
    import numpy as np

    T = len(trait_index)
    LEVEL_LOOKUP = [[0] * (max_cap + 1) for _ in range(T)]
    LEVEL_VALUE = [[0.0] * (max_cap + 1) for _ in range(T)]

    # Pre-Berechnung: Für jeden Trait, sortiere Breakpoints. Ordinal-Level -> Index in Liste
    for tname, tid in trait_index.items():
        bps = sorted(trait_breakpoints.get(tname, []))
        if not bps:
            continue
        for cnt in range(1, max_cap + 1):
            # Finde höchsten Breakpoint <= cnt
            lvl_bp = 0
            for i, bp in enumerate(bps):
                if cnt >= bp:
                    lvl_bp = bp
                else:
                    break
            LEVEL_LOOKUP[tid][cnt] = lvl_bp
            if lvl_bp:
                # Gewicht: zuerst versuchen exakte Level-Weight nach count, sonst ordinal mapping
                val = level_weight_map.get(lvl_bp)
                if val is None:
                    # Fallback: Versuche mapping [2..8] zu ordinal (MIN(ord,8))
                    key = min(lvl_bp, 8)
                    val = level_weight_map.get(key, 0.0)
                LEVEL_VALUE[tid][cnt] = val
    # numpy Arrays für numba
    LEVEL_LOOKUP_NP = np.array(LEVEL_LOOKUP, dtype="int16")
    LEVEL_VALUE_NP = np.array(LEVEL_VALUE, dtype="float64")
    return LEVEL_LOOKUP, LEVEL_VALUE, LEVEL_LOOKUP_NP, LEVEL_VALUE_NP


# -----------------------------
# SCORE FUNCTION COMPONENTS
# -----------------------------
@dataclass
class ScoreParams:
    LEVEL_WEIGHT: Dict[int, float]
    HIGH_COST_UNIT_BONUS: float
    GOLD_UTIL_LINEAR: float
    LEFTOVER_PENALTY_FACTOR: float
    HIGH_COST_THRESHOLD: int
    TEAM_SIZE: int
    AVG_TARGET_COST: float


def compute_team_components(
    idxs: List[int],
    champions: List[Dict],
    trait_index: Dict[str, int],
    LEVEL_VALUE: List[List[float]],
    params: ScoreParams,
) -> Dict[str, Any]:
    T = len(trait_index)
    trait_counts = [0] * T
    cost = 0
    high_count = 0
    for i in idxs:
        c = champions[i]
        cost += c["cost"]
        if c["high"]:
            high_count += 1
        for tr in c["traits"]:
            trait_counts[trait_index[tr]] += 1
    # Trait Value
    trait_val = 0.0
    for t in range(T):
        cnt = trait_counts[t]
        if cnt >= len(LEVEL_VALUE[t]):
            cnt = len(LEVEL_VALUE[t]) - 1
        trait_val += LEVEL_VALUE[t][cnt]
    high_val = high_count * params.HIGH_COST_UNIT_BONUS
    gold_util_val = cost * params.GOLD_UTIL_LINEAR
    leftover = params.TEAM_SIZE * params.AVG_TARGET_COST - cost
    leftover_penalty = (
        params.LEFTOVER_PENALTY_FACTOR * leftover if leftover > 0 else 0.0
    )
    score = trait_val + high_val + gold_util_val - leftover_penalty
    return dict(
        cost=cost,
        size=len(idxs),
        high_count=high_count,
        trait_counts=trait_counts,
        trait_value=trait_val,
        high_value=high_val,
        gold_util_value=gold_util_val,
        leftover_penalty=leftover_penalty,
        score=score,
        idxs=tuple(idxs),
    )


def decompose_team_score(team: Dict[str, Any]) -> Dict[str, Any]:
    s = team["score"]
    tv = team["trait_value"]
    hv = team["high_value"]
    gv = team["gold_util_value"]
    pen = team["leftover_penalty"]
    pos_total = tv + hv + gv
    dd = {
        "score": s,
        "trait_value": tv,
        "high_value": hv,
        "gold_util_value": gv,
        "leftover_penalty": pen,
        "trait_value_pct": (tv / pos_total * 100.0) if pos_total else 0.0,
        "high_value_pct": (hv / pos_total * 100.0) if pos_total else 0.0,
        "gold_util_pct": (gv / pos_total * 100.0) if pos_total else 0.0,
        "penalty_pct_of_score": (pen / s * 100.0) if s else 0.0,
    }
    return dd


def print_score_decomposition(teams: List[Dict[str, Any]], k: int = 5):
    print(f"\n=== SCORE DECOMPOSITION (Top {k}) ===")
    for rank, team in enumerate(teams[:k], start=1):
        d = decompose_team_score(team)
        print(
            f"[#{rank}] Score={d['score']:.2f} "
            f"(Trait={d['trait_value']:.2f} [{d['trait_value_pct']:.1f}%], "
            f"High={d['high_value']:.2f} [{d['high_value_pct']:.1f}%], "
            f"Gold={d['gold_util_value']:.2f} [{d['gold_util_pct']:.1f}%], "
            f"Penalty={d['leftover_penalty']:.2f} [{d['penalty_pct_of_score']:.1f}% of final])"
        )


# -----------------------------
# NUMBA MERGE (C) + PY FALLBACK
# -----------------------------
if NUMBA_AVAILABLE:
    import numpy as np

    @njit(cache=True, nogil=True, fastmath=True)
    def merge_and_score_numba(
        countsA,
        countsB,
        costA,
        costB,
        highA,
        highB,
        high_cost_unit_bonus,
        gold_util_linear,
        leftover_penalty_factor,
        team_size,
        avg_target_cost,
        level_value,
    ):
        # counts arrays: int16
        T = countsA.shape[0]
        merged = np.empty(T, dtype=np.int16)
        for i in range(T):
            merged[i] = countsA[i] + countsB[i]
        total_cost = costA + costB
        total_high = highA + highB
        trait_val = 0.0
        for i in range(T):
            cnt = merged[i]
            if cnt >= level_value.shape[1]:
                cnt = level_value.shape[1] - 1
            trait_val += level_value[i, cnt]
        high_val = total_high * high_cost_unit_bonus
        gold_util_val = total_cost * gold_util_linear
        leftover = team_size * avg_target_cost - total_cost
        leftover_pen = leftover_penalty_factor * leftover if leftover > 0 else 0.0
        score = trait_val + high_val + gold_util_val - leftover_pen
        return (
            merged,
            total_cost,
            total_high,
            trait_val,
            high_val,
            gold_util_val,
            leftover_pen,
            score,
        )

else:

    def merge_and_score_numba(
        countsA,
        countsB,
        costA,
        costB,
        highA,
        highB,
        high_cost_unit_bonus,
        gold_util_linear,
        leftover_penalty_factor,
        team_size,
        avg_target_cost,
        level_value,
    ):
        T = len(countsA)
        merged = [0] * T
        for i in range(T):
            merged[i] = countsA[i] + countsB[i]
        total_cost = costA + costB
        total_high = highA + highB
        trait_val = 0.0
        for i in range(T):
            cnt = merged[i]
            if cnt >= len(level_value[i]):
                cnt = len(level_value[i]) - 1
            trait_val += level_value[i][cnt]
        high_val = total_high * high_cost_unit_bonus
        gold_util_val = total_cost * gold_util_linear
        leftover = team_size * avg_target_cost - total_cost
        leftover_pen = leftover_penalty_factor * leftover if leftover > 0 else 0.0
        score = trait_val + high_val + gold_util_val - leftover_pen
        return (
            merged,
            total_cost,
            total_high,
            trait_val,
            high_val,
            gold_util_val,
            leftover_pen,
            score,
        )


# -----------------------------
# SIGNATURE / MiM Enumeration (vereinfachtes Beispiel)
# -----------------------------
@dataclass
class HalfSignature:
    idxs: Tuple[int, ...]
    size: int
    cost: int
    high: int
    base_mask: int
    trait_counts: Tuple[int, ...]
    trait_value: float  # bereits berechnet (nur Traits, keine anderen Komponenten)


def enumerate_half_signatures(
    champions: List[Dict],
    trait_index: Dict[str, int],
    LEVEL_VALUE: List[List[float]],
    params: ScoreParams,
    exact_size: int,
    side: str = "A",
) -> List[HalfSignature]:
    """
    Erzeugt alle Subsets genau der Größe ``exact_size`` der Championsliste
    (dies ist bewusst reduziert – für sehr große N bräuchtest Du zusätzliche Filter).
    """
    n = len(champions)
    # Für Demonstration: wir nehmen einfach ALLE Champions (kann groß werden!) – ggf. filtern.
    # Besser: Aufteilen (erste Hälfte für side A, zweite für side B)
    if side == "A":
        indices = list(range(0, n // 2))
    else:
        indices = list(range(n // 2, n))

    T = len(trait_index)
    # Schnelle Zugriffe
    champ_cost = [c["cost"] for c in champions]
    champ_high = [1 if c["high"] else 0 for c in champions]
    champ_traits = [c["traits"] for c in champions]
    champ_base_index = [c["base_index"] for c in champions]

    signatures: List[HalfSignature] = []

    # Backtracking
    def dfs(
        start: int,
        chosen: List[int],
        size: int,
        cost: int,
        high: int,
        base_mask: int,
        trait_counts: List[int],
    ):
        if size == exact_size:
            trait_val = 0.0
            for ti in range(T):
                cnt = trait_counts[ti]
                if cnt >= len(LEVEL_VALUE[ti]):
                    cnt = len(LEVEL_VALUE[ti]) - 1
                trait_val += LEVEL_VALUE[ti][cnt]
            signatures.append(
                HalfSignature(
                    idxs=tuple(chosen),
                    size=size,
                    cost=cost,
                    high=high,
                    base_mask=base_mask,
                    trait_counts=tuple(trait_counts),
                    trait_value=trait_val,
                )
            )
            return
        if size > exact_size:
            return
        for pos in range(start, len(indices)):
            ci = indices[pos]
            # Champion Index = ci
            bidx = champ_base_index[ci]
            if (base_mask >> bidx) & 1:
                continue  # Variante schon gewählt
            new_cost = cost + champ_cost[ci]
            # (Optional: cost-bound / pruning)
            new_high = high + champ_high[ci]
            chosen.append(ci)
            for tr in champ_traits[ci]:
                trait_counts[trait_index[tr]] += 1
            dfs(
                pos + 1,
                chosen,
                size + 1,
                new_cost,
                new_high,
                base_mask | (1 << bidx),
                trait_counts,
            )
            for tr in champ_traits[ci]:
                trait_counts[trait_index[tr]] -= 1
            chosen.pop()

    dfs(0, [], 0, 0, 0, 0, [0] * T)

    # Basic Dominance-Pruning:
    # Für gleiche (size, high, cost_bucket) – behalte max trait_value.
    trimmed: Dict[Tuple[int, int, int], HalfSignature] = {}
    for sig in signatures:
        key = (sig.size, sig.high, sig.cost)
        prev = trimmed.get(key)
        if (prev is None) or (sig.trait_value > prev.trait_value):
            trimmed[key] = sig
    final = list(trimmed.values())
    return final


# -----------------------------
# MiM Combine Phase (nutzt merge_and_score_numba)
# -----------------------------
def combine_half_signatures(
    halfA: List[HalfSignature],
    halfB: List[HalfSignature],
    params: ScoreParams,
    LEVEL_VALUE_NP,  # numpy array
    top_k: int = 50,
) -> List[Dict[str, Any]]:
    """
    Kombiniert zwei halb-Signatur Listen zu vollständigen Teams of TEAM_SIZE.
    Sehr vereinfachte Variante (keine starken Bounds) – demonstriert Integration.
    """
    TEAM_SIZE = params.TEAM_SIZE
    best_heap: List[Tuple[float, int, Dict[str, Any]]] = []
    unique_counter = count()
    best_floor = -1e18

    # Sortiere halfB nach trait_value für leichte heuristische Bound (optional)
    halfB_sorted = sorted(halfB, key=lambda x: x.trait_value, reverse=True)

    iter_total = 0
    for a in prog(halfA, desc="Combine", unit="A"):
        slots_needed = TEAM_SIZE - a.size
        if slots_needed < 0:
            continue
        # Filter: nimm nur B-Sigs mit passender Größe
        for b in halfB_sorted:
            if b.size != slots_needed:
                continue
            # Varianten-Konflikt?
            if (a.base_mask & b.base_mask) != 0:
                continue
            # Merge trait counts & Score
            (
                merged_counts,
                total_cost,
                total_high,
                trait_val,
                high_val,
                gold_val,
                leftover_pen,
                score,
            ) = merge_and_score_numba(
                (a.trait_counts if NUMBA_AVAILABLE else a.trait_counts),
                (b.trait_counts if NUMBA_AVAILABLE else b.trait_counts),
                a.cost,
                b.cost,
                a.high,
                b.high,
                params.HIGH_COST_UNIT_BONUS,
                params.GOLD_UTIL_LINEAR,
                params.LEFTOVER_PENALTY_FACTOR,
                params.TEAM_SIZE,
                params.AVG_TARGET_COST,
                LEVEL_VALUE_NP,
            )
            iter_total += 1
            if score > best_floor:
                entry = dict(
                    idxs=tuple(a.idxs + b.idxs),
                    cost=total_cost,
                    size=TEAM_SIZE,
                    high_count=total_high,
                    trait_counts=tuple(int(x) for x in merged_counts),
                    trait_value=trait_val,
                    high_value=high_val,
                    gold_util_value=gold_val,
                    leftover_penalty=leftover_pen,
                    score=score,
                )
                import heapq

                heapq.heappush(best_heap, (score, next(unique_counter), entry))
                if len(best_heap) > top_k:
                    heapq.heappop(best_heap)
                best_floor = best_heap[0][0]
    print(f"Combine Iterationen: {iter_total}")
    return [e for _, _, e in sorted(best_heap, key=lambda x: x[0], reverse=True)]


# -----------------------------
# RE-SCORING (E)
# -----------------------------
def build_name_index(champions: List[Dict]) -> Dict[str, int]:
    name_idx = {}
    for i, c in enumerate(champions):
        # mappe verschiedene Variationen: original name, base, id
        name_idx[c["name"]] = i
        name_idx.setdefault(c["base"], i)
        name_idx.setdefault(c["id"], i)
    return name_idx


def score_from_indices(
    idxs: List[int],
    champions: List[Dict],
    trait_index: Dict[str, int],
    LEVEL_VALUE: List[List[float]],
    params: ScoreParams,
):
    return compute_team_components(idxs, champions, trait_index, LEVEL_VALUE, params)


def rescore_from_json(
    path: str,
    champions: List[Dict],
    trait_index: Dict[str, int],
    LEVEL_VALUE: List[List[float]],
    params: ScoreParams,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    name_idx = build_name_index(champions)
    results = []
    for row in data:
        names = row.get("champions") or row.get("team") or row.get("units") or []
        idxs = []
        valid = True
        for n in names:
            n2 = n.strip()
            if n2 not in name_idx:
                print(f"[WARN] Name nicht gefunden beim Rescore: {n2}")
                valid = False
                break
            idxs.append(name_idx[n2])
        if not valid:
            continue
        comp = score_from_indices(idxs, champions, trait_index, LEVEL_VALUE, params)
        comp["idxs"] = tuple(idxs)
        comp["champions"] = names
        results.append(comp)
    results.sort(key=lambda x: x["score"], reverse=True)
    print_score_decomposition(results, k=min(top_k, len(results)))
    return results


def rescore_from_csv(
    path: str,
    champions: List[Dict],
    trait_index: Dict[str, int],
    LEVEL_VALUE: List[List[float]],
    params: ScoreParams,
    top_k: int = 20,
    sep: str = ",",
) -> List[Dict[str, Any]]:
    name_idx = build_name_index(champions)
    results = []
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.reader(f, delimiter=sep)
        for row in rdr:
            names = [r.strip() for r in row if r.strip()]
            idxs = []
            valid = True
            for n in names:
                if n not in name_idx:
                    print(f"[WARN] Name nicht gefunden in CSV: {n}")
                    valid = False
                    break
                idxs.append(name_idx[n])
            if not valid:
                continue
            comp = score_from_indices(idxs, champions, trait_index, LEVEL_VALUE, params)
            comp["idxs"] = tuple(idxs)
            comp["champions"] = names
            results.append(comp)
    results.sort(key=lambda x: x["score"], reverse=True)
    print_score_decomposition(results, k=min(top_k, len(results)))
    return results


# -----------------------------
# PARAMETER SWEEP (Optional)
# -----------------------------
def quick_parameter_sweep(
    base_params: ScoreParams,
    champions: List[Dict],
    trait_index: Dict[str, int],
    LEVEL_VALUE: List[List[float]],
    sample_teams: List[List[int]],
    vary_high_bonus=(6.0, 8.0, 10.0),
    vary_gold_util=(0.45, 0.55, 0.65),
    vary_penalty=(0.4, 0.6, 0.8),
):
    """
    Kurzer heuristischer Sweep: vergleicht Score-Rangliste über sample_teams.
    """
    print("\n=== PARAMETER SWEEP (Heuristischer Vergleich) ===")
    results = []
    for hb in vary_high_bonus:
        for gu in vary_gold_util:
            for lp in vary_penalty:
                sp = ScoreParams(
                    LEVEL_WEIGHT=base_params.LEVEL_WEIGHT,
                    HIGH_COST_UNIT_BONUS=hb,
                    GOLD_UTIL_LINEAR=gu,
                    LEFTOVER_PENALTY_FACTOR=lp,
                    HIGH_COST_THRESHOLD=base_params.HIGH_COST_THRESHOLD,
                    TEAM_SIZE=base_params.TEAM_SIZE,
                    AVG_TARGET_COST=base_params.AVG_TARGET_COST,
                )
                scores = []
                for team in sample_teams:
                    comp = compute_team_components(
                        team, champions, trait_index, LEVEL_VALUE, sp
                    )
                    scores.append(comp["score"])
                avg_score = sum(scores) / len(scores) if scores else 0.0
                results.append((hb, gu, lp, avg_score))
    results.sort(key=lambda x: x[3], reverse=True)
    for r in results[:10]:
        print(f"HIGH={r[0]:.2f}, GOLD={r[1]:.2f}, PEN={r[2]:.2f} -> AVG={r[3]:.2f}")
    print("=== Ende Sweep ===")


# -----------------------------
# MAIN EXECUTION
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="TFT Set 14 Optimizer")
    parser.add_argument("--team-size", type=int, help="Team size override")
    parser.add_argument(
        "--param-set", choices=list(PARAM_SETS.keys()), default=ACTIVE_PARAM_SET
    )
    parser.add_argument("--top-k", type=int, default=20, help="Number of teams")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity 0-2")
    parser.add_argument(
        "--force-refresh", action="store_true", help="Ignore cached data"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][
            max(0, min(2, args.verbose))
        ],
        format="%(message)s",
    )
    write_run_meta(args.param_set)
    logging.info(f"Python {sys.version.split()[0]}")
    try:
        import numba
        import numpy

        logging.info(f"numba {numba.__version__}, numpy {numpy.__version__}")
    except Exception:
        pass
    random.seed(args.seed)

    # 1) Lade Parameter
    param_cfg = PARAM_SETS[args.param_set]
    if args.team_size:
        param_cfg = dict(param_cfg)
        param_cfg["TEAM_SIZE"] = args.team_size
    params = ScoreParams(**param_cfg)

    print(f"== Lade Data Dragon (Set14) – Aktives Param Set: {ACTIVE_PARAM_SET} ==")
    champions, trait_breakpoints = load_set14_data(force_refresh=args.force_refresh)
    print(f"Geladen: {len(champions)} Champions, {len(trait_breakpoints)} Traits")

    # 2) Normalisieren + Varianten markieren
    champions = normalize_and_tag_variants(champions, params.HIGH_COST_THRESHOLD)
    champions = validate_champions(champions)
    print("Variante/High Tags gesetzt.")

    # 3) Trait Index + Level Tabellen
    trait_index = build_trait_indices(trait_breakpoints)
    LEVEL_LOOKUP, LEVEL_VALUE, LEVEL_LOOKUP_NP, LEVEL_VALUE_NP = compute_level_tables(
        champions, trait_breakpoints, trait_index, params.LEVEL_WEIGHT, max_cap=32
    )
    print(f"Level-Tabellen aufgebaut. NUMBA={NUMBA_AVAILABLE}")

    # 4) Falls vorhandene Resultats-Datei -> Rescore (E)
    existing_files = []
    if os.path.isfile("tft_full_bruteforce_results.json"):
        existing_files.append("tft_full_bruteforce_results.json")
    if os.path.isfile("tft_full_bruteforce_results.csv"):
        existing_files.append("tft_full_bruteforce_results.csv")

    rescored_all = []
    if existing_files:
        print("\n== Re-Scoring vorhandener Ergebnisdateien ==")
        for fp in existing_files:
            if fp.endswith(".json"):
                r = rescore_from_json(
                    fp, champions, trait_index, LEVEL_VALUE, params, top_k=10
                )
            else:
                r = rescore_from_csv(
                    fp, champions, trait_index, LEVEL_VALUE, params, top_k=10
                )
            rescored_all.extend(r)
        # Aggregiert sortieren
        rescored_all.sort(key=lambda x: x["score"], reverse=True)
        if rescored_all:
            print_score_decomposition(rescored_all, k=10)
    else:
        print("Keine vorhandenen Ergebnisdateien gefunden – fahre mit Demo-MiM fort.")

    # 5) Demo Meet-in-the-Middle (klein, damit es schnell läuft)
    #    Passe 'half_size' an – für komplette Suche sind viel stärkere Strategien nötig.
    half_size = params.TEAM_SIZE // 2  # 4 bei 8
    print(f"\n== Enumeriere Halb-Signaturen (Größe {half_size}) ==")
    halfA = enumerate_half_signatures(
        champions,
        trait_index,
        LEVEL_VALUE,
        params,
        half_size,
        side="A",
    )
    halfB = enumerate_half_signatures(
        champions,
        trait_index,
        LEVEL_VALUE,
        params,
        params.TEAM_SIZE - half_size,
        side="B",
    )
    print(
        f"Half A: {len(halfA)} Signaturen (gepruned) | Half B: {len(halfB)} Signaturen (gepruned)"
    )

    # 6) Kombinieren
    start_combine = time.time()
    top_combined = combine_half_signatures(
        halfA,
        halfB,
        params,
        LEVEL_VALUE_NP if NUMBA_AVAILABLE else LEVEL_VALUE,
        top_k=args.top_k,
    )
    dur_combine = time.time() - start_combine
    top_combined.sort(key=lambda x: x["score"], reverse=True)

    print(
        f"\n== Combine fertig in {dur_combine:.2f}s – Top {len(top_combined)} Teams =="
    )
    for i, team in enumerate(top_combined[:10], start=1):
        # Rekonstruiere Trait Anzeige
        trait_counts = team["trait_counts"]
        inv_trait = {v: k for k, v in trait_index.items()}
        trait_parts = []
        for tid, cnt in enumerate(trait_counts):
            if cnt > 0:
                trait_parts.append(f"{inv_trait[tid]}={cnt}")
        champ_names = [champions[idx]["name"] for idx in team["idxs"]]
        print(
            f"{i:2d}. Score={team['score']:.2f} Cost={team['cost']} "
            f"High={team['high_count']} | Traits: "
            + ", ".join(trait_parts)
            + " || "
            + ", ".join(champ_names)
        )

    print_score_decomposition(top_combined, k=min(10, len(top_combined)))

    # 7) Optional Parameter-Sweep über zufällige Beispielteams (klein)
    sample_teams = []
    if len(champions) >= params.TEAM_SIZE:
        for _ in range(8):
            cand = random.sample(range(len(champions)), params.TEAM_SIZE)
            sample_teams.append(cand)
        quick_parameter_sweep(params, champions, trait_index, LEVEL_VALUE, sample_teams)

    # 8) Speichern top_combined als JSON (als Demo)
    out_path = "demo_mim_top_combined.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(top_combined, f, indent=2)
    print(f"\nTop kombinierte Demo-Teams gespeichert unter {out_path}")


if __name__ == "__main__":
    main()
