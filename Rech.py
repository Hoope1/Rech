#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TFT Set 14 Meet-in-the-Middle Optimizer (TEAM_SIZE=8)
Exakte Suche aller 8er Teams über Kompression beider Hälften.
Ziel: Massiver Speedup ggü. naivem Enumerieren von C(n,8).

Phasen:
  1. Vorbereitung (Trait-Tabellen, Champion Vorverarbeitung)
  2. Enumeriere Hälfe A (alle Teilmengen Größe 0..TEAM_SIZE) -> Kompression
  3. Enumeriere Hälfte B -> Kompression
  4. Combine Phase (alle Paare (sigA, sigB) mit sizeA+sizeB=TEAM_SIZE)
  5. Auswertung / Top-K Ausgabe

Fortschritt:
  - Jede Phase hat eigenen Zähler + ETA.
  - Always-on Logging (keine Parameter nötig).
  - '[BEST]' Zeilen bei Verbesserungen.
  - '[PROGRESS]' Zeilen mit virtueller Abdeckung (äquivalente rohe Teams).

Anpassbare Kernbereiche:
  - CHAMPION_DATA  (bitte vollständig korrekt pflegen)
  - TRAIT_BREAKPOINTS (relevante Zähl-Breakpoints pro Trait)
  - SCORING (Funktion compute_score())
  - HIGH_COST_THRESHOLD (für Bonus / Analyse)

Exaktheit:
  - Kompression verwirft für identische (TraitSignature, size, high_cost_units)
    nur Einträge mit *niedrigerem* Cost -> Monotonie garantiert
    (mehr Cost erhöht nie negative Terms in deiner Score-Formel).
"""

from __future__ import annotations

import itertools
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import json
import re

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# ---------------------- KONFIGURATION -----------------------
# ============================================================

TEAM_SIZE = 8
HIGH_COST_THRESHOLD = 4
TOP_K = 25  # wie viele beste Teams am Ende anzeigen
LOG_INTERVAL_SECONDS = 2.0  # Fortschritt-Ausgabe Frequenz
ABORT_AFTER_SECONDS = None  # Optional: harte Abbruchzeit (None = aus)

# Gewichtung Trait-Level – Beispiel (Alternative Scoring aus deinem Wunsch)
# Du kannst die Werte hier verändern / erweitern.
LEVEL_WEIGHT = {
    2: 20,
    3: 30,
    4: 45,
    5: 62.5,
    6: 85,
    7: 100,
    8: 120,
}

# Gold-Utilisierung: Beispielterm – bitte an deine bisherige Formel angleichen.
# Der Score setzt sich unten aus (Summe Trait-Level-Werte) + evtl. Bonus zusammen.
# Passe diese Parameter an deine vorherige Definition an, damit Zahlen wieder passen.
GOLD_UTIL_LINEAR = 0.55
LEFTOVER_PENALTY_FACTOR = 0.50
AVG_TARGET_COST = 4.0
HIGH_COST_UNIT_BONUS = 7.0
MAX_POSSIBLE_GOLD = None

# ============================================================
# ------------------- CHAMPION DATENBLOCK --------------------
# ============================================================
# !!! WICHTIG !!!
# Ersetze den kompletten CHAMPION_DATA Block durch deinen *korrekten* Datensatz
# (alle Set 14 Champions mit ihren Traits und Kosten).
#
# Format je Eintrag:
#   ("Name", Kosten, ("Trait1","Trait2",...))
#
# Traits: Schreibe sie *genau* konsistent (Groß-/Kleinschreibung) – das bestimmt die Indizes.
#
# Unten ist ein *TEILWEISER* Platzhalter (NICHT vollständig / NICHT garantiert korrekt!).
# -> Du musst vervollständigen. Falls du mir deine Liste gibst, liefere ich dir eine Version.
#
CHAMPION_DATA: List[Tuple[str, int, Tuple[str, ...]]] = [
    # cost 1 Beispiele (PLATZHALTER!)
    ("Annie", 1, ("A.M.P.", "Street Demon")),
    ("Naafiri", 1, ("A.M.P.", "Overlord")),
    ("Sylas", 1, ("A.M.P.", "Anima Squad")),
    ("Vi", 1, ("A.M.P.", "Bastion")),
    ("Brand", 1, ("Street Demon", "Rapidfire")),
    ("Dr Mundo", 1, ("Street Demon", "Bruiser")),
    ("Ekko", 1, ("Street Demon", "Strategist")),
    ("Jinx", 1, ("Street Demon", "Marksman")),
    ("Neeko", 1, ("Street Demon", "Dynamo")),
    ("Rengar", 1, ("Street Demon", "Slayer")),
    ("Samira", 1, ("Street Demon", "Executioner")),
    ("Zyra", 1, ("Street Demon", "Vanguard")),
    # cost 2 Beispiele
    ("Aurora", 2, ("Anima Squad", "Dynamo")),
    ("Illaoi", 2, ("Anima Squad", "Bruiser")),
    ("Leona", 2, ("Anima Squad", "Bastion")),
    ("Seraphine", 2, ("Anima Squad", "Rapidfire")),
    ("Vayne", 2, ("Anima Squad", "Marksman")),
    ("Xayah", 2, ("Anima Squad", "Executioner")),
    ("Yuumi", 2, ("Anima Squad", "Strategist")),
    # cost 3 Beispiele
    ("Jax", 3, ("Exotech", "Bruiser")),
    ("Jhin", 3, ("Exotech", "Marksman")),
    ("Mordekaiser", 3, ("Exotech", "Vanguard")),
    ("Naafiri+", 3, ("Exotech", "Slayer")),  # Platzhalter
    ("Sejuani", 3, ("Exotech", "Bastion")),
    ("Varus", 3, ("Exotech", "Rapidfire")),
    ("Zeri", 3, ("Exotech", "Dynamo")),
    # cost 4 Beispiele
    ("Renekton", 4, ("Overlord", "Divinicorp", "Bastion")),
    ("Kobuko", 4, ("Cyberboss", "Bruiser")),
    ("Garen", 4, ("God of the Net", "Vanguard")),
    ("Urgot", 4, ("BoomBot", "Executioner")),
    ("Viego", 4, ("Soul Killer", "Golden Ox", "Techie")),
    ("Zac", 4, ("Virus", "Bruiser")),
    ("Kogmaw", 4, ("Virus", "Marksman")),
    # cost 5 Beispiele (PLATZHALTER!)
    ("Twisted Fate", 5, ("Syndicate", "Strategist")),
    ("Graves", 5, ("Syndicate", "Marksman")),
    ("Shaco", 5, ("Syndicate", "Slayer")),
    ("Zed", 5, ("Syndicate", "Executioner")),
    ("Draven", 5, ("Divinicorp", "Slayer")),
    ("Galio", 5, ("Divinicorp", "Bastion")),
    ("Poppy", 5, ("Divinicorp", "Vanguard")),
    ("Darius", 5, ("Golden Ox", "Vanguard")),
    ("Jarvan IV", 5, ("Golden Ox", "Strategist")),
    ("Kindred", 5, ("Golden Ox", "Rapidfire")),
    ("Shyvana", 5, ("Golden Ox", "Bruiser")),
    ("Rhaast", 5, ("Soul Killer", "Bruiser")),
    ("Senna", 5, ("Soul Killer", "Marksman")),
    ("Aurora+", 5, ("Anima Squad", "Dynamo")),  # ersetze
    ("Veigar", 5, ("Nitro", "Rapidfire")),
    ("Ziggs", 5, ("Nitro", "Techie")),
    ("Zeri+", 5, ("Nitro", "Marksman")),
    ("Skarner", 5, ("Cypher", "Bruiser")),
    ("Sylas+", 5, ("Cypher", "Bastion")),
    ("Vi+", 5, ("Cypher", "Vanguard")),
    ("Vex", 5, ("Cypher", "Dynamo")),
    ("Ekko+", 5, ("Cypher", "Strategist")),
    ("Leblanc", 5, ("Overlord", "Rapidfire")),
    ("Miss Fortune", 5, ("Overlord", "Marksman")),
    ("Morgana", 5, ("Overlord", "Executioner")),
    ("Nidalee", 5, ("Overlord", "Slayer")),
    ("Aphelios", 5, ("God of the Net", "Marksman")),
    ("Aatrox?", 5, ("God of the Net", "Slayer")),  # Platzhalter falls existiert
    ("Braum", 5, ("God of the Net", "Vanguard")),
    ("Alistar", 5, ("God of the Net", "Bruiser")),
    # ... VERVOLLSTÄNDIGEN ...
]

# ============================================================
# ---- TRAIT BREAKPOINTS (max relevante Zählung pro Trait) ----
# Falls ein Trait keine höhere Stufe als z.B. 6 besitzt, deckeln wir bei 6.
# Passe diese Tabelle exakt zu den realen Trait Breakpoints an.
# (Werte >8 ignoriert das Score-Modell oben ohnehin.)
# ============================================================
TRAIT_BREAKPOINTS: Dict[str, List[int]] = {
    # Beispiel: (2,4,6) oder (2,3,4,5,6,7,8)
    "A.M.P.": [2, 3, 4, 5],
    "Street Demon": [2, 3, 4, 5, 6, 7, 8],
    "Anima Squad": [2, 3, 4, 5, 6, 7, 8],
    "Exotech": [2, 3, 4, 5, 6, 7, 8],
    "Overlord": [2, 3, 4, 5, 6, 7, 8],
    "Divinicorp": [2, 3, 4, 5, 6, 7, 8],
    "Cyberboss": [2, 3, 4, 5, 6, 7, 8],
    "God of the Net": [2, 3, 4, 5, 6, 7, 8],
    "Golden Ox": [2, 3, 4, 5, 6, 7, 8],
    "Soul Killer": [2, 3, 4, 5, 6, 7, 8],
    "Virus": [2, 3, 4, 5, 6, 7, 8],
    "Syndicate": [2, 3, 4, 5, 6, 7, 8],
    "Nitro": [2, 3, 4, 5, 6, 7, 8],
    "Cypher": [2, 3, 4, 5, 6, 7, 8],
    "BoomBot": [2, 3, 4, 5, 6, 7, 8],
    "Bruiser": [2, 4, 6, 8],
    "Bastion": [2, 4, 6, 8],
    "Vanguard": [2, 4, 6, 8],
    "Marksman": [2, 4, 6, 8],
    "Rapidfire": [2, 4, 6, 8],
    "Executioner": [2, 4, 6, 8],
    "Slayer": [2, 4, 6, 8],
    "Strategist": [2, 4, 6, 8],
    "Dynamo": [2, 3, 4, 5, 6, 7, 8],
    "Techie": [2, 3, 4, 5, 6, 7, 8],
    # ... ggf. fehlende Traits ergänzen ...
}

# ============================================================
# -------------- INTERN: TRAIT-INDEX / LOOKUPS ---------------
# ============================================================
# Baue Traitliste aus Championdaten (vereint) + Sortierung für stabile Indexe
all_traits_set = set()
for _, _, ts in CHAMPION_DATA:
    for t in ts:
        all_traits_set.add(t)

# Prüfe: Alle in TRAIT_BREAKPOINTS definierten Traits wirklich vorhanden?
missing_in_champs = [t for t in TRAIT_BREAKPOINTS if t not in all_traits_set]
if missing_in_champs:
    logger.warning(
        "WARNUNG: Traits in TRAIT_BREAKPOINTS ohne Champion: %s", missing_in_champs
    )

# Traits, die bei Champions vorkommen aber keine Breakpoints definiert haben -> Standard (2..8)
undefined_break = [t for t in all_traits_set if t not in TRAIT_BREAKPOINTS]
for t in undefined_break:
    TRAIT_BREAKPOINTS[t] = [2, 3, 4, 5, 6, 7, 8]

TRAITS = sorted(TRAIT_BREAKPOINTS.keys())
TRAIT_INDEX = {t: i for i, t in enumerate(TRAITS)}
T = len(TRAITS)

# Maximal relevante Zählung (Deckel) pro Trait -> letzter Breakpoint
TRAIT_CAP = [max(TRAIT_BREAKPOINTS[t]) for t in TRAITS]

# Precompute LEVEL_FROM_COUNT: count -> levelValue (hier direkt Weight)
# Für Zählungen zwischen Breakpoints interpolieren wir *nicht*, sondern nutzen die vorherige Stufe.
LEVEL_VALUE = [
    [0.0] * (cap + 1) for cap in TRAIT_CAP
]  # LEVEL_VALUE[traitIndex][count] -> Score-Beitrag dieses Traits
for tname, bps in TRAIT_BREAKPOINTS.items():
    ti = TRAIT_INDEX[tname]
    cap = TRAIT_CAP[ti]
    prev_val = 0.0
    prev_bp = 0
    for bp in range(0, cap + 1):
        # Setze Level sobald Breakpoint erreicht
        # Level ist Anzahl Breakpoints <= bp
        level = sum(1 for b in bps if bp >= b)
        if level in LEVEL_WEIGHT:
            prev_val = LEVEL_WEIGHT[level]
        LEVEL_VALUE[ti][bp] = prev_val

# NumPy Tabellen für Numba-Optimierung
MAX_COUNT_PER_TRAIT = max(TRAIT_CAP)
LEVEL_LOOKUP_T = np.zeros((T, MAX_COUNT_PER_TRAIT + 1), dtype=np.int16)
LEVEL_VALUE_T = np.zeros((T, MAX_COUNT_PER_TRAIT + 1), dtype=np.float64)
for tname, bps in TRAIT_BREAKPOINTS.items():
    ti = TRAIT_INDEX[tname]
    for cnt in range(1, MAX_COUNT_PER_TRAIT + 1):
        lvl = 0
        for bp in bps:
            if cnt >= bp:
                lvl = bp
        LEVEL_LOOKUP_T[ti, cnt] = lvl
        if lvl:
            LEVEL_VALUE_T[ti, cnt] = LEVEL_WEIGHT.get(bps.index(lvl) + 2, 0)


# ============================================================
# ------------------ CHAMPION VORVERARBEITEN -----------------
# ============================================================

variant_groups: Dict[str, List[int]] = defaultdict(list)
CHAMPIONS: List[Dict] = []

for idx, (name, cost, traits) in enumerate(CHAMPION_DATA):
    base = re.sub(r"\++$", "", name)
    champ = {
        "id": f"ch{idx}",
        "idx": idx,
        "name": name,
        "base": base,
        "cost": cost,
        "traits": tuple(TRAIT_INDEX[t] for t in traits),
        "high": cost >= HIGH_COST_THRESHOLD,
    }
    CHAMPIONS.append(champ)
    variant_groups[base].append(idx)

BASES = sorted(variant_groups.keys())
BASE_INDEX = {b: i for i, b in enumerate(BASES)}
for champ in CHAMPIONS:
    champ["base_index"] = BASE_INDEX[champ["base"]]

VARIANT_GROUPS = list(variant_groups.values())

N = len(CHAMPIONS)
if N > 64:
    logger.info("HINWEIS: N > 64 – Bitmask passt noch (Python int), kein Problem.")
logger.info(f"Champions geladen: {N} | Traits: {T}")

# Aufteilen in zwei Hälften (balanced)
mid = N // 2
LEFT = CHAMPIONS[:mid]
RIGHT = CHAMPIONS[mid:]


# ============================================================
# -------------------- SCORE FUNKTION ------------------------
# ============================================================
def compute_score(
    trait_counts: List[int], total_cost: int, high_cost_units: int
) -> float:
    """Berechnet den Score eines Teams nach den aktuellen Parametern."""

    trait_sum = 0.0
    for ti, cnt in enumerate(trait_counts):
        cnt = min(cnt, TRAIT_CAP[ti])
        trait_sum += LEVEL_VALUE[ti][cnt]

    gold_util = total_cost * GOLD_UTIL_LINEAR
    leftover = TEAM_SIZE * AVG_TARGET_COST - total_cost
    leftover_penalty = LEFTOVER_PENALTY_FACTOR * leftover if leftover > 0 else 0.0
    high_cost_bonus = high_cost_units * HIGH_COST_UNIT_BONUS

    return trait_sum + gold_util + high_cost_bonus - leftover_penalty


if NUMBA_AVAILABLE:

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
        gold_val = total_cost * gold_util_linear
        leftover = team_size * avg_target_cost - total_cost
        leftover_penalty = leftover_penalty_factor * leftover if leftover > 0 else 0.0
        score = trait_val + high_val + gold_val - leftover_penalty
        return (
            merged,
            total_cost,
            total_high,
            trait_val,
            high_val,
            gold_val,
            leftover_penalty,
            score,
        )

else:

    def merge_and_score_numba(*args, **kwargs):
        raise RuntimeError("Numba nicht verf\u00fcgbar")


# ============================================================
# ---------------- SIGNATURE / KOMPRESSION -------------------
# ============================================================
@dataclass(frozen=True)
class Signature:
    size: int
    cost: int
    high: int
    trait_counts: tuple[int, ...]
    bitmask: int
    base_mask: int
    idxs: tuple[int, ...]


def enumerate_half(champs: List[Dict], label: str):
    """
    Erzeugt alle Teilmengen Größe 0..TEAM_SIZE.
    Kompression:
        key = (size, high_cost, tuple(capped trait counts))
        value = Signature mit MAX cost (alle anderen überschrieben)
    """
    start = time.time()
    L = len(champs)
    # Prebuild arrays für Traits pro Champion
    trait_lists = [c["traits"] for c in champs]
    costs = [c["cost"] for c in champs]
    highs = [1 if c["high"] else 0 for c in champs]
    bases = [c["base_index"] for c in champs]

    compressed: Dict[Tuple[int, int, Tuple[int, ...]], Signature] = {}
    total_subsets = 0

    # Vorab Anzahl Kombinationen für Fortschritts-ETA
    # Sum_{k=0..TEAM_SIZE, k<=L} C(L,k)
    total_target = sum(math.comb(L, k) for k in range(0, min(TEAM_SIZE, L) + 1))

    last_log = start
    processed = 0

    for k in range(0, min(TEAM_SIZE, L) + 1):
        for combo in itertools.combinations(range(L), k):
            processed += 1
            total_subsets += 1
            # Fortschritt
            now = time.time()
            if now - last_log >= LOG_INTERVAL_SECONDS:
                pct = processed / total_target * 100
                rate = processed / (now - start + 1e-9)
                eta = (total_target - processed) / rate if rate > 0 else 0
                logger.info(
                    "[%s] %5.2f%% k=%d subs=%s/%s rate=%s/s eta=%5.1fm unique=%s",
                    label,
                    pct,
                    k,
                    f"{processed:,}",
                    f"{total_target:,}",
                    f"{rate:,.0f}",
                    eta / 60,
                    f"{len(compressed):,}",
                )
                last_log = now

            # Akkumulieren
            cost = 0
            highc = 0
            trait_counts = [0] * T
            bitmask = 0
            base_mask = 0
            idxs = []
            for idx_local in combo:
                ch = champs[idx_local]
                cost += costs[idx_local]
                highc += highs[idx_local]
                idxs.append(ch["idx"] if "idx" in ch else idx_local)
                for ti in trait_lists[idx_local]:
                    if trait_counts[ti] < TRAIT_CAP[ti]:
                        trait_counts[ti] += 1
                # Globale Bitposition = real champion index
                bitmask |= 1 << (ch["idx"] if "idx" in ch else idx_local)
                base_mask |= 1 << bases[idx_local]

            key = (k, highc, tuple(trait_counts))
            prev = compressed.get(key)
            if prev is None or cost > prev.cost:
                compressed[key] = Signature(
                    k,
                    cost,
                    highc,
                    tuple(trait_counts),
                    bitmask,
                    base_mask,
                    tuple(idxs),
                )

    dur = time.time() - start
    logger.info(
        "[%s] DONE in %.1fs | raw subsets=%s | unique=%s compression=%.1fx",
        label,
        dur,
        f"{total_subsets:,}",
        f"{len(compressed):,}",
        total_subsets / len(compressed),
    )
    return list(compressed.values())


# ============================================================
# ------------------- COMBINE (MiM JOIN) ---------------------
# ============================================================
@dataclass
class BestTeam:
    score: float
    cost: int
    high: int
    trait_counts: Tuple[int, ...]
    bitmask: int
    idxs: tuple[int, ...]


def combine_and_search(left_sigs: List[Signature], right_sigs: List[Signature]):
    # Gruppiere nach Größe
    left_by_size: Dict[int, List[Signature]] = defaultdict(list)
    right_by_size: Dict[int, List[Signature]] = defaultdict(list)
    for s in left_sigs:
        left_by_size[s.size].append(s)
    for s in right_sigs:
        right_by_size[s.size].append(s)

    # Für einfache Obergrenzenanalyse Summen der Paare zählen
    total_pairings = 0
    for k in range(0, TEAM_SIZE + 1):
        a = len(left_by_size.get(k, []))
        b = len(right_by_size.get(TEAM_SIZE - k, []))
        total_pairings += a * b
    logger.info(f"[COMBINE] Erwartete effektive Paarungen: {total_pairings:,}")

    best_list: List[BestTeam] = []
    best_floor = -1e18
    improvements = 0

    last_log = time.time()
    processed_pairs = 0

    for k in range(0, TEAM_SIZE + 1):
        A_list = left_by_size.get(k, [])
        B_list = right_by_size.get(TEAM_SIZE - k, [])
        if not A_list or not B_list:
            continue

        # Optional: sortiere große Listen nach cost + high (heuristisch für schneller guten Floor)
        A_list.sort(key=lambda s: (-(s.cost + 2 * s.high)))
        B_list.sort(key=lambda s: (-(s.cost + 2 * s.high)))

        for sa in A_list:
            a_counts = np.array(sa.trait_counts, dtype=np.int16)
            a_cost = sa.cost
            a_high = sa.high
            for sb in B_list:
                processed_pairs += 1
                # Fortschritt
                now = time.time()
                if now - last_log >= LOG_INTERVAL_SECONDS:
                    pct = processed_pairs / total_pairings * 100
                    virt_cov = processed_pairs / math.comb(N, TEAM_SIZE) * 100
                    logger.info(
                        "[PROGRESS] %6.3f%% pairs=%s/%s virt_raw≈%.4f%% floor=%.2f",
                        pct,
                        f"{processed_pairs:,}",
                        f"{total_pairings:,}",
                        virt_cov,
                        best_floor,
                    )
                    last_log = now

                if sa.base_mask & sb.base_mask:
                    continue
                b_counts = np.array(sb.trait_counts, dtype=np.int16)

                (
                    merged,
                    total_cost,
                    total_high,
                    trait_val,
                    high_val,
                    gold_val,
                    leftover_penalty,
                    score,
                ) = merge_and_score_numba(
                    a_counts,
                    b_counts,
                    a_cost,
                    sb.cost,
                    a_high,
                    sb.high,
                    HIGH_COST_UNIT_BONUS,
                    GOLD_UTIL_LINEAR,
                    LEFTOVER_PENALTY_FACTOR,
                    TEAM_SIZE,
                    AVG_TARGET_COST,
                    LEVEL_VALUE_T,
                )

                if score > best_floor:
                    improvements += 1
                    best_floor = score
                    bt = BestTeam(
                        score=score,
                        cost=total_cost,
                        high=total_high,
                        trait_counts=tuple(int(x) for x in merged),
                        bitmask=sa.bitmask | sb.bitmask,
                        idxs=sa.idxs + sb.idxs,
                    )
                    best_list.append(bt)
                    best_list.sort(key=lambda x: x.score, reverse=True)
                    if len(best_list) > TOP_K:
                        best_list = best_list[:TOP_K]

                    logger.info(
                        "[BEST] Score=%.2f Cost=%d Floor=%.2f (Imp#%d)",
                        score,
                        total_cost,
                        best_floor,
                        improvements,
                    )

    return best_list


def decompose_team_score(team: BestTeam) -> Dict:
    comp = {
        "trait_value": 0.0,
        "high_value": team.high * HIGH_COST_UNIT_BONUS,
        "gold_util_value": team.cost * GOLD_UTIL_LINEAR,
        "leftover_penalty": 0.0,
    }
    for ti, cnt in enumerate(team.trait_counts):
        cnt = min(cnt, LEVEL_VALUE_T.shape[1] - 1)
        comp["trait_value"] += LEVEL_VALUE_T[ti, cnt]

    leftover = TEAM_SIZE * AVG_TARGET_COST - team.cost
    comp["leftover_penalty"] = (
        LEFTOVER_PENALTY_FACTOR * leftover if leftover > 0 else 0.0
    )
    score = (
        comp["trait_value"]
        + comp["high_value"]
        + comp["gold_util_value"]
        - comp["leftover_penalty"]
    )
    pos_total = comp["trait_value"] + comp["high_value"] + comp["gold_util_value"]
    comp_pct = {
        "trait_value_pct": (
            (comp["trait_value"] / pos_total * 100.0) if pos_total else 0.0
        ),
        "high_value_pct": (
            (comp["high_value"] / pos_total * 100.0) if pos_total else 0.0
        ),
        "gold_util_pct": (
            (comp["gold_util_value"] / pos_total * 100.0) if pos_total else 0.0
        ),
        "penalty_pct_of_score": (
            (comp["leftover_penalty"] / score * 100.0) if score else 0.0
        ),
    }
    comp.update(comp_pct)
    comp["score"] = score
    return comp


def print_score_decomposition(top_teams: List[BestTeam], k: int = 5) -> None:
    logger.info("\n=== SCORE DECOMPOSITION ===")
    for rank, team in enumerate(top_teams[:k], start=1):
        comp = decompose_team_score(team)
        logger.info(
            (
                "[#%d] Score=%.2f (Trait=%.2f [%.1f%%], High=%.2f [%.1f%%], "
                "GoldUtil=%.2f [%.1f%%], Penalty=%.2f [%.1f%% of final])"
            ),
            rank,
            comp["score"],
            comp["trait_value"],
            comp["trait_value_pct"],
            comp["high_value"],
            comp["high_value_pct"],
            comp["gold_util_value"],
            comp["gold_util_pct"],
            comp["leftover_penalty"],
            comp["penalty_pct_of_score"],
        )


def rescore_teams_from_json(json_path: str, top_k: int = 20) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    name_index = {c["name"]: c["idx"] for c in CHAMPIONS}
    rescored = []
    for entry in raw:
        names = entry.get("champions") or entry.get("team") or []
        idxs = []
        for n in names:
            n = n.strip()
            if n not in name_index:
                logger.warning("Unknown champion name in result file: %s", n)
                idxs = []
                break
            idxs.append(name_index[n])
        if not idxs:
            continue

        trait_counts = [0] * T
        cost = 0
        high = 0
        for i in idxs:
            c = CHAMPIONS[i]
            cost += c["cost"]
            if c["high"]:
                high += 1
            for t in c["traits"]:
                if trait_counts[t] < TRAIT_CAP[t]:
                    trait_counts[t] += 1

        score = compute_score(trait_counts, cost, high)
        rescored.append({"champions": names, "score": score})

    rescored.sort(key=lambda x: x["score"], reverse=True)
    return rescored[:top_k]


# ============================================================
# --------------- REKONSTRUKTION TEAMNAMEN -------------------
# ============================================================
def bitmask_to_team(bitmask: int) -> List[str]:
    names = []
    for ch in CHAMPIONS:
        if bitmask & (1 << ch["idx"]):
            names.append(ch["name"])
    return names


def summarize_traits(trait_counts: Tuple[int, ...]) -> str:
    parts = []
    for ti, c in enumerate(trait_counts):
        if c > 0:
            parts.append(f"{TRAITS[ti]}={c}")
    return ", ".join(parts)


# ============================================================
# --------------------------- MAIN ---------------------------
# ============================================================
start_global = time.time()
logger.info(
    "=== START Meet-in-the-Middle | N=%d | TEAM_SIZE=%d | Raw Combos=C(%d,%d)=%s ===",
    N,
    TEAM_SIZE,
    N,
    TEAM_SIZE,
    f"{math.comb(N, TEAM_SIZE):,}",
)

# Phase 1
left_sigs = enumerate_half(LEFT, "LEFT")

# Phase 2
right_sigs = enumerate_half(RIGHT, "RIGHT")

# Phase 3: Combine
combine_start = time.time()
best_teams = combine_and_search(left_sigs, right_sigs)
combine_dur = time.time() - combine_start

total_dur = time.time() - start_global

logger.info(
    "\n=== FERTIG in %.2f min (Combine Phase %.2f min) ===",
    total_dur / 60,
    combine_dur / 60,
)
logger.info("Top %d Teams:", len(best_teams))

for rank, bt in enumerate(best_teams, start=1):
    names = bitmask_to_team(bt.bitmask)
    logger.info(
        "%2d. Score=%.2f Cost=%2d High=%d | Traits: %s",
        rank,
        bt.score,
        bt.cost,
        bt.high,
        summarize_traits(bt.trait_counts),
    )
    logger.info("    %s", ", ".join(names))

print_score_decomposition(best_teams, k=min(5, len(best_teams)))

logger.info(
    "\nHinweis: Passe compute_score() und LEVEL_WEIGHT / Parameter an dein Original an,"
    " damit die Score-Werte exakt den bisherigen entsprechen."
)
