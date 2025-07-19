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
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# ---------------------- KONFIGURATION -----------------------
# ============================================================

TEAM_SIZE = 8
HIGH_COST_THRESHOLD = (
    4  # z.B. Cost >=4 als "High Cost" (anpassen falls anders gewünscht)
)
TOP_K = 25  # wie viele beste Teams am Ende anzeigen
LOG_INTERVAL_SECONDS = 2.0  # Fortschritt-Ausgabe Frequenz
ABORT_AFTER_SECONDS = None  # Optional: harte Abbruchzeit (None = aus)

# Gewichtung Trait-Level – Beispiel (Alternative Scoring aus deinem Wunsch)
# Du kannst die Werte hier verändern / erweitern.
LEVEL_WEIGHT = {
    0: 0.0,
    1: 0.0,  # Falls einzelne Traits schon bei 1 etwas geben -> anpassen
    2: 25.0,
    3: 37.5,
    4: 50.0,
    5: 62.5,
    6: 75.0,
    7: 87.5,
    8: 100.0,
}

# Gold-Utilisierung: Beispielterm – bitte an deine bisherige Formel angleichen.
# Der Score setzt sich unten aus (Summe Trait-Level-Werte) + evtl. Bonus zusammen.
# Passe diese Parameter an deine vorherige Definition an, damit Zahlen wieder passen.
GOLD_UTIL_LINEAR = 0.15  # Steigung für "mehr eingesetztes Gold"
GOLD_UTIL_OFFSET = 0.0  # Basis
LEFTOVER_PENALTY = 0.05  # Strafe für ungenutztes Gold (falls du das so hattest)
MAX_POSSIBLE_GOLD = (
    None  # Wenn du ein Gold-Limit hattest, trage es hier ein (z.B. 25). Sonst None.
)

HIGH_COST_UNIT_BONUS = 0.4  # Bonus * pro* High-Cost-Unit (Beispiel)

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


# ============================================================
# ------------------ CHAMPION VORVERARBEITEN -----------------
# ============================================================
@dataclass(frozen=True)
class Champion:
    idx: int
    name: str
    cost: int
    traits: Tuple[int, ...]  # trait indices
    high_cost: int


CHAMPIONS: List[Champion] = []
for idx, (name, cost, traits) in enumerate(CHAMPION_DATA):
    trait_ids = tuple(TRAIT_INDEX[t] for t in traits)
    CHAMPIONS.append(
        Champion(
            idx=idx,
            name=name,
            cost=cost,
            traits=trait_ids,
            high_cost=1 if cost >= HIGH_COST_THRESHOLD else 0,
        )
    )

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
    """
    trait_counts: *gedeckelte* counts (<= TRAIT_CAP[ti])
    Score = Sum TraitValues + GoldUtil + HighCostBonus - ggf. LeftoverPenalty
    Passe diese Funktion so an, dass sie identisch zur bisherigen in deinem Bruteforce Code ist!
    """
    trait_sum = 0.0
    for ti, cnt in enumerate(trait_counts):
        cnt = min(cnt, TRAIT_CAP[ti])
        trait_sum += LEVEL_VALUE[ti][cnt]

    # Gold / Leftover:
    gold_util = 0.0
    leftover_penalty = 0.0
    if MAX_POSSIBLE_GOLD is not None:
        used = total_cost
        leftover = max(0, MAX_POSSIBLE_GOLD - used)
        gold_util = GOLD_UTIL_LINEAR * used + GOLD_UTIL_OFFSET
        leftover_penalty = LEFTOVER_PENALTY * leftover
    else:
        # Wenn kein Limit definiert: nur leichte Belohnung für höheres Cost (optional)
        gold_util = GOLD_UTIL_LINEAR * total_cost

    high_cost_bonus = high_cost_units * HIGH_COST_UNIT_BONUS

    return trait_sum + gold_util + high_cost_bonus - leftover_penalty


# ============================================================
# ---------------- SIGNATURE / KOMPRESSION -------------------
# ============================================================
Signature = namedtuple("Signature", ("size", "cost", "high", "trait_counts", "bitmask"))


def enumerate_half(champs: List[Champion], label: str):
    """
    Erzeugt alle Teilmengen Größe 0..TEAM_SIZE.
    Kompression:
        key = (size, high_cost, tuple(capped trait counts))
        value = Signature mit MAX cost (alle anderen überschrieben)
    """
    start = time.time()
    L = len(champs)
    # Prebuild arrays für Traits pro Champion
    trait_lists = [c.traits for c in champs]
    costs = [c.cost for c in champs]
    highs = [c.high_cost for c in champs]

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
            for idx_local in combo:
                ch = champs[idx_local]
                cost += costs[idx_local]
                highc += highs[idx_local]
                for ti in trait_lists[idx_local]:
                    if trait_counts[ti] < TRAIT_CAP[ti]:
                        trait_counts[ti] += 1
                # Globale Bitposition = real champion index
                bitmask |= 1 << ch.idx

            key = (k, highc, tuple(trait_counts))
            prev = compressed.get(key)
            if prev is None or cost > prev.cost:
                compressed[key] = Signature(
                    k, cost, highc, tuple(trait_counts), bitmask
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
            # Pre-caches
            a_counts = sa.trait_counts
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

                b_counts = sb.trait_counts
                # Merge trait counts (gedeckelt)
                merged = [0] * T
                for ti in range(T):
                    c = a_counts[ti] + b_counts[ti]
                    if c > TRAIT_CAP[ti]:
                        c = TRAIT_CAP[ti]
                    merged[ti] = c
                total_cost = a_cost + sb.cost
                high = a_high + sb.high
                if (MAX_POSSIBLE_GOLD is not None) and total_cost > MAX_POSSIBLE_GOLD:
                    continue

                score = compute_score(merged, total_cost, high)

                if score > best_floor:
                    improvements += 1
                    best_floor = score
                    bt = BestTeam(
                        score=score,
                        cost=total_cost,
                        high=high,
                        trait_counts=tuple(merged),
                        bitmask=sa.bitmask | sb.bitmask,
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


# ============================================================
# --------------- REKONSTRUKTION TEAMNAMEN -------------------
# ============================================================
def bitmask_to_team(bitmask: int) -> List[str]:
    names = []
    for ch in CHAMPIONS:
        if bitmask & (1 << ch.idx):
            names.append(ch.name)
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

logger.info(
    "\nHinweis: Passe compute_score() und LEVEL_WEIGHT / Parameter an dein Original an,"
    " damit die Score-Werte exakt den bisherigen entsprechen."
)
