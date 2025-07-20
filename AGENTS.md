# AGENTS.md

This file provides guidance for AI agents (for example OpenAI Codex) when working on this repository. The project consists of a single Python script that performs a meet-in-the-middle search on TFT Set 14 data fetched from Riot's Data Dragon.

---

## 1. Project Structure

```
/ (repository root)
├─ Rech.py    # optimizer logic and entry point
├─ README.md  # usage and development notes
├─ AGENTS.md  # this guide for contributors
└─ .flake8    # flake8 configuration
```

`Rech.py` downloads champion and trait data, defines two parameter sets (`P1`, `P2`) and contains helper functions for rescoring and progress reporting. Parameters can now be overridden via CLI flags.

---

## 2. Coding Conventions

- Target **Python 3.10+**.
- Format code using `black`.
- Lint with `flake8` (max line length 100).
- Group imports: standard library, third‑party, local.
- Use type hints where practical.
- Prefer the `logging` module over `print` for any new output.

---

## 3. Testing Requirements

No automated tests exist yet. If you add new functionality, create tests under `tests/` using `pytest`.

---

## 4. Pull Request Guidelines

- Commit messages may start with `feat:`, `fix:`, `docs:` and similar prefixes.
- PR descriptions should explain the motivation, the changes and how they were tested.

---

## 5. Programmatic Checks

Run the following before submitting a PR:

```bash
black Rech.py --check
flake8 Rech.py
```

If a `tests/` directory is present also execute `pytest`.

---

## 6. Customisation Tips

- Switch scoring presets using the `--param-set` CLI flag (or change `ACTIVE_PARAM_SET`).
- Adjust scoring behaviour via the `ScoreParams` dataclass or by modifying `compute_team_components()`.
- Champion and trait data is downloaded from Data Dragon and cached in `dd_version_cache.json` and `tft_set14_cache.json.gz`.
- The script automatically rescans `tft_full_bruteforce_results.json` or `.csv` files and prints score decompositions.
- CLI flags like `--team-size`, `--top-k` and `--seed` allow quick parameter experiments.


Wenn möglich aollten all diese Punkte si vollständig wie möglich umgesetzt werden!:

1. Datenbeschaffung & Fallbacks
Mehrere DDragon Pfade + Retry: Aktuell nur einfache Schleife – ergänze exponentielles Backoff oder fixe kurze Retries pro URL.
CommunityDragon Parsing robuster: Überprüfe Struktur ("sets" -> "14"), logge wenn leer; fallback auf alternative Branch (raw.communitydragon.org/pbe).
Eingebetteten Snapshot versionieren: Feld snapshot_version + Datum; bei erfolgreichem Live-Fetch Merge + erneutes Persistieren als neue Embedded-Datei.
Diff-Log führen: Veränderungen (neu / entfernt / geänderte Traits oder Kosten) in tft_set14_diff.log (Maschinen- & menschenlesbar).
Validierung: Prüfen, ob jeder Champion min. 1 Trait besitzt – sonst markieren / filtern. Aktuell viele „leere“ Traits möglich (Test-/NPC-Einträge).
Filteroptionen: CLI Flags (--include-summons, --include-props, --exclude-npc) um Platzhalter / Summons / Props auszuschließen.
Normalization Mapping: Einheitliche Normalisierung von Schreibweisen (LeBlanc vs Leblanc, 'Kog'Maw'), bevor du Keys erzeugst.
Zwangs-Update Flag: --force-refresh ignoriert Cache & Snapshot.
Integrity Check Cache: Hash (SHA256) der geladenen JSON im Cache speichern – bei Korruption fallback ignorieren.
Graceful Partial Merge: Wenn ein Trait fehlt, aber Champions ihn referenzieren → Trait mit leeren Breakpoints initialisieren (und warnen).

2. Trait-Breakpoints & Gewichtung
Ordinal vs. Absolut entkoppeln: Breakpoint (z.B. 9) nicht direkt in LEVEL_WEIGHT suchen; stattdessen ordinales Level (1., 2., 3. …).
Fehlende Breakpoint-Gewichte dynamisch ergänzen: Interpolation (linear oder progressiv) zwischen bekannten Gewichten.
Marginal Value Tabelle: Für jedes zusätzliche Trait-Vorkommen den incremental gain speichern → für Bounds.
Trait Cap: Optionales Hard Cap (z.B. ignorieren von überzähligen Einheiten nach höchstem Breakpoint).
Synergie/Antisynergie Modell: Bonus oder Malus für bestimmte Trait-Kombinationen (Matrix oder Regel-DSL).
Trait Priorisierung: Liste „High Value Traits“ → bevorzugtes Pruning, wenn noch nicht erreicht.
Partielle Trait-Sättigung: Früh im DFS prüfen, ob weitere Einheiten dieses Traits keinen Score-Zuwachs mehr bringen → Branch Cut.

3. Scoring-Modell
Gold Over-Spend Penalty: Auch Überschreiten eines Zielbudgets minimal negativ werten (Soft Cap).
High-Cost Limit: Hard oder Soft-Limit für Anzahl high Einheiten (z.B. Diminishing Returns: Bonus = f(high_count)).
Doppelte Kostenfunktion: Lineares + leicht konkaves Element (z.B. a*cost - b*cost^2 / N) für feinere Balance.
Trait Diversitätsbonus: Leichter Bonus für Anzahl verschiedener aktiver Traits (Vermeidung von Overstack).
Upgrade/Variant Handling: Falls Varianten (mit +) implizit „ersetzen“, differenzierter Score (z.B. Basis + delta).
Normalisierung Score Komponenten: Vor dem Addieren jede Komponente z‑score oder min‑max skalieren (robuster bei Parameter-Sweeps).
Explainability: JSON Dump pro Team mit einzelnen Addenden (traits: {...}, gold: x, high: y).

4. Enumeration / Suchraum
Nur exakte Halbgrößen enumerieren statt 0..max_size → drastische Reduktion.
Optional Bandbreite (z.B. Halbgröße ±1) über CLI.
Upper-Bound Pruning im DFS: Optimistische Schätzung (trait_val + rest_slots*max_increment + estimated high + gold).
Dominance Key erweitern: Key = (size, cost_bucket, high, top3_trait_counts_sorted) → höheres Pruning.
Stufenweises Pruning: Erst Sammler, dann top-X pro (size, cost_range) behalten.
Meet-in-the-Middle Partitionierung: Nach Feature (z.B. Traits) statt halbem Index – Minimiert Überschneiden und verbessert Boundqualität.
Bitset Repräsentation: Base IDs → 64-bit Mask (wenn ≤64 Basen) für extrem schnellen Variant Check.
Vorberechnete Trait-Vektoren: Jeder Champion -> int16 Array; Merge per NumPy Vektoraddition.
Caching Halbsignaturen: Persistiere halfA/halfB in Datei (mit Hash der Parameter/Championliste), Reuse bei Parameteränderungen.
Iterative Deepening: Erst kleine Depth (Partial Teams), heuristisch erweitern nur vielversprechende Nodes.
Heuristische Seeds: Greedy Build (füge bestes marginal trait gain hinzu) → lokale Nachbarschaft (Swaps) → MIM nur auf diesen Kern.

5. Performance / Numba / Speicher
NumPy Vorbereitung: trait_counts als np.ndarray(dtype=int16) ablegen; Numba-Funktion erwartet gleiche Form → kein Python Overhead.
Batch Merge: Statt verschachtelter Python-Schleife, (A x B)-Gitter in Vektoren und in Chunks Numba-kalkulieren.
Memory Pooling: Re-Use Arrays (statt jedes Mal np.empty).
Parallelisierung: Multiprocessing / joblib für Combine (Chunking halfA).
Lazy Sorting: In Combine nur B subset mit passender size vorfiltern (Dict: size -> Liste).
Heap statt dauernd sortieren: heapq für Top-K statt best.sort() nach jedem Insert.
Adaptive Top-K: Start mit kleiner K (z.B. 20) und erhöhe, wenn Suchraum schrumpft.
Fast Cost Bucket: Precompute cost_bucket Feld in HalfSignature zur schnelleren Dominance-Kontrolle.
Pruning ab minimalem Score: Globaler Floor aktualisieren, Bound check in DFS/Combine.
Vectorized Marginal Gains: Tabelle (trait, count) -> incremental; vermeidet erneutes Summieren über alle Traits.

6. Parametertuning / Experimente
CLI Parameter Override: --param-set P2, --gold-weight 0.6, etc.
Grid + Random Hybrid: Erst grober Grid, dann Random Sampling in Top-Hypercube.
Bayesian Optimization (z.B. skopt) auf Durchschnitts-Score definierter Benchmark-Teams.
Fitness Stabilitätstest: Mehrere zufällige Team-Samples → Varianz messen; Parameter wählen, die Varianz reduzieren (robust).
Persistenz bester Parameter: JSON best_params.json mit Score Historie (timestamped).
Auto-Stop: Abbruch Tuning wenn Verbesserung < ε über N Iterationen.

7. Re-Scoring / Ergebnisverarbeitung
Mehr Inputformate: Unterstützung für TSV, Plain Lines, oder Pickle.
Fehlertoleranz: Fuzzy Matching (Casefold, Entfernen von Sonderzeichen) bei Champion-Namen.
Trait Coverage Bericht: Bei Re-Score zeigen: Aktivierte Breakpoints pro Team.
Delta Analyse: Anzeige Score alt vs. neu, wenn Parameter geändert.
Batch Re-Score: Ordner scan (Pattern *.teams.json).
Export Ranking: CSV & Markdown (README_teams.md).
Score Normalisierung: Zusätzlich Rang-Percentile ausgeben.

8. Logging & Monitoring
Verbosity Levels: --verbose 0|1|2.
Structured Logging: JSON Logs (Start Zeit, Parameter, Count enumerated, pruned).
Progress Metriken: In Combine: Rate (teams/sec), geschätzte Restzeit.
Warnungen klassifizieren: [WARN_DATA], [WARN_NAME] etc.
Stats Summary: Am Ende: enumerierte Halbs, pruned ratio, combine iterations, top_k floor.
Telemetry Hooks: Konsolidierte Kennzahlen in separate run_stats.json.

9. CLI / UX
argparse Integration: Alle Parameter + Flags (z.B. --team-size, --seed, --no-sweep, --top-k 50).
Konfig-Datei: --config config.yml lädt Parameter (YAML).
Dry-Run Mode: Lädt Daten & validiert, ohne Suche.
Quiet Mode: Nur Endergebnis + JSON Pfad.
Colorized Output (optional) via colorama für bessere Lesbarkeit.

10. Datenmodell / Struktur
Class Layer: Champion, Trait, DataStore, Optimizer, Scorer für klare Verantwortlichkeiten.
Hash Keys: Eindeutige numeric IDs für Champions & Traits (Mapping) → schneller & kompakter.
Immutable Structures: Nutzung von NamedTuple/dataclass(frozen=True) für inevitables (HalfSignature) zur Sicherheitsoptimierung.
Serialization: Separate modulare Funktionen für Laden/Speichern (io_utils.py).

11. Qualitätssicherung / Tests
Unit Tests: (a) Trait Level Table Korrektheit, (b) Score Monotonie bei Hinzufügen von Trait-Einheit, (c) Variant Exclusion.
Regression Snapshot: Fixiertes Eingabedataset + bekannte Top-K → Hash des Ergebnisses vergleichen nach Codeänderungen.
Property Tests: Hypothesis: Score darf nicht sinken, wenn nur neuer Trait-Breakpoint erreicht und sonstige Kosten gleich.
Benchmark Script: --benchmark führt definierte Szenarien aus (CPU Zeit, enumerations/s).

12. Reproduzierbarkeit
Determinismus: Setze random.seed(seed) am Anfang; seed via CLI.
Version Stamping: Schreibe run_meta.json (Git Commit Hash, Timestamp, Parameterversion).
Environment Check: Logge Python-Version, numba-Version, numpy-Version.

13. Fehlerbehandlung & Robustheit
Explizite Exceptions: Eigene Exception-Klassen (DataLoadError, ScoringError, PruningError).
Graceful Abort: Tastaturabbruch fängt auf, schreibt bis dahin gefundene Top-K in Datei.
Timeout Handling: Einzelne URL Timeout → sofort weiter, nicht gesamter Fail.
Sanity Checks: If TEAM_SIZE > len(unique_bases) → Warnung/Abruf stoppen.

14. Erweiterte Funktionalität
Teil-Rescore ohne Neu-Laden: Eingabe von Teamlisten direkt in CLI.
Live Mode: Automatisches periodisches Neuladen & Diff (Watch).
Explain JSON: --explain-champ CHAMP zeigt marginal Trait Gains bei Hinzufügen.
What-if Tool: --simulate remove=ChampionA add=ChampionB
Synergie Graph: Generiere Matrix Trait vs. Champion Count (CSV).
Export für UI: Kompakte JSON Struktur (IDs, Score, traitBreakpointsHit).
Score Decomposition Diagramm: (Falls Matplotlib optional) Balkendiagramm pro Team (abschaltbar).
API Mode: Optional Flask/FastAPI Endpoint /optimize.

15. Performance Feintuning
Pre-Sort Champions: Nach (trait richness, cost) für bessere Branching Orders.
Heuristic Ordering: In DFS zuerst Champions mit höchster erwarteter marginaler Trait-Steigerung.
Bitpacked Trait Counts: Wenn Traitanzahl ≤ 32, encode counts in 2 Bits (nur bei kleinen Caps) – experimentell.
Adaptive Bucket Size: Dominance Cost-Bucket dynamisch (kleinere Bins bei niedrigen Kosten).
Pruning Schwellwert adaptiv: Floor erhöht sich progressiv (Simulated Annealing Stil).
Memory Guard: Wenn Signaturen > Limit → random reservoir sampling behalten statt alles.

16. Dokumentation
README: Ziele, Datenschema, Parameterübersicht, Beispiele.
In-Code Param Docs: Jede Gewichtung mit Kurzbeschreibung (Einheit, erwartete Range).
CHANGELOG: Verfolge Änderungen an Scoring-Formel / Parametern.
Design Notes: Kurzer Architektur-Abschnitt (MIM, Pruning Strategie, Score Rationale).

17. Sicherheit / Stabilität
Timeout Konstante zentral: Eine Stelle für alle HTTP timeouts.
Rate-Limit Handling: Falls 429 → kurze Sleep & nächster Pfad.
Input Sanitizing bei externen Teamlisten.

18. Ergebnisqualität / Metamodelle
Diversity Set: Führe Auswahl-Algorithmus, der Top-K ohne starke Überschneidungen (Jaccard-Threshold) ausspuckt.
Pareto-Front: Neben Score auch (Cost, Trait-Diversität) – zeige nicht dominierte Teams.
Stabilitätsanalyse: Rauschen in Parametern (±ε) – Score-Rangstabilität messen.

19. Automatisiertes Parametertuning (vertieft)
Weighted Rank Correlation: Ziel: Maximiert Spearman ρ zwischen verschiedenen Random Splits von Teams (Robustheit).
Multi-Objective Tuning: Score Maximierung + Varianz Minimierung → Pareto Filter.
Early Stop Tuner: Keine Verbesserung der besten durchschnittlichen Top-Score über X Iterationen.

20. Zukunft / Skalierung
GPU Pfad: Optional, mit CuPy oder Numba CUDA für massives Pairwise Merge.
Distributed MIM: Sharding der halben Signaturen über mehrere Prozesse/Hosts.
Incremental Updates: Bei kleinem Daten-Diff nur betroffene Signaturen neu berechnen.
Cache Layer für TraitVal: Memoization (frozenset idxs → trait_val) bei wiederholter Nutzung (Achtung Speicher).

Bonus: Micro-Patches (Sofort umsetzbar)
(a) f-String Fix war schon korrigiert – ok.
(b) cost_bucket = sig.cost // 1 → einfach sig.cost oder sinnvolles Bucket.
(c) trait_names_sorted wird nicht genutzt → entfernen.
(d) LEVEL_LOOKUP ungenutzt → entfernen oder bei Decomposition anzeigen (erreichter Breakpoint).
(e) iter_total in Combine zählt, aber nicht ausgegeben → am Ende loggen (print(f"Combine Iterationen: {iter_total}")).
---
