# Arbeitsplan – River Crossing

## Ziel
Implementierung und vergleichende Evaluation mehrerer Pfadplanungsalgorithmen für ein Schiff im Fluss mit Strömung bei Einhaltung des Anfahrwinkels (30°–60°), mit Fokus auf minimaler Überquerungszeit und Reproduzierbarkeit.

## Reihenfolge der Umsetzung

1. Projektgrundstruktur und Konfigurationsbasis aufsetzen
- Verzeichnisstruktur gemäß README anlegen (`configs`, `core`, `algorithms`, `experiments`, `ui`, `analysis`).
- Zentrale Konfigurationsdateien erstellen: `env.yaml`, `algorithm.yaml`, `experiment.yaml`.
- Einheitliches Laden/Validieren von Konfigurationen implementieren.

2. Kernmodell des Problems implementieren
- Diskretes 2D-Gitter und Zustandsrepräsentation definieren.
- 8er-Aktionsraum und gültige Zustandsübergänge (`clip`) umsetzen.
- Strömungsmodell abstrahieren (zuerst konstante Strömung, später variabel).
- Start-, Ziel- und Ufer-/Anlegeorientierung formal im Environment abbilden.

3. Kosten- und Nebenbedingungslogik umsetzen
- Kostenfunktion `c(s,a) = alpha * t(s,a) + beta * E(s,a)` implementieren.
- Zeitkosten und Energieanteil als austauschbare Komponenten bauen.
- Anfahrwinkelbedingung als harte Zielvalidierung implementieren.

4. Referenzverfahren implementieren (Ground Truth)
- Dijkstra implementieren und als primäre Referenz verifizieren.
- Dynamic Programming (Value Iteration) ergänzen und Referenzwerte vergleichen.
- Konsistenztests: beide Verfahren sollen auf identischen Szenarien gleiche Optima liefern (innerhalb numerischer Toleranz).

5. Heuristische Suchverfahren implementieren
- A* mit zulässiger Heuristik (z. B. euklidische Distanz) umsetzen.
- Weighted A* mit Parameter `w >= 1` implementieren.
- Parameterstudie für `w` vorbereiten (Qualität vs. Laufzeit).

6. Potentialfeldverfahren implementieren
- APF mit attraktivem Potential und Strömungsterm aufsetzen.
- Diskrete Aktionswahl als Gradientenapproximation implementieren.
- Umgang mit lokalen Minima dokumentieren/absichern.

7. Reinforcement Learning implementieren
- Q-Learning-Agent mit `alpha`, `gamma`, `epsilon` entwickeln.
- Trainingsschleife mit Seed-Kontrolle, Logging und Policy-Export implementieren.
- Konvergenzmetriken (Reward-Verlauf, Stabilität, Varianz über Seeds) erfassen.

8. Experiment-Pipeline und Messung aufbauen
- `runner.py` für systematische Batch-Läufe implementieren.
- `evaluator.py` für Metriken: `J`, `Delta J`, Schritte, Laufzeiten, Winkelvalidität.
- Einheitliches Ergebnisformat (z. B. CSV/JSON) festlegen und speichern.

9. Visualisierung erstellen
- Gitter, Strömungsvektoren, Start/Ziel und Trajektorien darstellen.
- Vergleichsansicht für mehrere Algorithmen im gleichen Szenario bereitstellen.
- Optional: Animation der Trajektorie integrieren.

10. Robustheits- und Sensitivitätsanalysen durchführen
- Variation von Strömungsstärke/-richtung, Start/Ziel und Gittergröße.
- Einfluss des Anfahrwinkels auf Pfadstruktur und Rechenzeit untersuchen.
- Ergebnisse pro Algorithmus konsistent gegenüberstellen.

11. Ergebnisaufbereitung und Dokumentation abschließen
- Analyseplots/Tabellen in `analysis` erzeugen.
- Reproduzierbare Runs über Konfigurationen und Seeds dokumentieren.
- Abschlussbericht mit zentralen Erkenntnissen (Qualität, Effizienz, RL-Verhalten, Robustheit) erstellen.

## Definition of Done
- Alle genannten Algorithmen lauffähig und über identische Szenarien vergleichbar.
- Referenzoptimum verfügbar (Dijkstra/DP) und `Delta J` für alle Methoden berechenbar.
- Vollständige Messdaten, Visualisierungen und reproduzierbare Konfigurationen vorhanden.
- Anfahrwinkelbedingung wird in allen relevanten Verfahren korrekt eingehalten/geprüft.
