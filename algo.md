# Verwendete Algorithmen im Projekt River Crossing

Dieses Dokument beschreibt die in diesem Projekt verwendeten Planungs- und Lernalgorithmen:
- Wie sie funktionieren
- Welche Hyperparameter genutzt werden
- Wie sie in die Projektstruktur eingebettet sind

## 1. Problemrahmen und gemeinsame Bausteine

Alle Verfahren arbeiten auf derselben Umgebung:
- Diskreter Grid-Zustandsraum (nx x ny)
- 8er-Aktionsraum (Moore-Nachbarschaft)
- Start-/Zielzustand mit harter Docking-Bedingung
- Stroemungsfeld aus der Umgebungs-Konfiguration (`constant` oder `gaussian`)

Die Umgebung wird durch `RiverEnvironment` bereitgestellt und erzwingt:
- Start links vom Ziel
- Nur diagonale Abfahrt vom Start
- Zielerreichung nur bei diagonaler Endaktion
- Landzellen links vom Start und rechts vom Ziel als ungueltig

### Kostenfunktion

Der Basisausdruck ohne Vorgeschichte lautet:

c(s,a) = alpha * t(s,a) + beta * E(s,a)

Bei aktivierter Traegheitskomponente (inertia > 0) kommt ein Drehkostenterm hinzu:

c(s, a, hist) = alpha * t(s,a) + beta * E(s,a) + turn_penalty * (1 - cos theta) / 2

Mit:
- t(s,a): Zeitkosten (Euklidische Aktionslaenge)
- E(s,a): Energiekosten relativ zur Stroemung
- alpha, beta: globale Gewichte aus der Algorithmus-Konfiguration
- turn_penalty: Skalierungsfaktor fuer die Drehkostenkomponente
- cos theta: Kosinus des Winkels zwischen gemittelter Vorgeschichte (hist) und aktueller Aktion

Standardwerte in `configs/algorithm.yaml`:
- common.alpha = 1.0
- common.beta = 1.0
- common.inertia = 2
- common.turn_penalty = 1.0

Hinweis zur UI:
- Der UI-Regler Impact setzt zur Laufzeit beta (impact_override) und veraendert so den Energieanteil fuer alle Algorithmen gleichzeitig.

### Traegheitsbasierte Drehkosten

`inertia` bestimmt, wie viele zurueckliegende Aktionen den Referenz-Heading bilden:
- inertia = 0: kein Heading-Gedaechtnis, Drehkosten immer 0
- inertia = k: Durchschnitt der letzten k Aktionen bildet den Referenzvektor

Die Drehkosten werden ueber `TurnCost` und `CostFunction.with_history(state, action, history)` berechnet. Dijkstra, A*, Weighted A* und Q-Learning repraesentieren Knoten als `(State, history_tuple)`, um die Vorgeschichte im Suchgraph explizit zu verwalten. Dynamic Programming ignoriert die Vorgeschichte (Markov); APF nutzt direkt `cost_fn(state, action)`.

## 2. Dijkstra

### Idee und Funktionsweise

Dijkstra sucht den kostenoptimalen Pfad im gewichteten Graphen ohne Heuristik.

Ablauf:
1. Prioritaetswarteschlange mit aktuellem g-Wert
2. Immer den billigsten offenen Zustand expandieren
3. Nachbarn mit Kostenfunktion relaxieren
4. Vorgaenger speichern zur Pfadrekonstruktion

Docking-Integration:
- Uebergaenge in das Ziel werden nur akzeptiert, wenn die Endaktion die Docking-Regel erfuellt.

Inertia-Erweiterung:
- Knoten sind `(State, history_tuple)` Paare.
- Kantenkosten werden ueber `cost_fn.with_history(state, action, history)` berechnet.
- Das billigste Ziel-Arrival ueber alle Histories wird als Loesungsknoten gewaehlt.

Eigenschaften:
- Voll optimal (bei nichtnegativen Kosten)
- Referenzverfahren fuer delta_j in der Evaluation

### Hyperparameter

Keine algorithmusspezifischen Hyperparameter (`dijkstra: {}` in `algorithm.yaml`).

## 3. A*

### Idee und Funktionsweise

A* priorisiert mit:

f(s) = g(s) + h(s)

Im Projekt wird A* als Weighted A* mit weight = 1.0 umgesetzt.

Heuristik:
- Euklidische Distanz zum Ziel
- Skaliert mit alpha aus der Kostenfunktion
- Damit bleibt die Heuristik bei beta >= 0 als untere Schranke nutzbar

Docking-Integration:
- Zielkanten mit ungueltiger Endaktion werden verworfen.

Inertia-Erweiterung:
- Knoten sind `(State, history_tuple)` Paare; Kantenkosten ueber `with_history()`.

### Hyperparameter

Aus `configs/algorithm.yaml`:
- a_star.heuristic = euclidean

## 4. Weighted A*

### Idee und Funktionsweise

Weighted A* nutzt:

f(s) = g(s) + w * h(s), mit w >= 1

Wirkung:
- Hoeheres w: aggressivere Heuristik, oft schneller, potentiell suboptimal
- w = 1: identisch zu A*

Docking-Integration wie bei A*:
- Ungueltige Zielanfluege werden ausgeschlossen.

Inertia-Erweiterung:
- Wie A*: Knoten sind `(State, history_tuple)` Paare.

### Hyperparameter

Aus `configs/algorithm.yaml`:
- weighted_a_star.heuristic = euclidean
- weighted_a_star.weight = 1.5

## 5. Dynamic Programming (Value Iteration)

### Idee und Funktionsweise

Das Problem wird als deterministisches MDP geloest.

Bellman-Update:

V_{k+1}(s) = min_a [ c(s,a) + gamma * V_k(T(s,a)) ]

Implementierungsdetails:
- Sweeps ueber alle Grid-Zustaende
- V(goal) = 0
- Policy entsteht als greedy argmin ueber den aktuellen V-Wert
- Nach Konvergenz wird ein Pfad aus der Policy extrahiert

Docking-Integration:
- Uebergaenge ins Ziel mit ungueltigem Endwinkel werden ignoriert.

Hinweis zu Inertia:
- DP ist ein Markov-Verfahren; Vorgeschichte wird ignoriert und Drehkosten sind effektiv 0.

### Hyperparameter

Aus `configs/algorithm.yaml`:
- dynamic_programming.gamma = 1.0
- dynamic_programming.tolerance = 1.0e-6
- dynamic_programming.max_iterations = 10000

## 6. APF (Artificial Potential Field)

### Idee und Funktionsweise

APF nutzt ein Potentialfeld:

Phi(s) = Phi_att(s) + Phi_flow(s)

Mit:
- Phi_att: attraktive Komponente Richtung Ziel/Andock-Vorbereich
- Phi_flow: Stroemungseinfluss

Diskretisierung:
- Pro Schritt wird die gueltige Aktion mit bestmoeglicher Abwaertsrichtung im Potential gewaehlt.
- Bei Gleichstand werden naechstes Potential und Aktionstupel zur Stabilisierung genutzt.

Docking-spezifische Erweiterungen im Projekt:
- Gueltige Zielvorgaenger (ein Schritt vor dem Ziel mit erlaubter Endaktion) werden explizit berechnet.
- Ein Guidance-Target fuehrt den Agenten in den gueltigen Anfahrkorridor.
- Zusatzstrafe auf Distanz zu gueltigen Zielvorgaengern.
- Lokale Minima/Plateaus/Zyklen werden erkannt und als Fehlschlag terminiert.

### Hyperparameter

Aus `configs/algorithm.yaml`:
- apf.k_att = 1.0
- apf.lambda_flow = 0.2

Interne Grenzen:
- max_steps standardmaessig = 4 * nx * ny (falls nicht gesetzt)

Hinweis zu Inertia:
- APF nutzt direkt `cost_fn(state, action)` ohne Vorgeschichte; Drehkosten sind effektiv 0.

## 7. Q-Learning (tabellarisch)

### Idee und Funktionsweise

Q-Learning lernt Aktionswerte Q(s,a) off-policy.

Update-Regel:

Q(s,a) <- Q(s,a) + learning_rate * (reward + gamma * max_a' Q(s',a') - Q(s,a))

Aktionswahl:
- epsilon-greedy

Wichtige Implementationsdetails:
- Aktionen, die das Ziel mit ungueltiger Endaktion erreichen, werden schon bei der Aktionsmenge ausgeschlossen.
- Bei inertia > 0 sind Q-Tabellen-Schluessel `(State, history_tuple)`; die extrahierte Policy enthaelt Fallback-Eintraege fuer unbekannte Histories.
- Nach dem Training wird eine greedy Policy extrahiert und separat als Inferenz-Rollout bewertet.

Reward Shaping im Projekt:
- Basis: -cost_fn(state, action)
- Fortschrittsbonus in Richtung Ziel/Andockkorridor
- Wiederbesuchsstrafe gegen Schleifen
- Bonus fuer gueltigen Vor-Andockzustand
- Zusaetzlicher goal_reward bei erfolgreichem Docking

Adaptive Exploration:
- Epsilon wird nur reduziert, wenn die Erfolgsquote im success_window mindestens min_success_rate_for_decay erreicht.

### Hyperparameter

Aus `configs/algorithm.yaml`:
- q_learning.episodes = 1200
- q_learning.learning_rate = 0.1
- q_learning.gamma = 0.99
- q_learning.epsilon_start = 1.0
- q_learning.epsilon_end = 0.02
- q_learning.epsilon_decay = 0.998
- q_learning.goal_reward = 50.0
- q_learning.progress_reward_scale = 2.5
- q_learning.revisit_penalty = 0.3
- q_learning.approach_bonus = 5.0
- q_learning.max_steps_per_episode = 6400
- q_learning.min_success_rate_for_decay = 0.05
- q_learning.success_window = 100
- q_learning.max_snapshots = 200

Geteilte Hyperparameter via `common:`:
- common.inertia wirkt als Q-Tabellen-Schluesseldimension (0 deaktiviert den Verlauf)
- q_learning.progress_reward_scale = 2.5
- q_learning.revisit_penalty = 0.3
- q_learning.approach_bonus = 5.0
- q_learning.max_steps_per_episode = 6400
- q_learning.min_success_rate_for_decay = 0.05
- q_learning.success_window = 100
- q_learning.max_snapshots = 200

## 8. Einbettung in das Projekt

### Konfigurationsschicht

- `configs/env.yaml`: Grid, Stroemung, Start/Ziel, Docking
- `configs/algorithm.yaml`: globale Kostengewichte + algorithmenspezifische Hyperparameter
- `configs/experiment.yaml`: Seeds, Algorithmenliste, Metriken

### Ausfuehrungsschicht

Zentrale Orchestrierung liegt in `experiments/runner.py`:
- Laedt Konfigurationen
- Baut fuer jeden Lauf `RiverEnvironment` und `CostFunction`
- Fuehrt je nach Algorithmus den passenden Solver aus
- Erfasst Zeiten (planning/training/inference)
- Vereinheitlicht Ergebnisse in `ExperimentRunRecord`

Algorithmus-Dispatch:
- dijkstra -> `dijkstra(...)`
- a_star -> `astar_from_config(...)`
- weighted_a_star -> `weighted_astar_from_config(...)`
- dynamic_programming -> `value_iteration(...)` + `extract_path(...)`
- apf -> `apf_from_config(...)`
- q_learning -> `q_learning_from_config(...)` + `rollout_policy(...)`

### Evaluationsschicht

`experiments/evaluator.py` berechnet:
- Aggregation pro Algorithmus
- Erfolg, Laufzeiten, mittlere Kosten
- delta_j relativ zum Referenzalgorithmus (standardmaessig dijkstra)

### UI- und CLI-Einbindung

- CLI (`main.py`):
  - check: Konfiguration validieren
  - ui: Desktop-UI starten
- UI (`ui/visualization.py`):
  - Einzelrun oder Batch ausfuehren
  - Ergebnisdarstellung und Vergleich
  - Flow-Vektor und Impact (beta-Override) interaktiv anpassen

## 9. Praktische Interpretation der Verfahren

- Dijkstra: beste Referenz fuer optimale Kosten, aber oft hoher Suchaufwand
- A*: guter Standard, wenn Heuristik passt
- Weighted A*: schneller bei moeglicher Suboptimalitaet
- Value Iteration: policy-orientiert, gut fuer vollstaendige MDP-Sicht
- APF: sehr schnell und anschaulich, aber Risiko lokaler Minima
- Q-Learning: flexibel und robust bei Shaping, braucht Trainingszeit

Fuer wissenschaftlichen Vergleich im Projekt:
- Kostenqualitaet via total_cost und delta_j
- Effizienz via plan_time, training_time, inference_time
- Robustheit via mehrere Seeds und veraenderbare Stroemung/Impact
