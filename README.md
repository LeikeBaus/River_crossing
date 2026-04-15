# River Crossing – Path Planning in Flow Fields

## 1. Motivation

Ziel dieses Projekts ist die Untersuchung und der Vergleich verschiedener Pfadplanungsalgorithmen für die Navigation eines Schiffes in einem Fluss mit Strömung.

Das Szenario orientiert sich an der realen Fährverbindung zwischen zwei leicht versetzten Anlegestellen. Das Schiff startet an einem Punkt A und muss einen Zielpunkt B am gegenüberliegenden Ufer erreichen. Aufgrund der Lage der Anlegestellen ist eine direkte frontale Anfahrt nicht möglich. Stattdessen muss das Ziel in einem Winkelbereich von 30° bis 60° relativ zur Uferlinie angefahren werden.

Optimiert wird ausschließlich die Überquerungszeit.

## 2. Mathematische Problemformulierung

### 2.1 Zustandsraum

Die Umgebung wird als diskretes 2D-Gitter modelliert:

S = { (i, j) | i ∈ {0, …, N_x − 1}, j ∈ {0, …, N_y − 1} }

Jeder Zustand s ∈ S entspricht einer Position des Schiffes im Fluss.

Jedem Zustand ist ein Strömungsvektor zugeordnet:

u_flow : S → ℝ²

Im einfachsten Fall ist die Strömung konstant:

u_flow(s) = u₀ für alle s ∈ S

### 2.2 Aktionsraum

Das Schiff bewegt sich in einem 8-Nachbarschaftsmodell.

A = { a₁, …, a₈ }, wobei a_k ∈ ℤ².

Beispiel:

A = {
(0,1), (1,1), (1,0), (1,−1),
(0,−1), (−1,−1), (−1,0), (−1,1)
}

### 2.3 Zustandsübergang

Die diskrete Dynamik ist definiert durch:

s_{t+1} = T(s_t, a_t)

mit

T(s, a) = clip(s + a)

wobei clip sicherstellt, dass der Folgezustand innerhalb des Gitters liegt.

### 2.4 Kostenfunktion

Die Übergangskosten sind definiert als:

c : S × A → ℝ_{≥0}

Allgemeine Form:

c(s, a) = α · t(s, a) + β · E(s, a)

mit

α, β ∈ ℝ_{≥0}

t(s, a) = Zeitkosten
E(s, a) = Energieverbrauch

Eine mögliche Modellierung:

t(s, a) = ||a||₂

E(s, a) = || a − u_flow(s) ||₂²

Gesamtkosten eines Pfades π = (s₀, …, s_T):

J(π) = Σ_{t=0}^{T−1} c(s_t, a_t)

Gesucht ist ein optimaler Pfad

π* = argmin_π J(π)

### 2.5 Randbedindung: Anfahrtwinkel

Das Ziel darf nur erreicht werden, wenn der letzte Bewegungsvektor a_T einen Winkel θ im Bereich

30° ≤ θ ≤ 60°

relativ zur Uferlinie bzw. Zielorientierung erfüllt.

Formal:

θ = arccos( ⟨a_T, n⟩ / (||a_T|| · ||n||) )

mit n als Normalenvektor der Anlegestelle.

Zustände, die diese Bedingung nicht erfüllen, gelten nicht als gültige Zielzustände.

## 3. Zielsetzung

Ziel des Projekts ist:
- Implementierung mehrerer Planungsalgorithmen.
- Systematischer Vergleich hinsichtlich:
    - Lösungsqualität,
    - Rechenzeit,
    - Konvergenzverhalten (bei RL),
    - Robustheit gegenüber veränderter Strömung.

## 4. Algorithmische Ansätze

### 4.1 Dijkstra

Dijkstra berechnet die optimale Lösung durch vollständige Exploration des Zustandsgraphen mit monoton wachsender Kostengrenze.

Eigenschaften:
- garantiert optimale Lösung,
- dient als Referenz („Ground Truth“).

### 4.2 A*

A* verwendet eine Heuristik h(s), beispielsweise die euklidische Distanz zum Ziel.

Bewertungsfunktion:

f(s) = g(s) + h(s)

mit

g(s) = bisher akkumulierte Kosten,
h(s) ≤ J*(s) (zulässige Heuristik).

Bei admissibler Heuristik bleibt A* optimal.

### 4.3 Weighted A*

Modifizierte Bewertungsfunktion:

f(s) = g(s) + w · h(s), mit w ≥ 1.

Eigenschaften:
- schnellere Planung,
- kontrollierte Suboptimalität,
- Analyse des Trade-offs zwischen Optimalität und Laufzeit.

### 4.4 Dynamic Programming (Value Iteration)

Das Problem wird als deterministisches Markov Decision Process (MDP) formuliert.

MDP: (S, A, P, R, γ)

Übergangswahrscheinlichkeit:

P(s' | s, a) = 1, falls s' = T(s, a),
P(s' | s, a) = 0 sonst.

Reward:

R(s, a) = −c(s, a)

Bellman-Optimalitätsgleichung:

V*(s) = min_{a ∈ A} [ c(s, a) + γ V*(T(s, a)) ]

Value Iteration:

V_{k+1}(s) = min_{a ∈ A} [ c(s, a) + γ V_k(T(s, a)) ]

mit Diskontfaktor γ ∈ (0,1].

### 4.5 Artificial Potential Field (APF)

Das APF verwendet weiterhin ein Potential der Form

Φ(s) = Φ_att(s) + Φ_flow(s),

wird jedoch in der aktuellen Implementierung zusätzlich **andockungsbewusst** geführt. Statt das Schiff nur direkt auf das Ziel auszurichten, wird auch ein gültiger Vorbereich des Ziels berücksichtigt, aus dem der letzte Schritt die Winkelbedingung erfüllen kann.

Attraktives Potential:

Φ_att(s) = 1/2 · k_att · || s − s_target ||²

wobei s_target je nach Situation entweder das eigentliche Ziel oder ein gültiger Vorzustand des Ziels ist.

Strömungseinfluss:

Φ_flow(s) = −λ · ⟨ s, u_flow(s) ⟩

Die diskrete Aktion wird entlang des negativen Gradienten gewählt und um eine Zusatzbewertung ergänzt, die Zustände außerhalb des gültigen Anfahrkorridors benachteiligt. Dadurch vermeidet der APF das frühere Verhalten, direkt geradeaus zum Ziel zu fahren und dort an der Winkelrestriktion zu scheitern.

### 4.6 Reinforcement Learning – Q-Learning

Q-Learning approximiert die optimale Aktionswertfunktion Q* weiterhin tabellarisch, verwendet inzwischen aber eine **Reward-Shaping-Strategie**, damit sich eine stabile Andockpolitik schneller ausbildet.

Update-Regel:

Q_{k+1}(s, a) = Q_k(s, a)

α [ r + γ max_{a'} Q_k(s', a') − Q_k(s, a) ]

mit

α ∈ (0,1] (Lernrate),
γ ∈ (0,1] (Diskontfaktor),
r = R(s, a).

Der Reward enthält dabei heute nicht nur die negativen Bewegungskosten, sondern auch:
- einen positiven Zielreward für erfolgreiches Andocken,
- einen Fortschrittsbonus in Richtung Ziel bzw. Andockkorridor,
- eine Strafe für Wiederbesuche bereits gesehener Zustände,
- einen Bonus für das Erreichen eines gültigen Vor-Andockzustands.

Dadurch werden Schleifen und zufälliges Pendeln reduziert und die gelernten Policies deutlich robuster.

## 5. Vergleichskriterien

### 5.1 Lösungsqualität

Optimalitätslücke:

ΔJ = J_algo − J_optimal

Dabei wird J_optimal durch Dijkstra oder Dynamic Programming bestimmt.

### 5.2 Rechenzeit

- Planungszeit (Graphverfahren),
- Trainingszeit (Q-Learning),
- Inferenzzeit.

### 5.3 Konvergenzverhalten (RL)

Reward-Verlauf über Episoden,

Stabilität der Policy,

Varianz über verschiedene Random Seeds.

### 5.4 Robustheit

Analyse unter Variation von:
- Strömungsstärke,
- Start- und Zielposition,
- Gittergröße.


## 6. Software-Architektur

Projekt ist modular aufgebaut.

river-crossing/
│
├── configs/
│   ├── env.yaml
│   ├── algorithm.yaml
│   └── experiment.yaml
│
├── core/
│   ├── environment/
│   ├── dynamics/
│   └── cost/
│
├── algorithms/
│   ├── graph_search/
│   └── rl/
│
├── experiments/
│   ├── runner.py
│   └── evaluator.py
│
├── ui/
│   └── visualization.py
│
├── test/
│
├── analysis/
├── main.py
└── README.md

## 7. Konfigurationsprinzip

Alle variablen Parameter werden ausschließlich über Konfigurationsdateien definiert.

Beispiele:
- Gittergröße (N_x, N_y)
- Strömungsmodell und Strömungsvektor
- Gewichtungsparameter α, β
- APF-Parameter (`k_att`, `lambda_flow`)
- RL-Parameter (α, γ, ε sowie Reward-Shaping)
- Anzahl Episoden
- Random Seeds

Ziele:
- Keine hardcodierten Parameter,
- vollständige Reproduzierbarkeit,
- Vergleichbarkeit von Experimenten,
- automatisierte Batch-Experimente.

## 8. Visualisierung (UI)

Die aktuelle PyQt6-Oberfläche bietet drei klar getrennte Ansichten:
- **Run**: zeigt den schrittweisen Aufbau einer Lösung,
- **Best path**: zeigt und animiert den final besten Pfad eines ausgewählten Laufs,
- **Compare**: vergleicht die besten Pfade mehrerer Algorithmen im selben Szenario.

Zusätzlich umfasst die UI:
- Gitterstruktur,
- Strömungsvektoren,
- Start- und Zielpunkt,
- Animation mit Play, Pause, Step und Reset,
- Flow-Steuerung über einen QDial für die Richtung,
- einen Slider für die Strömungsstärke,
- drei synchronisierte Float-Eingabefelder für x-Richtung, y-Richtung und Stärke.

Im Run-Tab werden Entscheidungsoptionen farblich hervorgehoben:
- rot = ungültige Aktion,
- gelb = gültige, aber nicht gewählte Aktion,
- grün = aktuell beste Aktion gemäß Planung.

Die Schaltfläche **Run Batch** ist derzeit bewusst noch als Platzhalter markiert.

## 9. Ergebnisartekfakt
Das Ergebnisartefakt dieses Projekts besteht aus einem konsistenten und reproduzierbaren Satz an Experimenten sowie deren Auswertung. Es dient dazu, die implementierten Algorithmen unter identischen Bedingungen vergleichbar zu machen und ihre Eigenschaften systematisch zu analysieren. Dabei stehen sowohl die Qualität der gefundenen Lösungen als auch der Rechenaufwand und – im Fall von Reinforcement Learning – das Lernverhalten im Fokus.

### 9.1 Experimentelle Datensätze

Für jede Kombination aus Environment, Algorithmus und Parametrierung wird ein eigenständiger Experimentlauf durchgeführt. Die zugrunde liegenden Konfigurationen umfassen insbesondere die Gittergröße, die Strömung, die Lage von Start- und Zielpunkt sowie die Definition des zulässigen Anfahrwinkels.

Ein einzelner Lauf erzeugt einen vollständigen Datensatz, der heute sowohl den **besten resultierenden Pfad** als auch den **Run-Trace** für die Visualisierung enthält. Dazu gehören insbesondere:
- `path` und `actions` als kompatible Standardfelder,
- `best_path` und `best_actions` für die explizite Best-Path-Darstellung,
- `run_trace` für die schrittweise UI-Visualisierung,
- Zeit- und Trainingsmetriken,
- Reward-Verlauf und Erfolgsrate beim Q-Learning.

Die Trajektorie ist eine Folge diskreter Zustände
π = (s₀, s₁, …, s_T)
und wird zusammen mit den zugehörigen Aktionen gespeichert. Aus ihr wird die Gesamtzeit berechnet:

J(π) = Σ t(s_t, a_t)

Zusätzlich werden die Anzahl der benötigten Schritte sowie der letzte Bewegungsvektor erfasst, um die Einhaltung der Anfahrbedingung überprüfen zu können.

Alle Daten werden in strukturierter Form abgelegt, sodass sie später automatisiert ausgewertet und in der UI getrennt als **Run**, **Best path** und **Compare** genutzt werden können.

### 9.2 Analyse und Vergleich

Auf Basis der erzeugten Datensätze erfolgt eine systematische Auswertung der Algorithmen. Ein zentraler Bezugspunkt ist dabei die optimale Lösung, die mit Dijkstra oder Dynamic Programming bestimmt wird. Für jeden Algorithmus wird die Abweichung von dieser Referenz berechnet:

ΔJ = J_algo − J_optimal

Diese Größe erlaubt eine direkte Aussage über die Lösungsqualität.

Ergänzend dazu wird die benötigte Rechenzeit betrachtet, sowohl in absoluten Werten als auch in Abhängigkeit von der Problemgröße. Dadurch lässt sich der typische Zielkonflikt zwischen Optimalität und Effizienz sichtbar machen, insbesondere im Vergleich zwischen A*, Weighted A* und den vollständig optimalen Verfahren.

Für das Q-Learning wird zusätzlich das Lernverhalten analysiert. Hierbei steht im Vordergrund, wie schnell sich eine stabile Strategie entwickelt und wie stark die Ergebnisse zwischen verschiedenen Durchläufen variieren. Der Verlauf der kumulierten Rewards pro Episode dient dabei als zentrales Diagnoseinstrument.

Ein weiterer Aspekt der Analyse ist die Sensitivität gegenüber der Strömung. Durch Variation von Richtung und Stärke des Strömungsvektors wird untersucht, wie sich die resultierenden Pfade verändern und wie robust die einzelnen Verfahren auf diese Änderungen reagieren. Ebenso wird betrachtet, welchen Einfluss die Einschränkung des Anfahrwinkels auf die Lösungsstruktur und die Planungszeit hat.

### 9.3 Visuelle Aufbereitung

Ein wesentlicher Bestandteil des Ergebnisartefakts ist die visuelle Darstellung der Ergebnisse. Für ausgewählte Szenarien werden die berechneten Trajektorien direkt im Gitter visualisiert. Dabei werden sowohl die Strömungsvektoren als auch der zulässige Anfahrkorridor am Zielpunkt dargestellt.

Diese Visualisierung ermöglicht es, Unterschiede zwischen den Algorithmen unmittelbar nachzuvollziehen. Insbesondere lassen sich typische Verhaltensweisen erkennen, etwa Umwege zur Einhaltung des Anfahrwinkels oder lokale Fehlentscheidungen beim Potentialfeldansatz.

Ergänzend dazu kann die Bewegung des Schiffes entlang der Trajektorie animiert werden, um den zeitlichen Verlauf der Entscheidungsschritte sichtbar zu machen.

### 9.4 Reproduzierbarkeit

Alle Experimente sind vollständig über Konfigurationsdateien definiert. Ein Lauf ist eindeutig bestimmt durch die Environment-Parameter, die Algorithmuskonfiguration und – im Fall von Reinforcement Learning – den verwendeten Zufalls-Seed.

Dadurch lassen sich sämtliche Ergebnisse reproduzieren und gezielt variieren. Gleichzeitig wird sichergestellt, dass unterschiedliche Algorithmen unter exakt denselben Bedingungen verglichen werden.

### 9.5 Zusammenfassung

Das Ergebnisartefakt stellt einen strukturierten und nachvollziehbaren Vergleich verschiedener Pfadplanungsverfahren in einer strömungsbehafteten Umgebung dar. Es verbindet experimentelle Daten, quantitative Auswertung und visuelle Analyse zu einem konsistenten Gesamtbild und ermöglicht damit eine fundierte Bewertung der eingesetzten Methoden.