# River Crossing – Path Planning in Flow Fields

## 1. Motivation

Dieses Projekt untersucht das Problem der optimalen Navigation eines Schiffes in einem Fluss mit Strömung. Ein Schiff soll von einem Startpunkt A an einem Ufer zu einem Zielpunkt B am gegenüberliegenden Ufer navigieren.

Ziel ist es, eine Trajektorie zu bestimmen, die
- die Überquerungszeit minimiert,
- den Energieverbrauch minimiert,
- oder eine gewichtete Kombination beider Größen optimiert.

Das Problem wird als diskretes 2D-Gittermodell formuliert und mithilfe klassischer Pfadalgorithmen sowie Reinforcement Learning untersucht. Der Fokus liegt auf einer sauberen mathematischen Modellierung, modularer Software-Architektur und reproduzierbaren Experimenten.

## 2. Mathematische Problemformulierung

### 2.1 Zustandsraum

Die Environment ist ein diskretes Gitter:

S = { (i, j) | i ∈ {0, …, N_x − 1}, j ∈ {0, …, N_y − 1} }

Jeder Zustand s ∈ S repräsentiert eine Gitterzelle.

Jedem Zustand ist ein Strömungsvektor zugeordnet:

u_flow : S → ℝ²

u_flow(s) = ( u_x(s), u_y(s) )ᵀ

Im einfachsten Fall ist die Strömung konstant:

u_flow(s) = u₀ ∈ ℝ² für alle s ∈ S.

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

Die Strömung beeinflusst im diskreten Modell nicht direkt die Position, sondern geht in die Übergangskosten ein.

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

### 4.5 Reinforcement Learning – Q-Learning

Q-Learning approximiert die optimale Aktionswertfunktion Q*.

Update-Regel:

Q_{k+1}(s, a) = Q_k(s, a)

α [ r + γ max_{a'} Q_k(s', a') − Q_k(s, a) ]

mit

α ∈ (0,1] (Lernrate),
γ ∈ (0,1] (Diskontfaktor),
r = R(s, a).

Die optimale Politik ergibt sich aus:

π*(s) = argmax_{a ∈ A} Q*(s, a)

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
├── analysis/
├── main.py
└── README.py

## 7. Konfigurationsprinzip

Alle variablen Parameter werden ausschließlich über Konfigurationsdateien definiert.

Beispiele:
- Gittergröße (N_x, N_y)
- Strömungsmodell
- Gewichtungsparameter α, β
- RL-Parameter (α, γ, ε)
- Anzahl Episoden
- Random Seeds

Ziele:
- Keine hardcodierten Parameter,
- vollständige Reproduzierbarkeit,
- Vergleichbarkeit von Experimenten,
- automatisierte Batch-Experimente.

8. Visualisierung (UI)

Die UI visualisiert:
- Gitterstruktur,
- Strömungsvektoren,
- Start- und Zielpunkt,
- gefundene Trajektorien,
- Vergleich mehrerer Algorithmen.

Optional:
- animierte Bewegung,
- Filter nach Algorithmus, Run, Seed.