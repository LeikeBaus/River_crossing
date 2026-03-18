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

Definition eines Potentials über dem Zustandsraum:

Φ(s) = Φ_att(s) + Φ_flow(s)

Attraktives Potential:

Φ_att(s) = 1/2 · k_att · || s − s_goal ||²

Strömungseinfluss:

Φ_flow(s) = −λ · ⟨ s, u_flow(s) ⟩

Die Bewegung erfolgt entlang des negativen Gradienten:

a*(s) ≈ argmin_{a ∈ A} ⟨ a, ∇Φ(s) ⟩

Die diskrete Aktion wird als beste Approximation des kontinuierlichen Gradienten gewählt.

### 4.6 Reinforcement Learning – Q-Learning

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

## 8. Visualisierung (UI)

Die UI visualisiert:
- Gitterstruktur,
- Strömungsvektoren,
- Start- und Zielpunkt,
- gefundene Trajektorien,
- Vergleich mehrerer Algorithmen.

Optional:
- animierte Bewegung,
- Filter nach Algorithmus, Run, Seed.