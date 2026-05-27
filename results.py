#!/usr/bin/env python3
"""
results.py — Analyse sweep results and produce report figures.

Usage:
    python results.py [analysis_dir]      # defaults to ./analysis
    python results.py analysis --show     # also open figure windows

Output:
    analysis/plots/s{N}_*.png  — one PNG per section

Requires:
    pip install pandas matplotlib
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# ── dependency check ──────────────────────────────────────────────────────────
try:
    import pandas as pd
except ImportError:
    sys.exit("pandas not found. Run:  pip install pandas matplotlib")
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ImportError:
    sys.exit("matplotlib not found. Run:  pip install pandas matplotlib")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_FNAME_RE = re.compile(
    r"R-(?P<short>\w+)-V\d+-G(?P<dir>[ud])-s(?P<s10>\d+)-f(?P<f100>\d+)"
    r"-i(?P<inertia>\d+)(?:-g(?P<nx>\d+)x(?P<ny>\d+))?(?:-e(?P<ep>\d+))?-S(?P<seed>\d+)\.json"
)

_SHORT_TO_ALGO = {
    "dijk":  "dijkstra",
    "astar": "a_star",
    "wastar": "weighted_a_star",
    "dp":    "dynamic_programming",
    "ql":    "q_learning",
}

_LABELS = {
    "dijkstra":            "Dijkstra",
    "a_star":              "A*",
    "weighted_a_star":     "Wt. A*",
    "dynamic_programming": "DP",
    "q_learning":          "Q-Learning",
}

_ALGO_ORDER = ["dijkstra", "a_star", "weighted_a_star", "dynamic_programming", "q_learning"]
_ENV_ORDER  = ["small", "medium", "large"]

_COLORS = {
    "dijkstra":            "#1f77b4",
    "a_star":              "#ff7f0e",
    "weighted_a_star":     "#2ca02c",
    "dynamic_programming": "#9467bd",
    "q_learning":          "#d62728",
}

_ENV_COLORS = {"small": "#17becf", "medium": "#bcbd22", "large": "#7f7f7f"}

# Keys kept from each JSON record (heavy fields like path/actions dropped after use)
_KEEP = {
    "environment_name", "algorithm", "seed", "found", "total_cost",
    "path_length", "steps", "angle_valid", "plan_time", "training_time",
    "inference_time", "total_time", "nodes_expanded",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _parse_fname(name: str) -> dict | None:
    m = _FNAME_RE.match(name)
    if not m:
        return None
    nx = int(m["nx"]) if m["nx"] else 80
    ny = int(m["ny"]) if m["ny"] else 40
    return {
        "algorithm": _SHORT_TO_ALGO.get(m["short"], m["short"]),
        "flow_dir":  "up" if m["dir"] == "u" else "down",
        "sigma":     int(m["s10"]) / 10,
        "floor":     int(m["f100"]) / 100,
        "inertia":   int(m["inertia"]),
        "episodes":  int(m["ep"]) if m["ep"] else None,
        "env_size":  "small" if nx <= 10 else ("medium" if nx <= 40 else "large"),
        "nx": nx, "ny": ny,
    }


def load_data(analysis_dir: Path, floor_filter: float | None = 0.10) -> pd.DataFrame:
    """
    Load all per-algorithm R-*.json result files into a flat DataFrame.

    Sweep parameters (flow_dir, sigma, inertia, episodes, env_size) are
    parsed from filenames since they are not stored inside the JSON records.
    ``cost_gap_pct`` is added as the percentage cost difference vs Dijkstra
    under identical conditions.
    """
    rows: list[dict] = []
    for p in sorted(analysis_dir.glob("R-*.json")):
        params = _parse_fname(p.name)
        if params is None:
            continue
        if floor_filter is not None and params["floor"] != floor_filter:
            continue
        try:
            records: list[dict] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for rec in records:
            path = rec.get("path", [])
            ys = [pt[1] for pt in path] if path else None
            row = {k: rec[k] for k in _KEEP if k in rec}
            row["lateral_drift"] = float(max(ys) - min(ys)) if ys else None
            row.update(params)
            rows.append(row)

    if not rows:
        sys.exit(f"No matching R-*.json files found in {analysis_dir!r}")

    df = pd.DataFrame(rows)

    num_cols = [
        "total_cost", "steps", "path_length", "plan_time", "training_time",
        "inference_time", "total_time", "nodes_expanded", "lateral_drift",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["total_time"]     = df["plan_time"] + df["training_time"] + df["inference_time"]
    df["grid_cells"]     = df["nx"] * df["ny"]
    df["nodes_per_cell"] = df["nodes_expanded"] / df["grid_cells"]

    # Relative cost gap vs Dijkstra for identical experimental conditions
    ref_keys = ["env_size", "flow_dir", "sigma", "inertia", "seed"]
    ref = (
        df[df["algorithm"] == "dijkstra"]
        .drop_duplicates(subset=ref_keys)
        .set_index(ref_keys)["total_cost"]
        .rename("dijk_cost")
    )
    df = df.join(ref, on=ref_keys)
    df["cost_gap_pct"] = (df["total_cost"] - df["dijk_cost"]) / df["dijk_cost"] * 100

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lbl(algo: str) -> str:
    return _LABELS.get(algo, algo)

def _col(algo: str) -> str:
    return _COLORS.get(algo, "#555555")

def _algos(df: pd.DataFrame) -> list[str]:
    return [a for a in _ALGO_ORDER if a in df["algorithm"].unique()]

def _envs(df: pd.DataFrame) -> list[str]:
    return [e for e in _ENV_ORDER if e in df["env_size"].unique()]

def _save(fig: plt.Figure, name: str, out_dir: Path, show: bool) -> None:
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → {path}")
    if show:
        plt.show()
    plt.close(fig)


def _grouped_bars(ax: plt.Axes, algos: list[str], categories: list,
                  values: dict[str, list[float]], ylabel: str, title: str,
                  xlabels: list[str] | None = None, baseline: float | None = None) -> None:
    """Generic grouped bar chart: one group per category, one bar per algorithm."""
    n = len(algos)
    w = min(0.8 / n, 0.2)
    for i, algo in enumerate(algos):
        offsets = [x + (i - n / 2 + 0.5) * w for x in range(len(categories))]
        ax.bar(offsets, values[algo], width=w, label=_lbl(algo),
               color=_col(algo), alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(xlabels if xlabels else [str(c) for c in categories])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2, axis="y")
    if baseline is not None:
        ax.axhline(baseline, color="black", linewidth=0.8, linestyle="--")


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Overall algorithm ranking
# ─────────────────────────────────────────────────────────────────────────────

def section1_overview(df: pd.DataFrame, out_dir: Path, show: bool) -> None:
    print("\n── Section 1: Overall algorithm ranking ──")
    algos  = _algos(df)
    solved = df[df["found"] == True]

    rows = []
    for algo in algos:
        sub = df[df["algorithm"] == algo]
        sol = solved[solved["algorithm"] == algo]
        gap = sub["cost_gap_pct"].dropna()
        rows.append({
            "Algorithm":            _lbl(algo),
            "N":                    len(sub),
            "Success %":            f"{sub['found'].mean()*100:.1f}",
            "Mean cost":            f"{sol['total_cost'].mean():.3f}" if not sol.empty else "—",
            "Mean gap %":           f"{gap.mean():.2f}" if not gap.empty else "—",
            "Max gap %":            f"{gap.max():.2f}" if not gap.empty else "—",
            "Mean steps":           f"{sol['steps'].mean():.1f}" if not sol.empty else "—",
            "Mean time (s)":        f"{sub['total_time'].mean():.4f}",
            "Angle valid %":        f"{sub['angle_valid'].mean()*100:.1f}",
        })
    print(pd.DataFrame(rows).set_index("Algorithm").to_string())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Section 1 — Overall Algorithm Performance", fontsize=13, fontweight="bold")

    # Box plot: cost_gap_pct distribution
    data = [
        df.loc[(df["algorithm"] == a) & df["cost_gap_pct"].notna(), "cost_gap_pct"].values
        for a in algos
    ]
    bp = ax1.boxplot(data, patch_artist=True, notch=False,
                     medianprops=dict(color="black", linewidth=2),
                     flierprops=dict(marker=".", markersize=4, alpha=0.5))
    for patch, algo in zip(bp["boxes"], algos):
        patch.set_facecolor(_col(algo)); patch.set_alpha(0.75)
    ax1.set_xticks(range(1, len(algos) + 1))
    ax1.set_xticklabels([_lbl(a) for a in algos], rotation=15, ha="right")
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Cost gap vs Dijkstra (%)")
    ax1.set_title("Solution quality across all settings")

    # Stacked bar: time breakdown
    bottoms = [0.0] * len(algos)
    for label, field, color in [
        ("Plan",      "plan_time",      "#4C8BE2"),
        ("Training",  "training_time",  "#E25C4C"),
        ("Inference", "inference_time", "#4CAE4C"),
    ]:
        vals = [df[df["algorithm"] == a][field].mean() for a in algos]
        ax2.bar(range(len(algos)), vals, bottom=bottoms,
                label=label, color=color, alpha=0.85, edgecolor="white")
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax2.set_xticks(range(len(algos)))
    ax2.set_xticklabels([_lbl(a) for a in algos], rotation=15, ha="right")
    ax2.set_ylabel("Mean time (s)")
    ax2.set_title("Computation time breakdown (mean over all settings)")
    ax2.legend()

    fig.tight_layout()
    _save(fig, "s1_overview.png", out_dir, show)


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Scalability (environment size)
# ─────────────────────────────────────────────────────────────────────────────

def section2_scalability(df: pd.DataFrame, out_dir: Path, show: bool) -> None:
    print("\n── Section 2: Scalability ──")
    algos = _algos(df)
    envs  = _envs(df)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Section 2 — Scalability by Environment Size", fontsize=13, fontweight="bold")

    for ax, (col, ylabel, title) in zip(axes, [
        ("cost_gap_pct",    "Cost gap vs Dijkstra (%)",  "Solution quality"),
        ("total_cost",      "Mean total cost",            "Absolute path cost"),
        ("nodes_per_cell",  "Nodes expanded / grid cell", "Search complexity (normalised)"),
    ]):
        for algo in algos:
            sub = df[df["algorithm"] == algo]
            ys  = [sub[sub["env_size"] == e][col].mean() for e in envs]
            ax.plot(envs, ys, marker="o", label=_lbl(algo), color=_col(algo), linewidth=2)
        ax.set_xlabel("Environment size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if col == "cost_gap_pct":
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    fig.tight_layout()
    _save(fig, "s2_scalability.png", out_dir, show)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Flow sensitivity (sigma + direction)
# ─────────────────────────────────────────────────────────────────────────────

def section3_flow(df: pd.DataFrame, out_dir: Path, show: bool) -> None:
    print("\n── Section 3: Flow sensitivity ──")
    algos  = _algos(df)
    sigmas = sorted(df["sigma"].dropna().unique())
    dirs   = sorted(df["flow_dir"].dropna().unique())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Section 3 — Flow Sensitivity", fontsize=13, fontweight="bold")

    # 3a: cost_gap_pct vs sigma
    _grouped_bars(
        axes[0, 0], algos, sigmas,
        {a: [df[(df["algorithm"] == a) & (df["sigma"] == s)]["cost_gap_pct"].mean()
             for s in sigmas] for a in algos},
        "Mean cost gap (%)", "Effect of sigma on solution quality",
        xlabels=[f"σ = {s}" for s in sigmas], baseline=0.0,
    )

    # 3b: cost_gap_pct vs flow direction
    _grouped_bars(
        axes[0, 1], algos, dirs,
        {a: [df[(df["algorithm"] == a) & (df["flow_dir"] == d)]["cost_gap_pct"].mean()
             for d in dirs] for a in algos},
        "Mean cost gap (%)", "Effect of flow direction on solution quality",
        xlabels=[f"Flow {d}" for d in dirs], baseline=0.0,
    )

    # 3c: absolute cost vs sigma, one line per algo per env (large env only for clarity)
    ax = axes[1, 0]
    for algo in algos:
        for env, ls in [("large", "-"), ("medium", "--"), ("small", ":")]:
            if env not in df["env_size"].unique():
                continue
            sub = df[(df["algorithm"] == algo) & (df["env_size"] == env)]
            ys  = [sub[sub["sigma"] == s]["total_cost"].mean() for s in sigmas]
            ax.plot(sigmas, ys, marker="o", linestyle=ls, color=_col(algo),
                    linewidth=1.8, alpha=0.85)
    ax.set_xlabel("Sigma"); ax.set_ylabel("Mean total cost")
    ax.set_title("Absolute cost vs sigma (line style = env size)")
    ax.grid(True, alpha=0.3)
    legend_handles = [
        plt.Line2D([0], [0], color=_col(a), linewidth=2, label=_lbl(a)) for a in algos
    ] + [
        plt.Line2D([0], [0], color="gray", linewidth=2, linestyle=ls, label=env.capitalize())
        for env, ls in [("large", "-"), ("medium", "--"), ("small", ":")]
        if env in df["env_size"].unique()
    ]
    ax.legend(handles=legend_handles, fontsize=7, ncol=2)

    # 3d: lateral drift vs flow direction
    ax = axes[1, 1]
    if "lateral_drift" in df.columns:
        _grouped_bars(
            ax, algos, dirs,
            {a: [df[(df["algorithm"] == a) & (df["flow_dir"] == d)]["lateral_drift"].mean()
                 for d in dirs] for a in algos},
            "Mean vertical path excursion (cells)", "Lateral drift by flow direction",
            xlabels=[f"Flow {d}" for d in dirs],
        )
    else:
        ax.text(0.5, 0.5, "No path data", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    _save(fig, "s3_flow.png", out_dir, show)


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Inertia constraint
# ─────────────────────────────────────────────────────────────────────────────

def section4_inertia(df: pd.DataFrame, out_dir: Path, show: bool) -> None:
    print("\n── Section 4: Inertia constraint ──")
    algos    = _algos(df)
    inertias = sorted(df["inertia"].dropna().unique().astype(int))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Section 4 — Effect of Inertia Constraint", fontsize=13, fontweight="bold")

    for ax, (col, ylabel, title, baseline) in zip(axes, [
        ("cost_gap_pct", "Mean cost gap (%)",           "Solution quality",         0.0),
        ("steps",        "Mean path steps",              "Path length",              None),
        ("total_time",   "Mean total computation (s)",  "Computation time",         None),
    ]):
        _grouped_bars(
            ax, algos, inertias,
            {a: [df[(df["algorithm"] == a) & (df["inertia"] == k)][col].mean()
                 for k in inertias] for a in algos},
            ylabel, title,
            xlabels=[f"Inertia = {k}" for k in inertias], baseline=baseline,
        )

    fig.tight_layout()
    _save(fig, "s4_inertia.png", out_dir, show)


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Q-Learning convergence
# ─────────────────────────────────────────────────────────────────────────────

def section5_ql(df: pd.DataFrame, out_dir: Path, show: bool) -> None:
    print("\n── Section 5: Q-Learning convergence ──")
    ql = df[(df["algorithm"] == "q_learning") & df["episodes"].notna()].copy()
    if ql.empty:
        print("  No Q-Learning data with episode info found.")
        return

    episodes = sorted(ql["episodes"].unique().astype(int))
    envs     = _envs(ql)

    # Print per-env per-episode table
    tbl_rows = []
    for env in envs:
        for ep in episodes:
            sub = ql[(ql["env_size"] == env) & (ql["episodes"] == ep)]
            tbl_rows.append({
                "Env": env, "Episodes": ep,
                "Mean cost gap %": f"{sub['cost_gap_pct'].mean():.2f}",
                "Mean train time (s)": f"{sub['training_time'].mean():.3f}",
            })
    print(pd.DataFrame(tbl_rows).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Section 5 — Q-Learning: Convergence vs Training Episodes",
                 fontsize=13, fontweight="bold")

    # 5a: cost_gap_pct vs episodes per env
    ax = axes[0]
    for env in envs:
        sub = ql[ql["env_size"] == env]
        ys  = [sub[sub["episodes"] == ep]["cost_gap_pct"].mean() for ep in episodes]
        ax.plot(episodes, ys, marker="o", label=env.capitalize(),
                color=_ENV_COLORS.get(env, "gray"), linewidth=2)
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Cost gap vs Dijkstra (%)")
    ax.set_title("Solution quality vs episodes")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    # 5b: training_time vs episodes per env
    ax = axes[1]
    for env in envs:
        sub = ql[ql["env_size"] == env]
        ys  = [sub[sub["episodes"] == ep]["training_time"].mean() for ep in episodes]
        ax.plot(episodes, ys, marker="s", label=env.capitalize(),
                color=_ENV_COLORS.get(env, "gray"), linewidth=2)
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Mean training time (s)")
    ax.set_title("Training time vs episodes")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 5c: efficiency frontier — training_time vs cost_gap_pct, annotated by episode count
    ax = axes[2]
    for env in envs:
        sub  = ql[ql["env_size"] == env]
        xs   = [sub[sub["episodes"] == ep]["training_time"].mean() for ep in episodes]
        ys   = [sub[sub["episodes"] == ep]["cost_gap_pct"].mean()  for ep in episodes]
        ax.plot(xs, ys, marker="o", label=env.capitalize(),
                color=_ENV_COLORS.get(env, "gray"), linewidth=1.5)
        for x, y, ep in zip(xs, ys, episodes):
            ax.annotate(f"{ep:,}", (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=7)
    ax.set_xlabel("Mean training time (s)")
    ax.set_ylabel("Cost gap vs Dijkstra (%)")
    ax.set_title("Training efficiency frontier (ep. labels)")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    fig.tight_layout()
    _save(fig, "s5_ql_convergence.png", out_dir, show)


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Time vs quality Pareto
# ─────────────────────────────────────────────────────────────────────────────

def section6_pareto(df: pd.DataFrame, out_dir: Path, show: bool) -> None:
    print("\n── Section 6: Time vs quality trade-off ──")
    algos = _algos(df)
    envs  = _envs(df)

    fig, axes = plt.subplots(1, len(envs), figsize=(5 * len(envs), 5), sharey=True)
    if len(envs) == 1:
        axes = [axes]
    fig.suptitle("Section 6 — Time vs Quality Trade-off", fontsize=13, fontweight="bold")

    for ax, env in zip(axes, envs):
        env_df = df[df["env_size"] == env]
        for algo in algos:
            sub = env_df[env_df["algorithm"] == algo]
            if sub.empty:
                continue
            # For QL, pick the best-episode run (lowest mean cost_gap_pct)
            ep_label = ""
            if algo == "q_learning" and sub["episodes"].notna().any():
                best_ep = (sub.groupby("episodes")["cost_gap_pct"]
                           .mean().idxmin())
                sub = sub[sub["episodes"] == best_ep]
                ep_label = f"\n({int(best_ep)} ep)"
            x = sub["total_time"].mean()
            y = sub["cost_gap_pct"].mean()
            ax.scatter(x, y, s=130, color=_col(algo), zorder=5,
                       edgecolors="white", linewidths=0.8)
            ax.annotate(_lbl(algo) + ep_label, (x, y),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Mean total time (s)")
        ax.set_title(env.capitalize())
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean cost gap vs Dijkstra (%)")
    fig.tight_layout()
    _save(fig, "s6_pareto.png", out_dir, show)


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Path quality (angle validity + lateral drift)
# ─────────────────────────────────────────────────────────────────────────────

def section7_path_quality(df: pd.DataFrame, out_dir: Path, show: bool) -> None:
    print("\n── Section 7: Path quality ──")
    algos = _algos(df)
    envs  = _envs(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Section 7 — Path Quality", fontsize=13, fontweight="bold")

    # 7a: docking angle validity rate per algorithm
    ax = axes[0]
    rates = [df[df["algorithm"] == a]["angle_valid"].mean() * 100 for a in algos]
    bars  = ax.bar(range(len(algos)), rates,
                   color=[_col(a) for a in algos], alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels([_lbl(a) for a in algos], rotation=15, ha="right")
    ax.set_ylim(0, 108)
    ax.set_ylabel("Valid docking angle (%)")
    ax.set_title("Docking angle validity rate (all settings)")
    ax.axhline(100, color="gray", linewidth=0.8, linestyle="--")
    for bar, v in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=9)

    # 7b: lateral drift per algo × env_size (grouped bars)
    ax = axes[1]
    if "lateral_drift" in df.columns and df["lateral_drift"].notna().any():
        n = len(algos)
        w = min(0.8 / (n * len(envs)), 0.12)
        env_hatches = {e: h for e, h in zip(envs, ["", "///", "xxx"])}
        legend_patches = []
        for i, algo in enumerate(algos):
            for j, env in enumerate(envs):
                val = df[(df["algorithm"] == algo) & (df["env_size"] == env)]["lateral_drift"].mean()
                x   = i + (j - len(envs) / 2 + 0.5) * w * 1.2
                hatch = env_hatches.get(env, "")
                bar = ax.bar(x, val, width=w, color=_col(algo),
                             hatch=hatch, alpha=0.80, edgecolor="white")
        # Legend: algo colors
        legend_algo = [Patch(facecolor=_col(a), label=_lbl(a)) for a in algos]
        # Legend: env hatches
        legend_env  = [Patch(facecolor="white", edgecolor="black",
                             hatch=env_hatches.get(e, ""), label=e.capitalize())
                       for e in envs]
        ax.legend(handles=legend_algo + legend_env, fontsize=7, ncol=2)
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels([_lbl(a) for a in algos], rotation=15, ha="right")
        ax.set_ylabel("Mean vertical path excursion (cells)")
        ax.set_title("Lateral drift by algorithm and environment size")
        ax.grid(True, alpha=0.2, axis="y")
    else:
        ax.text(0.5, 0.5, "Path data not available", ha="center", va="center",
                transform=ax.transAxes)

    fig.tight_layout()
    _save(fig, "s7_path_quality.png", out_dir, show)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]
    show = "--show" in args
    dirs = [a for a in args if not a.startswith("--")]

    analysis_dir = Path(dirs[0]) if dirs else Path("analysis")
    out_dir      = analysis_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {analysis_dir!r} …")
    df = load_data(analysis_dir)

    n_total  = len(df)
    n_solved = int(df["found"].sum())
    print(f"  {n_total} runs loaded  ({n_solved} solved, {n_total - n_solved} unsolved)")
    print(f"  Algorithms : {sorted(df['algorithm'].unique())}")
    print(f"  Env sizes  : {sorted(df['env_size'].unique())}")
    print(f"  Sigmas     : {sorted(df['sigma'].unique())}")
    print(f"  Inertias   : {sorted(df['inertia'].unique().astype(int))}")
    print(f"  Output     : {out_dir}/")

    section1_overview(df, out_dir, show)
    section2_scalability(df, out_dir, show)
    section3_flow(df, out_dir, show)
    section4_inertia(df, out_dir, show)
    section5_ql(df, out_dir, show)
    section6_pareto(df, out_dir, show)
    section7_path_quality(df, out_dir, show)

    print(f"\nDone. All plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
