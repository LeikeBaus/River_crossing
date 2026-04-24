"""Experiment identifier encoding / decoding utilities.

Encoding scheme
---------------
Format (tokens joined by ``-``):

    [{algo}-]V{NNNN}-{flow_type}{dir}[v{NN}][-s{NNN}][-f{NN}][-a{NN}][-b{NN}]-i{N}[-t{NN}][-g{NX}x{NY}][-e{N}]-S{seeds}

Token legend
~~~~~~~~~~~~
algo        : dijk | astar | wastar | dp | apf | ql  (omitted for batch runs)
V{NNNN}     : version from common.version × 1000, zero-padded to 4 digits
              e.g. version 0.100 (stored as float 0.1 by YAML) → V0100; always present
flow_type   : G = gaussian, C = constant
dir         : u = bottom→top (vj ≥ 0), d = top→bottom (vj < 0)
v{NN}       : flow strength × 10, zero-padded to 2 digits; **omitted when strength == 1.0**
s{NNN}      : Gaussian sigma × 10  (gaussian only)
f{NN}       : Gaussian floor × 100 (gaussian only)
a{NN}       : alpha × 10; **omitted when alpha == 1.0**
b{NN}       : beta  × 10; **omitted when beta  == 1.0**
i{N}        : inertia (always included)
t{NN}       : turn_penalty × 10; **omitted when turn_penalty == 1.0**
g{NX}x{NY} : grid dimensions; **omitted when nx=40, ny=20**
e{N}        : Q-learning episodes (e.g. e200, e500, e1500);
              **omitted when ql_episodes == 1200** (the config default)
S{seeds}    : seed encoding:
                  contiguous  → {first}k{last}  e.g. S1k5
                  single      → {seed}           e.g. S3
                  non-contig, all < 10 → concatenated digits  e.g. S135
                  non-contig, any ≥ 10 → hyphen-joined        e.g. S1-3-10

Only characters in [A-Za-z0-9-] are used — safe on Linux, Windows (NTFS/FAT32), and macOS.
"""

from __future__ import annotations

import math
from pathlib import Path

_DEFAULT_NX = 40
_DEFAULT_NY = 20

_ALGO_ABBREV: dict[str, str] = {
    "dijkstra": "dijk",
    "a_star": "astar",
    "weighted_a_star": "wastar",
    "dynamic_programming": "dp",
    "apf": "apf",
    "q_learning": "ql",
}


def build_experiment_id(
    flow_cfg: dict,
    algo_cfg: dict,
    seeds: list[int],
    *,
    algorithm: str | None = None,
    grid_nx: int = _DEFAULT_NX,
    grid_ny: int = _DEFAULT_NY,
    ql_episodes: int = 1200,
) -> str:
    """Return a compact, filesystem-safe experiment identifier string.

    Parameters
    ----------
    flow_cfg:
        The flow sub-section of the environment config (type, vector, sigma, floor).
    algo_cfg:
        The full algorithm config dict (after any UI overrides have been applied).
    seeds:
        List of integer seed values used in the experiment.
    algorithm:
        Algorithm name for a single-algorithm run.  Pass ``None`` for batch runs.
    grid_nx, grid_ny:
        Grid dimensions.  Omitted from the ID when equal to the defaults (40 × 20).
    ql_episodes:
        Number of Q-learning training episodes.  Omitted from the ID when equal
        to 1200 (the ``algorithm.yaml`` default).
    """
    parts: list[str] = []

    # ── Algorithm prefix (single runs only) ─────────────────────────────────
    if algorithm is not None:
        parts.append(_ALGO_ABBREV.get(algorithm, algorithm[:6]))

    # ── Version ───────────────────────────────────────────────────────────────
    common = algo_cfg.get("common", {})
    version_raw = float(common.get("version", 0.0))
    version_int = int(round(version_raw * 1000))
    parts.append(f"V{version_int:04d}")

    # ── Flow ─────────────────────────────────────────────────────────────────
    flow_type = str(flow_cfg.get("type", "constant"))
    vec = flow_cfg.get("vector", [0.0, 1.0])
    vi, vj = float(vec[0]), float(vec[1])
    strength = math.hypot(vi, vj)

    type_char = "G" if flow_type == "gaussian" else "C"
    dir_char = "d" if vj < 0 else "u"
    flow_token = type_char + dir_char

    strength_int = int(round(strength * 10))
    if strength_int != 10:
        flow_token += f"v{strength_int:02d}"
    parts.append(flow_token)

    # ── Gaussian-specific parameters ─────────────────────────────────────────
    if flow_type == "gaussian":
        sigma = float(flow_cfg.get("sigma", 5.0))
        floor_frac = float(flow_cfg.get("floor", 0.1))
        parts.append(f"s{int(round(sigma * 10))}")
        parts.append(f"f{int(round(floor_frac * 100)):02d}")

    # ── Cost-function parameters ─────────────────────────────────────────────
    # (common already read above for version)
    alpha = float(common.get("alpha", 1.0))
    beta = float(common.get("beta", 1.0))
    inertia = int(common.get("inertia", 0))
    turn_penalty = float(common.get("turn_penalty", 1.0))

    alpha_int = int(round(alpha * 10))
    beta_int = int(round(beta * 10))
    tp_int = int(round(turn_penalty * 10))

    if alpha_int != 10:
        parts.append(f"a{alpha_int:02d}")
    if beta_int != 10:
        parts.append(f"b{beta_int:02d}")
    parts.append(f"i{inertia}")
    if tp_int != 10:
        parts.append(f"t{tp_int:02d}")

    # ── Grid (omit when default) ──────────────────────────────────────────────
    if grid_nx != _DEFAULT_NX or grid_ny != _DEFAULT_NY:
        parts.append(f"g{grid_nx}x{grid_ny}")
    # ── Q-learning episodes (omit when default 1200) ───────────────────────────
    if int(ql_episodes) != 1200:
        parts.append(f"e{int(ql_episodes)}")
    # ── Seeds ─────────────────────────────────────────────────────────────────
    seeds_sorted = sorted(set(int(s) for s in seeds))
    if not seeds_sorted:
        seed_str = "none"
    elif seeds_sorted == list(range(seeds_sorted[0], seeds_sorted[-1] + 1)):
        if len(seeds_sorted) == 1:
            seed_str = str(seeds_sorted[0])
        else:
            seed_str = f"{seeds_sorted[0]}k{seeds_sorted[-1]}"
    elif all(s < 10 for s in seeds_sorted):
        seed_str = "".join(str(s) for s in seeds_sorted)
    else:
        seed_str = "-".join(str(s) for s in seeds_sorted)
    parts.append(f"S{seed_str}")

    return "-".join(parts)


def results_paths(exp_id: str, analysis_dir: str | Path = "analysis") -> tuple[Path, Path]:
    """Return ``(raw_results_path, evaluation_summary_path)`` for the given experiment ID.

    Files are named ``R-{exp_id}.json`` and ``EV-{exp_id}.json`` inside *analysis_dir*.
    """
    d = Path(analysis_dir)
    return d / f"R-{exp_id}.json", d / f"EV-{exp_id}.json"
