"""Tune the poorly constrained aircraft-model parameters using optimizer behavior.

This script treats p_base, ks_base, and kv_base as calibration parameters. For
one candidate parameter set, it solves the normal deterministic aircraft design
problem *without a CL constraint*. It then scores the resulting optimum against
broad behavioral envelopes.

Two Latin-hypercube sweeps are performed:
    1. a coarse sweep over broad parameter ranges;
    2. a refined sweep around the feasible/best region from sweep 1.

The important distinction is that the envelopes are guardrails, not point
calibration targets. Any optimum inside every envelope receives zero violation
score, so the tuning does not secretly force CL (or any other design variable)
to a single reference-aircraft value.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import openmdao.api as om
from scipy.stats import qmc

from fixed import parameters, tuning
from .organize import configure_subsystems, initialize


# -----------------------------------------------------------------------------
# User-editable calibration setup
# -----------------------------------------------------------------------------

# These are SEARCH bounds, not claims about physically correct values.
# p_base is sampled linearly; ks_base and kv_base are sampled logarithmically
# because their plausible numerical scale can span orders of magnitude.
SEARCH_BOUNDS = {
    "p_base":  (3.0, 15.0, "linear"),
    "ks_base": (3.0e-5, 1.0e-3, "log"),
    "kv_base": (50.0, 2500.0, "log"),
}

# Broad guardrails for the UNCONSTRAINED deterministic optimum. These should be
# changed deliberately as you decide what "737-esque" means for this example.
# CL is intentionally an interval, not a target value.
OPTIMUM_ENVELOPE = {
    "S":          (110.0, 160.0),
    "AR":         (8.0, 13.0),
    "V_cruise":   (210.0, 255.0),
    "SFC_tech":   (-0.90, 0.90),
    "CL":         (0.30, 0.80),
}

# The inner optimization retains physical/model constraints that are not being
# used as calibration targets.
INNER_DV_BOUNDS = {
    "S":        (100.0, 180.0),
    "AR":       (7.0, 50.0),
    "V_cruise": (200.0, 260.0),
    "SFC_tech": (-1.0, 1.0),
}

# Use the reference-aircraft fuel-capacity value already stored in fixed.py.
# Set this to 50000.0 if you want to reproduce the old loose model constraint.
FUEL_UPPER = float(parameters.get("m_fuel_max", 50000.0))


@dataclass(frozen=True)
class CandidateResult:
    p_base: float
    ks_base: float
    kv_base: float
    S: float
    AR: float
    V_cruise: float
    SFC_tech: float
    CL: float
    DOC: float
    m_fuel: float
    m_total: float
    violation_score: float
    center_score: float
    feasible: bool
    success: bool

    def as_dict(self) -> Dict[str, float | bool]:
        return self.__dict__.copy()


def _scalar(prob: om.Problem, name: str) -> float:
    return float(np.asarray(prob.get_val(name)).reshape(-1)[0])


def interval_violation(value: float, lower: float, upper: float) -> float:
    """Dimensionless squared hinge penalty; zero anywhere inside [lower, upper]."""
    width = upper - lower
    if width <= 0.0:
        raise ValueError(f"Invalid interval [{lower}, {upper}].")

    if value < lower:
        return ((lower - value) / width) ** 2
    if value > upper:
        return ((value - upper) / width) ** 2
    return 0.0


def envelope_scores(outputs: Mapping[str, float]) -> Tuple[float, float]:
    """Return (violation score, center score).

    violation_score defines admissibility and is exactly zero inside all
    envelopes. center_score is only a secondary ranking metric among feasible
    candidates; it does not affect whether a candidate is accepted.
    """
    violation = 0.0
    center = 0.0

    for name, (lower, upper) in OPTIMUM_ENVELOPE.items():
        value = outputs[name]
        violation += interval_violation(value, lower, upper)

        midpoint = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower)
        center += ((value - midpoint) / half_width) ** 2

    return violation, center


def build_inner_problem() -> om.Problem:
    """Create the deterministic design problem with NO CL constraint."""
    prob = om.Problem()
    configure_subsystems(prob)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["maxiter"] = 500
    prob.driver.options["tol"] = 1.0e-7
    prob.driver.options["disp"] = False

    prob.model.add_design_var(
        "S", lower=INNER_DV_BOUNDS["S"][0], upper=INNER_DV_BOUNDS["S"][1],
        ref=parameters["S"],
    )
    prob.model.add_design_var(
        "AR", lower=INNER_DV_BOUNDS["AR"][0], upper=INNER_DV_BOUNDS["AR"][1],
        ref=parameters["AR"],
    )
    prob.model.add_design_var(
        "V_cruise",
        lower=INNER_DV_BOUNDS["V_cruise"][0],
        upper=INNER_DV_BOUNDS["V_cruise"][1],
        ref=parameters["V_cruise"],
    )
    prob.model.add_design_var(
        "SFC_tech",
        lower=INNER_DV_BOUNDS["SFC_tech"][0],
        upper=INNER_DV_BOUNDS["SFC_tech"][1],
        ref=1.0,
    )

    prob.model.add_objective("DOC", ref=1.0e4)
    prob.model.add_constraint("m_fuel", lower=1000.0, upper=FUEL_UPPER, ref=16000.0)

    prob.setup()
    return prob


def evaluate_candidate(p_base: float, ks_base: float, kv_base: float) -> CandidateResult:
    """Run one nested deterministic optimization and score its optimum."""
    prob = build_inner_problem()

    try:
        # Start the design at the stored reference-aircraft values and initialize
        # all fixed/tuning inputs through the project's normal helper.
        initialize(prob, parameters)

        # Override only the three parameters being calibrated.
        prob.set_val("p_base", p_base)
        prob.set_val("ks_base", ks_base)
        prob.set_val("kv_base", kv_base)

        prob.run_driver()

        outputs = {
            "S": _scalar(prob, "S"),
            "AR": _scalar(prob, "AR"),
            "V_cruise": _scalar(prob, "V_cruise"),
            "SFC_tech": _scalar(prob, "SFC_tech"),
            "CL": _scalar(prob, "CL"),
            "DOC": _scalar(prob, "DOC"),
            "m_fuel": _scalar(prob, "m_fuel"),
            "m_total": _scalar(prob, "m_total"),
        }

        finite = all(np.isfinite(value) for value in outputs.values())
        if finite:
            violation, center = envelope_scores(outputs)
        else:
            violation, center = np.inf, np.inf

        return CandidateResult(
            p_base=p_base,
            ks_base=ks_base,
            kv_base=kv_base,
            **outputs,
            violation_score=violation,
            center_score=center,
            feasible=bool(finite and violation <= 1.0e-12),
            success=bool(finite),
        )

    except Exception:
        # A failed nonlinear solve/optimizer evaluation is simply an inadmissible
        # tuning candidate. Keeping it in the CSV makes failure regions visible.
        return CandidateResult(
            p_base=p_base,
            ks_base=ks_base,
            kv_base=kv_base,
            S=np.nan,
            AR=np.nan,
            V_cruise=np.nan,
            SFC_tech=np.nan,
            CL=np.nan,
            DOC=np.nan,
            m_fuel=np.nan,
            m_total=np.nan,
            violation_score=np.inf,
            center_score=np.inf,
            feasible=False,
            success=False,
        )
    finally:
        try:
            prob.cleanup()
        except Exception:
            pass


def _transform_sample(u: np.ndarray, bounds: Mapping[str, Tuple[float, float, str]]):
    values = {}
    for j, (name, (lower, upper, scale)) in enumerate(bounds.items()):
        if scale == "linear":
            values[name] = lower + u[j] * (upper - lower)
        elif scale == "log":
            log_lo = np.log10(lower)
            log_hi = np.log10(upper)
            values[name] = 10.0 ** (log_lo + u[j] * (log_hi - log_lo))
        else:
            raise ValueError(f"Unknown scale '{scale}' for {name}.")
    return values


def latin_hypercube_candidates(
    n_samples: int,
    bounds: Mapping[str, Tuple[float, float, str]],
    seed: int,
):
    sampler = qmc.LatinHypercube(d=len(bounds), seed=seed)
    unit_samples = sampler.random(n=n_samples)
    return [_transform_sample(row, bounds) for row in unit_samples]


def refined_bounds(
    results: Iterable[CandidateResult],
    original_bounds: Mapping[str, Tuple[float, float, str]],
    padding_fraction: float = 0.15,
):
    """Build sweep-2 bounds around feasible points, or around the best points."""
    results = [r for r in results if r.success]
    if not results:
        return dict(original_bounds)

    feasible = [r for r in results if r.feasible]
    if feasible:
        focus = feasible
    else:
        focus = sorted(results, key=lambda r: r.violation_score)[: max(5, len(results) // 10)]

    new_bounds = {}
    for name, (orig_lo, orig_hi, scale) in original_bounds.items():
        vals = np.asarray([getattr(r, name) for r in focus])

        if scale == "linear":
            lo = float(np.min(vals))
            hi = float(np.max(vals))
            pad = padding_fraction * (orig_hi - orig_lo)
            lo = max(orig_lo, lo - pad)
            hi = min(orig_hi, hi + pad)
        else:
            log_vals = np.log10(vals)
            orig_log_lo = np.log10(orig_lo)
            orig_log_hi = np.log10(orig_hi)
            pad = padding_fraction * (orig_log_hi - orig_log_lo)
            lo = 10.0 ** max(orig_log_lo, float(np.min(log_vals)) - pad)
            hi = 10.0 ** min(orig_log_hi, float(np.max(log_vals)) + pad)

        # Avoid a degenerate second-stage interval.
        if np.isclose(lo, hi):
            lo, hi = orig_lo, orig_hi

        new_bounds[name] = (lo, hi, scale)

    return new_bounds


def run_sweep(candidates, label: str):
    results = []
    total = len(candidates)
    for i, candidate in enumerate(candidates, start=1):
        result = evaluate_candidate(**candidate)
        results.append(result)
        print(
            f"[{label} {i:4d}/{total}] "
            f"score={result.violation_score:.3e} "
            f"CL={result.CL:.4f} "
            f"S={result.S:.2f} AR={result.AR:.2f} V={result.V_cruise:.2f} "
            f"{'FEASIBLE' if result.feasible else ''}"
        )
    return results


def write_csv(results: Iterable[CandidateResult], path: Path):
    results = list(results)
    if not results:
        return

    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0].as_dict().keys()))
        writer.writeheader()
        writer.writerows(r.as_dict() for r in results)


def summarize(results: Iterable[CandidateResult]):
    results = [r for r in results if r.success]
    feasible = [r for r in results if r.feasible]

    print("\n" + "=" * 78)
    print(f"Successful inner optimizations: {len(results)}")
    print(f"Feasible tuning candidates:      {len(feasible)}")

    if feasible:
        print("\nObserved admissible parameter ranges from the sampled feasible cloud:")
        for name in SEARCH_BOUNDS:
            vals = np.asarray([getattr(r, name) for r in feasible])
            print(f"  {name:8s}: [{np.min(vals):.8g}, {np.max(vals):.8g}]")

        representative = min(feasible, key=lambda r: r.center_score)
        print("\nRepresentative interior candidate (smallest center score):")
        for key, value in representative.as_dict().items():
            print(f"  {key:16s}: {value}")

        print(
            "\nNOTE: the independent min/max values above do not form a guaranteed "
            "feasible box.\nThe three tuning parameters can be correlated; preserve "
            "the feasible point cloud when\nchoosing final values."
        )
    elif results:
        best = min(results, key=lambda r: r.violation_score)
        print("\nNo fully feasible candidates were found. Best sampled candidate:")
        for key, value in best.as_dict().items():
            print(f"  {key:16s}: {value}")
    else:
        print("\nNo inner optimization completed successfully.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n1", type=int, default=60, help="coarse sweep sample count")
    parser.add_argument("--n2", type=int, default=60, help="refined sweep sample count")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("tune_results.csv"))
    args = parser.parse_args()

    print("Stored tuning values used only as context/initial reference:")
    for name in SEARCH_BOUNDS:
        print(f"  {name:8s} = {tuning[name]}")

    sweep1_candidates = latin_hypercube_candidates(args.n1, SEARCH_BOUNDS, args.seed)
    sweep1 = run_sweep(sweep1_candidates, "sweep 1")

    bounds2 = refined_bounds(sweep1, SEARCH_BOUNDS)
    print("\nRefined sweep bounds:")
    for name, bounds in bounds2.items():
        print(f"  {name:8s}: {bounds}")

    sweep2_candidates = latin_hypercube_candidates(args.n2, bounds2, args.seed + 1)
    sweep2 = run_sweep(sweep2_candidates, "sweep 2")

    all_results = sweep1 + sweep2
    write_csv(all_results, args.output)
    summarize(all_results)
    print(f"\nWrote: {args.output.resolve()}")


if __name__ == "__main__":
    main()
