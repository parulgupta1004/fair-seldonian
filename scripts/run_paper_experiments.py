"""Regenerate every figure and table in the paper.

Writes per-variant panels into ``<out>/<variant>/`` using the filenames the
LaTeX source already references, plus ``summary.csv`` per variant and a combined
``tables/*.csv`` for the numeric tables.

Usage::

    python scripts/run_paper_experiments.py --out exp/paper \\
        --trials 40 --jobs 8

``--out`` is required and may be any directory; ``exp/`` is gitignored.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from fair_seldonian.config import SeldonianConfig  # noqa: E402
from fair_seldonian.constraints.inequalities import Inequality  # noqa: E402
from fair_seldonian.experiments import (  # noqa: E402
    StudySpec,
    plot_all,
    run_study,
    save_summary,
    summarise,
)

logger = logging.getLogger(__name__)

#: The constraint from the paper: |TP(0) - TP(1)| - 0.2 * TP(1) <= 0.
PAPER_CONSTRAINT = "TP(1) TP(0) - abs 0.2 TP(1) * -"

MS = (2_000, 5_000, 10_000, 20_000, 40_000, 80_000, 160_000, 320_000)

#: Directory name -> (seldonian_type, config overrides, spec overrides).
#: Directory names match the ones the LaTeX source already includes.
VARIANTS: dict[str, tuple[str, dict, dict]] = {
    "base": ("base", {}, {}),
    "mod": ("mod", {}, {}),
    "const": ("const", {}, {}),
    "bound": ("bound", {}, {}),
    "opt": ("opt", {}, {}),
    "affine": ("affine", {}, {}),
    "bfgs": ("base", {"optimizer": "BFGS"}, {}),
    "force/powell": ("base", {}, {"force_start": True}),
    "force/bfgs": ("base", {"optimizer": "BFGS"}, {"force_start": True}),
    "ebernstein": ("base", {"inequality": Inequality.EMPIRICAL_BERNSTEIN}, {}),
    "ttest": ("base", {"inequality": Inequality.T_TEST}, {}),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--eval-size", type=int, default=200_000)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument(
        "--constraint",
        default=PAPER_CONSTRAINT,
        help="postfix constraint; defaults to the study constraint",
    )
    parser.add_argument(
        "--tag", default="", help="prefix for output directories, e.g. 'eo'"
    )
    parser.add_argument("--ms", nargs="*", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)

    all_rows: dict[str, list[dict[str, float]]] = {}
    names = args.only or list(VARIANTS)
    for name in names:
        seldonian_type, config_kwargs, spec_kwargs = VARIANTS[name]
        config = SeldonianConfig(
            constraint=args.constraint,
            delta=0.05,
            candidate_ratio=0.60,
            **config_kwargs,
        )
        spec = StudySpec(
            seldonian_type=seldonian_type,
            ms=tuple(args.ms) if args.ms else MS,
            num_trials=args.trials,
            eval_size=args.eval_size,
            config=config,
            **spec_kwargs,
        )
        started = time.time()
        records = run_study(spec, n_jobs=args.jobs)
        rows = summarise(records)
        all_rows[name] = rows

        out_dir = os.path.join(args.out, f"{args.tag}/{name}" if args.tag else name)
        os.makedirs(out_dir, exist_ok=True)
        plot_all(rows, out_dir)
        save_summary(rows, os.path.join(out_dir, "summary.csv"))
        logger.info(
            "%-14s %5.0fs  P(soln)@max=%.2f  loss@max=%s",
            name,
            time.time() - started,
            rows[-1]["p_solution"],
            "n/a"
            if not np.isfinite(rows[-1]["log_loss"])
            else f"{rows[-1]['log_loss']:.4f}",
        )

    suffix = f"_{args.tag}" if args.tag else ""
    with open(
        os.path.join(args.out, "tables", f"all_variants{suffix}.json"), "w"
    ) as fh:
        json.dump(all_rows, fh, indent=1)

    # Main table: one row per variant at each size, for the paper.
    table_path = os.path.join(args.out, "tables", f"main_table{suffix}.csv")
    with open(table_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "variant",
                "m",
                "p_solution",
                "p_solution_lo",
                "p_solution_hi",
                "p_violation",
                "p_violation_hi",
                "log_loss",
                "log_loss_se",
                "n_solutions",
                "g_mean",
                "frac_candidate_infeasible",
                "frac_safety_rejected",
            ]
        )
        for name, rows in all_rows.items():
            for r in rows:
                writer.writerow(
                    [
                        name,
                        int(r["m"]),
                        f"{r['p_solution']:.3f}",
                        f"{r['p_solution_lo']:.3f}",
                        f"{r['p_solution_hi']:.3f}",
                        f"{r['p_violation']:.3f}",
                        f"{r['p_violation_hi']:.3f}",
                        ""
                        if not np.isfinite(r["log_loss"])
                        else f"{r['log_loss']:.4f}",
                        ""
                        if not np.isfinite(r["log_loss_se"])
                        else f"{r['log_loss_se']:.4f}",
                        int(r["n_solutions"]),
                        "" if not np.isfinite(r["g_mean"]) else f"{r['g_mean']:.4f}",
                        f"{r['frac_candidate_infeasible']:.2f}",
                        f"{r['frac_safety_rejected']:.2f}",
                    ]
                )
    logger.info("Wrote %s", table_path)


if __name__ == "__main__":
    main()
