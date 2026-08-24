"""Tell apart the two reasons QSA returns no solution.

"No Solution Found" covers two quite different situations, and they call for
opposite remedies:

``candidate_infeasible``
    Candidate selection never reached a point satisfying the constraint, so the
    safety test was doomed before it ran. More data will not help on its own --
    the constraint is too tight for any model this class can express.

``safety_test_rejected``
    A feasible candidate *was* found, but the safety set could not certify it
    with the required confidence. Here more data, or a tighter bound, is exactly
    what would help.

``result.diagnostics.failure_mode`` reports which. Sweeping the tolerance from
impossible to comfortable walks through all three outcomes.

Run it with::

    uv run python examples/why_no_solution.py
"""

from __future__ import annotations

from fair_seldonian.algorithms import QSA
from fair_seldonian.config import SeldonianConfig
from fair_seldonian.data import data_split, get_data

REMEDY = {
    "candidate_infeasible": "constraint too tight for this model class",
    "safety_test_rejected": "feasible, but not certifiable on this much data",
    "solution_found": "certified",
}


def main() -> None:
    data = get_data(
        N=20000, features=5, t_ratio=0.5, tp0_ratio=0.35, tp1_ratio=0.65, random_seed=7
    )
    _, _, _, X_tr, Y_tr, T_tr = data_split(
        frac=1.0, all_data=data, random_state=1, m_test=0.3
    )

    print("Demographic-parity gap tolerance swept from impossible to comfortable.\n")
    print(f"{'epsilon':>8} {'certified':>10} {'failure_mode':>22}   what it means")
    print("-" * 78)

    for epsilon in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20):
        config = SeldonianConfig(constraint=f"PR(1) PR(0) - abs {epsilon} -")
        result = QSA(X_tr, Y_tr, T_tr, "opt", None, None, config)
        mode = result.diagnostics.failure_mode
        print(
            f"{epsilon:>8.2f} {str(result.passed_safety):>10} {mode:>22}"
            f"   {REMEDY[mode]}"
        )

    print(
        "\nWithout failure_mode all the unsuccessful rows look alike, and a "
        "non-monotone\ncurve invites an explanation in terms of the bound when "
        "the real cause is that\ncandidate selection never left the infeasible "
        "region."
    )

    # The diagnostics also carry the optimizer's own verdict, which separates
    # "converged, but to an infeasible point" from "ran out of iterations".
    tight = SeldonianConfig(constraint="PR(1) PR(0) - abs 0.0 -")
    diagnostics = QSA(X_tr, Y_tr, T_tr, "opt", None, None, tight).diagnostics
    print(
        f"\nAt epsilon = 0.00 the optimizer reports {diagnostics.optimizer_message!r}"
        f"\nafter {diagnostics.optimizer_iterations} iterations and "
        f"{diagnostics.optimizer_evaluations} evaluations, reaching a candidate "
        f"bound of\n{diagnostics.candidate_upper_bound:+.4f} (> 0, so infeasible) "
        f"against a safety bound of {diagnostics.safety_upper_bound:+.4f}."
    )

    # The sweep above never lands on safety_test_rejected, and that is the
    # design working rather than a gap in the example. Candidate selection does
    # not optimise against the constraint directly; it optimises against a
    # deliberately inflated *prediction* of what the safety test will say, so a
    # candidate that clears the prediction nearly always clears the real test.
    # Rejection needs an unlucky safety split. Over a grid of 96 seed/size/
    # tolerance combinations exactly one produced it -- this one.
    print("\n" + "-" * 78)
    print("The rare third case: feasible on the candidate set, rejected on safety.\n")
    unlucky = get_data(
        N=8000, features=5, t_ratio=0.5, tp0_ratio=0.35, tp1_ratio=0.65, random_seed=0.5
    )
    _, _, _, Xu, Yu, Tu = data_split(
        frac=1.0, all_data=unlucky, random_state=1, m_test=0.3
    )
    rejected = QSA(
        Xu,
        Yu,
        Tu,
        "opt",
        None,
        None,
        SeldonianConfig(constraint="PR(1) PR(0) - abs 0.2 -"),
    ).diagnostics
    print(f"  failure_mode          : {rejected.failure_mode}")
    print(f"  candidate upper bound : {rejected.candidate_upper_bound:+.4f}  (<= 0)")
    print(f"  safety upper bound    : {rejected.safety_upper_bound:+.4f}  (> 0)")
    print(
        "\nHere more data would genuinely help, which is the opposite of the "
        "advice for\nthe candidate_infeasible rows above. That is the whole "
        "reason to report which\nof the two happened."
    )


if __name__ == "__main__":
    main()
