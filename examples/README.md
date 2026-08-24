# Examples

Runnable examples for the `fair-seldonian` library. The `.py` scripts are
self-contained and print their results; run any of them with:

```bash
uv run python examples/<script>.py
# or, once the package is installed:
python examples/<script>.py
```

The `.ipynb` notebooks ship with saved outputs (so they render on GitHub);
open them with `jupyter lab` after installing the notebook extras
(`pip install "fair-seldonian[notebook]"`).

| Example | What it shows |
|---------|---------------|
| [`quickstart.py`](quickstart.py) | Minimal end-to-end flow: generate data → `data_split` → `QSA` → read the certified fairness bound. |
| [`fairness_guarantee.py`](fairness_guarantee.py) | The one-sided guarantee: a fair dataset is certified, while an unfair one returns **No Solution Found**. |
| [`custom_constraint.py`](custom_constraint.py) | Customizing `SeldonianConfig` — `delta`, the concentration `inequality`, `candidate_ratio`, and the postfix `constraint` string. |
| [`tighter_bounds.py`](tighter_bounds.py) | Where the slack goes: the same fixed model certified by each `seldonian_type` (`base`, `opt`, `affine`) crossed with each inequality (Hoeffding, empirical Bernstein, betting). |
| [`why_no_solution.py`](why_no_solution.py) | Using `result.diagnostics.failure_mode` to tell `candidate_infeasible` from `safety_test_rejected` — two refusals that call for opposite remedies. |
| [`cells_vs_rates.py`](cells_vs_rates.py) | `TP(g)` is a joint probability over the whole group; `TPR(g)` is a rate over its positive rows. Constraining one is not constraining the other. |
| [`minority_group.py`](minority_group.py) | Why a small protected group is expensive however large the dataset, and which inequality softens it. |
| [`real_world_adult.ipynb`](real_world_adult.ipynb) | A real dataset (UCI Adult income): an unconstrained model's demographic-parity gap vs. QSA refusing to certify it. Ships with saved outputs so results render on GitHub; needs network access to re-run. |
| [`quickstart.ipynb`](quickstart.ipynb) | The full guided notebook: data generation, certification, fair vs. unfair, constraint decoding, all the algorithm variants, and accuracy-vs-fairness plots. |

## Notes

- The synthetic generator (`get_data`) adds Gaussian noise to the label to form
  its signal feature, giving a Bayes error of about 16% — so unconstrained
  accuracy lands near 0.84, and enforcing a fairness constraint has a real
  accuracy cost to pay. `real_world_adult.ipynb` shows the same on real data.
- Throughout the library, the **last column of `X` is the sensitive attribute
  `T`**, and `T` is also passed separately for computing group rates.
- A negative `eval_ghat` value is a satisfied constraint (an upper bound on
  `g(theta)` that sits below zero).
