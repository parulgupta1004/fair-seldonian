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
| [`real_world_adult.ipynb`](real_world_adult.ipynb) | A real dataset (UCI Adult income): an unconstrained model's demographic-parity gap vs. QSA refusing to certify it. Ships with saved outputs so results render on GitHub; needs network access to re-run. |
| [`quickstart.ipynb`](quickstart.ipynb) | The full guided notebook: data generation, certification, fair vs. unfair, constraint decoding, all five algorithm variants, and accuracy-vs-fairness plots. |

## Notes

- The synthetic generator (`get_data`) produces a trivially separable feature,
  so accuracy on synthetic data is ~1.0; it is meant to illustrate the fairness
  machinery, not model difficulty. `real_world_adult.ipynb` shows realistic numbers.
- Throughout the library, the **last column of `X` is the sensitive attribute
  `T`**, and `T` is also passed separately for computing group rates.
- A negative `eval_ghat` value is a satisfied constraint (an upper bound on
  `g(theta)` that sits below zero).
