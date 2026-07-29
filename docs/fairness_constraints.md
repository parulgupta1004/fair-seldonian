# Fairness constraints

`fair-seldonian` ships builders for the most common fairness definitions, so you
can pass a named criterion straight to
{class}`~fair_seldonian.config.SeldonianConfig` instead of hand-writing the
reverse-Polish (postfix) constraint string:

```python
from fair_seldonian import SeldonianConfig, demographic_parity

config = SeldonianConfig(constraint=demographic_parity(epsilon=0.1))
```

Each builder takes a tolerance `epsilon` (smaller is stricter). The group-parity
builders also take a `groups=(g1, g0)` pair (default `("1", "0")`), while
`error_rate` takes a single `group`. Labels are matched against `str(T)`, so the
defaults line up with a 0/1 sensitive column.

The constraints are written over the per-group confusion-matrix cells `TP(g)`,
`FP(g)`, `FN(g)`, `TN(g)` — each a fraction of group `g` (the four sum to 1
within a group), so the predicted-positive rate is `TP + FP`, the true-positive
rate is `TP / (TP + FN)`, the false-positive rate is `FP / (FP + TN)`, and the
error rate is `FP + FN`.

## Available constraints

| Builder | Criterion | Bounds (`<= epsilon`) | Reference |
|---|---|---|---|
| `demographic_parity` | Statistical parity / independence | `\|PPR(g1) - PPR(g0)\|`, PPR = `TP + FP` | Dwork et al. (2012) |
| `equal_opportunity` | Equal true-positive rate | `\|TPR(g1) - TPR(g0)\|`, TPR = `TP / (TP + FN)` | Hardt et al. (2016) |
| `equalized_odds` | Separation (TPR **and** FPR) | `\|ΔTPR\| + \|ΔFPR\|`, FPR = `FP / (FP + TN)` | Hardt et al. (2016) |
| `error_rate` | Misclassification bound (one group) | `err(g)`, err = `FP + FN` | Thomas et al. (2019) |
| `error_rate_parity` | Overall accuracy equality | `\|err(g1) - err(g0)\|` | Berk et al. (2021) |

```{note}
`equal_opportunity` and `equalized_odds` divide by per-group base rates to form
conditional rates. Confidence bounds on a ratio are looser than on a difference,
so they need more data (or a tighter `inequality` such as `Inequality.T_TEST`) to
certify than the division-free constraints (`demographic_parity`, `error_rate`,
`error_rate_parity`).
```

## Custom constraints

The builders are just conveniences: `SeldonianConfig.constraint` accepts any
postfix string, so you can write your own over the same `TP/FP/FN/TN` primitives.
For example, demographic parity by hand is:

```python
SeldonianConfig(constraint="TP(1) FP(1) + TP(0) FP(0) + - abs 0.1 -")
```

`SeldonianConfig` validates the string on construction (via
{func}`~fair_seldonian.constraints.expression_tree.validate_constraint`), so a
malformed expression raises `ValueError` immediately instead of failing inside
QSA.

## References

- Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012). Fairness
  through awareness. *ITCS '12*. <https://arxiv.org/abs/1104.3913>
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in
  supervised learning. *NeurIPS 2016*. <https://arxiv.org/abs/1610.02413>
- Berk, R., Heidari, H., Jabbari, S., Kearns, M., & Roth, A. (2021). Fairness in
  criminal justice risk assessments: The state of the art. *Sociological Methods
  & Research*, 50(1), 3–44. <https://arxiv.org/abs/1703.09207>
- Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and Machine Learning:
  Limitations and Opportunities*. MIT Press. <https://fairmlbook.org>
- Thomas, P. S., da Silva, B. C., Barto, A. G., Giguère, S., Brun, Y., &
  Brunskill, E. (2019). Preventing undesirable behavior of intelligent machines.
  *Science*, 366(6468), 999–1004. <https://doi.org/10.1126/science.aag3311>

See {mod}`fair_seldonian.constraints.fairness` for the full descriptions of each
builder.
