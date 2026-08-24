# Examples

Runnable examples for the `fair-seldonian` library. The notebooks below are
rendered from their **saved outputs** — figures and printed results appear
exactly as committed, without re-executing the code.

To run them yourself, install the notebook extras and open them with Jupyter:

```bash
pip install "fair-seldonian[notebook]"
jupyter lab
```

`quickstart` is the guided tour; `demographic_parity_guarantee` walks one
criterion end to end -- define it, choose the tolerance and the confidence, and
read the verdict; `real_world_adult` applies the same to a real dataset and
needs network access on first run to download it. The source `.py` scripts and full listing live in the
[`examples/` directory](https://github.com/parulgupta1004/fair-seldonian/tree/master/examples)
of the repository.

```{toctree}
:maxdepth: 1

quickstart
demographic_parity_guarantee
real_world_adult
```
