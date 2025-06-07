# Approximate exp function stress test

The `compare_exp_accuracy.py` script evaluates the `approx_exp` function on 100k
values in `[-20, 20]`. A recent run produced:

```
max ULP error: 1
mean ULP error: 0.00769
```

A separate stress test (`stress_test_approx_exp.py`) samples one million random
inputs from `[-120, 120]` and checks special cases. The summary from a run is:

```
Tested 1000008 samples
Within range max ULP: 1
Within range mean ULP: 0.00661
NaN mismatches: 0, Inf mismatches: 0
```

These results confirm that `approx_exp` matches the precision of `torch.exp`
within the normal floating‑point range while correctly handling overflow,
underflow and special values.
