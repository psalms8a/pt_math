# pt_math

This repository contains resources for experiments with PyTorch-based math models.

## Setting up the `ptmath` CPU environment

Use the provided setup script to create a Python virtual environment with CPU-only PyTorch:

```bash
bash setup_ptmath_cpu.sh
```

Activate the environment with:

```bash
source ptmath/bin/activate
```

## Measuring `torch.exp` accuracy

Run the `measure_exp_accuracy.py` script inside the environment to compute the ULP error of `torch.exp` in float32 against a double precision reference:

```bash
python measure_exp_accuracy.py
```

The script prints summary statistics and saves the ULP errors to `exp_accuracy.csv`.

