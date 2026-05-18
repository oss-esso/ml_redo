# oncology-sbi-pkpd

Amortized simulation-based inference for oncology PKPD tumor-growth models.

This package implements Neural Posterior Estimation (NPE) for patient-specific
parameter inference in Simeoni-style tumor growth inhibition models, with
hierarchical partial pooling for sparse longitudinal observations.

## Quickstart

```bash
# Create environment
conda env create -f environment.yml
conda activate oncology-sbi

# Install package in development mode
pip install -e ".[dev]"

# Run tests (do this FIRST)
pytest

# Milestone 1: Smoke test — generate synthetic data and plot
python -m src.pkpd_sbi.experiments.smoke_simulate

# Milestone 2: Train NPE and run full validation
python -m src.pkpd_sbi.experiments.train_and_evaluate --n-train 10000 --n-sbc 100

# Full run (takes ~30 min on CPU, ~5 min on GPU)
python -m src.pkpd_sbi.experiments.train_and_evaluate --n-train 50000 --n-sbc 300 --device cuda
```

## Project structure

```
oncology-sbi-pkpd/
├── src/pkpd_sbi/
│   ├── simulators/
│   │   ├── simeoni.py          # Tumor-growth ODE (the core simulator)
│   │   ├── dosing.py           # Drug dosing schedules
│   │   ├── observation.py      # Log-normal observation model
│   │   └── priors.py           # Biologically calibrated prior distributions
│   ├── data/
│   │   └── synthetic.py        # Virtual patient cohort generator
│   ├── inference/
│   │   ├── npe.py              # Neural Posterior Estimation (sbi wrapper)
│   │   └── summaries.py        # Trajectory encoders for SBI
│   ├── validation/
│   │   ├── sbc.py              # Simulation-Based Calibration
│   │   ├── coverage.py         # Credible interval coverage + metrics
│   │   └── plots.py            # Paper figure generation
│   └── experiments/
│       ├── smoke_simulate.py   # MVP: generate and plot synthetic data
│       └── train_and_evaluate.py  # Full Phase 1 pipeline
├── configs/
│   └── default.yaml            # All hyperparameters in one place
├── tests/
│   └── test_simulator.py       # Simulator sanity tests
├── outputs/                    # Generated figures and CSVs (gitignored)
├── environment.yml
├── pyproject.toml
└── README.md
```

## Milestones

| # | Milestone | Command | What it proves |
|---|-----------|---------|----------------|
| 1 | Smoke test | `smoke_simulate` | Simulator works, data pipeline works |
| 2 | NPE training | `train_and_evaluate --n-train 10000` | SBI learns something |
| 3 | SBC validation | `train_and_evaluate --n-sbc 300` | Posteriors are calibrated |
| 4 | Parameter recovery | Check `parameter_recovery.png` | Inference is accurate |
| 5 | PPC | Check `ppc_patient_*.png` | Predictions match data |

## Mapping from LaTeX to code

Every equation in the project plan maps to a function:

| LaTeX | Function | File |
|-------|----------|------|
| `dx1/dt = g(x1) - k_kill C(t) x1` | `simeoni_rhs()` | `simulators/simeoni.py` |
| `g(x1) = lam0*x1 / (...)` | `growth_term()` | `simulators/simeoni.py` |
| `log y = log V + epsilon` | `observe_lognormal()` | `simulators/observation.py` |
| `C(t) = sum doses * exp(...)` | `DoseSchedule.concentration()` | `simulators/dosing.py` |
| `q_phi(theta|x) ≈ p(theta|x)` | `build_npe()` | `inference/npe.py` |
| `SBC rank uniformity` | `run_sbc()` | `validation/sbc.py` |
| `Coverage of CI` | `compute_coverage()` | `validation/coverage.py` |

## References

- Simeoni et al., Cancer Research, 2004 (tumor growth model)
- Talts et al., arXiv:1804.06788, 2018 (SBC validation)
- Cranmer et al., PNAS, 2020 (SBI review)
- Boelts et al., JOSS, 2025 (sbi package)
