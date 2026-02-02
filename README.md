# QSVM-EEG

Project structure

```shell
qsvm-eeg/
├── configs/                   # [Control Center]
│   └── default.yaml           # Define patients, window size, bands, model params here
│
├── data/                      # [Data Layer]
│   ├── raw/                   # Immutable inputs (e.g., patient48_eeg.csv)
│   └── processed/             # Cached features (e.g., features_48_411.npy)
│
├── reports/                   # [Report]
│   ├── figures/               # Plots for report
│   └── logs/                  # Logs for report
│
├── src/                       # [Package]
│   └── qsvm_eeg/
│       ├── __init__.py
│       ├── data.py            # Logic to load & combine specific patients
│       ├── features.py        # Signal processing
│       ├── quantum_kernel.py  #
│       │
│       └── models/            # Strategy
│           ├── __init__.py
│           ├── base.py        # Base (Train/Predict/Save)
│           ├── svr_rbf.py     # Classical Wrapper
│           ├── svr_qkernel.py # Quantum Wrapper
│           └── registry.py    # Factory to select 'svr_rbf' or 'svr_qkernel'
│
├── main.py                    # [Orchestrator] Handles CLI and loops over models
├── pyproject.toml             # Dependency management
└── README.md
```

## Prerequisites

- Python version >= 3.12

```shell
pip install -r requirements.txt
```

## Experiments
