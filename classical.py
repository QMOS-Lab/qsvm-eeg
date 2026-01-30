"""
Classical SVR Experiment Runner.

DESCRIPTION:
    This script runs the baseline classical SVR pipeline for comparison.
    It uses a standard Radial Basis Function (RBF) kernel SVM to predict BIS.

USAGE:
    # Run standalone
    python classical.py -p 48 411 -j 8

    # Run via main.py
    python main.py --mode classical -p 48 411

    # Check help
    python classical.py -h

NOTE: Running Classical SVR (RBF Kernel)
      Running with arguments:
          -p, --patients :    List of patient IDs (e.g., "48" "411").
                              Default: ["48", "411"]
          -n, --samples  :    Total number of samples to use (subsampled evenly).
                              Default: All available data.
          -j, --jobs     :    Number of CPU cores for Grid Search.
                              Default: -1 (Use all available cores).
"""

import sys
import time
import argparse
from typing import Optional, Dict, Tuple, List

import joblib
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

from qsvm_eeg.data import load_raw_data, trim_zero_ends
from qsvm_eeg.features import extract_features

FS: int = 128
AVAILABLE_PATIENTS: List[str] = ["48", "411"]

ROOT_DIR = Path.cwd()
DATA_DIR = ROOT_DIR / "data" / "raw"
REPORT_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORT_DIR / "figures"
LOGS_DIR = REPORT_DIR / "logs"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan> - <level>{message}</level>",
)
logger.add(LOGS_DIR / "benchmark_classical_{time}.log", rotation="50 MB", level="INFO")


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments to configure the classical experiment.
    """
    parser = argparse.ArgumentParser(description="Run Classical SVR (RBF) Experiment.")

    parser.add_argument(
        "-p", "--patients",
        nargs="+",
        default=AVAILABLE_PATIENTS,
        help=f"List of Patients. Default: {AVAILABLE_PATIENTS}",
    )

    parser.add_argument(
        "-n", "--samples",
        type=int,
        default=None,
        help="Total number of samples to use (distributed equally). Default: Use all data.",
    )

    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=-1,
        help="Number of CPU cores to use. Default: -1 (All cores).",
    )

    return parser.parse_args()


def log_results(metrics: Dict[str, float], params: Dict[str, any], experiment_id: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = LOGS_DIR / "experiment_log.csv"

    header = "Timestamp,Experiment_ID,Sample_N,MSE,RMSE,R2,Pearson_R,CI_95,C,Epsilon,Kernel,Train_Kernel_Sec,Infer_Kernel_Sec\n"
    if not log_file.exists():
        with open(log_file, "w") as f:
            f.write(header)

    with open(log_file, "a") as f:
        # Note: Kernel is labeled 'Classical_RBF' for comparison
        line = (
            f"{timestamp},{experiment_id},{params['n_samples']},{metrics['mse']:.5f},"
            f"{metrics['rmse']:.5f},{metrics['r2']:.5f},"
            f"{metrics['pearson']:.5f},{metrics['ci']:.5f},"
            f"{params['C']},{params['epsilon']},Classical_RBF,"
            f"{metrics['train_time']:.4f},"
            f"{metrics['infer_time']:.4f}\n"
        )
        f.write(line)
    logger.success(f"Logged to CSV: {log_file}")


def process_single_patient(pid: str, limit_per_patient: Optional[int]) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads, cleans, and extracts features for a single patient.

    Args:
        pid (str): Patient ID.
        limit_per_patient (Optional[int]): Number of samples to take.

    Returns:
        tuple: (X_features, y_labels)
    """
    logger.info(f"Processing Patient {pid}")
    eeg_path = DATA_DIR / f"patient{pid}_eeg.csv"
    bis_path = DATA_DIR / f"patient{pid}_bis.csv"

    eeg_raw, bis_raw = load_raw_data(eeg_path, bis_path)
    if eeg_raw is None:
        return None, None

    # Trim silent periods
    eeg, bis = trim_zero_ends(eeg_raw, bis_raw, fs_eeg=FS)

    # Extract spectral/entropy features
    X = extract_features(eeg, fs=FS)

    # Align labels (Predict future BIS)
    advance_steps = 60
    y = bis[advance_steps:]
    min_len = min(len(X), len(y))
    X, y = X[:min_len], y[:min_len]

    # Subsampling
    if limit_per_patient is not None:
        if len(X) > limit_per_patient:
            logger.info(f"Patient {pid}: Subsampling {len(X)} -> {limit_per_patient}")
            indices = np.linspace(0, len(X) - 1, limit_per_patient).astype(int)
            X, y = X[indices], y[indices]

    return X, y


def run_classical(args: argparse.Namespace) -> dict:
    # Create Experiment ID
    if len(args.patients) == 1:
        experiment_id = f"Single_{args.patients[0]}"
    else:
        experiment_id = f"Mix_{'_'.join(args.patients)}"

    logger.info(f"Starting Classical SVR (RBF) Experiment: {experiment_id}")
    logger.info(f"Config: Samples={args.samples if args.samples else 'ALL'} | Jobs={args.jobs}")

    # Start: Load & Process Data
    X_combined, y_combined = [], []
    limit = args.samples // len(args.patients) if args.samples else None

    for pid in args.patients:
        X_p, y_p = process_single_patient(pid, limit)
        if X_p is not None:
            X_combined.append(X_p)
            y_combined.append(y_p)

    if not X_combined:
        return

    X = np.vstack(X_combined)
    y = np.concatenate(y_combined)

    logger.info("Shuffling combined dataset")
    p = np.random.RandomState(42).permutation(len(X))
    X, y = X[p], y[p]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    # Start: Scaling
    # Classical SVM works best with StandardScaler (Mean=0, Std=1)
    logger.info("Scaling Data (StandardScaler)")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Start: Hyperparameter tunning
    logger.info("Starting Grid Search (Classical)")

    param_grid = {
        "C": [0.1, 1, 10, 50, 100, 500, 1000],
        "epsilon": [0.1, 0.5, 1.0, 2.0, 4.0],
        "gamma": ["scale", "auto"],
    }

    t0_train = time.perf_counter()

    # Use args.jobs to control parallelism
    n_jobs = getattr(args, 'jobs', -1)

    grid_search = GridSearchCV(
        SVR(kernel="rbf"),
        param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=n_jobs,
        verbose=0,
    )

    grid_search.fit(X_train_scaled, y_train)
    train_time = time.perf_counter() - t0_train

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    logger.success(
        f"Best Params: {best_params} | Best CV RMSE: {-grid_search.best_score_:.4f}"
    )
    logger.info(f"BENCHMARK | Total Tuning Time: {train_time:.4f}s")

    # Start: Test
    t0_infer = time.perf_counter()
    y_pred = best_model.predict(X_test_scaled)
    infer_time = time.perf_counter() - t0_infer

    # Start: Evaluation
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    r_val, _ = pearsonr(y_test, y_pred)
    ci = 1.96 * np.std(y_pred - y_test) / np.sqrt(len(y_pred))

    logger.success(f"CLASSICAL RESULTS | RMSE: {rmse:.4f} | R2: {r2:.4f} | 95% CI: {ci:.4f}")

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "pearson": r_val,
        "ci": ci,
        "train_time": train_time,
        "infer_time": infer_time,
    }

    params = {
        "n_samples": len(X),
        "C": best_params["C"],
        "epsilon": best_params["epsilon"],
    }

    log_results(metrics, params, experiment_id)

    # Optional Plotting
    fig = plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Classical RBF', linestyle='--')
    plt.title(f"Classical RBF: {experiment_id} (N={len(X)}) | R2={r2:.2f}")
    plt.legend()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = FIGURES_DIR / f"classical_pred_{experiment_id}_{timestamp}.png"
    fig.savefig(filename, dpi=300)

    return metrics


def main():
    args = parse_arguments()
    run_classical(args)


if __name__ == "__main__":
    main()
