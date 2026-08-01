# -*- coding: utf-8 -*-
"""
Simulation experiment for robust ICP under data scarcity.

Experimental settings:
- Dataset: 6,000 two-dimensional Gaussian samples, binary classes.
- Fixed test set across repeats.
- Training-pool sizes: n = 100, 200, 400, 800.
- For each n, repeat 10 stratified samplings from the available training pool.
- Inside each sampled training pool, split proper/calibration as 80:20.
- Calibration is pooled across all calibration samples.
- Methods: ICP, BICP, RICP, OOB-ICP.
- BICP/RICP/OOB-ICP use 100 resampling/repartitioning iterations.

Outputs are CSV files under ./results and are designed to be consumed by
plot_all_paper_results.py.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================
# Configuration
# =========================
SEED = 42
N_TOTAL = 6000
TEST_SIZE = 0.20
N_LIST = [100, 200, 400, 800]
N_REPEATS = 10
N_BOOT = 100
PROPER_RATIO = 0.80
ALPHA = 0.10
N_BINS = 5
OUTPUT_DIR = "results"
METHOD_ORDER = ["ICP", "BICP", "RICP", "OOB-ICP"]
DATASET_NAME = "Simulation"


# =========================
# Basic utilities
# =========================
def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def generate_overlapping_data(n_samples: int = 6000, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the two-class overlapping Gaussian simulation used in the manuscript."""
    rng = np.random.default_rng(random_state)
    mean0 = np.array([0.5, 0.5])
    mean1 = np.array([-0.5, -0.5])
    cov = np.array([[1.5, 0.3], [0.3, 1.5]])
    n0 = n_samples // 2
    n1 = n_samples - n0
    X0 = rng.multivariate_normal(mean0, cov, size=n0)
    X1 = rng.multivariate_normal(mean1, cov, size=n1)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    return X, y


def stratified_sample_indices(y: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """Sample n_samples from y while approximately preserving class proportions."""
    if n_samples > len(y):
        raise ValueError(f"Requested n={n_samples}, but only {len(y)} samples are available.")
    idx_all = np.arange(len(y))
    _, idx_sample = train_test_split(
        idx_all,
        test_size=n_samples,
        stratify=y,
        random_state=seed,
    )
    return np.asarray(idx_sample, dtype=int)


def split_proper_cal_indices(y_pool: np.ndarray, seed: int, proper_ratio: float = 0.80) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified 80:20 split inside the current training pool."""
    idx = np.arange(len(y_pool))
    proper_idx, cal_idx = train_test_split(
        idx,
        train_size=proper_ratio,
        stratify=y_pool,
        random_state=seed,
    )
    return np.asarray(proper_idx, dtype=int), np.asarray(cal_idx, dtype=int)


# =========================
# Pooled conformal prediction
# =========================
def cpsc_nonconformity(proba_row: np.ndarray, class_col: int) -> float:
    """
    Manuscript nonconformity score:
        alpha(x,y) = 0.5 - (p_y - max_{y' != y} p_y') / 2.
    """
    p_y = float(proba_row[class_col])
    if proba_row.shape[0] == 1:
        max_other = 0.0
    else:
        mask = np.ones(proba_row.shape[0], dtype=bool)
        mask[class_col] = False
        max_other = float(np.max(proba_row[mask]))
    return 0.5 - (p_y - max_other) / 2.0


def pvalues_from_pooled_calibration(model, X_test: np.ndarray, pooled_cal_scores: np.ndarray) -> np.ndarray:
    """Compute p-values for all candidate labels using one pooled calibration distribution."""
    proba_test = model.predict_proba(X_test)
    n_test, n_classes = proba_test.shape
    pvals = np.zeros((n_test, n_classes), dtype=float)
    m = len(pooled_cal_scores)
    for i in range(n_test):
        for k in range(n_classes):
            score = cpsc_nonconformity(proba_test[i], k)
            pvals[i, k] = (np.sum(pooled_cal_scores >= score) + 1.0) / (m + 1.0)
    return pvals


def fit_lr(X_train: np.ndarray, y_train: np.ndarray, seed: int):
    # Scaling is learned only from the fitting subset in each ICP iteration.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="liblinear", max_iter=2000, random_state=seed)),
    ])
    model.fit(X_train, y_train)
    return model


def pooled_calibration_scores(model, X_cal: np.ndarray, y_cal: np.ndarray) -> np.ndarray:
    """Compute one pooled calibration score array using each calibration sample's true label."""
    proba_cal = model.predict_proba(X_cal)
    class_to_col = {int(c): j for j, c in enumerate(model.classes_)}
    scores = np.array(
        [cpsc_nonconformity(proba_cal[i], class_to_col[int(y_cal[i])]) for i in range(len(y_cal))],
        dtype=float,
    )
    return scores


@dataclass
class ICPOutput:
    p_values: np.ndarray
    pred_labels: np.ndarray
    credibility: np.ndarray
    confidence: np.ndarray


def output_from_pvalues(pvals: np.ndarray, classes: np.ndarray) -> ICPOutput:
    pred = classes[np.argmax(pvals, axis=1)].astype(int)
    cred = np.max(pvals, axis=1)
    sorted_p = np.sort(pvals, axis=1)
    second = sorted_p[:, -2] if pvals.shape[1] > 1 else np.zeros_like(cred)
    conf = 1.0 - second
    return ICPOutput(p_values=pvals, pred_labels=pred, credibility=cred, confidence=conf)


def run_icp(X_pool: np.ndarray, y_pool: np.ndarray, X_test: np.ndarray, seed: int) -> ICPOutput:
    proper_idx, cal_idx = split_proper_cal_indices(y_pool, seed=seed, proper_ratio=PROPER_RATIO)
    X_proper, y_proper = X_pool[proper_idx], y_pool[proper_idx]
    X_cal, y_cal = X_pool[cal_idx], y_pool[cal_idx]
    model = fit_lr(X_proper, y_proper, seed=seed)
    cal_scores = pooled_calibration_scores(model, X_cal, y_cal)
    pvals = pvalues_from_pooled_calibration(model, X_test, cal_scores)
    return output_from_pvalues(pvals, model.classes_)


def stratified_bootstrap(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Class-wise bootstrap to avoid losing minority classes in small samples."""
    xs, ys = [], []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        boot_idx = rng.choice(idx_c, size=len(idx_c), replace=True)
        xs.append(X[boot_idx])
        ys.append(y[boot_idx])
    Xb = np.vstack(xs)
    yb = np.concatenate(ys)
    perm = rng.permutation(len(yb))
    return Xb[perm], yb[perm]


def run_bicp(X_pool: np.ndarray, y_pool: np.ndarray, X_test: np.ndarray, seed: int, n_boot: int = 100) -> ICPOutput:
    proper_idx, cal_idx = split_proper_cal_indices(y_pool, seed=seed, proper_ratio=PROPER_RATIO)
    X_proper, y_proper = X_pool[proper_idx], y_pool[proper_idx]
    X_cal, y_cal = X_pool[cal_idx], y_pool[cal_idx]
    rng = np.random.default_rng(seed + 10000)
    classes = np.unique(y_pool)
    p_sum = np.zeros((len(X_test), len(classes)), dtype=float)
    for b in range(n_boot):
        Xb, yb = stratified_bootstrap(X_proper, y_proper, rng)
        model = fit_lr(Xb, yb, seed=seed + b + 1)
        cal_scores = pooled_calibration_scores(model, X_cal, y_cal)
        p_sum += pvalues_from_pooled_calibration(model, X_test, cal_scores)
    return output_from_pvalues(p_sum / float(n_boot), classes)


def run_ricp(X_pool: np.ndarray, y_pool: np.ndarray, X_test: np.ndarray, seed: int, n_random: int = 100) -> ICPOutput:
    """Repeatedly re-partition the same training pool into 80:20 proper/calibration subsets."""
    classes = np.unique(y_pool)
    p_sum = np.zeros((len(X_test), len(classes)), dtype=float)
    for r in range(n_random):
        proper_idx, cal_idx = split_proper_cal_indices(y_pool, seed=seed + 20000 + r, proper_ratio=PROPER_RATIO)
        X_proper, y_proper = X_pool[proper_idx], y_pool[proper_idx]
        X_cal, y_cal = X_pool[cal_idx], y_pool[cal_idx]
        model = fit_lr(X_proper, y_proper, seed=seed + 20000 + r)
        cal_scores = pooled_calibration_scores(model, X_cal, y_cal)
        p_sum += pvalues_from_pooled_calibration(model, X_test, cal_scores)
    return output_from_pvalues(p_sum / float(n_random), classes)


def oob_split_from_pool(X_pool: np.ndarray, y_pool: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Class-wise bootstrap: in-bag samples train the model, out-of-bag samples calibrate it."""
    x_train, y_train, x_cal, y_cal = [], [], [], []
    for c in np.unique(y_pool):
        idx_c = np.where(y_pool == c)[0]
        n_c = len(idx_c)
        for _ in range(500):
            sampled_local = rng.choice(np.arange(n_c), size=n_c, replace=True)
            unique_local = np.unique(sampled_local)
            oob_mask = np.ones(n_c, dtype=bool)
            oob_mask[unique_local] = False
            oob_local = np.where(oob_mask)[0]
            if len(oob_local) > 0:
                x_train.append(X_pool[idx_c[sampled_local]])
                y_train.append(y_pool[idx_c[sampled_local]])
                x_cal.append(X_pool[idx_c[oob_local]])
                y_cal.append(y_pool[idx_c[oob_local]])
                break
        else:
            raise RuntimeError(f"Failed to create non-empty OOB calibration set for class {c}.")
    Xtr = np.vstack(x_train)
    ytr = np.concatenate(y_train)
    Xca = np.vstack(x_cal)
    yca = np.concatenate(y_cal)
    p1 = rng.permutation(len(ytr))
    p2 = rng.permutation(len(yca))
    return Xtr[p1], ytr[p1], Xca[p2], yca[p2]


def run_oobicp(X_pool: np.ndarray, y_pool: np.ndarray, X_test: np.ndarray, seed: int, n_boot: int = 100) -> ICPOutput:
    rng = np.random.default_rng(seed + 30000)
    classes = np.unique(y_pool)
    p_sum = np.zeros((len(X_test), len(classes)), dtype=float)
    for b in range(n_boot):
        Xtr, ytr, Xca, yca = oob_split_from_pool(X_pool, y_pool, rng)
        model = fit_lr(Xtr, ytr, seed=seed + 30000 + b)
        cal_scores = pooled_calibration_scores(model, Xca, yca)
        p_sum += pvalues_from_pooled_calibration(model, X_test, cal_scores)
    return output_from_pvalues(p_sum / float(n_boot), classes)


# =========================
# Evaluation and output
# =========================
def make_output_df(y_true: np.ndarray, out: ICPOutput, method: str, repeat: int, n_train: int) -> pd.DataFrame:
    data = {
        "dataset": DATASET_NAME,
        "n_train": n_train,
        "repeat": repeat,
        "method": method,
        "sample_idx": np.arange(len(y_true)),
        "y_true": y_true.astype(int),
        "y_pred": out.pred_labels.astype(int),
        "credibility": out.credibility,
        "confidence": out.confidence,
        "is_correct": (out.pred_labels.astype(int) == y_true.astype(int)).astype(int),
    }
    for k in range(out.p_values.shape[1]):
        data[f"p{k}"] = out.p_values[:, k]
    return pd.DataFrame(data)


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "credibility_mean": float(df["credibility"].mean()),
        "credibility_std": float(df["credibility"].std(ddof=0)),
        "credibility_correct_mean": float(df.loc[df["is_correct"] == 1, "credibility"].mean()) if (df["is_correct"] == 1).any() else np.nan,
        "credibility_wrong_mean": float(df.loc[df["is_correct"] == 0, "credibility"].mean()) if (df["is_correct"] == 0).any() else np.nan,
    }


def grouped_metrics(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    tmp = df.copy()
    # Rank first so tied conformal p-values still produce exactly five equal-frequency bins.
    tmp["group"] = pd.qcut(
        tmp["credibility"].rank(method="first"), q=n_bins, labels=False
    ).astype(int) + 1
    rows = []
    for g in sorted(tmp["group"].unique()):
        sub = tmp[tmp["group"] == g]
        rows.append({
            "dataset": DATASET_NAME,
            "n_train": int(sub["n_train"].iloc[0]),
            "repeat": int(sub["repeat"].iloc[0]),
            "method": str(sub["method"].iloc[0]),
            "group": int(g),
            "count": int(len(sub)),
            "cred_mean": float(sub["credibility"].mean()),
            "accuracy": float(accuracy_score(sub["y_true"], sub["y_pred"])),
            "f1": float(f1_score(sub["y_true"], sub["y_pred"], average="macro", zero_division=0)),
            "f1_macro": float(f1_score(sub["y_true"], sub["y_pred"], average="macro", zero_division=0)),
        })
    return pd.DataFrame(rows)


def main():
    out_dir = os.path.join(script_dir(), OUTPUT_DIR)
    ensure_dir(out_dir)

    X, y = generate_overlapping_data(N_TOTAL, random_state=SEED)

    # Build one fixed test set and one available training pool.
    X_pool_all, X_test, y_pool_all, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    all_outputs: List[pd.DataFrame] = []
    metric_rows: List[Dict[str, float]] = []
    grouped_rows: List[pd.DataFrame] = []

    for n_train in N_LIST:
        for repeat in range(N_REPEATS):
            seed_rep = SEED + 1000 * n_train + repeat
            idx_pool = stratified_sample_indices(y_pool_all, n_train, seed=seed_rep)
            X_pool = X_pool_all[idx_pool]
            y_pool = y_pool_all[idx_pool]

            method_outputs = {
                "ICP": run_icp(X_pool, y_pool, X_test, seed_rep + 11),
                "BICP": run_bicp(X_pool, y_pool, X_test, seed_rep + 22, n_boot=N_BOOT),
                "RICP": run_ricp(X_pool, y_pool, X_test, seed_rep + 33, n_random=N_BOOT),
                "OOB-ICP": run_oobicp(X_pool, y_pool, X_test, seed_rep + 44, n_boot=N_BOOT),
            }

            for method in METHOD_ORDER:
                df_out = make_output_df(y_test, method_outputs[method], method, repeat, n_train)
                all_outputs.append(df_out)
                met = compute_metrics(df_out)
                metric_rows.append({"dataset": DATASET_NAME, "n_train": n_train, "repeat": repeat, "method": method, **met})
                grouped_rows.append(grouped_metrics(df_out, n_bins=N_BINS))

            print(f"Finished {DATASET_NAME}: n={n_train}, repeat={repeat + 1}/{N_REPEATS}", flush=True)

    df_outputs = pd.concat(all_outputs, ignore_index=True)
    df_metrics = pd.DataFrame(metric_rows)
    df_grouped = pd.concat(grouped_rows, ignore_index=True)

    df_outputs.to_csv(os.path.join(out_dir, "all_test_outputs_4methods.csv"), index=False)
    df_metrics.to_csv(os.path.join(out_dir, "per_repeat_metrics_4methods.csv"), index=False)
    df_grouped.to_csv(os.path.join(out_dir, "grouped_metrics_4methods.csv"), index=False)

    summary = df_metrics.groupby(["dataset", "n_train", "method"], as_index=False).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", lambda x: x.std(ddof=0)),
        f1_mean=("f1", "mean"),
        f1_std=("f1", lambda x: x.std(ddof=0)),
        credibility_mean=("credibility_mean", "mean"),
        credibility_std=("credibility_mean", lambda x: x.std(ddof=0)),
    )
    summary.to_csv(os.path.join(out_dir, "summary_metrics_4methods.csv"), index=False)
    print(f"All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
