# -*- coding: utf-8 -*-
"""
Chinese herbal medicine classification experiment for robust ICP.

Preserved settings from the original implementation:
- Dataset: 600 samples, 12 classes, 50 samples per class.
- Each outer repeat uses exactly:
    * 10 test samples per class
    * 30 proper-training samples per class
    * 10 calibration samples per class
- 10 outer repeats.
- Class-conditional conformal prediction.
- CPSC-style nonconformity score.
- Logistic regression classifier.
- BICP and OOB-ICP use 100 bootstrap iterations.

Revisions in this version:
- RICP now performs 100 proper/calibration repartitions inside the fixed
  training pool of each outer repeat and averages the resulting p-values.
- The outer test set remains fixed throughout all RICP iterations.
- Method names are standardized to ICP / BICP / RICP / OOB-ICP.
- Plotting code has been removed.
- Results are saved as CSV files under ./results.
"""

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =========================================================
# Configuration
# =========================================================
SEED = 42

N_CLASSES = 12
SAMPLES_PER_CLASS = 50

# Per class in each outer repeat:
# 10 test + 30 proper-training + 10 calibration = 50
TEST_PER_CLASS = 10
PROPER_PER_CLASS = 30
CAL_PER_CLASS = 10

N_BINS = 5
N_REPEATS = 10
N_BOOT = 100
N_RANDOM = 100

DATA_CSV_NAME = "original_dataset_stdlzed.csv"
OUTPUT_DIR = "results"
DATASET_NAME = "Chinese herbal medicine classification"

METHOD_ORDER = ["ICP", "BICP", "RICP", "OOB-ICP"]


# =========================================================
# Basic utilities
# =========================================================
def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def log_step(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def load_dataset_same_folder(csv_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the feature matrix from a CSV in the same folder.

    Labels are reconstructed according to the original dataset ordering:
    12 consecutive classes, with 50 samples per class.
    """
    path = os.path.join(script_dir(), csv_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find dataset in the same folder: {path}")

    df = pd.read_csv(path)
    X = df.values.astype(float)

    expected_rows = N_CLASSES * SAMPLES_PER_CLASS
    if X.shape[0] != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows ({N_CLASSES}*{SAMPLES_PER_CLASS}), "
            f"but got {X.shape[0]} rows."
        )

    y = np.repeat(np.arange(N_CLASSES), SAMPLES_PER_CLASS).astype(int)
    return X, y


# =========================================================
# Stratified splitting and resampling
# =========================================================
def split_test_proper_cal_per_class(
    y: np.ndarray,
    test_per_class: int,
    proper_per_class: int,
    cal_per_class: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create one outer split from the complete dataset.

    For every class:
    - test: test_per_class samples
    - proper-training: proper_per_class samples
    - calibration: cal_per_class samples
    """
    rng = np.random.RandomState(seed)

    test_idx: List[int] = []
    proper_idx: List[int] = []
    cal_idx: List[int] = []

    need = test_per_class + proper_per_class + cal_per_class

    for c in range(N_CLASSES):
        idx_c = np.where(y == c)[0].copy()
        rng.shuffle(idx_c)

        if len(idx_c) != need:
            raise ValueError(
                f"Class {c} has {len(idx_c)} samples, expected {need}."
            )

        test_idx.extend(idx_c[:test_per_class].tolist())
        proper_idx.extend(
            idx_c[test_per_class:test_per_class + proper_per_class].tolist()
        )
        cal_idx.extend(
            idx_c[test_per_class + proper_per_class:need].tolist()
        )

    return (
        np.asarray(test_idx, dtype=int),
        np.asarray(proper_idx, dtype=int),
        np.asarray(cal_idx, dtype=int),
    )


def split_proper_cal_from_fixed_pool_per_class(
    y_pool: np.ndarray,
    proper_per_class: int,
    cal_per_class: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Repartition only the fixed training pool used by RICP.

    The input pool contains 40 samples per class. For every RICP iteration,
    each class is repartitioned into:
    - 30 proper-training samples
    - 10 calibration samples

    Returned indices are local indices relative to X_pool/y_pool.
    """
    rng = np.random.RandomState(seed)

    proper_local: List[int] = []
    cal_local: List[int] = []
    expected_per_class = proper_per_class + cal_per_class

    for c in range(N_CLASSES):
        idx_c = np.where(y_pool == c)[0].copy()

        if len(idx_c) != expected_per_class:
            raise ValueError(
                f"RICP pool class {c} has {len(idx_c)} samples; "
                f"expected {expected_per_class}."
            )

        rng.shuffle(idx_c)
        proper_local.extend(idx_c[:proper_per_class].tolist())
        cal_local.extend(idx_c[proper_per_class:].tolist())

    return (
        np.asarray(proper_local, dtype=int),
        np.asarray(cal_local, dtype=int),
    )


def stratified_bootstrap_resample_per_class(
    X_proper: np.ndarray,
    y_proper: np.ndarray,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    BICP bootstrap: sample with replacement separately within every class.
    """
    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    expected_classes = np.arange(N_CLASSES)
    observed_classes = np.unique(y_proper)

    if not np.array_equal(observed_classes, expected_classes):
        raise ValueError(
            f"Proper-training labels are incomplete: got {observed_classes}, "
            f"expected {expected_classes}."
        )

    for c in expected_classes:
        idx_c = np.where(y_proper == c)[0]
        n_c = len(idx_c)

        if n_c == 0:
            raise ValueError(f"Class {c} has no proper-training samples.")

        boot_idx_c = rng.choice(idx_c, size=n_c, replace=True)
        X_parts.append(X_proper[boot_idx_c])
        y_parts.append(y_proper[boot_idx_c])

    X_boot = np.vstack(X_parts)
    y_boot = np.concatenate(y_parts)

    perm = rng.permutation(len(y_boot))
    return X_boot[perm], y_boot[perm]


def bootstrap_train_oob_from_pool_per_class(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    OOB-ICP on the fixed proper+calibration pool.

    Within each class:
    - Draw n_c samples with replacement.
    - Use the full bootstrap in-bag sample, retaining repeated observations,
      for model training.
    - Use observations never selected as OOB calibration data.
    """
    X_train_parts: List[np.ndarray] = []
    y_train_parts: List[np.ndarray] = []
    X_cal_parts: List[np.ndarray] = []
    y_cal_parts: List[np.ndarray] = []

    for c in range(N_CLASSES):
        idx_c = np.where(y_pool == c)[0]
        n_c = len(idx_c)

        if n_c == 0:
            raise ValueError(f"Class {c} has no samples in the training pool.")

        for _ in range(500):
            sampled_local = rng.choice(np.arange(n_c), size=n_c, replace=True)
            unique_inbag_local = np.unique(sampled_local)

            oob_mask = np.ones(n_c, dtype=bool)
            oob_mask[unique_inbag_local] = False
            oob_local = np.where(oob_mask)[0]

            if len(oob_local) > 0:
                train_idx_c = idx_c[sampled_local]
                cal_idx_c = idx_c[oob_local]

                X_train_parts.append(X_pool[train_idx_c])
                y_train_parts.append(y_pool[train_idx_c])
                X_cal_parts.append(X_pool[cal_idx_c])
                y_cal_parts.append(y_pool[cal_idx_c])
                break
        else:
            raise RuntimeError(
                f"Failed to generate a non-empty OOB calibration set for class {c}."
            )

    X_train = np.vstack(X_train_parts)
    y_train = np.concatenate(y_train_parts)
    X_cal = np.vstack(X_cal_parts)
    y_cal = np.concatenate(y_cal_parts)

    train_perm = rng.permutation(len(y_train))
    cal_perm = rng.permutation(len(y_cal))

    return (
        X_train[train_perm],
        y_train[train_perm],
        X_cal[cal_perm],
        y_cal[cal_perm],
    )


def fit_scaler_from_proper_and_transform(
    X: np.ndarray,
    proper_idx: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    pool_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Preserve the original preprocessing procedure:
    fit StandardScaler on the current outer proper-training set only.
    """
    scaler = StandardScaler()
    scaler.fit(X[proper_idx])

    X_proper = scaler.transform(X[proper_idx])
    X_cal = scaler.transform(X[cal_idx])
    X_test = scaler.transform(X[test_idx])
    X_pool = scaler.transform(X[pool_idx])

    return X_proper, X_cal, X_test, X_pool, scaler


# =========================================================
# Class-conditional conformal prediction
# =========================================================
@dataclass
class ICPOutputs:
    p_values: np.ndarray
    pred_labels: np.ndarray
    credibility: np.ndarray
    confidence: np.ndarray


class StandardICP:
    """
    Standard inductive conformal predictor with class-conditional calibration.
    """

    def __init__(self, random_state: int = 42):
        self.model = LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            random_state=random_state,
        )
        self.classes_: np.ndarray | None = None
        self.cal_scores_by_class_: Dict[int, np.ndarray] | None = None

    @staticmethod
    def _nonconformity_from_proba_vector(
        proba_row: np.ndarray,
        class_index: int,
    ) -> float:
        """
        CPSC-style nonconformity score:

            a(x,y) = 0.5 - (p(y|x) - max_{y' != y} p(y'|x)) / 2

        Smaller values indicate stronger conformity.
        """
        p_y = float(proba_row[class_index])

        if proba_row.shape[0] <= 1:
            max_other = 0.0
        else:
            other_mask = np.ones(proba_row.shape[0], dtype=bool)
            other_mask[class_index] = False
            max_other = float(np.max(proba_row[other_mask]))

        return 0.5 - (p_y - max_other) / 2.0

    @staticmethod
    def _p_value(cal_scores: np.ndarray, test_score: float) -> float:
        n = len(cal_scores)
        return (np.sum(cal_scores >= test_score) + 1.0) / (n + 1.0)

    def fit(
        self,
        X_proper: np.ndarray,
        y_proper: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> "StandardICP":
        self.model.fit(X_proper, y_proper)
        self.classes_ = self.model.classes_

        proba_cal = self.model.predict_proba(X_cal)
        self.cal_scores_by_class_ = {}

        for cls in self.classes_:
            mask = y_cal == cls
            if not np.any(mask):
                raise ValueError(f"Calibration set for class {cls} is empty.")

            cls_col = int(np.searchsorted(self.classes_, cls))
            proba_cal_cls = proba_cal[mask]

            self.cal_scores_by_class_[int(cls)] = np.asarray(
                [
                    self._nonconformity_from_proba_vector(row, cls_col)
                    for row in proba_cal_cls
                ],
                dtype=float,
            )

        return self

    def predict(self, X_test: np.ndarray) -> ICPOutputs:
        if self.classes_ is None or self.cal_scores_by_class_ is None:
            raise RuntimeError("StandardICP must be fitted before prediction.")

        proba = self.model.predict_proba(X_test)
        n_test, n_classes = proba.shape
        p_values = np.zeros((n_test, n_classes), dtype=float)

        for i in range(n_test):
            for k in range(n_classes):
                cls = int(self.classes_[k])
                score = self._nonconformity_from_proba_vector(proba[i], k)
                p_values[i, k] = self._p_value(
                    self.cal_scores_by_class_[cls],
                    score,
                )

        pred = self.classes_[np.argmax(p_values, axis=1)].astype(int)
        credibility = np.max(p_values, axis=1)
        sorted_p = np.sort(p_values, axis=1)
        second = (
            sorted_p[:, -2]
            if n_classes > 1
            else np.zeros_like(credibility)
        )
        confidence = 1.0 - second

        return ICPOutputs(
            p_values=p_values,
            pred_labels=pred,
            credibility=credibility,
            confidence=confidence,
        )


class BootstrappedICP:
    """
    BICP:
    - Keep calibration fixed.
    - Bootstrap the proper-training set within class.
    - Average p-values over n_boot iterations.
    """

    def __init__(self, n_boot: int = 100, random_state: int = 42):
        self.n_boot = n_boot
        self.random_state = random_state

    def fit_predict(
        self,
        X_proper: np.ndarray,
        y_proper: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        X_test: np.ndarray,
        progress_prefix: str = "BICP",
    ) -> ICPOutputs:
        rng = np.random.RandomState(self.random_state)

        classes = np.unique(y_proper)
        p_sum = np.zeros((len(X_test), len(classes)), dtype=float)

        for b in range(self.n_boot):
            if b == 0 or (b + 1) % 10 == 0 or b + 1 == self.n_boot:
                log_step(f"{progress_prefix}: bootstrap {b + 1}/{self.n_boot}")

            X_boot, y_boot = stratified_bootstrap_resample_per_class(
                X_proper,
                y_proper,
                rng,
            )

            icp_b = StandardICP(random_state=self.random_state + b + 1)
            icp_b.fit(X_boot, y_boot, X_cal, y_cal)
            p_sum += icp_b.predict(X_test).p_values

        p_avg = p_sum / float(self.n_boot)
        return output_from_average_pvalues(p_avg, classes)


class RandomizedICP:
    """
    RICP:
    - Keep the outer test set fixed.
    - Keep the outer training pool fixed.
    - Repartition the fixed pool into 30 proper + 10 calibration samples
      per class for every iteration.
    - Fit preprocessing only on that iteration's proper-training subset.
    - Average p-values over n_random iterations.
    """

    def __init__(self, n_random: int = 100, random_state: int = 42):
        self.n_random = n_random
        self.random_state = random_state

    def fit_predict(
        self,
        X_pool_raw: np.ndarray,
        y_pool: np.ndarray,
        X_test_raw: np.ndarray,
        progress_prefix: str = "RICP",
    ) -> ICPOutputs:
        classes = np.unique(y_pool)
        p_sum = np.zeros((len(X_test_raw), len(classes)), dtype=float)

        for r in range(self.n_random):
            if r == 0 or (r + 1) % 10 == 0 or r + 1 == self.n_random:
                log_step(
                    f"{progress_prefix}: repartition {r + 1}/{self.n_random}"
                )

            proper_local, cal_local = split_proper_cal_from_fixed_pool_per_class(
                y_pool=y_pool,
                proper_per_class=PROPER_PER_CLASS,
                cal_per_class=CAL_PER_CLASS,
                seed=self.random_state + r,
            )

            # Each RICP repartition receives its own leakage-free scaler.
            scaler = StandardScaler()
            scaler.fit(X_pool_raw[proper_local])

            X_proper_r = scaler.transform(X_pool_raw[proper_local])
            X_cal_r = scaler.transform(X_pool_raw[cal_local])
            X_test_r = scaler.transform(X_test_raw)

            y_proper_r = y_pool[proper_local]
            y_cal_r = y_pool[cal_local]

            icp_r = StandardICP(random_state=self.random_state + r)
            icp_r.fit(X_proper_r, y_proper_r, X_cal_r, y_cal_r)
            p_sum += icp_r.predict(X_test_r).p_values

        p_avg = p_sum / float(self.n_random)
        return output_from_average_pvalues(p_avg, classes)


class OOBICP:
    """
    OOB-ICP:
    - Bootstrap within each class from the fixed proper+calibration pool.
    - Train on the full bootstrap in-bag sample, retaining repeated observations.
    - Calibrate on OOB observations.
    - Average p-values over n_boot iterations.
    """

    def __init__(self, n_boot: int = 100, random_state: int = 42):
        self.n_boot = n_boot
        self.random_state = random_state

    def fit_predict(
        self,
        X_pool: np.ndarray,
        y_pool: np.ndarray,
        X_test: np.ndarray,
        progress_prefix: str = "OOB-ICP",
    ) -> ICPOutputs:
        rng = np.random.RandomState(self.random_state)
        classes = np.unique(y_pool)
        p_sum = np.zeros((len(X_test), len(classes)), dtype=float)

        for b in range(self.n_boot):
            if b == 0 or (b + 1) % 10 == 0 or b + 1 == self.n_boot:
                log_step(
                    f"{progress_prefix}: OOB bootstrap {b + 1}/{self.n_boot}"
                )

            X_train_b, y_train_b, X_cal_b, y_cal_b = (
                bootstrap_train_oob_from_pool_per_class(
                    X_pool,
                    y_pool,
                    rng,
                )
            )

            icp_b = StandardICP(random_state=self.random_state + b + 1)
            icp_b.fit(X_train_b, y_train_b, X_cal_b, y_cal_b)
            p_sum += icp_b.predict(X_test).p_values

        p_avg = p_sum / float(self.n_boot)
        return output_from_average_pvalues(p_avg, classes)


def output_from_average_pvalues(
    p_values: np.ndarray,
    classes: np.ndarray,
) -> ICPOutputs:
    pred = classes[np.argmax(p_values, axis=1)].astype(int)
    credibility = np.max(p_values, axis=1)
    sorted_p = np.sort(p_values, axis=1)
    second = (
        sorted_p[:, -2]
        if p_values.shape[1] > 1
        else np.zeros_like(credibility)
    )
    confidence = 1.0 - second

    return ICPOutputs(
        p_values=p_values,
        pred_labels=pred,
        credibility=credibility,
        confidence=confidence,
    )


# =========================================================
# Evaluation and output tables
# =========================================================
def make_output_df(
    y_true: np.ndarray,
    out: ICPOutputs,
    method: str,
    repeat: int,
) -> pd.DataFrame:
    data = {
        "dataset": DATASET_NAME,
        "n_train": "full",
        "repeat": repeat,
        "method": method,
        "sample_idx": np.arange(len(y_true)),
        "y_true": y_true.astype(int),
        "y_pred": out.pred_labels.astype(int),
        "credibility": out.credibility,
        "confidence": out.confidence,
        "is_correct": (
            out.pred_labels.astype(int) == y_true.astype(int)
        ).astype(int),
    }

    for k in range(out.p_values.shape[1]):
        data[f"p{k}"] = out.p_values[:, k]

    return pd.DataFrame(data)


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "credibility_mean": float(df["credibility"].mean()),
        "credibility_std": float(df["credibility"].std(ddof=0)),
        "credibility_correct_mean": (
            float(df.loc[df["is_correct"] == 1, "credibility"].mean())
            if (df["is_correct"] == 1).any()
            else np.nan
        ),
        "credibility_wrong_mean": (
            float(df.loc[df["is_correct"] == 0, "credibility"].mean())
            if (df["is_correct"] == 0).any()
            else np.nan
        ),
    }


def grouped_metrics_by_credibility(
    df: pd.DataFrame,
    n_bins: int,
) -> pd.DataFrame:
    """
    Preserve the original per-repeat equal-frequency credibility grouping.
    """
    tmp = df.copy()
    tmp["bin"] = pd.qcut(
        tmp["credibility"],
        q=n_bins,
        labels=False,
        duplicates="drop",
    )
    tmp["group"] = tmp["bin"].astype(int) + 1

    rows: List[Dict[str, object]] = []

    for group in sorted(tmp["group"].unique()):
        sub = tmp[tmp["group"] == group]
        y_true = sub["y_true"].to_numpy()
        y_pred = sub["y_pred"].to_numpy()

        rows.append({
            "dataset": DATASET_NAME,
            "n_train": "full",
            "repeat": int(sub["repeat"].iloc[0]),
            "method": str(sub["method"].iloc[0]),
            "group": int(group),
            "count": int(len(sub)),
            "cred_mean": float(sub["credibility"].mean()),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "f1_macro": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
        })

    return pd.DataFrame(rows)


# =========================================================
# Main experiment
# =========================================================
def main() -> None:
    np.random.seed(SEED)

    out_dir = os.path.join(script_dir(), OUTPUT_DIR)
    ensure_dir(out_dir)

    log_step(f"Loading {DATA_CSV_NAME}")
    X, y = load_dataset_same_folder(DATA_CSV_NAME)
    log_step(f"Loaded data: X={X.shape}, y={len(y)}")

    all_output_dfs: List[pd.DataFrame] = []
    metric_rows: List[Dict[str, object]] = []
    grouped_rows: List[pd.DataFrame] = []

    log_step(
        f"Starting experiment: outer repeats={N_REPEATS}; "
        f"BICP/RICP/OOB-ICP iterations={N_BOOT}"
    )

    for repeat in range(N_REPEATS):
        split_seed = SEED + 1000 * repeat
        log_step(f"Repeat {repeat + 1}/{N_REPEATS}: creating outer split")

        test_idx, proper_idx, cal_idx = split_test_proper_cal_per_class(
            y=y,
            test_per_class=TEST_PER_CLASS,
            proper_per_class=PROPER_PER_CLASS,
            cal_per_class=CAL_PER_CLASS,
            seed=split_seed,
        )

        pool_idx = np.concatenate([proper_idx, cal_idx])

        # Preserve original scaling for ICP, BICP, and OOB-ICP.
        X_proper, X_cal, X_test, X_pool, _ = (
            fit_scaler_from_proper_and_transform(
                X=X,
                proper_idx=proper_idx,
                cal_idx=cal_idx,
                test_idx=test_idx,
                pool_idx=pool_idx,
            )
        )

        y_proper = y[proper_idx]
        y_cal = y[cal_idx]
        y_test = y[test_idx]
        y_pool = y[pool_idx]

        method_outputs: Dict[str, ICPOutputs] = {}

        log_step(f"Repeat {repeat + 1}/{N_REPEATS}: running ICP")
        icp = StandardICP(random_state=split_seed + 11)
        icp.fit(X_proper, y_proper, X_cal, y_cal)
        method_outputs["ICP"] = icp.predict(X_test)

        log_step(f"Repeat {repeat + 1}/{N_REPEATS}: running BICP")
        bicp = BootstrappedICP(
            n_boot=N_BOOT,
            random_state=split_seed + 33,
        )
        method_outputs["BICP"] = bicp.fit_predict(
            X_proper=X_proper,
            y_proper=y_proper,
            X_cal=X_cal,
            y_cal=y_cal,
            X_test=X_test,
            progress_prefix=f"Repeat {repeat + 1}/{N_REPEATS} | BICP",
        )

        log_step(f"Repeat {repeat + 1}/{N_REPEATS}: running RICP")
        ricp = RandomizedICP(
            n_random=N_RANDOM,
            random_state=split_seed + 22,
        )
        method_outputs["RICP"] = ricp.fit_predict(
            X_pool_raw=X[pool_idx],
            y_pool=y_pool,
            X_test_raw=X[test_idx],
            progress_prefix=f"Repeat {repeat + 1}/{N_REPEATS} | RICP",
        )

        log_step(f"Repeat {repeat + 1}/{N_REPEATS}: running OOB-ICP")
        oobicp = OOBICP(
            n_boot=N_BOOT,
            random_state=split_seed + 44,
        )
        method_outputs["OOB-ICP"] = oobicp.fit_predict(
            X_pool=X_pool,
            y_pool=y_pool,
            X_test=X_test,
            progress_prefix=f"Repeat {repeat + 1}/{N_REPEATS} | OOB-ICP",
        )

        for method in METHOD_ORDER:
            df_out = make_output_df(
                y_true=y_test,
                out=method_outputs[method],
                method=method,
                repeat=repeat,
            )
            all_output_dfs.append(df_out)

            metrics = compute_metrics(df_out)
            metric_rows.append({
                "dataset": DATASET_NAME,
                "n_train": "full",
                "repeat": repeat,
                "method": method,
                **metrics,
            })

            grouped_rows.append(
                grouped_metrics_by_credibility(df_out, n_bins=N_BINS)
            )

        log_step(
            f"Finished {DATASET_NAME}: repeat={repeat + 1}/{N_REPEATS}"
        )

    df_outputs = pd.concat(all_output_dfs, ignore_index=True)
    df_metrics = pd.DataFrame(metric_rows)
    df_grouped = pd.concat(grouped_rows, ignore_index=True)

    summary = (
        df_metrics
        .groupby(["dataset", "n_train", "method"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", lambda x: x.std(ddof=0)),
            f1_mean=("f1", "mean"),
            f1_std=("f1", lambda x: x.std(ddof=0)),
            credibility_mean=("credibility_mean", "mean"),
            credibility_std=(
                "credibility_mean",
                lambda x: x.std(ddof=0),
            ),
        )
    )

    df_outputs.to_csv(
        os.path.join(out_dir, "all_test_outputs_4methods.csv"),
        index=False,
    )
    df_metrics.to_csv(
        os.path.join(out_dir, "per_repeat_metrics_4methods.csv"),
        index=False,
    )
    df_grouped.to_csv(
        os.path.join(out_dir, "grouped_metrics_4methods.csv"),
        index=False,
    )
    summary.to_csv(
        os.path.join(out_dir, "summary_metrics_4methods.csv"),
        index=False,
    )

    log_step(f"All outputs saved to: {out_dir}")
    print("\nSummary:")
    for _, row in summary.iterrows():
        print(
            f"{row['method']:>8s}: "
            f"acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f}, "
            f"f1={row['f1_mean']:.4f}±{row['f1_std']:.4f}, "
            f"mean_cred={row['credibility_mean']:.4f}±"
            f"{row['credibility_std']:.4f}"
        )


if __name__ == "__main__":
    main()