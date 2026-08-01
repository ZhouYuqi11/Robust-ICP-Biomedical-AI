# -*- coding: utf-8 -*-
"""
Breast cancer subtyping experiment for robust ICP.

This version restores the original TCGA implementation, except that:
1. BICP/RICP/OOB-ICP each use 100 internal iterations;
2. method names are unified as ICP / BICP / RICP / OOB-ICP;
3. plotting is removed; only CSV tables are saved.

Original experimental settings retained:
- Keep the three most frequent TCGA subtypes and remap them to 0, 1, 2.
- Use all samples from those three classes; no n-based downsampling.
- Repeat 10 stratified train/test splits.
- Train/test split: 75% / 25%.
- Proper-training/calibration split inside the training pool: 67% / 33%.
- High-dimensional model pipeline:
  median imputation -> variance filtering -> standardization ->
  SelectKBest(f_classif, k=50) -> L1 logistic regression (SAGA, balanced).
- Pooled calibration and CPSC nonconformity score.
- Class-stratified bootstrap for BICP and OOB-ICP.

Expected files in the same directory:
- TCGA_X.npy
- TCGA_Y.npy

Outputs are written to ./results.
"""

import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =========================================================
# Configuration
# =========================================================
SEED = 42
X_FILE = "TCGA_X.npy"
Y_FILE = "TCGA_Y.npy"
OUTPUT_DIR = "results"
DATASET_NAME = "Breast cancer subtyping"

KEEP_TOP_K = 3
N_REPEATS = 10
N_BOOT = 100
ALPHA = 0.10

# Restore the original split settings.
TEST_SIZE = 0.25
PROPER_TRAIN_RATIO = 0.67

K_FEATURES = 50
N_BINS = 5
METHOD_ORDER = ["ICP", "BICP", "RICP", "OOB-ICP"]


# =========================================================
# Basic utilities
# =========================================================
def log_step(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def resolve_path(filename: str) -> str:
    return os.path.join(script_dir(), filename)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def load_data(x_file: str = X_FILE, y_file: str = Y_FILE) -> Tuple[np.ndarray, np.ndarray]:
    x_path = resolve_path(x_file)
    y_path = resolve_path(y_file)

    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Cannot find X file: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Cannot find Y file: {y_path}")

    X = np.load(x_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)

    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)

    if X.ndim != 2:
        raise ValueError(f"X must be two-dimensional, got shape {X.shape}.")
    if X.shape[0] != len(y):
        raise ValueError(f"X rows {X.shape[0]} do not match y length {len(y)}.")

    # Keep the original behavior: no additional log transformation is applied here.
    return X.astype(float), y


def summarize_labels(y: np.ndarray) -> pd.DataFrame:
    labels, counts = np.unique(y, return_counts=True)
    return (
        pd.DataFrame({"label": labels, "count": counts})
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def format_label_counts(y: np.ndarray) -> str:
    labels, counts = np.unique(y, return_counts=True)
    return ", ".join(f"{int(label)}:{int(count)}" for label, count in zip(labels, counts))


def filter_top_k_classes(
    X: np.ndarray,
    y: np.ndarray,
    keep_top_k: int = KEEP_TOP_K,
) -> Tuple[np.ndarray, np.ndarray, List, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original_distribution = summarize_labels(y)
    kept_labels = original_distribution.head(keep_top_k)["label"].tolist()

    mask = np.isin(y, kept_labels)
    X_filtered = X[mask]
    y_filtered = y[mask]

    sorted_labels = sorted(np.unique(y_filtered).tolist())
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}
    y_remapped = np.array([label_mapping[label] for label in y_filtered], dtype=int)

    mapping_df = pd.DataFrame(
        {
            "original_label": list(label_mapping.keys()),
            "new_label": list(label_mapping.values()),
        }
    )

    filtered_distribution = summarize_labels(y_filtered)
    filtered_distribution["new_label"] = filtered_distribution["label"].map(label_mapping)

    return (
        X_filtered,
        y_remapped,
        kept_labels,
        mapping_df,
        original_distribution,
        filtered_distribution,
    )


def split_train_test_indices(
    y: np.ndarray,
    seed: int,
    test_size: float = TEST_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )
    train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y))
    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


def split_proper_cal_indices(
    y_train: np.ndarray,
    seed: int,
    proper_ratio: float = PROPER_TRAIN_RATIO,
) -> Tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1.0 - proper_ratio,
        random_state=seed,
    )
    proper_idx, cal_idx = next(splitter.split(np.zeros(len(y_train)), y_train))
    return np.asarray(proper_idx, dtype=int), np.asarray(cal_idx, dtype=int)


# =========================================================
# Model
# =========================================================
def get_effective_k_features(X_train: np.ndarray, requested_k: int) -> int:
    return int(max(1, min(requested_k, X_train.shape[1])))


def build_base_model(k_features: int = K_FEATURES, random_state: int = SEED) -> Pipeline:
    """Restore the original high-dimensional TCGA classification pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold()),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=k_features)),
            (
                "clf",
                LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def fit_model(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> Pipeline:
    model = build_base_model(
        k_features=get_effective_k_features(X_train, K_FEATURES),
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


# =========================================================
# Pooled conformal prediction
# =========================================================
def cpsc_nonconformity_from_proba_row(proba_row: np.ndarray, class_index: int) -> float:
    """
    CPSC nonconformity score:
        a(x,y) = 0.5 - (p(y|x) - max_{y' != y} p(y'|x)) / 2.
    """
    p_y = float(proba_row[class_index])
    if proba_row.shape[0] <= 1:
        max_other = 0.0
    else:
        mask = np.ones(proba_row.shape[0], dtype=bool)
        mask[class_index] = False
        max_other = float(np.max(proba_row[mask]))
    return 0.5 - (p_y - max_other) / 2.0


def pooled_calibration_scores(
    model: Pipeline,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
) -> np.ndarray:
    proba_cal = model.predict_proba(X_cal)
    class_to_column = {int(label): column for column, label in enumerate(model.classes_)}

    return np.array(
        [
            cpsc_nonconformity_from_proba_row(
                proba_cal[i],
                class_to_column[int(y_cal[i])],
            )
            for i in range(len(y_cal))
        ],
        dtype=float,
    )


def p_values_from_calibration(
    model: Pipeline,
    X_test: np.ndarray,
    calibration_scores: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    proba_test = model.predict_proba(X_test)
    p_values = np.zeros((X_test.shape[0], n_classes), dtype=float)
    m = len(calibration_scores)

    # Preserve the original implementation exactly.
    for i in range(X_test.shape[0]):
        for candidate_label in range(n_classes):
            test_score = cpsc_nonconformity_from_proba_row(
                proba_test[i],
                candidate_label,
            )
            p_values[i, candidate_label] = (
                np.sum(calibration_scores >= test_score) + 1.0
            ) / (m + 1.0)

    return p_values


def prediction_set_from_p_values(
    p_values: np.ndarray,
    alpha: float = ALPHA,
) -> List[List[int]]:
    return [np.where(row > alpha)[0].tolist() for row in p_values]


def credibility_confidence(p_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    credibility = np.max(p_values, axis=1)
    sorted_p = np.sort(p_values, axis=1)
    second_largest = (
        sorted_p[:, -2]
        if p_values.shape[1] > 1
        else np.zeros(len(credibility), dtype=float)
    )
    confidence = 1.0 - second_largest
    return credibility, confidence


def evaluate_prediction_sets(
    y_true: np.ndarray,
    p_values: np.ndarray,
    alpha: float,
    method: str,
    repeat_id: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    prediction_sets = prediction_set_from_p_values(p_values, alpha=alpha)
    predicted_labels = np.argmax(p_values, axis=1).astype(int)
    credibility, confidence = credibility_confidence(p_values)

    rows = []
    for i in range(len(y_true)):
        prediction_set = prediction_sets[i]
        row = {
            "dataset": DATASET_NAME,
            "n_train": "full",
            "repeat": repeat_id,
            "sample_idx": i,
            "method": method,
            "y_true": int(y_true[i]),
            "y_pred": int(predicted_labels[i]),
            "pred_set": str(prediction_set),
            "set_size": int(len(prediction_set)),
            "contains_true": int(int(y_true[i]) in prediction_set),
            "singleton_correct": int(
                len(prediction_set) == 1
                and prediction_set[0] == int(y_true[i])
            ),
            "credibility": float(credibility[i]),
            "confidence": float(confidence[i]),
            "is_correct": int(predicted_labels[i] == int(y_true[i])),
        }
        for class_index in range(p_values.shape[1]):
            row[f"p{class_index}"] = float(p_values[i, class_index])
        rows.append(row)

    output_df = pd.DataFrame(rows)

    metrics = {
        "dataset": DATASET_NAME,
        "n_train": "full",
        "repeat": repeat_id,
        "method": method,
        "accuracy": float(accuracy_score(y_true, predicted_labels)),
        "f1": float(
            f1_score(y_true, predicted_labels, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, predicted_labels, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, predicted_labels, average="weighted", zero_division=0)
        ),
        "coverage": float(output_df["contains_true"].mean()),
        "avg_set_size": float(output_df["set_size"].mean()),
        "singleton_rate": float((output_df["set_size"] == 1).mean()),
        "singleton_accuracy": (
            float(
                output_df.loc[
                    output_df["set_size"] == 1,
                    "is_correct",
                ].mean()
            )
            if (output_df["set_size"] == 1).any()
            else np.nan
        ),
        "credibility_mean": float(output_df["credibility"].mean()),
        "credibility_std": float(output_df["credibility"].std(ddof=0)),
        "credibility_correct_mean": (
            float(
                output_df.loc[
                    output_df["is_correct"] == 1,
                    "credibility",
                ].mean()
            )
            if (output_df["is_correct"] == 1).any()
            else np.nan
        ),
        "credibility_wrong_mean": (
            float(
                output_df.loc[
                    output_df["is_correct"] == 0,
                    "credibility",
                ].mean()
            )
            if (output_df["is_correct"] == 0).any()
            else np.nan
        ),
    }

    return output_df, metrics


# =========================================================
# Four methods — original implementations with new names
# =========================================================
def run_icp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    rng_seed: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    proper_idx, cal_idx = split_proper_cal_indices(
        y_train,
        seed=rng_seed,
        proper_ratio=PROPER_TRAIN_RATIO,
    )

    X_proper, y_proper = X_train[proper_idx], y_train[proper_idx]
    X_cal, y_cal = X_train[cal_idx], y_train[cal_idx]

    model = fit_model(X_proper, y_proper, seed=rng_seed)
    calibration_scores = pooled_calibration_scores(model, X_cal, y_cal)
    p_values = p_values_from_calibration(
        model,
        X_test,
        calibration_scores,
        n_classes=len(np.unique(y_train)),
    )

    return evaluate_prediction_sets(
        y_test,
        p_values,
        alpha,
        method="ICP",
        repeat_id=rng_seed,
    )


def run_ricp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    rng_seed: int,
    n_random: int = N_BOOT,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    n_classes = len(np.unique(y_train))
    p_values_sum = np.zeros((len(y_test), n_classes), dtype=float)

    for iteration in range(n_random):
        if verbose and (
            iteration == 0
            or (iteration + 1) % 10 == 0
            or iteration + 1 == n_random
        ):
            log_step(f"RICP repartition {iteration + 1}/{n_random}")

        iteration_seed = rng_seed + 1000 + iteration
        proper_idx, cal_idx = split_proper_cal_indices(
            y_train,
            seed=iteration_seed,
            proper_ratio=PROPER_TRAIN_RATIO,
        )

        X_proper, y_proper = X_train[proper_idx], y_train[proper_idx]
        X_cal, y_cal = X_train[cal_idx], y_train[cal_idx]

        model = fit_model(X_proper, y_proper, seed=iteration_seed)
        calibration_scores = pooled_calibration_scores(model, X_cal, y_cal)
        p_values_sum += p_values_from_calibration(
            model,
            X_test,
            calibration_scores,
            n_classes=n_classes,
        )

    mean_p_values = p_values_sum / float(n_random)
    return evaluate_prediction_sets(
        y_test,
        mean_p_values,
        alpha,
        method="RICP",
        repeat_id=rng_seed,
    )


def stratified_bootstrap_resample_per_class(
    X_input: np.ndarray,
    y_input: np.ndarray,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    X_parts = []
    y_parts = []

    for class_label in np.unique(y_input):
        class_indices = np.where(y_input == class_label)[0]
        bootstrap_indices = rng.choice(
            class_indices,
            size=len(class_indices),
            replace=True,
        )
        X_parts.append(X_input[bootstrap_indices])
        y_parts.append(y_input[bootstrap_indices])

    X_bootstrap = np.vstack(X_parts)
    y_bootstrap = np.concatenate(y_parts)
    permutation = rng.permutation(len(y_bootstrap))
    return X_bootstrap[permutation], y_bootstrap[permutation]


def run_bicp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    rng_seed: int,
    n_boot: int = N_BOOT,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    proper_idx, cal_idx = split_proper_cal_indices(
        y_train,
        seed=rng_seed,
        proper_ratio=PROPER_TRAIN_RATIO,
    )

    X_proper, y_proper = X_train[proper_idx], y_train[proper_idx]
    X_cal, y_cal = X_train[cal_idx], y_train[cal_idx]

    n_classes = len(np.unique(y_train))
    p_values_sum = np.zeros((len(y_test), n_classes), dtype=float)
    rng = np.random.RandomState(rng_seed + 2222)

    for iteration in range(n_boot):
        if verbose and (
            iteration == 0
            or (iteration + 1) % 10 == 0
            or iteration + 1 == n_boot
        ):
            log_step(f"BICP bootstrap {iteration + 1}/{n_boot}")

        X_bootstrap, y_bootstrap = stratified_bootstrap_resample_per_class(
            X_proper,
            y_proper,
            rng,
        )

        model = fit_model(
            X_bootstrap,
            y_bootstrap,
            seed=rng_seed + 2222 + iteration,
        )
        calibration_scores = pooled_calibration_scores(model, X_cal, y_cal)
        p_values_sum += p_values_from_calibration(
            model,
            X_test,
            calibration_scores,
            n_classes=n_classes,
        )

    mean_p_values = p_values_sum / float(n_boot)
    return evaluate_prediction_sets(
        y_test,
        mean_p_values,
        alpha,
        method="BICP",
        repeat_id=rng_seed,
    )


def bootstrap_train_oob_from_pool_per_class(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train_parts = []
    y_train_parts = []
    X_cal_parts = []
    y_cal_parts = []

    for class_label in np.unique(y_pool):
        class_indices = np.where(y_pool == class_label)[0]
        class_size = len(class_indices)

        for _ in range(500):
            sampled_positions = rng.choice(
                np.arange(class_size),
                size=class_size,
                replace=True,
            )
            bootstrap_indices = class_indices[sampled_positions]

            used_positions = np.unique(sampled_positions)
            oob_mask = np.ones(class_size, dtype=bool)
            oob_mask[used_positions] = False
            oob_positions = np.where(oob_mask)[0]

            if len(oob_positions) > 0:
                oob_indices = class_indices[oob_positions]
                X_train_parts.append(X_pool[bootstrap_indices])
                y_train_parts.append(y_pool[bootstrap_indices])
                X_cal_parts.append(X_pool[oob_indices])
                y_cal_parts.append(y_pool[oob_indices])
                break
        else:
            raise RuntimeError(
                f"Failed to generate non-empty OOB samples for class {class_label}."
            )

    X_bootstrap_train = np.vstack(X_train_parts)
    y_bootstrap_train = np.concatenate(y_train_parts)
    X_oob_cal = np.vstack(X_cal_parts)
    y_oob_cal = np.concatenate(y_cal_parts)

    train_permutation = rng.permutation(len(y_bootstrap_train))
    cal_permutation = rng.permutation(len(y_oob_cal))

    return (
        X_bootstrap_train[train_permutation],
        y_bootstrap_train[train_permutation],
        X_oob_cal[cal_permutation],
        y_oob_cal[cal_permutation],
    )


def run_oobicp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    rng_seed: int,
    n_boot: int = N_BOOT,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    n_classes = len(np.unique(y_train))
    p_values_sum = np.zeros((len(y_test), n_classes), dtype=float)
    rng = np.random.RandomState(rng_seed + 3333)

    for iteration in range(n_boot):
        if verbose and (
            iteration == 0
            or (iteration + 1) % 10 == 0
            or iteration + 1 == n_boot
        ):
            log_step(f"OOB-ICP bootstrap {iteration + 1}/{n_boot}")

        (
            X_bootstrap_train,
            y_bootstrap_train,
            X_oob_cal,
            y_oob_cal,
        ) = bootstrap_train_oob_from_pool_per_class(X_train, y_train, rng)

        model = fit_model(
            X_bootstrap_train,
            y_bootstrap_train,
            seed=rng_seed + 3333 + iteration,
        )
        calibration_scores = pooled_calibration_scores(
            model,
            X_oob_cal,
            y_oob_cal,
        )
        p_values_sum += p_values_from_calibration(
            model,
            X_test,
            calibration_scores,
            n_classes=n_classes,
        )

    mean_p_values = p_values_sum / float(n_boot)
    return evaluate_prediction_sets(
        y_test,
        mean_p_values,
        alpha,
        method="OOB-ICP",
        repeat_id=rng_seed,
    )


# =========================================================
# Credibility grouping and CSV summaries
# =========================================================
def grouped_metrics_per_repeat(
    output_df: pd.DataFrame,
    n_bins: int = N_BINS,
) -> pd.DataFrame:
    """Create five credibility groups within one repeat for unified plotting."""
    grouped_df = output_df.copy()

    # Ranking only resolves ties; it does not change the credibility values used
    # for means and metrics.
    grouped_df["group"] = (
        pd.qcut(
            grouped_df["credibility"].rank(method="first"),
            q=n_bins,
            labels=False,
        ).astype(int)
        + 1
    )

    rows = []
    for group_number in sorted(grouped_df["group"].unique()):
        subset = grouped_df[grouped_df["group"] == group_number]
        rows.append(
            {
                "dataset": DATASET_NAME,
                "n_train": "full",
                "repeat": int(subset["repeat"].iloc[0]),
                "method": str(subset["method"].iloc[0]),
                "group": int(group_number),
                "count": int(len(subset)),
                "cred_mean": float(subset["credibility"].mean()),
                "accuracy": float(
                    accuracy_score(subset["y_true"], subset["y_pred"])
                ),
                "f1": float(
                    f1_score(
                        subset["y_true"],
                        subset["y_pred"],
                        average="macro",
                        zero_division=0,
                    )
                ),
                "f1_macro": float(
                    f1_score(
                        subset["y_true"],
                        subset["y_pred"],
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def grouped_metrics_original_global(
    all_outputs: pd.DataFrame,
    n_bins: int = N_BINS,
) -> pd.DataFrame:
    """Also save the old code's global, across-repeat binning result."""
    rows = []

    for method in METHOD_ORDER:
        method_df = all_outputs[all_outputs["method"] == method].copy()
        method_df["group"] = (
            pd.qcut(
                method_df["credibility"],
                q=n_bins,
                labels=False,
                duplicates="drop",
            ).astype(int)
            + 1
        )

        for group_number in sorted(method_df["group"].unique()):
            subset = method_df[method_df["group"] == group_number]
            rows.append(
                {
                    "dataset": DATASET_NAME,
                    "n_train": "full",
                    "method": method,
                    "group": int(group_number),
                    "count": int(len(subset)),
                    "cred_mean": float(subset["credibility"].mean()),
                    "accuracy": float(
                        accuracy_score(subset["y_true"], subset["y_pred"])
                    ),
                    "f1": float(
                        f1_score(
                            subset["y_true"],
                            subset["y_pred"],
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "f1_macro": float(
                        f1_score(
                            subset["y_true"],
                            subset["y_pred"],
                            average="macro",
                            zero_division=0,
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


# =========================================================
# Main
# =========================================================
def main() -> None:
    set_seed(SEED)
    output_dir = resolve_path(OUTPUT_DIR)
    ensure_dir(output_dir)

    log_step("Loading TCGA_X.npy and TCGA_Y.npy")
    X_raw, y_raw = load_data()
    log_step(f"Loaded X shape={X_raw.shape}, y length={len(y_raw)}")

    (
        X,
        y,
        kept_labels,
        mapping_df,
        original_distribution,
        filtered_distribution,
    ) = filter_top_k_classes(X_raw, y_raw, keep_top_k=KEEP_TOP_K)

    log_step(f"Kept original labels: {kept_labels}")
    log_step(f"Top-{KEEP_TOP_K} remapped counts: {format_label_counts(y)}")
    log_step("No additional log transformation is applied, matching the original code")

    original_distribution.to_csv(
        os.path.join(output_dir, "label_distribution_original.csv"),
        index=False,
    )
    mapping_df.to_csv(
        os.path.join(output_dir, "top3_label_mapping.csv"),
        index=False,
    )
    filtered_distribution.to_csv(
        os.path.join(output_dir, "label_distribution_top3.csv"),
        index=False,
    )

    all_output_frames: List[pd.DataFrame] = []
    metric_rows: List[Dict[str, float]] = []
    grouped_frames: List[pd.DataFrame] = []

    log_step(
        f"Starting experiment: repeats={N_REPEATS}, internal iterations={N_BOOT}, "
        f"test_size={TEST_SIZE:.2f}, proper_ratio={PROPER_TRAIN_RATIO:.2f}"
    )

    for repeat in range(N_REPEATS):
        seed_repeat = SEED + repeat

        log_step(f"Repeat {repeat + 1}/{N_REPEATS}: stratified train/test split")
        train_idx, test_idx = split_train_test_indices(
            y,
            seed=seed_repeat,
            test_size=TEST_SIZE,
        )

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        log_step(
            f"Repeat {repeat + 1}/{N_REPEATS}: train={len(y_train)} "
            f"({format_label_counts(y_train)}); test={len(y_test)} "
            f"({format_label_counts(y_test)})"
        )

        log_step(f"Repeat {repeat + 1}/{N_REPEATS}: running ICP")
        df_icp, metrics_icp = run_icp(
            X_train,
            y_train,
            X_test,
            y_test,
            alpha=ALPHA,
            rng_seed=seed_repeat,
        )

        log_step(f"Repeat {repeat + 1}/{N_REPEATS}: running BICP")
        df_bicp, metrics_bicp = run_bicp(
            X_train,
            y_train,
            X_test,
            y_test,
            alpha=ALPHA,
            rng_seed=seed_repeat,
            n_boot=N_BOOT,
            verbose=True,
        )

        log_step(f"Repeat {repeat + 1}/{N_REPEATS}: running RICP")
        df_ricp, metrics_ricp = run_ricp(
            X_train,
            y_train,
            X_test,
            y_test,
            alpha=ALPHA,
            rng_seed=seed_repeat,
            n_random=N_BOOT,
            verbose=True,
        )

        log_step(f"Repeat {repeat + 1}/{N_REPEATS}: running OOB-ICP")
        df_oob, metrics_oob = run_oobicp(
            X_train,
            y_train,
            X_test,
            y_test,
            alpha=ALPHA,
            rng_seed=seed_repeat,
            n_boot=N_BOOT,
            verbose=True,
        )

        method_results = {
            "ICP": (df_icp, metrics_icp),
            "BICP": (df_bicp, metrics_bicp),
            "RICP": (df_ricp, metrics_ricp),
            "OOB-ICP": (df_oob, metrics_oob),
        }

        for method in METHOD_ORDER:
            output_df, metrics = method_results[method]

            # Replace the internal seed identifier with the actual outer-repeat ID.
            output_df["repeat"] = repeat
            metrics["repeat"] = repeat

            all_output_frames.append(output_df)
            metric_rows.append(metrics)
            grouped_frames.append(
                grouped_metrics_per_repeat(output_df, n_bins=N_BINS)
            )

        log_step(f"Repeat {repeat + 1}/{N_REPEATS}: finished all four methods")

    all_outputs = pd.concat(all_output_frames, ignore_index=True)
    per_repeat_metrics = pd.DataFrame(metric_rows)
    grouped_metrics = pd.concat(grouped_frames, ignore_index=True)

    summary_metrics = (
        per_repeat_metrics.groupby(
            ["dataset", "n_train", "method"],
            as_index=False,
        )
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", lambda values: values.std(ddof=0)),
            f1_mean=("f1", "mean"),
            f1_std=("f1", lambda values: values.std(ddof=0)),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", lambda values: values.std(ddof=0)),
            f1_weighted_mean=("f1_weighted", "mean"),
            f1_weighted_std=(
                "f1_weighted",
                lambda values: values.std(ddof=0),
            ),
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", lambda values: values.std(ddof=0)),
            avg_set_size_mean=("avg_set_size", "mean"),
            avg_set_size_std=(
                "avg_set_size",
                lambda values: values.std(ddof=0),
            ),
            credibility_mean=("credibility_mean", "mean"),
            credibility_std=(
                "credibility_mean",
                lambda values: values.std(ddof=0),
            ),
        )
    )

    # Unified filenames expected by the current plotting program.
    all_outputs.to_csv(
        os.path.join(output_dir, "all_test_outputs_4methods.csv"),
        index=False,
    )
    per_repeat_metrics.to_csv(
        os.path.join(output_dir, "per_repeat_metrics_4methods.csv"),
        index=False,
    )
    grouped_metrics.to_csv(
        os.path.join(output_dir, "grouped_metrics_4methods.csv"),
        index=False,
    )
    summary_metrics.to_csv(
        os.path.join(output_dir, "summary_metrics_4methods.csv"),
        index=False,
    )

    # Also retain the old code's global-across-repeat credibility grouping.
    original_global_grouping = grouped_metrics_original_global(
        all_outputs,
        n_bins=N_BINS,
    )
    original_global_grouping.to_csv(
        os.path.join(output_dir, "grouped_metrics_original_global.csv"),
        index=False,
    )

    log_step(f"All CSV outputs saved to: {output_dir}")
    print("=== DONE: Breast cancer subtyping experiment ===", flush=True)
    print(f"Top-3 original labels: {kept_labels}", flush=True)
    print(f"Repeats: {N_REPEATS}", flush=True)
    print(f"BICP/RICP/OOB-ICP iterations: {N_BOOT}", flush=True)
    print(f"Train/test: {1.0 - TEST_SIZE:.2f}/{TEST_SIZE:.2f}", flush=True)
    print(
        f"Proper/calibration within training pool: "
        f"{PROPER_TRAIN_RATIO:.2f}/{1.0 - PROPER_TRAIN_RATIO:.2f}",
        flush=True,
    )
    print(
        f"Model: VarianceThreshold + SelectKBest(k={K_FEATURES}) + "
        "L1 LogisticRegression(saga, balanced, max_iter=1000)",
        flush=True,
    )


if __name__ == "__main__":
    main()