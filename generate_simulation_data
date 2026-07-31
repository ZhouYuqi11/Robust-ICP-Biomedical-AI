# -*- coding: utf-8 -*-
"""Generate the exact two-class Gaussian dataset used by Simulation.py.

Dataset design
--------------
- Total sample size: 6,000
- Class 0: 3,000 observations from N([0.5, 0.5], Sigma)
- Class 1: 3,000 observations from N([-0.5, -0.5], Sigma)
- Shared covariance matrix: [[1.5, 0.3], [0.3, 1.5]]
- NumPy generator: np.random.default_rng(42)
- Fixed stratified split: sklearn.model_selection.train_test_split,
  with test_size=0.20 and random_state=42

The generated CSV contains the following columns:
sample_id, source_index, feature_1, feature_2, label, split, split_order

Rows in each split retain exactly the order returned by train_test_split in
the experiment. ``source_index`` identifies the row in the original class-
ordered 6,000-sample array before splitting.

Run:
    python generate_simulation_data.py

By default, ``simulation_dataset.csv`` is written next to this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42
N_TOTAL = 6000
N_CLASSES = 2
TEST_FRACTION = 0.20

CLASS_MEANS = {
    0: np.array([0.5, 0.5], dtype=float),
    1: np.array([-0.5, -0.5], dtype=float),
}

COVARIANCE = np.array(
    [
        [1.5, 0.3],
        [0.3, 1.5],
    ],
    dtype=float,
)


def validate_configuration() -> None:
    """Check that the requested design is internally consistent."""
    if N_TOTAL % N_CLASSES != 0:
        raise ValueError("N_TOTAL must be divisible by N_CLASSES.")

    if not 0.0 < TEST_FRACTION < 1.0:
        raise ValueError("TEST_FRACTION must be between 0 and 1.")

    if COVARIANCE.shape != (2, 2):
        raise ValueError("COVARIANCE must be a 2 x 2 matrix.")

    if not np.allclose(COVARIANCE, COVARIANCE.T):
        raise ValueError("COVARIANCE must be symmetric.")

    if np.any(np.linalg.eigvalsh(COVARIANCE) <= 0):
        raise ValueError("COVARIANCE must be positive definite.")


def fixed_split_indices(
    y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the exact fixed split used in the experiment code."""
    source_indices = np.arange(len(y), dtype=int)
    pool_indices, test_indices = train_test_split(
        source_indices,
        test_size=TEST_FRACTION,
        stratify=y,
        random_state=seed,
    )
    return (
        np.asarray(pool_indices, dtype=int),
        np.asarray(test_indices, dtype=int),
    )


def generate_dataset(seed: int = SEED) -> pd.DataFrame:
    """Generate and return the complete simulation dataset."""
    validate_configuration()
    # This must remain default_rng to match Simulation.py exactly.
    rng = np.random.default_rng(seed)
    samples_per_class = N_TOTAL // N_CLASSES

    feature_blocks = []
    label_blocks = []
    for class_label in sorted(CLASS_MEANS):
        features = rng.multivariate_normal(
            mean=CLASS_MEANS[class_label],
            cov=COVARIANCE,
            size=samples_per_class,
        )
        feature_blocks.append(features)
        label_blocks.append(
            np.full(samples_per_class, class_label, dtype=int)
        )

    X = np.vstack(feature_blocks)
    y = np.concatenate(label_blocks)
    pool_indices, test_indices = fixed_split_indices(y, seed)

    # Preserve the order returned by train_test_split for each subset.
    ordered_indices = np.concatenate([pool_indices, test_indices])
    split = np.concatenate(
        [
            np.full(len(pool_indices), "development_pool", dtype=object),
            np.full(len(test_indices), "test", dtype=object),
        ]
    )
    split_order = np.concatenate(
        [
            np.arange(len(pool_indices), dtype=int),
            np.arange(len(test_indices), dtype=int),
        ]
    )

    dataset = pd.DataFrame(
        {
            "sample_id": [f"sample_{i + 1:04d}" for i in ordered_indices],
            "source_index": ordered_indices,
            "feature_1": X[ordered_indices, 0],
            "feature_2": X[ordered_indices, 1],
            "label": y[ordered_indices],
            "split": split,
            "split_order": split_order,
        }
    )
    return dataset


def verify_dataset(dataset: pd.DataFrame) -> None:
    """Verify sample counts, class balance, and fixed split sizes."""
    if len(dataset) != N_TOTAL:
        raise RuntimeError(f"Expected {N_TOTAL} rows, got {len(dataset)}.")

    if dataset["source_index"].nunique() != N_TOTAL:
        raise RuntimeError("source_index must contain every source row once.")

    expected_per_class = N_TOTAL // N_CLASSES
    class_counts = dataset["label"].value_counts().sort_index().to_dict()
    expected_counts = {
        class_label: expected_per_class for class_label in CLASS_MEANS
    }
    if class_counts != expected_counts:
        raise RuntimeError(
            f"Unexpected class counts: {class_counts}; expected {expected_counts}."
        )

    expected_test_per_class = int(round(expected_per_class * TEST_FRACTION))
    split_counts = (
        dataset.groupby(["label", "split"]).size().unstack(fill_value=0)
    )
    for class_label in CLASS_MEANS:
        if split_counts.loc[class_label, "test"] != expected_test_per_class:
            raise RuntimeError(
                f"Unexpected test count for class {class_label}: "
                f"{split_counts.loc[class_label, 'test']}."
            )

    expected_pool_size = N_TOTAL - int(round(N_TOTAL * TEST_FRACTION))
    pool = dataset[dataset["split"] == "development_pool"]
    test = dataset[dataset["split"] == "test"]
    if len(pool) != expected_pool_size or len(test) != N_TOTAL - expected_pool_size:
        raise RuntimeError("Unexpected development/test split sizes.")

    if not np.array_equal(pool["split_order"].to_numpy(), np.arange(len(pool))):
        raise RuntimeError("Development-pool row order is not contiguous.")
    if not np.array_equal(test["split_order"].to_numpy(), np.arange(len(test))):
        raise RuntimeError("Test-set row order is not contiguous.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Gaussian simulation dataset."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().with_name("simulation_dataset.csv"),
        help="Output CSV path (default: next to this script).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = generate_dataset(seed=args.seed)
    verify_dataset(dataset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Seventeen significant digits preserve IEEE-754 float64 values on reload.
    dataset.to_csv(args.output, index=False, float_format="%.17g")

    split_summary = (
        dataset.groupby(["label", "split"]).size().unstack(fill_value=0)
    )
    print(f"Saved {len(dataset)} samples to: {args.output}")
    print("Class and split counts:")
    print(split_summary.to_string())


if __name__ == "__main__":
    main()
