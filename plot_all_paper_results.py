# -*- coding: utf-8 -*-
"""
Plot all manuscript figures and export all manuscript-style tables from the four experiment folders.

Expected folder layout:

project_root/
├── plot_all_paper_results_modified.py
├── Simulation/
│   ├── Simulation.py
│   └── results/
│       ├── all_test_outputs_4methods.csv
│       ├── per_repeat_metrics_4methods.csv
│       └── grouped_metrics_4methods.csv
├── Pneumonia diagnosis/
│   ├── Pneumonia diagnosis.py
│   └── results/
│       ├── all_test_outputs_4methods.csv
│       ├── per_repeat_metrics_4methods.csv
│       └── grouped_metrics_4methods.csv
├── Chinese_herbal_medicine/
│   ├── Chinese_herbal_medicine.py
│   └── results/
│       ├── all_test_outputs_4methods.csv
│       ├── per_repeat_metrics_4methods.csv
│       └── grouped_metrics_4methods.csv
└── Breast_cancer_subtyping/
    ├── Breast_cancer_subtyping.py
    └── results/
        ├── all_test_outputs_4methods.csv
        ├── per_repeat_metrics_4methods.csv
        └── grouped_metrics_4methods.csv

The script outputs figures and tables under ./paper_outputs/.
For the main 2x2 figures, Simulation and Pneumonia use n=200 by default,
while the two small datasets use the full repeated train/test experiments.

Modification in this version:
- In the stability histograms, each method has a separate colored line above
  the histogram showing mean ± 1 standard deviation of the per-sample
  credibility-STD distribution.
- The central marker on each line is the mean.
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_rel


# =========================
# Global settings
# =========================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT_DIR, "paper_outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TABLE_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["savefig.dpi"] = 300

METHOD_ORDER = ["ICP", "BICP", "RICP", "OOB-ICP"]
PANEL_LABELS_4 = ["(a)", "(b)", "(c)", "(d)"]

METHOD_COLORS = {
    "ICP": "#5DA5DA",
    "BICP": "#DEB887",
    "RICP": "#8CD17D",
    "OOB-ICP": "#D4A6E8",
}

METHOD_MARKERS = {
    "ICP": "o",
    "BICP": "s",
    "RICP": "^",
    "OOB-ICP": "D",
}

METHOD_LINESTYLES = {
    "ICP": "-",
    "BICP": "--",
    "RICP": "-.",
    "OOB-ICP": ":",
}

METHOD_NAME_MAP = {
    "ICP": "ICP",
    "BICP": "BICP",
    "B-ICP": "BICP",
    "BootICP": "BICP",
    "Bootstrapped ICP": "BICP",
    "Bootstrapping ICP": "BICP",
    "Bootstrap ICP": "BICP",
    "RICP": "RICP",
    "R-ICP": "RICP",
    "Randomized ICP": "RICP",
    "RandomizedICP": "RICP",
    "OOB-ICP": "OOB-ICP",
    "OOBICP": "OOB-ICP",
    "oob_icp": "OOB-ICP",
    "oobicp": "OOB-ICP",
}

DATASETS = {
    "Simulation": {
        "folder_candidates": ["Simulation", "simulation"],
        "display": "Simulation",
        "main_n": "200",
        "fixed_test": True,
        "binary": True,
    },
    "Pneumonia diagnosis": {
        "folder_candidates": ["Pneumonia diagnosis", "Pneumonia_diagnosis", "Pneumonia"],
        "display": "Pneumonia diagnosis",
        "main_n": "200",
        "fixed_test": True,
        "binary": True,
    },
    "Chinese herbal medicine classification": {
        "folder_candidates": ["Chinese_herbal_medicine", "Chinese herbal medicine", "Chinese"],
        "display": "Chinese herbal medicine classification",
        "main_n": "full",
        "fixed_test": False,
        "binary": False,
    },
    "Breast cancer subtyping": {
        "folder_candidates": ["Breast_cancer_subtyping", "Breast cancer subtyping", "Breast"],
        "display": "Breast cancer subtyping",
        "main_n": "full",
        "fixed_test": False,
        "binary": False,
    },
}


# =========================
# Utility functions
# =========================
def normalize_method_name(x: str) -> str:
    x = str(x).strip()
    return METHOD_NAME_MAP.get(x, x)


def resolve_dataset_folder(info: Dict) -> str:
    for folder in info["folder_candidates"]:
        p = os.path.join(ROOT_DIR, folder)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        "Cannot find dataset folder. Tried: " + ", ".join(info["folder_candidates"])
    )


def read_result_csv(dataset_name: str, file_name: str) -> pd.DataFrame:
    info = DATASETS[dataset_name]
    folder = resolve_dataset_folder(info)
    path = os.path.join(folder, "results", file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing result file: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "method" in df.columns:
        df["method"] = df["method"].apply(normalize_method_name)
    if "dataset" not in df.columns:
        df["dataset"] = dataset_name
    df["dataset"] = dataset_name
    if "n_train" in df.columns:
        df["n_train"] = df["n_train"].astype(str)
    return df


def load_all_metrics() -> pd.DataFrame:
    frames = []
    for dataset_name in DATASETS:
        df = read_result_csv(dataset_name, "per_repeat_metrics_4methods.csv")
        if "f1" not in df.columns:
            if "f1_macro" in df.columns:
                df["f1"] = df["f1_macro"]
            elif "f1_weighted" in df.columns:
                df["f1"] = df["f1_weighted"]
            else:
                raise ValueError(f"{dataset_name}: missing f1/f1_macro/f1_weighted.")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out[out["method"].isin(METHOD_ORDER)].copy()


def load_all_outputs() -> pd.DataFrame:
    frames = []
    for dataset_name in DATASETS:
        df = read_result_csv(dataset_name, "all_test_outputs_4methods.csv")
        if "sample_idx" not in df.columns:
            df = df.sort_values(["method", "n_train", "repeat"]).copy()
            df["sample_idx"] = df.groupby(["method", "n_train", "repeat"]).cumcount()
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out[out["method"].isin(METHOD_ORDER)].copy()


def load_all_grouped() -> pd.DataFrame:
    frames = []
    for dataset_name in DATASETS:
        df = read_result_csv(dataset_name, "grouped_metrics_4methods.csv")
        if "f1" not in df.columns:
            if "f1_macro" in df.columns:
                df["f1"] = df["f1_macro"]
            else:
                raise ValueError(f"{dataset_name}: grouped metrics missing f1/f1_macro.")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out[out["method"].isin(METHOD_ORDER)].copy()


def main_subset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    main_n = DATASETS[dataset_name]["main_n"]
    if "n_train" in df.columns:
        return df[
            (df["dataset"] == dataset_name)
            & (df["n_train"].astype(str) == str(main_n))
        ].copy()
    return df[df["dataset"] == dataset_name].copy()


def p_to_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def save_figure(fig, output_stem: str):
    png_path = os.path.join(FIG_DIR, f"{output_stem}.png")
    pdf_path = os.path.join(FIG_DIR, f"{output_stem}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", png_path)
    print("Saved:", pdf_path)


def set_box_style(bp, colors: List[str]):
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.70)
        patch.set_linewidth(1.2)
    for whisker in bp["whiskers"]:
        whisker.set_linewidth(1.0)
    for cap in bp["caps"]:
        cap.set_linewidth(1.0)
    for median in bp["medians"]:
        median.set_linewidth(1.4)
        median.set_color("black")


def upper_whisker(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    inside = vals[vals <= upper_bound]
    return float(np.max(inside)) if len(inside) else float(np.max(vals))


def add_method_stars_paired(
    ax,
    df_dataset: pd.DataFrame,
    metric: str,
    data: List[np.ndarray],
):
    """Paired t-test against ICP across repeats."""
    icp = df_dataset[df_dataset["method"] == "ICP"].sort_values("repeat")
    y_min, y_max = ax.get_ylim()
    y_range = max(y_max - y_min, 1e-6)
    star_ys = []

    for idx, method in enumerate(METHOD_ORDER[1:], start=1):
        other = df_dataset[df_dataset["method"] == method].sort_values("repeat")
        merged = icp[["repeat", metric]].merge(
            other[["repeat", metric]],
            on="repeat",
            suffixes=("_icp", "_other"),
        )
        if len(merged) < 2:
            continue
        _, p = ttest_rel(
            merged[f"{metric}_icp"],
            merged[f"{metric}_other"],
            nan_policy="omit",
        )
        stars = p_to_stars(p)
        if not stars:
            continue
        x = idx + 1
        y = upper_whisker(data[idx]) + 0.035 * y_range
        star_ys.append(y)
        ax.text(
            x,
            y,
            stars,
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="black",
        )

    if star_ys:
        ax.set_ylim(y_min, max(y_max, max(star_ys) + 0.10 * y_range))


# =========================
# Tables
# =========================
def export_metric_tables(df_metrics: pd.DataFrame):
    summary = df_metrics.groupby(
        ["dataset", "n_train", "method"], as_index=False
    ).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", lambda x: x.std(ddof=0)),
        f1_mean=("f1", "mean"),
        f1_std=("f1", lambda x: x.std(ddof=0)),
        credibility_mean=("credibility_mean", "mean"),
        credibility_std=("credibility_mean", lambda x: x.std(ddof=0)),
    )
    summary.to_csv(
        os.path.join(TABLE_DIR, "summary_all_metrics_mean_std.csv"), index=False
    )

    rows = []
    for dataset_name in DATASETS:
        sub = summary[summary["dataset"] == dataset_name].copy()
        n_order = (
            ["100", "200", "400", "800"]
            if DATASETS[dataset_name]["fixed_test"]
            else ["full"]
        )
        for n in n_order:
            sn = sub[sub["n_train"].astype(str) == n]
            if sn.empty:
                continue
            row_name = dataset_name if n == "full" else f"{dataset_name} (n = {n})"
            row = {"Dataset": row_name}
            for method in METHOD_ORDER:
                sm = sn[sn["method"] == method]
                row[method] = (
                    ""
                    if sm.empty
                    else f"{sm['accuracy_mean'].iloc[0]:.3f} ± {sm['accuracy_std'].iloc[0]:.3f}"
                )
            rows.append(row)

    table2 = pd.DataFrame(rows)
    table2.to_csv(
        os.path.join(TABLE_DIR, "table2_accuracy_mean_std.csv"), index=False
    )
    table2.to_markdown(
        os.path.join(TABLE_DIR, "table2_accuracy_mean_std.md"), index=False
    )

    rows_f1 = []
    for dataset_name in DATASETS:
        sub = summary[summary["dataset"] == dataset_name].copy()
        n_order = (
            ["100", "200", "400", "800"]
            if DATASETS[dataset_name]["fixed_test"]
            else ["full"]
        )
        for n in n_order:
            sn = sub[sub["n_train"].astype(str) == n]
            if sn.empty:
                continue
            row_name = dataset_name if n == "full" else f"{dataset_name} (n = {n})"
            row = {"Dataset": row_name}
            for method in METHOD_ORDER:
                sm = sn[sn["method"] == method]
                row[method] = (
                    ""
                    if sm.empty
                    else f"{sm['f1_mean'].iloc[0]:.3f} ± {sm['f1_std'].iloc[0]:.3f}"
                )
            rows_f1.append(row)

    f1_table = pd.DataFrame(rows_f1)
    f1_table.to_csv(
        os.path.join(TABLE_DIR, "supplementary_f1_mean_std.csv"), index=False
    )
    f1_table.to_markdown(
        os.path.join(TABLE_DIR, "supplementary_f1_mean_std.md"), index=False
    )
    print("Saved metric tables to:", TABLE_DIR)


def export_credibility_tables(df_outputs: pd.DataFrame):
    rows = []
    for dataset_name in DATASETS:
        sub = main_subset(df_outputs, dataset_name)
        for method in METHOD_ORDER:
            sm = sub[sub["method"] == method]
            repeat_means = sm.groupby(
                ["repeat", "is_correct"], as_index=False
            )["credibility"].mean()
            correct = repeat_means.loc[
                repeat_means["is_correct"] == 1, "credibility"
            ].dropna().values
            wrong = repeat_means.loc[
                repeat_means["is_correct"] == 0, "credibility"
            ].dropna().values
            _, p = (
                ttest_ind(correct, wrong, equal_var=False, nan_policy="omit")
                if len(correct) > 1 and len(wrong) > 1
                else (np.nan, np.nan)
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "n_train_for_figure": DATASETS[dataset_name]["main_n"],
                    "method": method,
                    "correct_repeat_n": len(correct),
                    "wrong_repeat_n": len(wrong),
                    "correct_median": np.nanmedian(correct) if len(correct) else np.nan,
                    "correct_q1": np.nanpercentile(correct, 25) if len(correct) else np.nan,
                    "correct_q3": np.nanpercentile(correct, 75) if len(correct) else np.nan,
                    "wrong_median": np.nanmedian(wrong) if len(wrong) else np.nan,
                    "wrong_q1": np.nanpercentile(wrong, 25) if len(wrong) else np.nan,
                    "wrong_q3": np.nanpercentile(wrong, 75) if len(wrong) else np.nan,
                    "correct_vs_wrong_pvalue": p,
                    "stars": p_to_stars(p),
                }
            )
    pd.DataFrame(rows).to_csv(
        os.path.join(TABLE_DIR, "correct_wrong_credibility_summary.csv"),
        index=False,
    )


def export_stability_table(df_outputs: pd.DataFrame):
    rows = []
    for dataset_name, info in DATASETS.items():
        if not info["fixed_test"]:
            continue
        sub = main_subset(df_outputs, dataset_name)
        for method in METHOD_ORDER:
            sm = sub[sub["method"] == method]
            std_df = sm.groupby("sample_idx")["credibility"].std(ddof=0).reset_index()
            vals = std_df["credibility"].dropna().values
            rows.append(
                {
                    "dataset": dataset_name,
                    "n_train_for_figure": info["main_n"],
                    "method": method,
                    "mean_per_sample_credibility_std": np.mean(vals),
                    "median_per_sample_credibility_std": np.median(vals),
                    "q1": np.percentile(vals, 25),
                    "q3": np.percentile(vals, 75),
                }
            )
    pd.DataFrame(rows).to_csv(
        os.path.join(TABLE_DIR, "stability_summary.csv"), index=False
    )


# =========================
# Figure A1/A2: Accuracy/F1 boxplots
# =========================
def plot_metric_boxplots(
    df_metrics: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_stem: str,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), dpi=300)
    axes = axes.flatten()

    for ax, (dataset_name, _info), panel_label in zip(
        axes, DATASETS.items(), PANEL_LABELS_4
    ):
        sub = main_subset(df_metrics, dataset_name)
        data = [
            sub.loc[sub["method"] == method, metric].dropna().values
            for method in METHOD_ORDER
        ]
        colors = [METHOD_COLORS[m] for m in METHOD_ORDER]
        bp = ax.boxplot(
            data,
            labels=METHOD_ORDER,
            patch_artist=True,
            widths=0.60,
            showfliers=False,
        )
        set_box_style(bp, colors)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", linestyle="--", alpha=0.30)
        add_method_stars_paired(ax, sub, metric, data)
        ax.text(
            0.5,
            -0.24,
            panel_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
        )

    fig.tight_layout(w_pad=2.0, h_pad=2.8)
    save_figure(fig, output_stem)


# =========================
# Figure 3: Correct vs Wrong credibility
# =========================
def add_pair_bracket(
    ax,
    x1: float,
    x2: float,
    y: float,
    h: float,
    text: str,
    fontsize: int = 12,
):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color="black")
    ax.text(
        (x1 + x2) / 2.0,
        y + h,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
    )


def plot_correct_wrong(
    df_outputs: pd.DataFrame,
    output_stem: str = "fig4_credibility_correct_wrong",
):
    fig, axes = plt.subplots(2, 2, figsize=(20, 10), dpi=300)
    axes = axes.flatten()

    for ax, dataset_name, panel_label in zip(
        axes, DATASETS.keys(), PANEL_LABELS_4
    ):
        sub = main_subset(df_outputs, dataset_name)
        data, labels, colors = [], [], []
        pair_pvalues = []

        for method in METHOD_ORDER:
            sm = sub[sub["method"] == method]
            repeat_means = sm.groupby(
                ["repeat", "is_correct"], as_index=False
            )["credibility"].mean()
            correct = repeat_means.loc[
                repeat_means["is_correct"] == 1, "credibility"
            ].dropna().values
            wrong = repeat_means.loc[
                repeat_means["is_correct"] == 0, "credibility"
            ].dropna().values
            data.extend([correct, wrong])
            labels.extend([f"{method}-Correct", f"{method}-Wrong"])
            colors.extend([METHOD_COLORS[method], METHOD_COLORS[method]])
            _, p = (
                ttest_ind(correct, wrong, equal_var=False, nan_policy="omit")
                if len(correct) > 1 and len(wrong) > 1
                else (np.nan, np.nan)
            )
            pair_pvalues.append(p)

        bp = ax.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            showfliers=False,
            widths=0.55,
        )
        set_box_style(bp, colors)
        ax.set_ylabel("Mean Credibility per Repeat")
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", linestyle="--", alpha=0.25)

        for i, vals in enumerate(data, start=1):
            vals = np.asarray(vals, dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.text(
                i,
                q3 + 0.035,
                f"median={med:.3f}\nIQR=[{q1:.3f},{q3:.3f}]",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        y_min, y_max = ax.get_ylim()
        y_range = max(y_max - y_min, 1e-6)
        base_y = max(1.00, y_max - 0.06 * y_range)
        h = 0.025 * y_range
        for k, p in enumerate(pair_pvalues):
            stars = p_to_stars(p)
            if stars:
                add_pair_bracket(
                    ax,
                    2 * k + 1,
                    2 * k + 2,
                    base_y,
                    h,
                    stars,
                    fontsize=12,
                )
        ax.set_ylim(0, max(1.08, base_y + h + 0.04))
        ax.text(
            0.5,
            -0.22,
            panel_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13,
        )

    fig.tight_layout(w_pad=2.5, h_pad=3.2)
    save_figure(fig, output_stem)


# =========================
# Figure 4: TP/TN/FP/FN credibility for binary fixed-test tasks
# =========================
def compute_confusion_group_meancred(
    df_outputs: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    sub = main_subset(df_outputs, dataset_name)
    rows = []
    for method in METHOD_ORDER:
        sm = sub[sub["method"] == method]
        for r in sorted(sm["repeat"].unique()):
            sr = sm[sm["repeat"] == r]
            masks = {
                "TP": (sr["y_true"] == 1) & (sr["y_pred"] == 1),
                "TN": (sr["y_true"] == 0) & (sr["y_pred"] == 0),
                "FP": (sr["y_true"] == 0) & (sr["y_pred"] == 1),
                "FN": (sr["y_true"] == 1) & (sr["y_pred"] == 0),
            }
            for group_name, mask in masks.items():
                vals = sr.loc[mask, "credibility"].dropna().values
                rows.append(
                    {
                        "dataset": dataset_name,
                        "method": method,
                        "repeat": r,
                        "conf_group": group_name,
                        "mean_credibility": float(np.mean(vals)) if len(vals) else np.nan,
                        "count": int(len(vals)),
                    }
                )
    return pd.DataFrame(rows)


def plot_confusion_groups(
    df_outputs: pd.DataFrame,
    output_stem: str = "fig4_confusion_groups",
):
    datasets_for_conf = [d for d, info in DATASETS.items() if info["binary"]]
    fig, axes = plt.subplots(
        len(datasets_for_conf), 1, figsize=(12, 10.8), dpi=300
    )
    if len(datasets_for_conf) == 1:
        axes = [axes]

    group_order = ["TP", "TN", "FP", "FN"]
    centers = np.arange(len(group_order)) + 1
    offsets = {"ICP": -0.27, "BICP": -0.09, "RICP": 0.09, "OOB-ICP": 0.27}

    all_conf = []
    for ax, dataset_name, panel_label in zip(
        axes, datasets_for_conf, ["(a)", "(b)"]
    ):
        df_conf = compute_confusion_group_meancred(df_outputs, dataset_name)
        all_conf.append(df_conf)
        all_vals = []

        for method in METHOD_ORDER:
            data, positions = [], []
            for i, group in enumerate(group_order):
                vals = df_conf.loc[
                    (df_conf["method"] == method)
                    & (df_conf["conf_group"] == group),
                    "mean_credibility",
                ].dropna().values
                data.append(vals)
                positions.append(centers[i] + offsets[method])
                all_vals.extend(vals.tolist())
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=0.16,
                patch_artist=True,
                showfliers=False,
            )
            set_box_style(bp, [METHOD_COLORS[method]] * len(data))

        ax.set_xticks(centers)
        ax.set_xticklabels(group_order)
        ax.set_ylabel("Mean Credibility within Group (per repeat)")
        ax.grid(axis="y", linestyle="--", alpha=0.25)

        if all_vals:
            y_min, y_max = np.nanmin(all_vals), np.nanmax(all_vals)
            y_range = max(y_max - y_min, 1e-6)
            ax.set_ylim(
                max(0, y_min - 0.18 * y_range),
                min(1.05, y_max + 0.12 * y_range),
            )

        handles = [
            plt.Line2D([0], [0], color=METHOD_COLORS[m], lw=6, label=m)
            for m in METHOD_ORDER
        ]
        ax.legend(
            handles=handles,
            loc="lower right",
            bbox_to_anchor=(0.995, 0.03),
            ncol=4,
            frameon=True,
        )
        ax.text(
            0.5,
            -0.18,
            panel_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13,
        )

    pd.concat(all_conf, ignore_index=True).to_csv(
        os.path.join(TABLE_DIR, "confusion_group_meancred_summary.csv"),
        index=False,
    )
    fig.tight_layout(h_pad=2.8)
    fig.subplots_adjust(bottom=0.08)
    save_figure(fig, output_stem)


# =========================
# Figure 5/A3: Accuracy/F1 by credibility bins
# =========================
def plot_bins(
    df_grouped: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_stem: str,
):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=300)
    axes = axes.flatten()

    for ax, dataset_name, panel_label in zip(
        axes, DATASETS.keys(), PANEL_LABELS_4
    ):
        sub = main_subset(df_grouped, dataset_name)
        for method in METHOD_ORDER:
            sm = sub[sub["method"] == method]
            if sm.empty:
                continue

            if "repeat" in sm.columns:
                for _, row in sm.iterrows():
                    ax.scatter(
                        row["group"],
                        row[metric],
                        color=METHOD_COLORS[method],
                        alpha=0.18,
                        s=16,
                        edgecolor="none",
                    )
                mean_df = (
                    sm.groupby("group", as_index=False)[metric]
                    .mean()
                    .sort_values("group")
                )
            else:
                mean_df = sm[["group", metric]].sort_values("group")

            ax.plot(
                mean_df["group"],
                mean_df[metric],
                label=method,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linestyle=METHOD_LINESTYLES[method],
                linewidth=2.0,
                markersize=5,
            )

        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(["Bin1", "Bin2", "Bin3", "Bin4", "Bin5"])
        ax.set_xlabel("Credibility Group")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(loc="lower right", frameon=True, ncol=2)
        ax.text(
            0.5,
            -0.18,
            panel_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13,
        )

    fig.tight_layout(w_pad=2.2, h_pad=2.8)
    save_figure(fig, output_stem)


# =========================
# Figure 6: Stability histograms for fixed-test tasks
# =========================
def add_mean_std_line(
    ax,
    mean_value: float,
    std_value: float,
    y_value: float,
    method: str,
    cap_height: float,
):
    """
    Draw one colored mean ± 1 SD line.

    Horizontal line: mean - SD to mean + SD
    Central marker: mean
    Vertical caps: two endpoints
    """
    left = max(0.0, mean_value - std_value)
    right = mean_value + std_value
    color = METHOD_COLORS[method]

    ax.hlines(
        y=y_value,
        xmin=left,
        xmax=right,
        color=color,
        linewidth=3.0,
        zorder=8,
        clip_on=False,
    )
    ax.vlines(
        x=[left, right],
        ymin=y_value - cap_height,
        ymax=y_value + cap_height,
        color=color,
        linewidth=1.7,
        zorder=8,
        clip_on=False,
    )
    ax.scatter(
        [mean_value],
        [y_value],
        color=color,
        marker=METHOD_MARKERS[method],
        s=42,
        edgecolor="black",
        linewidth=0.45,
        zorder=9,
        clip_on=False,
    )


def plot_stability_hist(
    df_outputs: pd.DataFrame,
    output_stem: str = "fig6_stability",
):
    datasets_for_stab = [d for d, info in DATASETS.items() if info["fixed_test"]]
    fig, axes = plt.subplots(
        1,
        len(datasets_for_stab),
        figsize=(17, 5.8),
        dpi=300,
    )
    if len(datasets_for_stab) == 1:
        axes = [axes]

    for ax, dataset_name, panel_label in zip(
        axes,
        datasets_for_stab,
        ["(a)", "(b)"],
    ):
        sub = main_subset(df_outputs, dataset_name)

        # Cache each method's per-sample STD values so that the histogram and
        # the mean ± 1 SD annotation are computed from exactly the same data.
        method_values = {}
        all_values = []

        for method in METHOD_ORDER:
            sm = sub[sub["method"] == method]
            std_df = (
                sm.groupby("sample_idx")["credibility"]
                .std(ddof=0)
                .reset_index()
            )
            vals = std_df["credibility"].dropna().to_numpy(dtype=float)
            method_values[method] = vals
            all_values.extend(vals.tolist())

        if not all_values:
            ax.text(
                0.5,
                0.5,
                "No stability data",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            continue

        # Use common bin edges for all four methods. This makes the overlaid
        # histograms directly comparable.
        combined = np.asarray(all_values, dtype=float)
        bin_edges = np.histogram_bin_edges(combined, bins=35)

        for method in METHOD_ORDER:
            vals = method_values[method]
            if len(vals) == 0:
                continue
            ax.hist(
                vals,
                bins=bin_edges,
                alpha=0.38,
                color=METHOD_COLORS[method],
                label=method,
                edgecolor="none",
            )

        ax.set_xlabel("Std of Credibility across Repeats (per Test Sample)")
        ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle="--", alpha=0.25)

        # Reserve extra vertical space above the histogram for the four
        # mean ± 1 SD lines. Each method occupies a separate row.
        hist_ymax = ax.get_ylim()[1]
        ax.set_ylim(0, hist_ymax * 1.34)

        row_y = {
            "ICP": hist_ymax * 1.27,
            "BICP": hist_ymax * 1.20,
            "RICP": hist_ymax * 1.13,
            "OOB-ICP": hist_ymax * 1.06,
        }
        cap_height = hist_ymax * 0.018

        for method in METHOD_ORDER:
            vals = method_values[method]
            if len(vals) == 0:
                continue
            mean_value = float(np.mean(vals))
            std_value = float(np.std(vals, ddof=0))
            y_value = row_y[method]

            add_mean_std_line(
                ax=ax,
                mean_value=mean_value,
                std_value=std_value,
                y_value=y_value,
                method=method,
                cap_height=cap_height,
            )

        # The legend identifies the method colors; the upper annotations
        # therefore need no additional method labels on the left.
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.995, 0.78),
            frameon=True,
            ncol=2,
        )
        ax.text(
            0.5,
            -0.20,
            panel_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13,
        )

    fig.tight_layout(w_pad=3.4)
    fig.subplots_adjust(top=0.92, bottom=0.18)
    save_figure(fig, output_stem)


# =========================
# Main
# =========================
def main():
    print("Loading results from four experiment folders...")
    df_metrics = load_all_metrics()
    df_outputs = load_all_outputs()
    df_grouped = load_all_grouped()

    df_metrics.to_csv(
        os.path.join(TABLE_DIR, "merged_per_repeat_metrics_4datasets.csv"),
        index=False,
    )
    df_outputs.to_csv(
        os.path.join(TABLE_DIR, "merged_all_test_outputs_4datasets.csv"),
        index=False,
    )
    df_grouped.to_csv(
        os.path.join(TABLE_DIR, "merged_grouped_metrics_4datasets.csv"),
        index=False,
    )

    export_metric_tables(df_metrics)
    export_credibility_tables(df_outputs)
    export_stability_table(df_outputs)

    plot_metric_boxplots(
        df_metrics,
        metric="accuracy",
        ylabel="Accuracy",
        output_stem="fig2_accuracy_boxplot",
    )
    plot_metric_boxplots(
        df_metrics,
        metric="f1",
        ylabel="F1",
        output_stem="fig3_f1_boxplot",
    )
    plot_correct_wrong(
        df_outputs,
        output_stem="fig4_credibility_correct_wrong",
    )
    plot_confusion_groups(
        df_outputs,
        output_stem="fig5_confusion_groups",
    )
    plot_bins(
        df_grouped,
        metric="accuracy",
        ylabel="Accuracy",
        output_stem="fig6_accuracy_bins",
    )
    plot_stability_hist(
        df_outputs,
        output_stem="fig7_stability",
    )
    plot_bins(
        df_grouped,
        metric="f1",
        ylabel="F1",
        output_stem="figS1_f1_bins",
    )

    print("\nDone.")
    print("Figures:", FIG_DIR)
    print("Tables:", TABLE_DIR)


if __name__ == "__main__":
    main()