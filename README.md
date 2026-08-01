# Robust ICP for Biomedical AI

[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21731235.svg)](https://doi.org/10.5281/zenodo.21731235)

This repository contains the experimental and plotting code for the manuscript:

**Robust Uncertainty Quantification in Biomedical AI Under Data Scarcity Using Conformal Prediction and Statistical Randomization**

## Overview

This study investigates robust uncertainty quantification for biomedical artificial intelligence under limited-data conditions using inductive conformal prediction (ICP) and statistical randomization.

Four conformal prediction methods are evaluated:

- **ICP**: Standard Inductive Conformal Prediction
- **BICP**: Bootstrapped Inductive Conformal Prediction
- **RICP**: Randomized Inductive Conformal Prediction
- **OOB-ICP**: Out-of-Bag Inductive Conformal Prediction

The methods are evaluated on one simulated dataset and three biomedical classification tasks:

1. Statistical simulation
2. Pediatric pneumonia diagnosis
3. Chinese herbal medicine classification
4. Breast cancer subtyping

The experiments evaluate predictive performance, credibility-based uncertainty quantification, and across-repeat uncertainty stability.

## Repository Structure

```text
Robust-ICP-Biomedical-AI/
|
|-- Simulation.py
|-- Pneumonia diagnosis.py
|-- Chinese_herbal_medicine.py
|-- Breast_cancer_subtyping.py
|-- plot_all_paper_results.py
`-- README.md
```

### Files

- `Simulation.py`: Runs the binary Gaussian simulation experiments.
- `Pneumonia diagnosis.py`: Runs the pediatric pneumonia diagnosis experiments using precomputed BiomedCLIP image embeddings.
- `Chinese_herbal_medicine.py`: Runs the 12-class Chinese herbal medicine classification experiments.
- `Breast_cancer_subtyping.py`: Runs the breast cancer molecular subtyping experiments using gene-expression features.
- `plot_all_paper_results.py`: Generates the main and supplementary figures and performs the statistical analyses reported in the manuscript.

## Methods

All experiments compare four ICP-based uncertainty quantification methods.

### ICP

Standard ICP uses a single proper-training/calibration partition. A classifier is fitted using the proper-training subset, and nonconformity scores are calculated using the calibration subset.

### BICP

BICP repeatedly performs class-stratified bootstrap resampling of the proper-training subset while keeping the calibration subset fixed. The conformal P-values obtained from the bootstrap iterations are averaged to produce the final P-values.

### RICP

RICP repeatedly repartitions the available training data into proper-training and calibration subsets. A new classifier is fitted for each randomized partition, and the resulting conformal P-values are averaged.

### OOB-ICP

OOB-ICP repeatedly performs class-stratified bootstrap resampling of the available training data. The complete bootstrap in-bag sample, including repeated observations, is used for classifier fitting, while observations not selected in the bootstrap sample are used for calibration. The resulting conformal P-values are averaged across bootstrap iterations.

## Experimental Settings

All experiments use logistic regression as the base classifier and are repeated over 10 outer experimental repetitions.

For BICP, RICP, and OOB-ICP, 100 internal resampling or repartitioning iterations are performed within each outer repetition.

The final predicted class is determined by the candidate label with the largest method-specific conformal P-value.

### Simulation

The simulation dataset contains 6,000 observations generated from two overlapping Gaussian distributions.

A fixed stratified test set containing 20% of the observations is constructed. The remaining 80% forms the available training pool.

For each outer repetition, training subsets of the following sizes are sampled:

- 100
- 200
- 400
- 800

Each sampled training subset is subsequently divided into proper-training and calibration subsets at an 80%/20% ratio.

### Pediatric Pneumonia Diagnosis

The pneumonia experiment uses the official training, validation, and test partitions of PneumoniaMNIST in MedMNIST v2. Precomputed BiomedCLIP image embeddings are used as input features.

The released training and validation partitions are merged to form the available training pool, while the released test partition is retained as a fixed test set throughout the experiments.

Training subsets of the following sizes are evaluated:

- 100
- 200
- 400
- 800

Each sampled training subset is divided into proper-training and calibration subsets at an 80%/20% ratio.

### Chinese Herbal Medicine Classification

The Chinese herbal medicine dataset contains 600 observations from 12 classes.

For each outer repetition, the following samples are randomly selected from each class:

- 30 proper-training samples
- 10 calibration samples
- 10 test samples

Class-conditional calibration is used for this experiment.

### Breast Cancer Subtyping

The breast cancer experiment uses gene-expression features for three molecular subtypes.

Ten repeated stratified 75%/25% train-test splits are performed. Within each outer training partition, 67% of the observations are used for proper training and 33% for calibration.

The classification pipeline includes:

- Median imputation
- Removal of zero-variance features
- Standardization
- ANOVA-based selection of up to 50 features
- Class-weighted L1-regularized logistic regression using the SAGA solver

## Nonconformity Score

The class probability-based nonconformity score used in the experiments is:

```text
alpha(x, y) = 0.5 - [p_y(x) - max_{y' != y} p_{y'}(x)] / 2
```

where `p_y(x)` is the predicted probability of candidate class `y`.

For the simulation, pneumonia diagnosis, and breast cancer subtyping experiments, pooled calibration is used.

For the Chinese herbal medicine experiment, class-conditional calibration is used.

## Evaluation Metrics

The experiments evaluate the following outcomes:

- Classification accuracy
- Macro-averaged F1-score
- Credibility distributions for correct and incorrect predictions
- Credibility distributions for TP, TN, FP, and FN predictions in binary classification tasks
- Accuracy across credibility bins
- Macro-F1 across credibility bins
- Across-repeat per-sample credibility standard deviation for fixed-test experiments

Credibility is defined as the largest conformal P-value assigned to a test observation.

For the fixed-test simulation and pneumonia experiments, uncertainty stability is evaluated using the standard deviation of each test sample's credibility across the 10 outer repetitions.

## Statistical Analysis

Paired t-tests across matched outer repetitions are used to compare classification accuracy and macro-F1 between each robust ICP variant and standard ICP.

For the correct-versus-incorrect credibility analysis, the mean credibility of each prediction group is first calculated within each outer repetition. Welch's two-sample t-test is then applied to the resulting repeat-level mean credibility values.

TP, TN, FP, and FN credibility distributions are examined descriptively for the two binary classification tasks.

## Requirements

The experiments were implemented in Python. The main required packages are:

- NumPy
- pandas
- SciPy
- scikit-learn
- Matplotlib

## Data Availability

The simulated and processed datasets used in this study are openly available from Zenodo:

**Dataset DOI:** [10.5281/zenodo.21731235](https://doi.org/10.5281/zenodo.21731235)

The Zenodo record contains the following four archives:

| Archive | Contents | Corresponding script |
| --- | --- | --- |
| `simulation_data.zip` | Generated binary Gaussian simulation dataset and data-generation code | `Simulation.py` |
| `pneumonia_data.zip` | Precomputed BiomedCLIP embeddings and labels derived from the official PneumoniaMNIST partitions | `Pneumonia diagnosis.py` |
| `chinese_herbal_medicine_data.zip` | Processed electronic-nose dataset containing 600 observations from 12 herbal medicine categories | `Chinese_herbal_medicine.py` |
| `breast_cancer_data.zip` | Processed TCGA-BRCA gene-expression features and molecular subtype labels | `Breast_cancer_subtyping.py` |

### Original Data Sources

#### PneumoniaMNIST

The pneumonia feature embeddings were derived from the official PneumoniaMNIST partitions in MedMNIST v2:

- MedMNIST v2 dataset: [https://doi.org/10.5281/zenodo.6496656](https://doi.org/10.5281/zenodo.6496656)
- MedMNIST v2 reference: [https://doi.org/10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8)

#### Chinese Herbal Medicine

The Chinese herbal medicine electronic-nose dataset was originally collected and described in:

- Zhan, X. et al. *Discrimination between Alternative Herbal Medicines from Different Categories with the Electronic Nose*. Sensors, 2018, 18, 2936. [https://doi.org/10.3390/s18092936](https://doi.org/10.3390/s18092936)

#### Breast Cancer

The processed breast cancer feature and label matrices were derived from the TCGA Breast Invasive Carcinoma project:

- TCGA-BRCA project: [https://portal.gdc.cancer.gov/projects/TCGA-BRCA](https://portal.gdc.cancer.gov/projects/TCGA-BRCA)
- TCGA reference: The Cancer Genome Atlas Network. *Comprehensive molecular portraits of human breast tumours*. Nature, 2012. [https://doi.org/10.1038/nature11412](https://doi.org/10.1038/nature11412)

The deposited files contain no direct patient identifiers. The source datasets remain subject to the attribution requirements and data-use conditions specified by their original providers.

## Downloading and Preparing the Data

1. Download the four data archives from [Zenodo](https://doi.org/10.5281/zenodo.21731235).
2. Extract the required archives.
3. Copy the extracted files into the repository root, alongside the corresponding experiment scripts, unless a different path has been configured in the script.
4. Keep the original filenames because the experiment scripts expect specific filenames.

After preparing the data, the repository may contain additional data files such as the precomputed pneumonia embeddings, `original_dataset_stdlzed.csv`, `TCGA_X.npy`, and `TCGA_Y.npy`. These large data files are distributed through Zenodo rather than duplicated in the GitHub repository.

## Running the Experiments

Run each experiment script separately from the repository root:

```bash
python Simulation.py
```

```bash
python "Pneumonia diagnosis.py"
```

```bash
python Chinese_herbal_medicine.py
```

```bash
python Breast_cancer_subtyping.py
```

After all experimental results have been generated, run:

```bash
python plot_all_paper_results.py
```

The plotting script reads the generated experimental results and produces the figures and statistical analyses used in the manuscript.

## Reproducibility Notes

The experiments use fixed random seeds where specified in the scripts to support reproducibility.

All four methods use the same sampled training pool and test set within each outer repetition, while method-specific random seeds govern the internal partitioning and resampling procedures.

The simulation and pneumonia diagnosis experiments use fixed test sets, allowing per-sample uncertainty stability to be evaluated across repeated experiments.

## Citation

If you use the datasets, please cite the Zenodo record:

```text
Zhou, Y., & Zhan, X. (2026). Datasets for Robust Uncertainty Quantification
in Biomedical AI Under Data Scarcity [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.21731235
```

If you use the code or methods, please cite the corresponding manuscript:

```text
Zhou, Y., & Zhan, X. Robust Uncertainty Quantification in Biomedical AI
Under Data Scarcity Using Conformal Prediction and Statistical Randomization.
```

Full journal citation information will be added after publication.

## License

The deposited dataset is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. The source datasets remain subject to the terms specified by their original providers.

The source code is provided for academic and research purposes.
