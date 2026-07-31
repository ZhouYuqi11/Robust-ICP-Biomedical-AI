# Robust ICP for Biomedical AI

This repository contains the experimental, data-generation, and plotting code for the manuscript:

**Robust Uncertainty Quantification in Biomedical AI Under Data Scarcity Using Conformal Prediction and Statistical Randomization**

## Overview

This study investigates robust uncertainty quantification for biomedical artificial intelligence under limited-data conditions using inductive conformal prediction (ICP) and statistical randomization.

Four conformal prediction methods are evaluated:

* **ICP**: Standard Inductive Conformal Prediction
* **BICP**: Bootstrapped Inductive Conformal Prediction
* **RICP**: Randomized Inductive Conformal Prediction
* **OOB-ICP**: Out-of-Bag Inductive Conformal Prediction

The methods are evaluated on one simulated dataset and three biomedical classification tasks:

1. Statistical simulation
2. Pediatric pneumonia diagnosis
3. Chinese herbal medicine classification
4. Breast cancer subtyping

The experiments evaluate predictive performance, credibility-based uncertainty quantification, and uncertainty stability across repeated experiments.

## Repository Structure

```text
Robust-ICP-Biomedical-AI/
│
├── Simulation.py
├── generate_simulation_data.py
├── simulation_dataset.csv
├── Pneumonia diagnosis.py
├── Chinese_herbal_medicine.py
├── Breast_cancer_subtyping.py
├── plot_all_paper_results.py
└── README.md
```

## Files

* `Simulation.py`: Runs the binary Gaussian simulation experiments.
* `generate_simulation_data.py`: Reproduces the complete Gaussian simulation dataset and the fixed stratified development/test split used in `Simulation.py`.
* `simulation_dataset.csv`: Contains the 6,000 generated simulation observations, class labels, original sample indices, and fixed development/test split assignments.
* `Pneumonia diagnosis.py`: Runs the pediatric pneumonia diagnosis experiments using precomputed BiomedCLIP image embeddings.
* `Chinese_herbal_medicine.py`: Runs the 12-class Chinese herbal medicine classification experiments using electronic-nose features.
* `Breast_cancer_subtyping.py`: Runs the breast cancer molecular subtyping experiments using RNA-sequencing gene-expression features.
* `plot_all_paper_results.py`: Generates the main and supplementary figures and performs the statistical analyses reported in the manuscript.

## Methods

All experiments compare four ICP-based uncertainty quantification methods.

### ICP

Standard ICP uses a single proper-training/calibration partition. A classifier is fitted using the proper-training subset, and nonconformity scores are calculated using the calibration subset.

### BICP

BICP repeatedly performs class-stratified bootstrap resampling of the proper-training subset while keeping the calibration subset fixed. The conformal P-values obtained from the bootstrap iterations are averaged to produce the final P-values.

### RICP

RICP repeatedly repartitions the available training pool into proper-training and calibration subsets. A new classifier is fitted for each randomized partition, and the resulting conformal P-values are averaged.

### OOB-ICP

OOB-ICP repeatedly performs class-stratified bootstrap resampling of the available training pool. The complete bootstrap in-bag sample, including repeated observations, is used for classifier fitting, while observations not selected in the bootstrap sample are used for calibration. The resulting conformal P-values are averaged across bootstrap iterations.

## Experimental Settings

All experiments use logistic regression as the base classifier and are evaluated over 10 outer experimental repetitions.

For BICP, RICP, and OOB-ICP, 100 internal bootstrap or repartitioning iterations are performed within each outer repetition.

The final predicted class is determined by the candidate label with the largest method-specific conformal P-value.

### Simulation

The complete simulation dataset is included in this repository as `simulation_dataset.csv`.

The dataset contains 6,000 observations generated from two overlapping bivariate Gaussian distributions:

* Class 0 mean: `(0.5, 0.5)`
* Class 1 mean: `(-0.5, -0.5)`
* Shared covariance matrix: `[[1.5, 0.3], [0.3, 1.5]]`
* Number of observations per class: 3,000
* NumPy random generator: `np.random.default_rng(42)`

A fixed stratified test set containing 20% of the observations is constructed using `train_test_split` with `random_state=42`. The remaining 80% forms the available development pool.

The included `generate_simulation_data.py` script reproduces both the complete dataset and the fixed development/test split used in `Simulation.py`.

For each outer repetition, stratified training subsets of the following sizes are sampled from the fixed development pool:

* 100
* 200
* 400
* 800

Each sampled training subset is subsequently divided into proper-training and calibration subsets at an 80%/20% ratio.

### Pediatric Pneumonia Diagnosis

The pneumonia experiment uses PneumoniaMNIST, a pediatric chest X-ray dataset included in MedMNIST v2.

The released training, validation, and test partitions are retained. The released training and validation partitions are merged to form the available training pool, while the released test partition is used as a fixed test set throughout the experiments.

Precomputed BiomedCLIP image embeddings are used as input features.

Stratified training subsets of the following sizes are evaluated:

* 100
* 200
* 400
* 800

Each sampled training subset is divided into proper-training and calibration subsets at an 80%/20% ratio.

### Chinese Herbal Medicine Classification

The Chinese herbal medicine dataset contains 600 electronic-nose observations from 12 classes, with 50 observations per class.

For each outer repetition, the following observations are randomly selected from each class:

* 30 proper-training observations
* 10 calibration observations
* 10 test observations

Class-conditional conformal calibration is used for this experiment.

### Breast Cancer Subtyping

The breast cancer experiment uses TCGA RNA-sequencing gene-expression features and molecular subtype labels.

The analysis retains the three most frequent molecular subtypes:

* Luminal A
* Luminal B
* Basal-like

Ten repeated stratified 75%/25% train-test splits are performed. Within each outer training partition, approximately 67% of the observations are used for proper training and 33% are used for calibration.

The classification pipeline includes:

* Median imputation
* Removal of zero-variance features
* Standardization
* ANOVA-based selection of up to 50 features
* Class-weighted L1-regularized logistic regression using the SAGA solver

## Nonconformity Score

The class-probability-based nonconformity score used in the experiments is:

```text
alpha(x, y) = 0.5 - [p_y(x) - max_{y' != y} p_{y'}(x)] / 2
```

where `p_y(x)` is the predicted probability assigned to candidate class `y`.

For the simulation, pneumonia diagnosis, and breast cancer subtyping experiments, pooled calibration is used.

For the Chinese herbal medicine experiment, class-conditional calibration is used.

## Evaluation Metrics

The experiments evaluate the following outcomes:

* Classification accuracy
* Macro-averaged F1-score
* Credibility distributions for correct and incorrect predictions
* Credibility distributions for TP, TN, FP, and FN predictions in binary classification tasks
* Accuracy across five equal-frequency credibility bins
* Macro-F1 across five equal-frequency credibility bins
* Across-repeat per-sample credibility standard deviation for fixed-test experiments

Credibility is defined as the largest conformal P-value assigned to a test observation.

For the fixed-test simulation and pneumonia experiments, uncertainty stability is evaluated using the standard deviation of each test sample's credibility across the 10 outer repetitions.

## Statistical Analysis

Paired t-tests across matched outer repetitions are used to compare classification accuracy and macro-F1 between each robust ICP variant and standard ICP.

For the correct-versus-incorrect credibility analysis, the mean credibility of each prediction group is first calculated within each outer repetition. Welch's two-sample t-test is then applied to the resulting repeat-level mean credibility values.

TP, TN, FP, and FN credibility distributions are examined descriptively for the two binary classification tasks.

## Requirements

The experiments were implemented in Python.

The main required Python packages are:

* NumPy
* pandas
* SciPy
* scikit-learn
* Matplotlib

The packages can be installed using:

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

## Data Availability

### Simulation Dataset

The simulation dataset generated in this study is included in this repository as `simulation_dataset.csv`.

The complete data-generation procedure is provided in `generate_simulation_data.py`. Running the script with its default settings reproduces the dataset and the fixed stratified development/test split used in `Simulation.py`.

### Pediatric Pneumonia Diagnosis

The pediatric pneumonia experiment is based on the publicly available PneumoniaMNIST dataset from MedMNIST v2:

* Dataset: https://zenodo.org/records/6496656
* Project website: https://medmnist.com/
* Publication: https://doi.org/10.1038/s41597-022-01721-8

Reference:

Yang, J.; Shi, R.; Wei, D.; Liu, Z.; Zhao, L.; Ke, B.; Pfister, H.; Ni, B. MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification. *Scientific Data* **2023**, *10*, 41.

The released training, validation, and test partitions are used in this study. The `Pneumonia diagnosis.py` script operates on precomputed BiomedCLIP image embeddings rather than directly on the original images.

Users should place the required feature and label files in the same directory as the experiment script, using the filenames specified in the script.

### Chinese Herbal Medicine Classification

The electronic-nose dataset used for Chinese herbal medicine classification is available from the original research repository:

* Dataset repository: https://github.com/xzhan96-stf/Herbal-medicine-origin-e-nose
* Publication: https://doi.org/10.3390/s18092936

Reference:

Zhan, X.; Guan, X.; Wu, R.; Wang, Z.; Wang, Y.; Li, G. Discrimination between Alternative Herbal Medicines from Different Categories with the Electronic Nose. *Sensors* **2018**, *18*, 2936.

The required processed electronic-nose dataset should be placed in the same directory as `Chinese_herbal_medicine.py`, using the filename specified in the script.

### Breast Cancer Subtyping

The breast cancer RNA-sequencing data are publicly available through the NCI Genomic Data Commons TCGA-BRCA project:

* Dataset portal: https://portal.gdc.cancer.gov/projects/TCGA-BRCA
* Publication: https://doi.org/10.1038/nature11412

Reference:

The Cancer Genome Atlas Network. Comprehensive Molecular Portraits of Human Breast Tumours. *Nature* **2012**, *490*, 61-70.

The `Breast_cancer_subtyping.py` script uses preprocessed RNA-sequencing feature and molecular-subtype label matrices. These files should be placed in the same directory as the experiment script, using the filenames specified in the script.

The original biomedical datasets are not redistributed in this repository. Users are responsible for complying with the access requirements and licenses specified by the original data providers.

## Running the Experiments

### 1. Reproduce the Simulation Dataset

The generated dataset is already included as `simulation_dataset.csv`.

To reproduce it from the specified random seed and distribution parameters, run:

```bash
python generate_simulation_data.py
```

The generated file should match the included `simulation_dataset.csv`.

### 2. Run the Simulation Experiment

```bash
python Simulation.py
```

### 3. Run the Pneumonia Experiment

Place the required precomputed BiomedCLIP feature and label files in the directory expected by the script, and then run:

```bash
python "Pneumonia diagnosis.py"
```

### 4. Run the Chinese Herbal Medicine Experiment

Place the processed electronic-nose dataset in the directory expected by the script, and then run:

```bash
python Chinese_herbal_medicine.py
```

### 5. Run the Breast Cancer Experiment

Place the preprocessed RNA-sequencing feature and subtype-label files in the directory expected by the script, and then run:

```bash
python Breast_cancer_subtyping.py
```

### 6. Generate Figures and Statistical Results

After the required experimental result files have been generated and placed in the locations expected by the plotting script, run:

```bash
python plot_all_paper_results.py
```

The plotting script generates the main and supplementary figures and performs the statistical analyses reported in the manuscript.

## Reproducibility Notes

The experiments use fixed random seeds specified in the scripts to support reproducibility.

All four conformal prediction methods use the same sampled training pool and test set within each outer repetition. Method-specific random seeds govern internal partitioning, bootstrap resampling, and randomized repartitioning.

The simulation and pneumonia diagnosis experiments use fixed test sets, allowing per-sample uncertainty stability to be evaluated across repeated experiments.

The complete synthetic dataset and its fixed split are included in this repository. The three biomedical experiments require datasets or processed feature matrices obtained from the publicly available sources listed above.

## Citation

If you use this code in your research, please cite the corresponding manuscript:

```text
Robust Uncertainty Quantification in Biomedical AI Under Data Scarcity Using Conformal Prediction and Statistical Randomization
```

Full citation information will be added after publication.

## License

The source code is provided for academic and research purposes.

