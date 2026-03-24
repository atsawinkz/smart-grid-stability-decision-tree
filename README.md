# Decision Tree Classifier for Smart Grid Stability Analysis

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Special Requirements](#special-requirements)

---

## Project Overview

This project explores the application of **Decision Tree Classification** to predict smart grid stability status. The analysis includes:

- **Exploratory Data Analysis (EDA)** on the augmented smart grid dataset
- **Feature/Label Separation** — all features used; last column as the target label
- **Label Encoding** — encoding the categorical target variable
- **Train/Test Split** — 70% training, 30% testing (`random_state=42`)
- **Hyperparameter Tuning** — using `GridSearchCV` with 5-fold cross-validation across parameters: `max_depth`, `criterion`, `max_features`, and `ccp_alpha`
- **Model Evaluation** — accuracy, precision, recall, F1-score, and confusion matrix
- **Tree Visualization** — exported as `.dot` and `.png` using `sklearn.tree.export_graphviz`

---

## Dataset

| Property | Value |
|---|---|
| **Dataset Name** | Smart Grid Stability Augmented |
| **File** | `data/smart_grid_stability_augmented.csv` |
| **Instances** | 60,000 |
| **Features** | 14 numerical attributes |
| **Target** | `stabf` — grid stability status (`stable`, `unstable`) |

> **Attribution**: Arzamasov, Vadim, Böhm, Klemens, & Jochem, Patrick. (2018). *Towards Concise Models of Grid Stability*. IEEE PES Innovative Smart Grid Technologies Conference Europe (ISGT-Europe).

---

## Project Structure

```text
LAB_03/
│
├── data/
│   └── smart_grid_stability_augmented.csv    # Raw training dataset
│
├── notebooks/
│   └── Decision_Tree_smart_grid_stability_augmented.ipynb  # Main experiment notebook
│
├── outputs/
│   ├── smart_grid_stability_augmented1.dot   # Decision tree source (Graphviz)
│   └── smart_grid_stability_augmented1.png   # Decision tree visualization
│
├── .gitignore
├── README.md                                 # Project documentation
└── requirements.txt                          # Python dependencies
```

---

## Installation

### Prerequisites
- Python >= 3.12
- pip or conda package manager

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd LAB_03
```

### 2. Create a Virtual Environment *(Recommended)*

```bash
# Using venv
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Run the Jupyter Notebook

```bash
jupyter notebook notebooks/Decision_Tree_smart_grid_stability_augmented.ipynb
```

Execute all cells top-to-bottom to reproduce the full analysis pipeline.

---

## Results

The model achieved near-perfect accuracy after GridSearchCV tuning:

| Metric | Value |
|---|---|
| **Best CV Score** | ~99.99% |
| **Criterion** | `gini` / `entropy` |
| **Depth** | ≥ 3 |

> Refer to the notebook for full classification reports and visualizations.

---

## Special Requirements

| Requirement | Notes |
|---|---|
| **No GPU required** | All computations are CPU-based |
| **Memory** | ~1 GB RAM recommended for 60,000 samples with GridSearchCV |
| **Python Version** | Tested on Python 3.12 |
| **sklearn version** | >= 1.3.0 (the `max_features='auto'` parameter is deprecated and removed; use `'sqrt'` or `'log2'`) |

---