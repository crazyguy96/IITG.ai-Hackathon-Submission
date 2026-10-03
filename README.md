# IITG.ai Hackathon Submission

## Overview

This repository contains our submission for the **IITG.ai Hackathon** based on the **Atlantis Citizens** dataset.

The project consists of two major tasks:

- **Task 1:** Exploratory Data Analysis and statistical investigation
- **Task 2:** Machine Learning-based occupation prediction

The overall workflow goes from understanding the dataset and discovering patterns to building a machine learning model for classification.

---

## Repository Structure

```text
IITG.ai-Hackathon-Submission/
│
├── Task 1.ipynb
├── Task 2.ipynb
└── README.md
```

### Files

- `Task 1.ipynb` — Exploratory Data Analysis
- `Task 2.ipynb` — Machine Learning and Occupation Prediction
- `README.md` — Project documentation

---

# Dataset

The project uses the **Atlantis Citizens** dataset containing demographic, socioeconomic, geographic, and biometric information about citizens.

### Features

| Feature | Description |
|---|---|
| `Citizen_ID` | Unique citizen identifier |
| `Diet_Type` | Citizen's diet category |
| `District_Name` | Citizen's residential district |
| `Occupation` | Citizen's occupation |
| `Wealth_Index` | Wealth-related numerical index |
| `House_Size_sq_ft` | House size in square feet |
| `Life_Expectancy` | Life expectancy |
| `Vehicle_Owned` | Type of vehicle owned |
| `Work_District` | District where the citizen works |
| `Bio_Hash` | Biometric hash associated with the citizen |

---

# Task 1 — Exploratory Data Analysis

The first task focuses on understanding the dataset and identifying relationships between different socioeconomic and demographic variables.

## 1. Commuting Analysis

A `Commute_Out` feature is created by comparing the residential district with the work district.

```text
1 → Citizen works outside their residential district
0 → Citizen works within their residential district
```

The analysis compares commuting patterns across districts and occupations.

### Outward Commuting by District

| District | Outward Commuting |
|---|---:|
| The Golden Reef | 72.87% |
| Coral Slums | 66.47% |
| Deep Trench | 63.27% |
| Mariana Plaza | 63.25% |

This helps identify districts where residents are more likely to travel outside their residential district for work.

---

## 2. Wealth Distribution

The average `Wealth_Index` is calculated for each district.

This allows us to compare wealth levels between districts and understand differences in the socioeconomic distribution of the population.

---

## 3. House Size vs Life Expectancy

The relationship between `House_Size_sq_ft` and `Life_Expectancy` is investigated using Pearson correlation.

The analysis obtains a correlation of approximately:

**0.7978**

This indicates a strong positive association between house size and life expectancy within the dataset.

> Correlation indicates association and does not establish causation.

---

## 4. Wealth vs Diet Type

Average `Wealth_Index` is calculated for each `Diet_Type`.

The results are visualized to compare the wealth distribution associated with different diet categories.

---

## 5. Biometric Hash Analysis

The uniqueness of `Bio_Hash` values is investigated.

The analysis shows that the biometric hashes function as unique identifiers. Since they do not represent a meaningful socioeconomic or demographic characteristic, they are excluded from the machine learning model.

---

# Task 2 — Machine Learning

The second task formulates **occupation prediction as a supervised classification problem**.

## Objective

Predict a citizen's occupation using demographic, socioeconomic, and geographic information.

### Target Variable

```text
Occupation
```

### Input Features

```text
Diet_Type
District_Name
Wealth_Index
House_Size_sq_ft
Life_Expectancy
Vehicle_Owned
Work_District
```

`Citizen_ID` and `Bio_Hash` are excluded because they are identifier-like features.

---

# Data Preprocessing

## Missing Value Handling

Numerical features are handled using mean imputation:

- `Wealth_Index`
- `House_Size_sq_ft`
- `Life_Expectancy`

Categorical features are handled using the most frequent category.

---

## Categorical Encoding

Categorical features are transformed using One-Hot Encoding:

```python
OneHotEncoder(handle_unknown="ignore")
```

The target variable `Occupation` is encoded using:

```python
LabelEncoder
```

---

# Random Forest Classifier

A **Random Forest Classifier** is used for occupation prediction.

The initial model configuration is:

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=4,
    min_samples_split=10,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42
)
```

The preprocessing steps and classifier are combined into a Scikit-learn pipeline.

---

# Hyperparameter Optimization

Hyperparameter optimization is performed using `RandomizedSearchCV`.

### Configuration

- **Parameter combinations:** 12
- **Cross-validation:** 3-fold
- **Evaluation metric:** F1 Macro

The following parameters are explored:

```text
n_estimators
max_depth
min_samples_leaf
max_features
```

The selected configuration in the notebook is approximately:

```text
n_estimators     = 200
max_depth        = 10
min_samples_leaf = 2
max_features     = 0.7
```

---

# Project Workflow

```text
                 Atlantis Citizens Dataset
                            │
                            ▼
                  ┌──────────────────┐
                  │ Data Exploration │
                  └────────┬─────────┘
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
       Task 1: EDA               Task 2: Machine Learning
             │                           │
             ▼                           ▼
    Statistical Analysis         Data Preprocessing
             │                           │
             ▼                           ▼
    Feature Relationships       Missing Value Handling
                                         │
                                         ▼
                                  One-Hot Encoding
                                         │
                                         ▼
                                  Random Forest
                                         │
                                         ▼
                                Hyperparameter Tuning
                                         │
                                         ▼
                              Occupation Prediction
```

---

# Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Matplotlib | Data visualization |
| Scikit-learn | Machine learning |
| Jupyter Notebook | Development and analysis |

---

# Key Insights

### Exploratory Analysis

- Commuting behavior varies significantly across districts.
- Districts show differences in average wealth.
- House size and life expectancy show a strong positive association in the dataset.
- Wealth distributions vary across different diet categories.
- Biometric hashes behave as identifiers and are excluded from predictive modeling.

### Machine Learning

- Both numerical and categorical variables are incorporated into a unified ML pipeline.
- Missing values are handled before model training.
- One-hot encoding enables categorical variables to be used by the Random Forest model.
- Hyperparameter optimization is performed using macro F1 cross-validation.
- The resulting model is designed to predict occupation from citizen-level attributes.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/crazyguy96/IITG.ai-Hackathon-Submission.git
cd IITG.ai-Hackathon-Submission
```

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
Task 1.ipynb
Task 2.ipynb
```

and run the notebooks sequentially.

---

# Limitations

- The analysis is based on the provided Atlantis Citizens dataset.
- Correlation does not imply causation.
- Model performance may vary on unseen or differently distributed data.
- Identifier-like features are intentionally excluded from the machine learning model.
- The current implementation is notebook-based and is not deployed as a production application.

---

# Hackathon Submission

**Hackathon:** IITG.ai Hackathon

**Repository:**  
https://github.com/crazyguy96/IITG.ai-Hackathon-Submission

---

## Team

Developed as part of the **IITG.ai Hackathon**.

---

## License

This project was developed as a hackathon submission and is intended primarily for educational and evaluation purposes.
