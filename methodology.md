# Methodology

## Research question

Which demographic and employment attributes are most strongly associated with an individual earning more than $50,000/year, and how well can standard supervised classifiers predict income class from these attributes alone?

## Data source

- **Dataset:** Adult / Census Income (UCI Machine Learning Repository; Kohavi, 1996), accessed via the Kaggle mirror [`uciml/adult-census-income`](https://www.kaggle.com/datasets/uciml/adult-census-income) as `adult.csv`.
- **Size:** 32,561 records, 14 predictor attributes (6 numeric, 8 categorical/binary) plus the target label `income` (`<=50K` / `>50K`).
- **Target balance:** ~75.9% `<=50K`, ~24.1% `>50K` — a moderately imbalanced binary classification problem.

## Data cleaning

1. **Missing-value encoding.** Missing values in this dataset are recorded as the literal string `"?"` rather than a null value. These are converted to `NaN` before any further processing.
2. **Row removal.** Rows containing any `NaN` are dropped rather than imputed, following common practice for this dataset in prior work (e.g., Chakrabarty & Biswas, 2018). This is a deliberate simplicity/robustness trade-off: it avoids introducing imputation bias into a demographic dataset already known to be bias-sensitive, at the cost of discarding ~7.4% of records (32,561 → 30,162). Missingness is concentrated in three columns: `workclass`, `occupation`, and `native.country`.
3. **Target encoding.** The text label `income` is converted to a binary integer column `income_binary` (1 = `>50K`, 0 = `<=50K`) for modeling.

## Feature selection and preprocessing

| Step | Detail |
|---|---|
| Dropped columns | `fnlwgt` (a census sampling weight with no demographic meaning for this task); `education` (a categorical string fully redundant with the already-ordinal `education.num`) |
| Numeric features | `age`, `education.num`, `capital.gain`, `capital.loss`, `hours.per.week` — standardized to zero mean / unit variance (`StandardScaler`) |
| Categorical features | `workclass`, `marital.status`, `occupation`, `relationship`, `race`, `sex`, `native.country` — one-hot encoded, with unknown categories at inference time handled via `handle_unknown="ignore"` |
| Implementation | `scikit-learn` `ColumnTransformer` composed with each model inside a single `Pipeline`, so all preprocessing is fit only on the training split (no leakage from the test set) |

## Train/test split

An 80/20 train-test split is used, **stratified on `income_binary`** to preserve the ~76/24 class balance in both splits. A fixed `random_state=42` is used throughout for reproducibility.

## Models

Two classifiers are trained and compared, chosen to contrast a linear baseline against a non-linear ensemble:

1. **Logistic Regression** — L2-regularized, `max_iter=1000`, default regularization strength. Serves as an interpretable linear baseline.
2. **Random Forest** — 200 trees (`n_estimators=200`), `max_depth=12` to limit overfitting, `random_state=42`. Chosen to capture non-linear interactions and to provide feature-importance scores.

Both models are trained on the identical preprocessed feature set so that any performance difference reflects model capacity rather than differing inputs.

## Evaluation

Models are scored on the held-out 20% test set using:

- **Accuracy** — overall correct-classification rate
- **Precision** and **Recall** (positive class = `>50K`) — chosen alongside accuracy because of class imbalance, where accuracy alone can be misleading
- **F1-score** — harmonic mean of precision and recall
- **ROC AUC** — threshold-independent measure of ranking quality, computed from predicted class-1 probabilities

**Feature importance** for the Random Forest model is extracted via its Gini-based `feature_importances_` attribute (mean decrease in impurity across all trees) to identify which encoded features (numeric and one-hot categorical) contribute most to predictions.

## Exploratory analysis

Alongside modeling, three descriptive charts are produced to characterize the relationship between individual attributes and income before any model is fit:

1. Share of individuals earning `>50K` by education level (bar chart)
2. Age distribution by income class (box plot)
3. Weekly hours worked by income class (violin plot)

These are generated with `matplotlib`/`seaborn` and are purely descriptive — they inform interpretation of the modeling results but are not themselves part of the predictive pipeline.

## Tools and environment

- Python 3, `pandas`, `numpy` for data handling
- `scikit-learn` for preprocessing, modeling, and evaluation
- `matplotlib` and `seaborn` for visualization
- Analysis is version-controlled as a Jupyter notebook (`notebook/income_analysis.ipynb`) for reproducibility

## Limitations of the approach (see paper's Discussion for full treatment)

- A single train/test split is used rather than cross-validation, so reported metrics carry some sampling variance.
- Dropping missing rows (rather than imputing) reduces the effective sample size by ~7.4%.
- The $50,000 income threshold is a fixed 1994-nominal figure with no inflation or cost-of-living adjustment.
- Several strong predictors (e.g., relationship status, sex) are known from prior fairness literature to encode historical social structure rather than a causal driver of earning capacity.
