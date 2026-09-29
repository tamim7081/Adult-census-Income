# Predicting Income Level from U.S. Census Data

## Abstract

This study examines which demographic and employment characteristics are associated with earning more than $50,000 per year, using the UCI Adult (Census Income) dataset drawn from the 1994 U.S. Census. After removing records with missing values, 30,162 individuals described by 14 attributes (age, education, occupation, hours worked, and others) were analyzed. Exploratory analysis shows that the share of high earners rises sharply with education level, from under 8% among those without a high-school diploma to roughly 75% among holders of a doctorate or professional degree, and that high earners tend to be older and work more hours per week. Two supervised classifiers, Logistic Regression and Random Forest, were trained to predict the binary income label using a standardized numeric and one-hot-encoded categorical feature set. Random Forest achieved the strongest performance (86.5% accuracy, ROC AUC 0.921), modestly outperforming Logistic Regression (85.3% accuracy, ROC AUC 0.913). Feature-importance analysis identified relationship status, capital gains, and age as the strongest predictors. The results confirm that education, hours worked, and marital/relationship status are the dominant correlates of income in this historical dataset, while also highlighting known limitations of the data, including its age and embedded demographic biases.

## Introduction

Predicting whether an individual's income exceeds a given threshold from demographic and employment data is a long-standing benchmark problem in machine learning, largely because of the enduring popularity of the Adult dataset. The dataset was originally extracted from the 1994 U.S. Census database by Barry Becker and was first used by Kohavi (1996) to evaluate a hybrid decision-tree/naive-Bayes classifier, after which it was donated to the UCI Machine Learning Repository. Because it combines a mix of continuous and categorical attributes with a moderately imbalanced binary target (roughly 25% of individuals earn above $50,000), it has since been used in dozens of studies as a standard testbed for classification algorithms.

The research question guiding this study is: **which demographic and employment attributes are most strongly associated with high income, and how well can standard classifiers predict income class from these attributes alone?** Beyond the original modeling question, the dataset has become equally well known in the algorithmic-fairness literature. Chakrabarty and Biswas (2018) applied a Random Forest classifier to the same data and used a tree-based feature-scoring method to rank predictors, finding that age, education, and hours worked were consistently influential. More recent work has shifted focus toward the dataset's embedded social biases: Besse et al. (2020) used the Adult dataset to illustrate how disparate-impact measures can quantify gender- and race-based prediction gaps in binary classifiers, and Girhepuje (2023) showed that ensemble models trained on this data produce systematically different income predictions for otherwise identical individuals when only the sex attribute is changed, with tree-based models showing the largest gender-based divergence. These findings motivate treating any accuracy result on this dataset with some caution, since a model that predicts income well may simultaneously be encoding historical inequities present in the 1994 labor market.

This study builds on that literature in two ways: first, by re-confirming, on a current mirror of the dataset, which attributes drive income prediction; second, by comparing a linear model (Logistic Regression) against a non-linear ensemble model (Random Forest) to see whether the more flexible model meaningfully improves predictive performance on this now three-decade-old data.

## Methodology

**Data source.** The dataset used is the Kaggle mirror of the UCI Adult / Census Income dataset (`adult.csv`), containing 32,561 records with 14 predictor attributes plus the binary income label (`<=50K` / `>50K`).

**Cleaning.** Missing values in this dataset are encoded as the literal string `"?"` rather than as null values. These were converted to `NaN` and dropped, following the common practice in prior work on this dataset (Chakrabarty & Biswas, 2018). This reduced the dataset from 32,561 to 30,162 complete records (a 7.4% reduction), concentrated in the `workclass`, `occupation`, and `native.country` fields.

**Feature preparation.** The sample-weight column `fnlwgt` and the redundant text column `education` (fully represented by the ordinal `education.num`) were dropped. The remaining features were split into:
- **Numeric:** age, education.num, capital.gain, capital.loss, hours.per.week — standardized with a `StandardScaler`.
- **Categorical:** workclass, marital.status, occupation, relationship, race, sex, native.country — transformed with one-hot encoding.

**Modeling.** The data were split 80/20 into training and test sets using stratified sampling on the income label to preserve class balance. Two classifiers were trained inside a single `scikit-learn` pipeline (preprocessing + model) to avoid data leakage:
1. **Logistic Regression** (L2-regularized, max 1,000 iterations) as a linear baseline.
2. **Random Forest** (200 trees, max depth 12) as a non-linear ensemble comparison.

**Evaluation.** Models were scored on the held-out test set using accuracy, precision, recall, F1-score, and ROC AUC. Feature importance for the Random Forest model was extracted from its Gini-based importance scores to identify the top predictors.

## Results

### Exploratory findings

Figure 1 shows a strong, monotonic relationship between educational attainment and the likelihood of earning more than $50,000 per year: the share of high earners rises from near 0% for those with only a preschool or early-elementary education to roughly 42% for Bachelor's degree holders and 75% for those with a doctorate or professional degree.

![Figure 1: Share earning >$50K by education level](https://github.com/tamim7081/Adult-census-Income/blob/main/fig1_education_income.png?raw=true)

Figure 2 shows that individuals earning above $50,000 tend to be noticeably older, with a higher median age and a narrower interquartile range than the lower-income group, consistent with income rising over the course of a career.

![Figure 2: Age distribution by income class](https://github.com/tamim7081/Adult-census-Income/blob/main/fig2_age_income.png?raw=true)

Figure 3 shows a similar pattern for hours worked: high earners are concentrated around and above the standard 40-hour work week, with a visibly fatter right tail, while lower earners show a wider, flatter distribution including a large share working fewer than 40 hours.

![Figure 3: Weekly hours worked by income class](https://github.com/tamim7081/Adult-census-Income/blob/main/fig3_hours_income.png?raw=true)

### Model performance

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.853 | 0.751 | 0.615 | 0.676 | 0.913 |
| Random Forest | 0.865 | 0.821 | 0.585 | 0.683 | 0.921 |

*Table 1. Test-set performance of the two classifiers (20% held-out test set, n ≈ 6,033).*

Random Forest achieved higher accuracy, precision, and ROC AUC than Logistic Regression, while Logistic Regression achieved slightly higher recall. Both models substantially outperform the majority-class baseline of 75.1% accuracy (always predicting `<=50K`), and both show a moderate gap between precision and recall on the minority (`>50K`) class, indicating that high earners are harder to identify correctly than low earners — an expected consequence of the roughly 3:1 class imbalance.

Figure 4 shows the ten most important features for the Random Forest model. Relationship status (particularly whether an individual is a husband or wife), capital gains, age, marital status, and education level are the strongest predictors, while race, native country, and workclass contribute comparatively little.

![Figure 4: Top 10 feature importances, Random Forest](https://github.com/tamim7081/Adult-census-Income/blob/main/fig4_feature_importance.png?raw=true)

## Discussion

The results are consistent with the exploratory patterns and with prior work on this dataset (Chakrabarty & Biswas, 2018): education, age, hours worked, and marital/relationship status are the dominant correlates of income in the 1994 Census sample, while capital gains — though held by a small share of individuals — is highly informative when present. The modest improvement of Random Forest over Logistic Regression (roughly 1.2 percentage points of accuracy, 0.008 in ROC AUC) suggests that the relationship between these features and income is largely additive and near-linear once the categorical variables are properly encoded; the added flexibility of a tree ensemble captures only a small amount of extra signal, mostly in the form of interactions such as those between relationship status and hours worked.

Several limitations should temper these conclusions. First, the data are three decades old and describe the 1994 U.S. labor market; the relationships identified — particularly around occupation, gender, and marital status — do not necessarily generalize to the present day. Second, the finding that relationship status (e.g., "Husband") is highly predictive should be interpreted with caution: as Besse et al. (2020) and Girhepuje (2023) both demonstrate, this and correlated attributes such as sex encode historical social structures rather than a causal link between marital status and earning capacity, and models trained on this data can reproduce or amplify gender-based prediction gaps. Third, the $50,000 threshold is a fixed nominal figure that does not account for inflation or regional cost of living, which likely inflates the apparent effect of variables like `native.country`. Finally, the 20% held-out test split used here, while standard practice, is a single split; a cross-validated estimate would give a more robust sense of variance in the reported metrics.

## Conclusion

Using the UCI Adult / Census Income dataset, this study found that educational attainment, age, hours worked, and relationship status are the strongest correlates of individual income exceeding $50,000 per year, and that a Random Forest classifier modestly outperforms Logistic Regression (86.5% vs. 85.3% accuracy, ROC AUC 0.921 vs. 0.913) on this prediction task. These results reaffirm findings from earlier work on the same dataset while underscoring, consistent with the fairness-focused literature this study reviewed, that strong predictive performance on demographic data does not imply that the underlying relationships are causal, current, or free of embedded social bias — a caveat that should inform any real-world application of models trained on this or similar census data.

## References

Besse, P., del Barrio, E., Gordaliza, P., Loubes, J.-M., & Risser, L. (2020). *A survey of bias in Machine Learning through the prism of Statistical Parity for the Adult Data Set*. arXiv:2003.14263. https://arxiv.org/abs/2003.14263

Chakrabarty, N., & Biswas, S. (2018). A statistical approach to adult census income level prediction. In *2018 Second International Conference on Electronics, Communication and Aerospace Technology (ICECA)* (pp. 207–212). IEEE.

Girhepuje, S. (2023). *Identifying and examining machine learning biases on Adult dataset*. arXiv:2310.09373. https://arxiv.org/abs/2310.09373

Kohavi, R. (1996). Scaling up the accuracy of naive-Bayes classifiers: A decision-tree hybrid. In *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96)* (pp. 202–207). AAAI Press.
