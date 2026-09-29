# README

# Predicting Income Level from U.S. Census Data

A supervised learning study on the [UCI Adult / Census Income dataset](https://archive.ics.uci.edu/dataset/20/census+income) (Kaggle mirror: [`uciml/adult-census-income`](https://www.kaggle.com/datasets/uciml/adult-census-income)), comparing Logistic Regression and Random Forest classifiers on the task of predicting whether an individual’s income exceeds $50,000/year.

## Contents

| Deliverable | File | Description |
| --- | --- | --- |
| 📄 Paper | [`paper.md`](./paper.md)  | Abstract, introduction, methodology, results, discussion, conclusion, and references |
| 📓 Notebook | [`income_analysis.ipynb`](./notebook/income_analysis.ipynb) | Full analysis: data cleaning, EDA charts, and model training/evaluation |
| 🖼️ Slides | [`income_presentation.pptx`](./slides/income_presentation.pptx) | 6-slide presentation — Problem, Method, Results, Takeaway |
| 📊 Figures | [`figures`](./figures/) | Generated charts (PNG) and model results (CSV) used in the paper and slides |

## Summary

- **Data:** 32,561 census records, 14 attributes; reduced to 30,162 complete records after dropping missing values.
- **Key finding:** Education level, age, hours worked, and relationship status are the strongest correlates of earning >$50K/year.
- **Models:** Random Forest (86.5% accuracy, ROC AUC 0.921) modestly outperformed Logistic Regression (85.3% accuracy, ROC AUC 0.913).
- **Caveat:** Consistent with prior fairness research on this dataset, results should be read as historical correlations from 1994 Census data, not causal or bias-free relationships — see the Discussion section in the paper.

## Reproducing the analysis

```bash
# from the notebook/ directory, with adult.csv alongside the notebook
jupyter nbconvert --to notebook --execute income_analysis.ipynb
```

## References

See the [References section of the paper](./paper.md#references) for full citations (Kohavi 1996; Chakrabarty & Biswas 2018; Besse et al. 2020; Girhepuje 2023).