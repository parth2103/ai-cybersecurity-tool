# Model Performance Comparison Report

**Generated:** 2025-11-10 15:42:13

## Executive Summary

This report compares the performance of models trained on:
- **Old Models:** CICIDS2017 dataset
- **New Models:** CIC IoT-IDAD 2024 + CICAPT-IIOT datasets

## Models Compared

1. **Random Forest**
2. **XGBoost**
3. **Isolation Forest**

## Performance Metrics

### Accuracy Comparison

| Model | Old (CICIDS2017) | New (IoT-IDAD + CICAPT) | Difference |
|-------|------------------|--------------------------|------------|
| Random Forest | 0.7545 | 0.5000 | -0.2545 |
| Xgboost | 0.7545 | 0.5000 | -0.2545 |
| Isolation Forest | 0.7545 | 0.9907 | +0.2362 |

### Detailed Metrics

#### Random Forest

| Metric | Old Model | New Model |
|--------|-----------|----------|
| Accuracy | 0.7545 | 0.5000 |
| Precision | 0.0000 | 0.5000 |
| Recall | 0.0000 | 1.0000 |
| F1-Score | 0.0000 | 0.6667 |
| ROC-AUC | 0.5854 | 1.0000 |

#### Xgboost

| Metric | Old Model | New Model |
|--------|-----------|----------|
| Accuracy | 0.7545 | 0.5000 |
| Precision | 0.0000 | 0.5000 |
| Recall | 0.0000 | 1.0000 |
| F1-Score | 0.0000 | 0.6667 |
| ROC-AUC | 0.6223 | 0.5000 |

#### Isolation Forest

| Metric | Old Model | New Model |
|--------|-----------|----------|
| Accuracy | 0.7545 | 0.9907 |
| Precision | 0.0000 | 0.9817 |
| Recall | 0.0000 | 1.0000 |
| F1-Score | 0.0000 | 0.9908 |
| ROC-AUC | 0.5557 | 0.9888 |

## Visualizations

1. **Metrics Comparison Chart** - `metrics_comparison.png`
2. **Confusion Matrices** - `confusion_matrices.png`
3. **ROC Curves** - `roc_curves.png`
4. **Metrics Table** - `metrics_table.png`

## Key Findings

### Performance Changes

- **Random Forest:** Decreased by 33.73%
- **Xgboost:** Decreased by 33.73%
- **Isolation Forest:** Improved by 31.31%

## Conclusions

### Recommendations

1. **Model Selection:** Based on the comparison, choose the model with best performance for your use case.
2. **Dataset Diversity:** New models trained on multiple datasets may have better generalization.
3. **Production Deployment:** Consider deploying the best-performing model to production.
4. **Continuous Monitoring:** Monitor model performance in production and retrain as needed.

