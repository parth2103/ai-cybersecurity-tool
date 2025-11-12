# Next Steps for Training on New Datasets

## ✅ Step 1: Place Your Datasets

Place your downloaded datasets in the following folders:

### 📁 Folder Structure

```
data/
├── cic_iot_idad_2024/          ← Place CIC IoT-IDAD Dataset 2024 here
│   └── Dataset/
│       └── (your CSV files)
│
├── cicapt_iiot/                ← Place CICAPT-IIOT dataset here
│   └── (your CSV files)
│
└── global_cybersecurity_threats/  ← Place Global Cybersecurity Threats here
    └── (your CSV files)
```

### 📋 Quick Commands

The folders have already been created. You just need to copy your CSV files:

```bash
# Example: If you downloaded datasets to Downloads folder
# CIC IoT-IDAD 2024
cp -r ~/Downloads/CIC-IoT-IDAD-2024/Dataset/* data/cic_iot_idad_2024/Dataset/

# CICAPT-IIOT
cp -r ~/Downloads/CICAPT-IIOT/* data/cicapt_iiot/

# Global Cybersecurity Threats
cp -r ~/Downloads/Global-Cybersecurity-Threats/* data/global_cybersecurity_threats/
```

## ✅ Step 2: Explore the Datasets

Before training, explore the datasets to understand their structure:

```bash
python explore_new_datasets.py
```

This will:
- ✅ List all available datasets
- ✅ Show feature counts and types
- ✅ Display label distributions
- ✅ Identify common features across datasets
- ✅ Save exploration results to `results/dataset_exploration.json`

**Expected Output:**
```
Available datasets: 3
  - cic_iot_2024: CIC IoT-IDAD 2024 (data/cic_iot_idad_2024/Dataset)
  - cicapt_iiot: CICAPT-IIOT (data/cicapt_iiot)
  - global_threats: Global Cybersecurity Threats (data/global_cybersecurity_threats)

Exploring: cic_iot_2024
  Files found: 5
  Total columns: 85
  Numeric features: 78
  Label column: Label
  Label classes: 2
```

## ✅ Step 3: Train Models on New Datasets

Once datasets are explored, train models:

```bash
python train_on_new_datasets.py
```

This will:
- ✅ Load and align features across all datasets
- ✅ Train XGBoost, Random Forest, Isolation Forest, and Ensemble models
- ✅ Compare performance with old CICIDS2017 models
- ✅ Save new models to `models/` folder
- ✅ Save training results to `results/` folder

**Expected Output:**
```
TRAINING ON NEW DATASETS
Loading datasets...
  Loaded: 50000 samples from CIC IoT-IDAD 2024
  Loaded: 50000 samples from CICAPT-IIOT
  Loaded: 50000 samples from Global Cybersecurity Threats

Aligning features...
  Combined dataset: 150000 samples
  Common features: 65

Training models...
  1. Training XGBoost... ✅ Accuracy: 0.9823
  2. Training Isolation Forest... ✅ Accuracy: 0.7545
  3. Training Random Forest... ✅ Accuracy: 0.9897
  4. Creating Ensemble... ✅ Accuracy: 0.9901
```

## ✅ Step 4: Compare Results

Compare new model performance with old models:

```bash
# Results are saved automatically to:
# results/training_results_new_datasets_YYYYMMDD_HHMMSS.json
```

## 📊 Expected Results

After training, you should have:

1. **New Models** (in `models/` folder):
   - `xgboost_model_new_datasets.pkl`
   - `random_forest_new_datasets.pkl`
   - `isolation_forest_new_datasets.pkl`
   - `ensemble_model_new_datasets.pkl`
   - `scaler_new_datasets.pkl`
   - `feature_names_new_datasets.pkl`

2. **Results** (in `results/` folder):
   - `dataset_exploration.json` - Dataset structure analysis
   - `training_results_new_datasets_*.json` - Training performance metrics

3. **Performance Metrics**:
   - Accuracy scores for each model
   - Classification reports
   - Comparison with old CICIDS2017 models

## 🔧 Configuration Options

### Adjust Sample Size

If datasets are too large, adjust the sample size in `train_on_new_datasets.py`:

```python
SAMPLE_SIZE_PER_DATASET = 50000  # Change this value
```

### Select Specific Datasets

To train on specific datasets only, modify the `DATASETS` list:

```python
DATASETS = [
    "cic_iot_2024",      # Only CIC IoT-IDAD 2024
    # "cicapt_iiot",     # Comment out to exclude
    # "global_threats"   # Comment out to exclude
]
```

### Include CICIDS2017

To include the existing CICIDS2017 dataset in training:

```python
DATASETS = ["cicids2017", "cic_iot_2024", "cicapt_iiot", "global_threats"]
```

## 🐛 Troubleshooting

### Issue: Dataset not found
**Solution:** Verify folder structure matches exactly:
```bash
ls -la data/cic_iot_idad_2024/Dataset/
ls -la data/cicapt_iiot/
ls -la data/global_cybersecurity_threats/
```

### Issue: Out of memory
**Solution:** Reduce sample size:
```python
SAMPLE_SIZE_PER_DATASET = 10000  # Smaller sample size
```

### Issue: Label column not found
**Solution:** Check exploration results and update label column name in `src/data_loader_multi.py`:
```python
label_column="YourLabelColumnName"  # Update this
```

### Issue: No common features
**Solution:** The feature alignment utility will handle this automatically by:
- Finding features that exist in ≥50% of datasets
- Filling missing features with zeros
- Using only numeric features

## 📚 Additional Resources

- **Dataset Setup Guide:** `DATASET_SETUP_GUIDE.md`
- **Exploration Script:** `explore_new_datasets.py`
- **Training Script:** `train_on_new_datasets.py`
- **Feature Aligner:** `src/utils/feature_aligner.py`
- **Multi-Dataset Loader:** `src/data_loader_multi.py`

## 🎯 Summary

1. ✅ **Place datasets** in the correct folders (already created)
2. ✅ **Explore datasets** with `python explore_new_datasets.py`
3. ✅ **Train models** with `python train_on_new_datasets.py`
4. ✅ **Review results** in `results/` folder
5. ✅ **Compare performance** with existing models

## 💡 Tips

- Start with a small sample size (10,000-50,000) for initial testing
- Use the exploration script to understand dataset structure first
- Check for common features across datasets before training
- Compare new models with old CICIDS2017 models to see improvements
- Save exploration results for future reference

Good luck with your training! 🚀

