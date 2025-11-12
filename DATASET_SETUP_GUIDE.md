# Dataset Setup Guide

This guide explains where to place your new cybersecurity datasets.

## Folder Structure

Place your datasets in the following directories within the `data/` folder:

```
data/
├── cicids2017/                    # Existing dataset (already present)
│   └── MachineLearningCCSV/
│       └── *.csv
│
├── cic_iot_idad_2024/            # NEW: CIC IoT-IDAD Dataset 2024
│   └── Dataset/
│       └── *.csv
│
├── cicapt_iiot/                   # NEW: CICAPT-IIOT Dataset
│   └── *.csv
│
└── global_cybersecurity_threats/  # NEW: Global Cybersecurity Threats (2015-2024)
    └── *.csv
```

## Detailed Instructions

### 1. CIC IoT-IDAD Dataset 2024

**Folder Path:** `data/cic_iot_idad_2024/Dataset/`

**Steps:**
1. Create the folder structure:
   ```bash
   mkdir -p data/cic_iot_idad_2024/Dataset
   ```

2. Place all CSV files from the CIC IoT-IDAD 2024 dataset in the `Dataset/` folder
3. The loader will automatically find all `.csv` files in this directory and subdirectories

**Expected Structure:**
```
data/cic_iot_idad_2024/
└── Dataset/
    ├── file1.csv
    ├── file2.csv
    ├── subfolder/
    │   └── file3.csv
    └── ...
```

### 2. CICAPT-IIOT Dataset

**Folder Path:** `data/cicapt_iiot/`

**Steps:**
1. Create the folder:
   ```bash
   mkdir -p data/cicapt_iiot
   ```

2. Place all CSV files from the CICAPT-IIOT dataset directly in this folder
3. You can also organize files in subdirectories - the loader will find them recursively

**Expected Structure:**
```
data/cicapt_iiot/
├── file1.csv
├── file2.csv
├── subfolder/
│   └── file3.csv
└── ...
```

### 3. Global Cybersecurity Threats (2015-2024)

**Folder Path:** `data/global_cybersecurity_threats/`

**Steps:**
1. Create the folder:
   ```bash
   mkdir -p data/global_cybersecurity_threats
   ```

2. Place all CSV files from the Global Cybersecurity Threats dataset in this folder
3. The loader supports nested folder structures

**Expected Structure:**
```
data/global_cybersecurity_threats/
├── 2015/
│   └── threats_2015.csv
├── 2016/
│   └── threats_2016.csv
├── ...
└── 2024/
    └── threats_2024.csv
```

## Quick Setup Commands

Run these commands from the project root to create the folder structure:

```bash
# Create all required directories
mkdir -p data/cic_iot_idad_2024/Dataset
mkdir -p data/cicapt_iiot
mkdir -p data/global_cybersecurity_threats

# Verify structure
tree data/ -L 2
```

## Dataset Requirements

### File Format
- **Format:** CSV files (`.csv`)
- **Encoding:** UTF-8, Latin-1, or ISO-8859-1 (auto-detected)
- **Separator:** Comma (`,`) - default, can be configured if needed

### Label Column
The datasets should have a label column indicating whether traffic is benign or an attack. Common column names:
- `Label`
- `label`
- ` Label` (with space)
- `LABEL`
- `target`
- `class`

The loader will automatically detect the label column.

### Features
- The datasets should contain numeric features (the loader will automatically extract numeric columns)
- Non-numeric columns (except labels) will be ignored
- Missing values will be handled during preprocessing

## Verification

After placing your datasets, verify they're detected by running:

```bash
python explore_new_datasets.py
```

This will:
1. List all available datasets
2. Explore each dataset's structure
3. Show feature counts, label distributions, etc.
4. Save exploration results to `results/dataset_exploration.json`

## Next Steps

1. **Place datasets** in the folders above
2. **Run exploration script:**
   ```bash
   python explore_new_datasets.py
   ```
3. **Review exploration results** in `results/dataset_exploration.json`
4. **Train models** on new datasets:
   ```bash
   python train_on_new_datasets.py
   ```

## Troubleshooting

### Dataset Not Found
If a dataset is not detected:
- Check that the folder path matches exactly (case-sensitive)
- Verify CSV files exist in the folder
- Check file permissions

### Encoding Errors
If you encounter encoding errors:
- The loader tries multiple encodings automatically
- If issues persist, convert files to UTF-8:
  ```bash
  iconv -f ISO-8859-1 -t UTF-8 input.csv > output.csv
  ```

### Label Column Not Found
If the label column is not detected:
- Check the exploration results to see what columns are available
- Update the `label_column` in `src/data_loader_multi.py` if needed
- Common label column names are automatically detected

### Missing Features
If features don't align across datasets:
- The feature alignment utility will handle this automatically
- Missing features will be filled with zeros
- Only common features across datasets will be used for training

## Notes

- **File Size:** Large datasets may take time to load. Consider using `sample_size` parameter for initial testing
- **Memory:** Loading multiple large datasets may require significant RAM
- **Subdirectories:** The loader searches recursively, so you can organize files in subdirectories
- **File Naming:** File names don't matter - the loader finds all `.csv` files

## Example Dataset Organization

```
data/
├── cicids2017/
│   └── MachineLearningCCSV/
│       └── (existing files)
│
├── cic_iot_idad_2024/
│   └── Dataset/
│       ├── IoT_Device_Attack_Data_2024.csv
│       ├── IoT_Traffic_2024.csv
│       └── ...
│
├── cicapt_iiot/
│   ├── IIoT_Attack_Data.csv
│   ├── IIoT_Normal_Data.csv
│   └── ...
│
└── global_cybersecurity_threats/
    ├── threats_2015_2024.csv
    ├── combined_threats.csv
    └── ...
```

After setting up, your datasets will be ready for training!

