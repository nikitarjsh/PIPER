# Immunogenicity Data

This folder contains the training and testing data used to build our models.

## Training Data: 8971 data instances (4059 positive + 4912 negative)

### Immunogenicity Data from IEDB
- 8971 data points
- Extracted from tested immunogenicity molecular assays (IEDB) using keywords: Linear epitope, T-cell assay, MHC class I, Human, and disease-related entries
- Redundant pMHC entries removed
- **File:** `data/raw/IEDB_pHLA_data.csv`

## Testing Data

### SARS-CoV-2 Peptides
- 100 peptides
- Tested for immunogenicity in convalescent and unexposed subjects
- **File:** `data/raw/sars_cov_2_result.csv`

## Data Folder Structure

- `raw/`  
  Original datasets (IEDB and SARS-CoV-2)

- `features/`  
  Processed datasets and selected features used for model training. Includes global and position-specific feature representations and feature selection files.

- `splits/`  
  Predefined train/validation/test indices (`.npy` files)

- `dataset_train.csv`, `dataset_val.csv`, `dataset_test.csv`  
  Final datasets used directly for model training and evaluation

- `processed_sars_cov_2_with_*`  
  Feature-engineered SARS-CoV-2 test datasets

- `deprecated/`  
  Older dataset versions (A/B/C splits), not used in the final pipeline