# Immunogenecity Data
This folder contains the training and testing data used to build our models. 
## Training Data: 8971 data instances (4059 positive reactive instances + 4912 negative reactive instances)
### Imuunogencity Data from IEDB
* 8971 data points
* tested immunogenecity molecular assays from IEDB. Extracted using the following keywrods: Linear epitope, T-cell assay, MHC class I, Human, and any disease
* all reduntant pMHC were removed
* **File:** [Immunogeneicity pHLA data from IEDB](/data/IEDB_pHLA_data.csv)

## Testing Data:

### Dengue Virus Data
* 408 dengue positive virus
* **File:** [Dengue Virus Data](data/dengue_test.csv)

### TESLA (Tumor Neoantigen Data)
* 608 experimentally tested tumot-specific neoantigens
* **File:** [Neoantigen Data](data/ori_test_cells.csv)

### SaRS-CoV-2 Peptides
* 100 peptides
* Tested for immunogeneicity in convalescent and unexposed subjected
* **File:** [SaRS-CoV-2 Data](data/sars_cov_2_result.csv)
