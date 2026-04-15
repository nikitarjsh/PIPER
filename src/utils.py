import regex as re
import peptides
import shap
import matplotlib.pyplot as plt
import peptidy
import numpy as np
import pandas as pd



def extract_features(sequence):
    peptide = peptides.Peptide(sequence)
    features = peptide.descriptors()
    features.update({'boman': peptide.boman()})
    features.update({'hydrophobicity': peptide.hydrophobicity()})
    features.update({'charge': peptide.charge()})
    features.update({'molecular_weight': peptide.molecular_weight()})
    features.update({'aliphatic_index': peptide.aliphatic_index()})
    features.update({'instability_index': peptide.instability_index()})
    features.update({'isoelectric_point': peptide.isoelectric_point()})
    features.update({'mz': peptide.mz()})
    #features.update({'structural_class': peptide.structural_class("Chou", distance="correlation")})     # Hamda: Removed becase it had only ne unique values for all our MHC psuedocode
    return features




def standardize_hla_alleles(df):
    def standardize_allele(allele):
        pattern = re.compile(r'^HLA-?([ABC]\*?\d{2,4})$')
        match = pattern.match(allele)

        if match:
            haplotype = match.group(1)
            haplotype = haplotype.replace('*', '')
            if len(haplotype) > 3:
                return f'HLA-{haplotype[0]}*{haplotype[1:3]}:{haplotype[3:5]}'
            elif len(haplotype) == 3:
                return f'HLA-{haplotype[0]}*0{haplotype[1]}:{haplotype[2]}'
        return allele

    df['HLA'] = df['HLA'].apply(standardize_allele)
    return df




def map_alleles(df):
    from mhc_pseudo import mhc_pseudo
    if 'HLA' in df.columns:
        df['hla_sequence'] = df['HLA'].map(mhc_pseudo)
        return df
    else:
        raise ValueError("Column 'HLA' not found in DataFrame")
    

## Instead of using regular feature importance, we can use SHAP values to understand the contribution of each feature to the model's predictions
## SHAP analysis function
## usage: run_shap_analysis(model, X_test)

def run_shap_analysis(model, X, max_display=20):
    """
    Compute and plot SHAP feature importance.

    Args:
        model: trained LightGBM model
        X: feature dataframe
        max_display: number of top features to show
    """

    # create explainer
    explainer = shap.TreeExplainer(model)

    # compute shap values
    shap_values = explainer.shap_values(X)

    # summary plot (like the one you showed)
    shap.summary_plot(shap_values, X, max_display=max_display)

    # bar plot (global importance)
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_display)



### Extracting position-specific features for peptides

def extract_peptidy_position_features(sequence, padding_len=10, method="aa_descriptors"):  # Hamda: Set to 10 to match the max length of peptides in our dataset
    """
    Convert a peptide sequence into position-specific features and return a flat dict.

    Parameters
    ----------
    sequence : str
        Peptide sequence.
    padding_len : int
        Final peptide length used by peptidy encoding.
    method : str
        One of:
        - "aa_descriptors" : position-specific amino acid descriptors
        - "one_hot"        : position-specific one-hot encoding
        - "blosum62"       : position-specific BLOSUM62 encoding

    Returns
    -------
    dict
        Flat feature dictionary ready to turn into a dataframe row.
    """

    # choose encoding
    if method == "aa_descriptors":
        if hasattr(peptidy.encoding, "aminoacid_descriptor_encoding"):
            encoded = peptidy.encoding.aminoacid_descriptor_encoding(
                sequence,
                padding_len=padding_len
            )
        else:
            raise AttributeError(
                "peptidy.encoding.amino_acid_descriptor_encoding not found. "
                "Check your installed peptidy version with dir(peptidy.encoding)."
            )
        prefix = "PeptidePos"

    elif method == "one_hot":
        encoded = peptidy.encoding.one_hot_encoding(
            sequence,
            padding_len=padding_len
        )
        prefix = "PeptideOH"

    elif method == "blosum62":
        if hasattr(peptidy.encoding, "blosum62_encoding"):
            encoded = peptidy.encoding.blosum62_encoding(
                sequence,
                padding_len=padding_len
            )
        else:
            raise AttributeError(
                "peptidy.encoding.blosum62_encoding not found. "
                "Check your installed peptidy version with dir(peptidy.encoding)."
            )
        prefix = "PeptideBLOSUM"

    else:
        raise ValueError("method must be one of: aa_descriptors, one_hot, blosum62")

    encoded = np.array(encoded)

    # flatten matrix into tabular columns
    features = {}
    for i in range(encoded.shape[0]):
        for j in range(encoded.shape[1]):
            features[f"{prefix}_p{i+1}_f{j+1}"] = float(encoded[i, j])

    return features


def build_peptidy_feature_df(df, peptide_col="peptide", padding_len=9):
    """
    Apply position-specific encoding to all peptides in dataframe.
    Returns a dataframe with one row per peptide.
    """

    feature_list = []

    for peptide in df[peptide_col]:
        feat_dict = extract_peptidy_position_features(
            sequence=peptide,
            padding_len=padding_len
        )
        feature_list.append(feat_dict)

    return pd.DataFrame(feature_list, index=df.index)