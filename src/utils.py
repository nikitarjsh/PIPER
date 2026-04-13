import regex as re
import peptides
import shap
import matplotlib.pyplot as plt



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
