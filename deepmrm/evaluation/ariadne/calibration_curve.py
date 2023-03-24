import numpy as np
from sklearn.linear_model import LinearRegression


def generate_calibration_curves(
                    smooth_quant_df, 
                    estimated_ratio_column,
                    peptide_id_col='peptide_id',
                    target_abundance_column='Heavy peptide abundance (fmole)',
                    Q_value_cutoff=None):

    cal_curves = dict()
    # cols = [peptide_id_col, 'Heavy peptide abundance (fmole)', estimated_ratio_column]
    # smooth_df = ariadne_ret['smooth'].merge(ret_df, 
    #                                     left_on=[peptide_id_col, 'File Name'],
    #                                     right_on=[peptide_id_col, 'filename'],
    #                                     how='left')
    # smooth_df['RatioLightToHeavy_DeepMRM'] = smooth_df.apply(extract_deepmrm_ratio, axis=1)
    # m = (smooth_quant_df[ratio_col].notna()) & (smooth_quant_df['RatioLightToHeavy_Skyline'] > 0)
    m = (smooth_quant_df[estimated_ratio_column].notna()) & \
        (smooth_quant_df[estimated_ratio_column] > 0)
    if Q_value_cutoff is not None:
        m &= (smooth_quant_df['annotation_QValue'] < Q_value_cutoff)

    for pep_id, sub_df  in smooth_quant_df[m].groupby(peptide_id_col):
        y_true = sub_df.loc[:, [target_abundance_column]]
        
        # In this experiment, light peptide is reference 
        # light/heavy ratio -> heavy/ligth ratio
        y_ratio = (1 / sub_df[[estimated_ratio_column]])  
        cal_curves[pep_id] = LinearRegression(fit_intercept=False).fit(y_ratio, y_true)

    return cal_curves

