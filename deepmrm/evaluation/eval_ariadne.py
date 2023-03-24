import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib_venn import venn3
from scipy.stats import pearsonr, spearmanr
from torchvision import transforms as T

from deepmrm import model_dir, private_project_dir
from deepmrm.data.dataset import MRMDataset
from deepmrm.data.transition import TransitionData
from deepmrm.data import obj_detection_collate_fn
from mstorch.data.mass_spec.tolerance import Tolerance, ToleranceUnit
from deepmrm.data_prep import ariadne
from deepmrm.predict.interface import _load_models
from deepmrm.transform.make_input import MakeInput
from deepmrm.utils.eval import create_prediction_results
from deepmrm.evaluation.ariadne import (
    generate_calibration_curves,
    extract_deepmrm_ratio,
)

reports_dir = private_project_dir / 'reports'
fig_dir = reports_dir / 'figures'
batch_size = 64
num_workers = 4

dataset_name = 'Ariadne'
peptide_id_col = ariadne.peptide_id_col
trans_df = ariadne.get_trans_df()
mzml_files = ariadne.get_mzml_files()

ariadne_ret = {
    data_name: ariadne.get_skyline_result_df(data_name) 
        for data_name in ariadne.DATASETS
}

# [NOTE] In the Ariadne benchmark, light peptides are reference.
# The concentration of heavy peptides varies.
heavy_is_reference = False
if not heavy_is_reference:
    trans_df['is_heavy'] = ~trans_df['is_heavy']
    save_path = reports_dir/f'{dataset_name}_output_df.pkl'
else:
    save_path = reports_dir/f'{dataset_name}_output_df_heavy-ref.pkl'

batch_size = 64
num_workers = 4
mz_tol = Tolerance(0.5, ToleranceUnit.MZ)
transform = T.Compose([MakeInput()])
transition_data = TransitionData(trans_df, peptide_id_col=peptide_id_col)

# save_path.unlink()

if save_path.exists():
    ret_df = pd.read_pickle(save_path)
else:
    model, model_qs = _load_models(model_dir)
    dfs = []
    for mzml_path in mzml_files:
        ds = MRMDataset(
                    mzml_path,
                    transition_data,
                    transform=transform)
        ds.extract_data(tolerance=mz_tol)
        m = np.in1d(ds.metadata_df[peptide_id_col], list(ds.chrom_index_map))
        ds.metadata_df = ds.metadata_df[m]

        data_loader = torch.utils.data.DataLoader(ds, 
                            batch_size=batch_size, 
                            num_workers=num_workers, 
                            collate_fn=obj_detection_collate_fn
                        )
        output_df = model.predict(data_loader)
        output_df = model_qs.predict(data_loader.dataset, output_df)
        
        result_dict = create_prediction_results(ds, output_df, quality_th=0.5)
        ret_df = ds.metadata_df.join(
                    pd.DataFrame.from_dict(
                        result_dict, orient='index'))
        
        # ret_df[['selected_transition_index', 'quantification_scores']]
        ret_df['File Name'] = mzml_path.stem
        dfs.append(ret_df)
        print(f'Completed {len(dfs)}/{len(mzml_files)} files')
    ret_df = pd.concat(dfs)
    ret_df.to_pickle(save_path)


ref_heavy_fmol = 100
Q_value_cutoff = 0.05
# score_th = 0.10
methods = ['Skyline default', 'Skyline FDR 5%', 'DeepMRM']
average_over_replicates = False

ret_df['DeepMRM_BD_score'] = ret_df['scores'].apply(lambda x : x[0] if len(x) > 0 else 0)
ret_df['DeepMRM_QS_score'] = ret_df['quantification_scores'].apply(lambda x : x[0] if len(x) > 0 else 0) 
kwargs={'heavy_is_reference': False, 'selected_xic': True}
ret_df['RatioLightToHeavy_DeepMRM'] = ret_df.apply(extract_deepmrm_ratio, axis=1, **kwargs)


#### Merge Ariadne's and DeepMRM's results
for dataset in ariadne.DATASETS:
    df = ariadne_ret[dataset]
    df = df.merge(ret_df, on=[peptide_id_col, 'File Name'], how='left')
    # add prediction columns for different methods
    for method in methods:
        df[f'{method} abundance prediction'] = np.float64(np.nan)
    ariadne_ret[dataset] = df


#### Generate calibration curves
calib_curves = dict()
smooth_df = ariadne_ret['Smooth']
calib_curves[methods[0]] = generate_calibration_curves(
                                smooth_df, 
                                estimated_ratio_column='RatioLightToHeavy_Skyline')
calib_curves[methods[1]] = generate_calibration_curves(
                                smooth_df, 
                                estimated_ratio_column='RatioLightToHeavy_Skyline',
                                Q_value_cutoff=0.05)
calib_curves[methods[2]] = generate_calibration_curves(
                                smooth_df, 
                                estimated_ratio_column='RatioLightToHeavy_DeepMRM')


#### Estimate abundance with calibrated curves
result_tables = dict()
for dataset in ariadne.DATASETS[:2]:
    quant_df = ariadne_ret[dataset]
    result_tables[dataset] = dict()
    # make predictions with calibration curves
    for k in range(len(methods)):
        method = methods[k]

        if k == 0:
            m = (quant_df['RatioLightToHeavy_Skyline'] > 0)
            ratio_col = 'RatioLightToHeavy_Skyline'
        elif k == 1:
            m = (quant_df['RatioLightToHeavy_Skyline'] > 0) & (quant_df['annotation_QValue'] < Q_value_cutoff)
            ratio_col = 'RatioLightToHeavy_Skyline'
        elif k == 2:
            m = (quant_df['RatioLightToHeavy_DeepMRM'].notna()) & \
                (quant_df['DeepMRM_QS_score'] > 0)
            ratio_col = 'RatioLightToHeavy_DeepMRM'
        else:
            raise ValueError('Not valid method')

        num_measurements = 0
        corr_ret = []
        for pep_id, sub_df  in quant_df.loc[m, :].groupby('peptide_id'):
            if average_over_replicates:
                sub_df = sub_df.groupby('Heavy peptide abundance (fmole)').mean().reset_index(drop=False)
            if sub_df.shape[0] < 2:
                print(f'================ {dataset} {pep_id} {method} skip ====================')
                continue
            num_measurements += sub_df.shape[0]
            if pep_id not in calib_curves[method]:
                continue

            y_true = sub_df.loc[:, 'Heavy peptide abundance (fmole)']
            y_ratio = (1 / (sub_df[[ratio_col]]*ariadne.MULTI_FACTORS[dataset]))
            y_pred = calib_curves[method][pep_id].predict(y_ratio)

            pcc = pearsonr(y_true, y_pred.reshape(y_true.shape))
            spc = spearmanr(y_true, y_pred)

            quant_df.loc[y_true.index, f'{method} abundance prediction'] = y_pred
            corr_ret.append([pep_id, pcc[0], spc[0]])

        corr_df = pd.DataFrame(corr_ret, columns=['peptide_id', 'PCC', 'SPC'])
        result_tables[dataset][method] = corr_df
#######################################################################################


#### Compute performance metrics 
BD_SCORE_TH = 0.05
QS_SCORE_TH = 0.1

dataset = ariadne.DATASETS[0]
quant_df = ariadne_ret[dataset]
corr_dfs = result_tables[dataset]
perf_stats = dict()
for method in methods:
    # method = methods[2]
    corr_df = corr_dfs[method]
    m = quant_df[f'{method} abundance prediction'].notnull()
    if method == 'DeepMRM':
        m &= (quant_df['DeepMRM_BD_score'] > BD_SCORE_TH)
        m &= (quant_df['DeepMRM_QS_score'] > QS_SCORE_TH)
    y_true = quant_df.loc[m, 'Heavy peptide abundance (fmole)']
    y_pred = quant_df.loc[m, f'{method} abundance prediction']
    ape = (y_true - y_pred).abs()/y_true
    # quant_df.loc[m, f'{method}_APE'] = ape
    aape = np.arctan(ape)
    perf_stats[method] = {
        '#peaks': np.sum(m),
        'PCC': corr_df['PCC'].mean(),
        'SPC': corr_df['SPC'].mean(),
        'MAPE': ape.mean(),
        'MdAPE': ape.median(),
        'MAAPE': aape.mean(),
    }

perf_stat_df = pd.DataFrame.from_dict(perf_stats, orient='index')
print(perf_stat_df)

#### DEBUG ####
# cols = [
#     peptide_id_col,
#     'Heavy peptide abundance (fmole)',
#     'Skyline default abundance prediction',
#     'DeepMRM abundance prediction',
#     'DeepMRM_BD_score', 
#     'DeepMRM_QS_score',
# ]
# quant_df[m][ape> 1].to_pickle('./temp/temp.pkl')
#### DEBUG ####

## quantification score


####################################################################
# Examples of DeepMRM, Skyline
# Why Skyline miss some detections, and how DeepMRM rescure them. 
# Even for peptides that both detected by DeepMRM and Skyline, 
# how DeepMRM gave a better quantification (better detected boundaries?). 
# Please show more instances for Fig. 1b-d and Fig. 2 
# other than just displaying the numbers and box plots.

# 1) Good peaks that are missed by Skyline but detected by DeepMRM


# 2) Poor peaks that are detected by Skyline, but filtered by DeepMRM


# 3) Detected by Both, but better quantified by DeepMRM




####################################################################

#### Create boxplots
datasets = ariadne.DATASETS[:2]
fig, axs = plt.subplots(len(datasets), 3, figsize=(17, 13))
for i, dataset in enumerate(datasets):
    quant_df = ariadne_ret[dataset]

    if dataset == 'Noisy': # sub-optimal without background
        ylim = [-12, 12]
    else: # optimal with or without background
        ylim = [-10, 10]
    
    for k in range(len(methods)):
        method = methods[k]
        # alg_name = method.split(' ')[0]
        m = quant_df[f'{method} abundance prediction'].notnull()
        if method == 'DeepMRM':
            m &= (quant_df['DeepMRM_BD_score'] > BD_SCORE_TH)
            m &= (quant_df['DeepMRM_QS_score'] > QS_SCORE_TH)

        quant_df['Log2 [Heavy peptide abundance]'] = np.log2(quant_df[f'{method} abundance prediction'])

        ax = axs[i, k]
        _ = quant_df[m].boxplot(
            'Log2 [Heavy peptide abundance]', 
            by='Heavy peptide abundance (fmole)',
            showfliers=True,
            flierprops=dict(markerfacecolor='grey', marker='.'),
            ax=ax,
            grid=False
        )

        x_ticks = ax.xaxis.get_majorticklocs()
        x_labels = ax.xaxis.get_majorticklabels()
        x_labels = [float(x.get_text()) for x in x_labels]
        ax.plot(x_ticks, np.log2(x_labels), 'r', linestyle='dotted', linewidth=2)
        ax.set_xticklabels(list(map(lambda x : x if x < 1 else int(x), x_labels)))
        ax.set_title(f'{method}', fontsize=18)
        ax.set_ylim(ylim)

for ax in axs.flat:
    ax.set(xlabel='Heavy peptide abundance (fmole)', 
           ylabel='Log2 [Heavy peptide abundance]')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    x_axis = ax.get_xaxis()
    y_axis = ax.get_yaxis()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(13)
    y_label = y_axis.get_label()
    x_label = x_axis.get_label()
    x_label.set_fontsize(16)
    y_label.set_fontsize(16)
    x_label.set_visible(ax.is_last_row())
    y_label.set_visible(ax.is_first_col())

fig.suptitle('')
fig.tight_layout()
plt.savefig(fig_dir / 'ariadne_boxplot.jpg')



##################### accuracy plot ############################
fig, axs = plt.subplots(len(datasets), 3, figsize=(17, 14))
for i, dataset in enumerate(datasets):
    quant_df = ariadne_ret[dataset]
    y_true = quant_df['Heavy peptide abundance (fmole)']

    for k, method in enumerate(methods):
        y_pred = quant_df[f'{method} abundance prediction']
        quant_df['APE'] = 100*(y_true - y_pred).abs()/y_true        

        m = y_pred.notnull()
        if method == 'DeepMRM':
            m &= (quant_df['DeepMRM_BD_score'] > BD_SCORE_TH)
            m &= (quant_df['DeepMRM_QS_score'] > QS_SCORE_TH)

        ax = axs[i, k]
        _ = quant_df[m].boxplot(
            'APE', 
            by='Heavy peptide abundance (fmole)',
            showfliers=True,
            flierprops=dict(markerfacecolor='grey', marker='.'),
            ax=ax,
            grid=False
        )
        x_labels = [float(x.get_text()) for x in ax.xaxis.get_majorticklabels()]
        ax.set_xticklabels(list(map(lambda x : x if x < 1 else int(x), x_labels)))
        ax.set_title(f'{method}', fontsize=17)
        ax.set_ylim([0, 1000])

for ax in axs.flat:
    ax.set(
        xlabel='Heavy peptide abundance (fmole)', 
        ylabel='Absolute percentage error (%)')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    x_axis = ax.get_xaxis()
    y_axis = ax.get_yaxis()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(13)
    y_label = y_axis.get_label()
    x_label = x_axis.get_label()
    x_label.set_fontsize(15)
    y_label.set_fontsize(15)
    x_label.set_visible(ax.is_last_row())
    y_label.set_visible(ax.is_first_col())

fig.suptitle('')
fig.tight_layout()
plt.savefig(fig_dir / f'ariadne_accuracy.jpg')

