from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr

from pathlib import Path
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib_venn import venn3
from torchvision import transforms as T

from deepmrm import model_dir
from deepmrm.data.dataset import MRMDataset
from deepmrm.data.transition import TransitionData
from deepmrm.data import obj_detection_collate_fn
from deepmrm.data_prep import ariadne
from deepmrm.transform.make_input import MakeInput, TransitionDuplicate
from deepmrm.train.run import task, transform
from deepmrm.predict.interface import create_prediction_results
from mstorch.data.mass_spec.tolerance import Tolerance, ToleranceUnit
from deepmrm.data_prep.ariadne import get_mzml_files, get_trans_df, get_skyline_result_df

fig_dir = Path('reports/figures')

model_name = 'ResNet34_Aug'
trans_df = get_trans_df()
mzml_files = get_mzml_files()
peptide_id_col='peptide_id'

trans_df['is_heavy'] = ~trans_df['is_heavy']

batch_size = 1
num_workers = 8
mz_tol = Tolerance(0.5, ToleranceUnit.MZ)

transform = T.Compose([MakeInput(), TransitionDuplicate()])
transition_data = TransitionData(trans_df, peptide_id_col=peptide_id_col)

ret_fname = './data/ariadne_result_ref-light.pkl'
ret_file = Path(ret_fname)

if ret_file.exists():
    ret_df = pd.read_pickle(ret_fname)
else:
    model_path = Path(model_dir) / f'{model_name}.pth'
    model = torch.load(model_path)
    dfs = []
    for mzml_path in mzml_files:
        ds = MRMDataset(
                    mzml_path,
                    transition_data,
                    transform=transform,
                    )
        ds.extract_data(tolerance=mz_tol)
        m = np.in1d(ds.metadata_df[peptide_id_col], list(ds.chrom_index_map))
        ds.metadata_df = ds.metadata_df[m]

        data_loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=1, collate_fn=obj_detection_collate_fn)
        output_df = model.evaluate(data_loader)
        result_dict = create_prediction_results(ds, output_df)
        ret_df = pd.DataFrame.from_dict(result_dict, orient='index').join(ds.metadata_df)
        ret_df['filename'] = mzml_path.stem
        dfs.append(ret_df)

    ret_df = pd.concat(dfs)
    ret_df.to_pickle(ret_fname)


ariadne_ret = {
    'smooth':get_skyline_result_df('Smooth'),
    'noisy':get_skyline_result_df('Noisy'),
    'bkg': get_skyline_result_df('SmoothBack')
} 

# peps = ['FHYQYDEDTILGR_2+', 'GGAVALDGR_2+', 'IGPPGDGETNPGGSISR_2+', 'TFEIDLDQPTDFSAR_2+', 'YAENQIESDLDISGPAGGQK_2+']
# tmp_df = ariadne_ret['noisy'].merge(
#                     ret_df, 
#                     left_on=[peptide_id_col, 'File Name'],
#                     right_on=[peptide_id_col, 'filename'],
#                     how='left')
# m = np.in1d(tmp_df['peptide_id'], peps)
# tmp_df[m].to_csv('tmp.tsv', sep='\t')


def extract_deepmrm_ratio(row):
    score_th = 0.1
    if len(row['boxes']) < 1:
        return np.nan
    if row['scores'][0] < score_th:
        return np.nan
    ratio = row['light_area'][0] / row['heavy_area'][0]
    return ratio

###########################################################
######## calibration curve fitting ########################
methods = ['Skyline default', 'Skyline at FDR 5%', 'DeepMRM']
cal_curves = dict()
for method in methods:

    alg_name = method.split(' ')[0]
    ratio_col = f'RatioLightToHeavy_{alg_name}'
    Q_value_cutoff = 0.05 if 'FDR' in method else -1
    cols = [peptide_id_col, 'Heavy peptide abundance (fmole)', ratio_col]
    smooth_df = ariadne_ret['smooth'].merge(
                    ret_df, 
                    left_on=[peptide_id_col, 'File Name'],
                    right_on=[peptide_id_col, 'filename'],
                    how='left')
    smooth_df['RatioLightToHeavy_DeepMRM'] = smooth_df.apply(extract_deepmrm_ratio, axis=1)
    
    cal_curves[method] = dict()
    quant_df = smooth_df
    m = (quant_df[ratio_col].notna()) & (quant_df['RatioLightToHeavy_Skyline'] > 0)
    
    if Q_value_cutoff > 0:
        m &= (quant_df['annotation_QValue'] < Q_value_cutoff)
    for pep_id, sub_df  in quant_df.loc[m, cols].groupby('peptide_id'):
        y_true = sub_df.loc[:, ['Heavy peptide abundance (fmole)']]
        y_ratio = (1 / sub_df[[ratio_col]])
        cal_curves[method][pep_id] = LinearRegression(fit_intercept=False).fit(y_ratio, y_true)
        #cal_curves[method][pep_id].predict(y_ratio)
###########################################################


cols = [
    peptide_id_col, 
    'Heavy peptide abundance (fmole)', 
    'RatioLightToHeavy_Skyline', 
    'RatioLightToHeavy_DeepMRM',
    'Score_DeepMRM']

ref_heavy_fmol = 100
Q_value_cutoff = 0.05
score_th = 0.10
methods = ['Skyline default', 'Skyline at FDR 5%', 'DeepMRM']
average_over_replicates = False

datasets = ['noisy', 'bkg']
multi_factors = [10, 1]
tbl_ret = dict()
quant_ret = dict()
fig, axs = plt.subplots(len(datasets), 3, figsize=(17, 14))
for i, dataset in enumerate(datasets):
    if dataset in ('bkg', 'smooth'): # optimal with or without background
        ylim = [-10, 10]
    elif dataset == 'noisy': # sub-optimal without background
        ylim = [-12, 12]

    df = ariadne_ret[dataset]
    tbl_ret[dataset] = dict()
    quant_ret[dataset] = dict()
    
    quant_df = df.merge(
                    ret_df, 
                    left_on=[peptide_id_col, 'File Name'],
                    right_on=[peptide_id_col, 'filename'],
                    how='left'
                )
    quant_df['RatioLightToHeavy_DeepMRM'] = quant_df.apply(extract_deepmrm_ratio, axis=1) 
    quant_df['Score_DeepMRM'] = quant_df['scores'].apply(lambda x : x[0] if len(x) > 0 else 0)

    for k, method in enumerate(methods):
        tbl_ret[dataset][method]  = dict()
        quant_ret[dataset][method] = dict()
        if k == 0:
            m = (quant_df['RatioLightToHeavy_Skyline'] > 0)
            ratio_col = 'RatioLightToHeavy_Skyline'
        elif k == 1:
            m = (quant_df['RatioLightToHeavy_Skyline'] > 0) & (quant_df['annotation_QValue'] < Q_value_cutoff)
            ratio_col = 'RatioLightToHeavy_Skyline'
        elif k == 2:
            m = quant_df['RatioLightToHeavy_DeepMRM'].notna()
            ratio_col = 'RatioLightToHeavy_DeepMRM'
        else:
            raise ValueError('Not valid method')

        num_measurements = 0
        corr_ret = []
        pep_quant_dfs = []
        for pep_id, sub_df  in quant_df.loc[m, cols].groupby('peptide_id'):
            if average_over_replicates:
                sub_df = sub_df.groupby('Heavy peptide abundance (fmole)').mean().reset_index(drop=False)
            if sub_df.shape[0] < 2:
                print(f'================ {dataset} {pep_id} {method} skip ====================')
                continue
            num_measurements += sub_df.shape[0]

            y_true = sub_df.loc[:, 'Heavy peptide abundance (fmole)']
            y_ratio = (1 / (sub_df[[ratio_col]]*multi_factors[i]))
            y_pred = cal_curves[method][pep_id].predict(y_ratio)

            pcc = pearsonr(y_true, y_pred.reshape(y_true.shape))
            spc = spearmanr(y_true, y_pred)

            pep_quant_df = pd.concat((y_true, pd.DataFrame(y_pred, columns=[ratio_col], index=y_true.index)), axis=1)
            
            ape = (pep_quant_df['Heavy peptide abundance (fmole)'] - pep_quant_df[ratio_col]).abs()/pep_quant_df['Heavy peptide abundance (fmole)']
            # Use MAAPE instead of MAPE (close-to-zero values)
            # https://www.sciencedirect.com/science/article/pii/S0169207016000121?via%3Dihub
            pep_quant_df['APE'] = ape
            pep_quant_df['AAPE'] = np.arctan(ape)
            pep_quant_df['peptide_id'] = pep_id
            # pep_quant_df.join(sub_df['Score_DeepMRM'])
            pep_quant_dfs.append(pep_quant_df)
            corr_ret.append([pep_id, pcc[0], spc[0]])

        corr_df = pd.DataFrame(corr_ret, columns=['peptide_id', 'PCC', 'SPC'])
        pep_quant_dfs = pd.concat(pep_quant_dfs)    

        tbl_ret[dataset][method]['#peaks'] = num_measurements # np.sum(m)
        tbl_ret[dataset][method]['PCC'] = corr_df['PCC'].mean()
        tbl_ret[dataset][method]['SPC'] = corr_df['SPC'].mean()
        tbl_ret[dataset][method]['MAPE'] = pep_quant_dfs['APE'].mean()
        tbl_ret[dataset][method]['MAAPE'] = pep_quant_dfs['AAPE'].mean()
        quant_ret[dataset][method] = pep_quant_dfs

        pep_quant_dfs['Log2 [Heavy peptide abundance]'] = \
                        np.log2(pep_quant_dfs[ratio_col])

        ax = axs[i, k]
        _ = pep_quant_dfs.boxplot(
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
        ax.set_title(f'{method}', fontsize=17)
        ax.set_ylim(ylim)

for ax in axs.flat:
    ax.set(xlabel='Heavy peptide abundance (fmole)', ylabel='Log2 [Heavy peptide abundance]')

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

if average_over_replicates: 
    fname = f'ariadne_boxplot_avg.jpg'
else:
    fname = f'ariadne_boxplot_.jpg'
plt.savefig(fig_dir / fname)


corr_df = pd.concat(
    [pd.DataFrame.from_dict(tbl_ret[ds], orient='index') for ds in list(tbl_ret)]
)

if average_over_replicates: 
    corr_df.to_csv('./reports/ariadne_corr_avg.csv')
else:
    corr_df.to_csv('./reports/ariadne_corr.csv')
    

##################### accuracy plot ############################
fig, axs = plt.subplots(len(datasets), 3, figsize=(17, 14))
for i, dataset in enumerate(datasets):
    for k, method in enumerate(methods):
        # k = 2
        # method = methods[k]
        pep_quant_df = quant_ret[dataset][method]
        alg_name = method.split(' ')[0]
        ratio_col = f'RatioLightToHeavy_{alg_name}'

        pep_quant_df['error'] = 100*(pep_quant_df['Heavy peptide abundance (fmole)'] - pep_quant_df[ratio_col]).abs()/pep_quant_df['Heavy peptide abundance (fmole)']
        # pep_quant_df.groupby('Heavy peptide abundance (fmole)')['error'].median()
        # pep_quant_df[pep_quant_df['error'] > 200]
        # pep_quant_df.loc[756:780, :]

        ax = axs[i, k]
        _ = pep_quant_df.boxplot(
            'error', 
            by='Heavy peptide abundance (fmole)',
            showfliers=True,
            flierprops=dict(markerfacecolor='grey', marker='.'),
            ax=ax,
            grid=False
        )
        ax.set_title(f'{method}', fontsize=17)
        ax.set_ylim([0, 200])
        

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

if average_over_replicates: 
    fname = f'ariadne_accuracy_avg.jpg'
else:
    fname = f'ariadne_accuracy.jpg'
plt.savefig(fig_dir / fname)





