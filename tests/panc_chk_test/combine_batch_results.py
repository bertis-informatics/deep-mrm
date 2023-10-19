
from pathlib import Path
import pandas as pd
import numpy as np


PANC_CHK_DIR = Path('/mnt/d/MassSpecData/PancCheck/verification')
SAVE_DIR = PANC_CHK_DIR / 'DeepMRM'
xlsx_path = PANC_CHK_DIR / 'PancCheck_Verification_Data(Manual).xlsx'


def get_transition_df():
    transition_path = PANC_CHK_DIR / 'Transition_Verification_1.xlsx'
    all_trans_df = pd.read_excel(transition_path, sheet_name=0).rename(
                        columns={'Q1': 'precursor_mz', 
                                'Q3': 'product_mz',
                                'ID': 'peptide_id',
                                'Retention time(min)': 'RT'})

    all_trans_df['is_heavy'] = all_trans_df['peptide_id'].apply(lambda x : x.endswith('.heavy'))
    all_trans_df['peptide_id'] = all_trans_df['peptide_id'].replace({'.light': '', '.heavy': ''}, regex=True)
    all_trans_df['RT'] *= 60
    return all_trans_df

manual_df = pd.read_excel(xlsx_path, sheet_name=1, header=[0,1])

# transition_path = PANC_CHK_DIR / 'PancCheck_Verification_Transition.csv'
# all_trans_df = pd.read_csv(transition_path)
# all_trans_df['peptide_id'] = all_trans_df['peptide_id'].apply(lambda x : '_'.join(x.split('_')[:-1])  )

all_trans_df = get_transition_df()

file_names = manual_df.iloc[:, 0]
target_df = all_trans_df[['peptide_id', 'RT']].drop_duplicates('peptide_id').reset_index(drop=True)

results = dict()
for sample_name in file_names:

    s = sample_name.split('_')
    sample_id = s[-3] if s[-1] == 'RR1' else s[-2]
    save_path = SAVE_DIR / f'{sample_id}.pkl'

    # if not save_path.exists():
    #     save_path = SAVE_DIR / f'20230602_PancCheck-RA_SER_TAR-HP-{sample_id}.pkl'

    deepmrm_df = pd.read_pickle(save_path)
    deepmrm_df = target_df.join(deepmrm_df)
    

    ret = []
    for idx, row in deepmrm_df.iterrows():
        rt_boxes = row['boxes']
        rt_error = np.abs(rt_boxes.mean(axis=1) - row['RT'])
        i = np.argmin(rt_error)

        boundary = rt_boxes[i, :]
        score = row['scores'][i]
        heavy_area = row['heavy_area'][i] - row['heavy_background'][i]
        light_area = row['light_area'][i] - row['light_background'][i]

        ret.append(
            {'peptide_id': row['peptide_id'], 
             'start_time': boundary[0], 
             'end_time': boundary[1], 
             'score': score, 
             'light_area': light_area, 
             'heavy_area': heavy_area})
        
    ret_df = pd.DataFrame.from_dict(ret)
    # ret_df.to_csv(SAVE_DIR / f'{sample_id}.csv', index=False)

    ret_df['ratio'] = ret_df['light_area']/ret_df['heavy_area']
    results[sample_id] = ret_df[['peptide_id', 'ratio']].set_index('peptide_id').to_dict()['ratio']
    
# save ratio results
result_df = pd.DataFrame.from_dict(results, orient='index')
result_df.to_csv(SAVE_DIR / 'DeepMRM_quant_result.csv')


########## combine results
# results = []
# for fname in file_names:
#     ret_df = pd.read_csv(SAVE_DIR / f'{fname}.csv', index_col='peptide_id')
#     s = fname.split('_')
#     sample_id = s[-3] if fname.endswith('RR1') else s[-2]
#     ret_df.columns = pd.MultiIndex.from_product([[sample_id], ret_df.columns] )
#     results.append(ret_df)
    
# df = pd.concat(results, axis=1)
# df.to_pickle(SAVE_DIR/'DeepMRM_quant_result_combined.pkl')
df = pd.read_pickle(SAVE_DIR/'DeepMRM_quant_result_combined.pkl')

sample_ids = df.columns.get_level_values(0)

rt_tables = []
for peptide_id, row in df.iterrows():
    
    elution_periods = np.array([row[(sample_id, 'end_time')] - row[(sample_id, 'start_time')] for sample_id in sample_ids])
    start_times = np.array([row[(sample_id, 'start_time')] for sample_id in sample_ids])
    end_times = np.array([row[(sample_id, 'end_time')] for sample_id in sample_ids])

    rt = (start_times + end_times)*0.5
    rt_med = np.median(rt)

    m = np.abs(rt - rt_med) < 1.5
    start_times = start_times[m]
    end_times = end_times[m]
    elution_periods = elution_periods[m]

    rt_tables.append(
        [peptide_id, np.min(start_times), np.max(end_times), rt_med, np.mean(elution_periods)]
    )
    
rt_df = pd.DataFrame(rt_tables, columns=['peptide_id', 'min_rt', 'max_rt', 'med_rt', 'avg_ep'])

from matplotlib import pyplot as plt

plt.figure(figsize=(18,8))
locs= []
for y, (_, row) in enumerate(rt_df.iterrows()):
    x1 = row['min_rt']
    x2 = row['max_rt']
    # x1 = row['med_rt'] - 2
    # x2 = row['med_rt'] + 2
    plt.plot( (x1, x2), (y+1, y+1), marker='*')
    locs.append(y+1)
# locs, labels = plt.yticks()
plt.xlabel('RT (min)')
plt.yticks(locs, rt_df['peptide_id'])
plt.savefig(SAVE_DIR/'rt_4wins.jpg')






rt_df['max_rt'] - rt_df['min_rt']

end_times = [row[(sample_id, 'end_time')] for sample_id in sample_ids]

for sample_id in sample_ids:
    if df.loc['PON3_YVYVADVAAK_+2y8', (sample_id, 'end_time')] > 15:
        break
    df.loc[:, 'CCA4']


    