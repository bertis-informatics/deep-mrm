import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import torch
from torchvision import transforms as T
from itertools import combinations

from mstorch.data.mass_spec.tolerance import Tolerance, ToleranceUnit
from deepmrm import get_yaml_config, model_dir
from deepmrm.data.dataset import PRMDataset
from deepmrm.transform.make_input import MakeInput, TransitionDuplicate
from deepmrm.transform.make_target import MakePeakQualityTarget
from deepmrm.data.transition import TransitionData
from deepmrm.data import obj_detection_collate_fn
from deepmrm.transform.make_input import MakeInput
from deepmrm.data_prep import p100_dia, p100_dia_ratio


all_trans_df = p100_dia.get_transition_df()
all_quant_df = p100_dia.get_quant_df()
sample_df = p100_dia.get_sample_df()

label_df = all_quant_df.loc[:, 'Manual'].reset_index(drop=False)

# merge reference RT column
ref_rt_df = all_trans_df[['modified_sequence', 'ref_rt']].drop_duplicates('modified_sequence').reset_index(drop=True)
label_df = label_df.merge(ref_rt_df, on='modified_sequence', how='left')

# evaluation parameters
peptide_id_col = 'modified_sequence'
mz_tol = Tolerance(20, ToleranceUnit.PPM)
transition_data = TransitionData(
                        all_trans_df,
                        peptide_id_col='modified_sequence',
                        rt_col='ref_rt')

model_name = 'DeepMRM_PQ'
model_path = Path(model_dir) / f'{model_name}.pth'
model = torch.load(model_path)


# top intense 8 transitions 
n = 8
n_select = 3

# list( combinations(range(n), 2) )

comb_score_matrix = np.zeros((n, n), dtype=np.float32)

comb_idx_matrix = np.arange(64).reshape((8, 8))
mask = comb_score_matrix % 3 == 0
comb_score_matrix[~mask] = 0
trans_idx_ordered = np.argsort(comb_score_matrix.sum(axis=0))[::-1]
trans_idx_ordered[:n_select]

# input:
#    xic_tensor, 
# output: 
#   transition indexes
#   averaged quantifiability score <------- interpretation

output_dfs = []
for mzml_idx, row in sample_df.iterrows():
    mzml_path = p100_dia.MZML_DIR / row['mzml_file']
    save_path = p100_dia.XIC_DIR / f'{mzml_path.stem}.pkl'
    sample_id = row['sample_id']

    m = (label_df['sample_id'] == sample_id) & (label_df['ratio'].notnull()) \
        & (label_df['start_time'].notnull()) & (label_df['end_time'].notnull())

    if ~np.any(m):
        continue
    
    metadata_df = label_df[m]
    
    transform = T.Compose([
                    MakeInput(ref_rt_key='ref_rt'),
                    # MakePeakQualityTarget()
                ])



    ds = PRMDataset(mzml_path, transition_data, metadata_df=metadata_df, transform=transform)
    ds.load_data(save_path)

    sample = ds[0]

    xic = sample['XIC']
    time = sample['TIME']
    start_time = sample['start_time']
    end_time = sample['end_time']

    new_boundary_idx = np.interp([start_time, end_time], time, np.arange(len(time)))
    st_idx, ed_idx = np.around(new_boundary_idx).astype(int)


    #ct_idx = int((st_idx+ed_idx)*0.5)
    ct_idx = int( np.median(xic[1, :, st_idx:ed_idx].argmax(axis=1)) ) + st_idx

    time[ct_idx]

    xic[1, :, ct_idx].max()

max_candidates = 6
trans_indexes = xic[1, :, ct_idx].argsort()[::-1][:max_candidates]

score_matrix = pd.DataFrame(0, 
                    index=trans_indexes, 
                    columns=trans_indexes, 
                    dtype=np.float32)

indexes = list(combinations(trans_indexes, 2))
xic_stacked = np.stack([
                xic[:, idx, st_idx:ed_idx] for idx in indexes
            ])

self = model

# prediction step
xic_tensors = self.batch_xics(self.device, xic_stacked)
model_outputs = model(xic_tensors)

for trans_tup, score in zip(indexes, model_outputs.sigmoid()):
    s = score.item()
    score_matrix.at[trans_tup[0], trans_tup[1]] = s
    score_matrix.at[trans_tup[1], trans_tup[0]] = s

predictions = score_matrix.mean(axis=1).to_dict()





    








    # def predict(self, sample_instance, input_transformed=False):
        
    #     model_inputs = self.prepare_input(sample_instance, input_transformed)
    #     model_outputs = self(model_inputs)

    #     prediction_result = dict()
    #     classes_ = self.task.labels
    #     pred = model_outputs[self.output_key][0]
    #     _, pred_label = torch.max(pred, dim=0)
    #     probs = torch.softmax(pred, dim=0)
    #     label_predicted = pred_label.item()

    #     pred = {
    #         'predicted_class': label_predicted
    #     }
    #     pred.update({
    #             f'probability_of_{class_}': prob.item()
    #                 for class_, prob in zip(classes_, probs)
    #     })

    #     prediction_result[self.task.label_column] = pred

    #     return prediction_result



from matplotlib import pyplot as plt

ls = 'solid'
fig, axs = plt.subplots(2, sharex=True)
for i in range(xic.shape[0]):
    # for j in range(xic.shape[1]):
    for j in trans_indexes:
        # frag_quality = sample[f'manual_frag_quality_t{j+1}']
        # ls = 'solid' if frag_quality > 0 else 'dashed'
        axs[i].plot(time, ((-1)**(i+2))*xic[i, j, :], linestyle=ls)
    
plt.xlim([start_time, end_time])
#plt.legend(['1st transition', '2nd transition', '3rd transition'])
#axs[0].set_title(f'{patient_id}-{replicate_id}, {peptide_id}, manual_quality: {label}')
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('./temp/dia_example.jpg')

    



    
    
    

