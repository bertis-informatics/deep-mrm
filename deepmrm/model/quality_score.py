import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from itertools import combinations

from mstorch.utils.cuda_memory import recursive_copy_to_device
from deepmrm.constant import TARGET_KEY, XIC_KEY
from deepmrm.transform.batch_xic import BatchXics
from deepmrm.model.resnet import ResNet1x3, BasicBlock1x3



class QualityScorer(ResNet1x3):
    def __init__(self,
                 name,
                 task,
                 layers=[2, 2, 2, 2],
                 block=BasicBlock1x3):
        
        super(QualityScorer, self).__init__(
            layers=layers,
            block=block,
            inplanes=64, 
            conv1_height=2
        )
        
        self.name = name
        self.task = task
        self.transform = None
        self.batch_xics = BatchXics(min_transitions=2)
        
        assert task.is_classification(), "task type should be ClassificationTask"
        self.criterion = torch.nn.BCELoss()

    def set_transform(self, transform):
        """
        A transform is set by Trainer, when training is started
        """
        self.transform = transform
        return self

    @property
    def device(self):
        device = next(self.parameters()).device
        return device

    def compute_loss(self, batched_samples, metrics=None):

        xic_list = batched_samples[XIC_KEY] 
        targets = batched_samples[self.task.label_column]
        xics, targets = self.batch_xics(self.device, xic_list, targets)

        # get prediction tensor
        preds = self(xics.tensors)
        probs = preds.sigmoid()
        targets = targets.reshape(-1, 1).to(torch.float32)
        loss = self.criterion(probs, targets)

        if isinstance(metrics, dict) and self.task in metrics:
            _ = metrics[self.task](probs, targets)
        
        return loss

    
    def score_peak_group(self, xic_array, peak_boundary, max_candidates=6):
        
        num_transitions, num_points = xic_array.shape[-2:]
        if num_transitions < 2:
            # there is only one transition. Can't scoring it
            return {idx: 0 for idx in range(num_transitions)}

        # elution_period = peak_boundary[1] - peak_boundary[0]
        # peak_boundary[0] -= elution_period
        # peak_boundary[1] += elution_period
        peak_boundary = peak_boundary.astype(np.int32).clip(min=0, max=num_points)
        xic_seg_array = xic_array[:, :, peak_boundary[0]:peak_boundary[1]]

        if num_transitions > max_candidates:
            ct_idx_ = int( np.median(xic_seg_array[1, :, :].argmax(axis=1)) )
            trans_indexes = xic_seg_array[1, :, ct_idx_].argsort()[::-1][:max_candidates]
        else:
            trans_indexes = np.arange(num_transitions)

        indexes = list(combinations(trans_indexes, 2))
        
        #(N-combs, 2-channel, 2-pair, len)
        xic_tensors = torch.stack([
                        self.batch_xics.normalize(xic_seg_array[:, idx, :]) 
                            for idx in indexes
                    ]).to(self.device)
        
        logits = self(xic_tensors)
        scores = logits.sigmoid()
        rep_idx = np.unique([ix for s, ix in zip(scores, indexes) if s > 0.5])
        if len(rep_idx) > 0:
            # found quantifiable XIC pair with high confidence
            xic_rep = xic_seg_array[:, rep_idx, :].sum(axis=1, keepdims=True)
        else:
            xic_rep = xic_seg_array.sum(axis=1, keepdims=True)

        xic_input = np.concatenate((xic_seg_array, xic_rep), axis=1)
        xic_tensors2 = torch.stack([
                        self.batch_xics.normalize(xic_input[:, (i, num_transitions), :]) 
                            for i in range(num_transitions)
                    ]).to(self.device)
        
        # from deepmrm.utils.plot import plot_heavy_light_pair        
        # from matplotlib import pyplot as plt
        # plt.figure()
        # ix = rep_idx
        # plot_heavy_light_pair(np.arange(xic_array.shape[-1]), xic_array[:,ix,:], pred_bd=peak_boundary)
        # #plot_heavy_light_pair(np.arange(len(time)), xic[:, [0, 2], :], manual_bd=target_boxes, pred_bd=pred_boxes)
        # plt.xlim([st_idx-40, ed_idx+40])
        # plt.savefig('./temp/temp.jpg')

        logits = self(xic_tensors2)
        scores = logits.sigmoid()
        predictions = scores.squeeze().cpu().numpy()

        return predictions
    

    def predict(self, dataset, bd_output_df):
        """_summary_

        Args:
            ds_loader (torch.utils.data.DataLoader): DataLoader instance
            bd_output_df (pandas.DataFrame): boundary detection output

        Returns:
            _type_: _description_
        """

        if len(dataset) != bd_output_df.shape[0]:
            raise ValueError('len(dataset) != len(bd_output_df)')
        
        peak_quality_results = []
        self.eval()
        with torch.no_grad():
            for i in tqdm(range(len(dataset))):
                sample = dataset[i]
                idx = sample[dataset.metadata_index_name]
                xic_array = sample[XIC_KEY]
                peak_quality_results.append(
                        np.array([
                            self.score_peak_group(xic_array, peak_boundary) 
                                for peak_boundary in bd_output_df.loc[idx, 'boxes']
                        ], dtype=np.float32)
                    )
        bd_output_df['peak_quality'] = peak_quality_results
        return bd_output_df


    def evaluate(self, testset_loader):
        
        assert isinstance(testset_loader, DataLoader)

        def convert_to_list(outputs):
            if torch.is_tensor(outputs):
                return outputs.cpu().detach().numpy().tolist()
            return outputs

        index_name = testset_loader.dataset.metadata_index_name

        dfs = []
        self.eval()
        with torch.no_grad():
            for batched_samples_ in tqdm(testset_loader, position=0):
                # index_values = batched_samples_[index_name]
                batched_samples = recursive_copy_to_device(batched_samples_, self.device)
                
                xic_list = batched_samples[XIC_KEY] 
                label_list = batched_samples[self.task.label_column]

                xics = self.batch_xics(self.device, xic_list)                

                model_outputs = self(xics.tensors)

                predictions = {
                        self.task.prediction_column: model_outputs.sigmoid().flatten(),
                        self.task.label_column: label_list
                    }
                #predictions = self._convert_model_output(model_outputs)
                predictions[index_name] = batched_samples[index_name]
                
                # batch_result = {index_name: index_values.numpy().tolist()}
                batch_result = {
                    col: convert_to_list(preds_) for col, preds_ in predictions.items()
                }

                # update with labeled data
                if 'manual_quality' in batched_samples:
                    batch_result['manual_quality'] = convert_to_list(batched_samples['manual_quality'])
                    # for k in ['boxes', 'labels']:
                    #     batch_result[f'target_{k}'] = [
                    #         target[k] for target in batched_samples[TARGET_KEY]]                

                dfs.append(pd.DataFrame.from_dict(batch_result))

        result_df = pd.concat(dfs).set_index(index_name)
        return result_df