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
                 block=BasicBlock1x3):
        
        super(QualityScorer, self).__init__(
            layers=[1, 1, 1, 1],
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

    # def forward(self, xic_tensors):
        # list of [Pairs, Channels, Length]
        # xic_list = batched_samples[XIC_KEY] 
        # xic_tensors = self.batch_xics(self.device, xic_list)
        # quality_scores = self.cnn(xic_tensors)        
        # outputs = {PEAK_QUALITY: quality_scores}
        # return outputs

    def compute_loss(self, batched_samples, metrics=None):

        xic_list = batched_samples[XIC_KEY] 
        targets = batched_samples[self.task.label_column]
        xic_tensors, targets = self.batch_xics(self.device, xic_list, targets)

        # get prediction tensor
        preds = self(xic_tensors)
        probs = preds.sigmoid()
        targets = targets.reshape(-1, 1).to(torch.float32)
        loss = self.criterion(probs, targets)

        if isinstance(metrics, dict) and self.task in metrics:
            _ = metrics[self.task](probs, targets)
        
        return loss

    
    def predict(self, xic_array, peak_boundary, max_candidates=6):
        
        num_transitions = xic_array.shape[1]
        if num_transitions < 2:
            # there is only one transition. Can't scoring it
            return {idx: 0 for idx in range(num_transitions)}

        st_idx, ed_idx = np.around(peak_boundary).astype(int)
        ct_idx = int( np.median(xic_array[1, :, st_idx:ed_idx].argmax(axis=1)) ) + st_idx
        trans_indexes = xic_array[1, :, ct_idx].argsort()[::-1][:max_candidates]

        indexes = list(combinations(trans_indexes, 2))
        xic_tensors = torch.stack([
                        self.batch_xics.normalize(xic_array[:, idx, st_idx:ed_idx]) 
                            for idx in indexes
                    ]).to(self.device)
        
        logits = self(xic_tensors)
        scores = logits.sigmoid()

        # Compute quality score for individual transitions
        predictions = {idx: 0 for idx in trans_indexes}
        num_trans = len(trans_indexes)
        for trans_tup, score in zip(indexes, scores):
            s = score.item()
            predictions[trans_tup[0]] += s/(num_trans-1)
            predictions[trans_tup[1]] += s/(num_trans-1)

        return predictions


    def evaluate(self, testset_loader):
        """
        Collect model outputs against testset, and 
        make results in tabular format (i.e. pandas.DataFrame)
        
        Args:
            testset_loader (torch.utils.data.DataLoader): DataLoader instance for test-set

        Returns:
            pandas.DataFrame: evaluation results in DataFrame
        """

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

                xic_tensors = self.batch_xics(self.device, xic_list)                

                model_outputs = self(xic_tensors)

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

                dfs.append(pd.DataFrame.from_dict(batch_result))

        result_df = pd.concat(dfs).set_index(index_name)
        return result_df