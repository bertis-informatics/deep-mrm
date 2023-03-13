import torch
from tqdm import tqdm
import pandas as pd

from torch.utils.data import DataLoader
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, LastLevelMaxPool, LastLevelP6P7

from mstorch.utils.cuda_memory import recursive_copy_to_device
from deepmrm.model.retinanet import RetinaNet
from deepmrm.model.utils import BoundaryAnchorGenerator
from deepmrm.constant import TARGET_KEY, XIC_KEY
from deepmrm.transform.batch_xic import BatchXics
from deepmrm.model.resnet import ResNet1x3, BasicBlock1x3, Bottleneck1x3


class DummyExtraFPNBlock(ExtraFPNBlock):
    def forward(self, results, x, names):
        return results, names


class BoundaryDetector(torch.nn.Module):

    def __init__(self, 
                 name, task, num_anchors=1, returned_layers=[2, 3, 4], 
                 backbone='resnet34'):
        super().__init__()
        self.name = name
        self.task = task
        self.transform = None

        if min(returned_layers) <= 0 or max(returned_layers) >= 5:
            raise ValueError(f"Each returned layer should be in the range [1,4]. Got {returned_layers}")

        num_classes = task.num_classes
        extra_blocks = LastLevelMaxPool()

        assert backbone in ['resnet18', 'resnet34', 'resnet50']
        self.backbone_name = backbone

        if backbone == 'resnet18':
            backbone = ResNet1x3([2, 2, 2, 2], block=BasicBlock1x3)
        elif backbone == 'resnet34':
            backbone = ResNet1x3([3, 4, 6, 3], block=BasicBlock1x3)
        elif backbone == 'resnet50':
            backbone = ResNet1x3([3, 4, 6, 3], block=Bottleneck1x3)
            
        backbone = _resnet_fpn_extractor(backbone, 
                        trainable_layers=5, 
                        returned_layers=returned_layers, 
                        extra_blocks=extra_blocks)

        # additional feature map produced by extra_blocks
        if isinstance(extra_blocks, DummyExtraFPNBlock):
            rt_layers = returned_layers
        elif isinstance(extra_blocks, LastLevelMaxPool):
            rt_layers = returned_layers + [5]
        elif isinstance(extra_blocks, LastLevelP6P7):
            rt_layers = returned_layers + [5, 6]
        
        def create_anchor_sizes(anchor_size, num_anchors):
            return tuple(int(anchor_size * 2 ** (i / num_anchors)) for i in range(num_anchors))        
        
        # [2, 3, 4, 5, 6] -> [32, 64, 128, 256, 512] (see retinanet_resnet50_fpn)
        # anchor_sizes = [2**(layer_idx+4) for layer_idx in rt_layers]
        anchor_sizes = [2**(layer_idx+3) for layer_idx in rt_layers]
        anchor_sizes = tuple(create_anchor_sizes(anchor_size, num_anchors) for anchor_size in anchor_sizes)

        # sizes and aspect_ratios should have the same number of elements, and it should
        # correspond to the number of feature maps.
        anchor_generator = BoundaryAnchorGenerator(anchor_sizes)
        self.detector = RetinaNet(backbone, anchor_generator, num_classes=num_classes, num_conv_layers_in_head=3)
        self.batch_xics = BatchXics(min_transitions=1)
        
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

    def set_detections_per_img(self, detections_per_img):
        self.detector.detections_per_img = detections_per_img
        return self

    def forward(self, batched_samples):
        # list of [Pairs, Channels, Length]
        xic_list = batched_samples[XIC_KEY] 
        xic_tensors = self.batch_xics(self.device, xic_list)
        detections = self.detector.predict(xic_tensors)
        
        return detections

    def _convert_model_output(self, detections):
        # detections: List[Dict[str, Tensor]]
        prediction = detections[0]
        detections2 = {k: list() for k in prediction}
        for det in detections:
            for k, v in det.items():
                detections2[k].append(v.cpu().detach().numpy())

        return detections2

    def compute_loss(self, batched_samples, metrics=None):
        
        xic_list = batched_samples[XIC_KEY] 
        targets = batched_samples[TARGET_KEY]
        xic_tensors, targets = self.batch_xics(self.device, xic_list, targets)
        loss_dict = self.detector.compute_loss(xic_tensors, targets)
        loss = loss_dict['classification'] + loss_dict['bbox_regression']
        
        return loss


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
                
                model_outputs = self(batched_samples)
                predictions = self._convert_model_output(model_outputs)
                predictions[index_name] = batched_samples[index_name]
                
                # batch_result = {index_name: index_values.numpy().tolist()}
                batch_result = {
                    col: convert_to_list(preds_) for col, preds_ in predictions.items()
                }

                dfs.append(pd.DataFrame.from_dict(batch_result))

        result_df = pd.concat(dfs).set_index(index_name)
        return result_df