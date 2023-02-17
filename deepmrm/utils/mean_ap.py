from torchmetrics.detection.mean_ap import MeanAveragePrecision, MAPMetricResults, MARMetricResults
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from torch import IntTensor, Tensor


class MeanAveragePrecisionRecall(MeanAveragePrecision):

    def __init__(
        self,
        box_format: str = "xyxy",
        iou_thresholds: Optional[List[float]] = None,
        rec_thresholds: Optional[List[float]] = None,
        max_detection_thresholds: Optional[List[int]] = None,
        class_metrics: bool = False,
        compute_on_step: Optional[bool] = None,
        **kwargs: Dict[str, Any],
    ) -> None:  # type: ignore
        super().__init__(box_format, 
                        iou_thresholds, rec_thresholds, 
                        max_detection_thresholds, class_metrics,
                        compute_on_step, **kwargs)
        self.bbox_area_ranges = {
            "all": (0**2, int(1e5**2)),
        }


    def _summarize_results(self, precisions: Tensor, recalls: Tensor) -> Tuple[MAPMetricResults, MARMetricResults]:
        
        # map_metrics, mar_metrics = super()._summarize_results(precisions, recalls)
        map_metrics = MAPMetricResults()
        results = dict(precision=precisions, recall=recalls)

        for max_det in self.max_detection_thresholds:
            map_metrics[f'map_det{max_det}'] = self._summarize(results, True, max_dets=max_det)
                
            for th in self.iou_thresholds:
                map_metrics[f'map_{int(th*100)}_det{max_det}'] = self._summarize(
                            results, 
                            True, 
                            iou_threshold=th, 
                            max_dets=max_det)
        
        # map_metrics.map_small = self._summarize(results, True, area_range="small", max_dets=last_max_det_thr)
        # map_metrics.map_medium = self._summarize(results, True, area_range="medium", max_dets=last_max_det_thr)
        # map_metrics.map_large = self._summarize(results, True, area_range="large", max_dets=last_max_det_thr)
        mar_metrics = MARMetricResults()
        for max_det in self.max_detection_thresholds:
            mar_metrics[f"mar_{max_det}"] = self._summarize(results, False, max_dets=max_det)
        
        # mar_metrics.mar_small = self._summarize(results, False, area_range="small", max_dets=last_max_det_thr)
        # mar_metrics.mar_medium = self._summarize(results, False, area_range="medium", max_dets=last_max_det_thr)
        # mar_metrics.mar_large = self._summarize(results, False, area_range="large", max_dets=last_max_det_thr)

        return map_metrics, mar_metrics



