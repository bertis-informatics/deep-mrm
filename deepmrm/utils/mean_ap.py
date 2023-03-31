from torchmetrics.detection.mean_ap import (
    MeanAveragePrecision, MAPMetricResults, MARMetricResults,
    COCOMetricResults
)
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor
import torch
import numpy as np


class MeanAveragePrecisionRecall(MeanAveragePrecision):

    def __init__(
        self,
        box_format: str = "xyxy",
        iou_type: str = "bbox",
        iou_thresholds: Optional[List[float]] = None,
        rec_thresholds: Optional[List[float]] = None,
        max_detection_thresholds: Optional[List[int]] = None,
        class_metrics: bool = False,
        compute_on_step: Optional[bool] = None,
        **kwargs: Dict[str, Any],
    ) -> None:  # type: ignore
        super().__init__(
                        box_format = box_format,
                        iou_type = iou_type,
                        iou_thresholds=iou_thresholds, 
                        rec_thresholds=rec_thresholds, 
                        max_detection_thresholds=max_detection_thresholds, 
                        class_metrics=class_metrics,
                        compute_on_step=compute_on_step, 
                        **kwargs)
        self.bbox_area_ranges = {
            "all": (0**2, int(1e5**2)),
        }

    def compute(self) -> dict:
        """Computes metric."""
        classes = self._get_classes()
        precisions, recalls, ious, recall_raw = self._calculate(classes)
        map_val, mar_val = self._summarize_results(precisions, recalls)

        # if class mode is enabled, evaluate metrics per class
        map_per_class_values: Tensor = torch.tensor([-1.0])
        mar_max_dets_per_class_values: Tensor = torch.tensor([-1.0])
        if self.class_metrics:
            raise NotImplementedError()
            # map_per_class_list = []
            # mar_max_dets_per_class_list = []

            # for class_idx, _ in enumerate(classes):
            #     cls_precisions = precisions[:, :, class_idx].unsqueeze(dim=2)
            #     cls_recalls = recalls[:, class_idx].unsqueeze(dim=1)
            #     cls_map, cls_mar = self._summarize_results(cls_precisions, cls_recalls)
            #     map_per_class_list.append(cls_map.map)
            #     mar_max_dets_per_class_list.append(cls_mar[f"mar_{self.max_detection_thresholds[-1]}"])

            # map_per_class_values = torch.tensor(map_per_class_list, dtype=torch.float)
            # mar_max_dets_per_class_values = torch.tensor(mar_max_dets_per_class_list, dtype=torch.float)

        precision_dict = dict()
        recall_dict = dict()
        for idx_max_det, max_det in enumerate(self.max_detection_thresholds):
            for idx_iou, th in enumerate(self.iou_thresholds):
                precision_dict[f'{int(th*100)}_det{max_det}'] = precisions[idx_iou, :, 0, 0, idx_max_det]
                recall_dict[f'{int(th*100)}_det{max_det}'] = recall_raw[idx_iou, :, 0, 0, idx_max_det]

        metrics = COCOMetricResults()
        metrics.update(map_val)
        metrics.update(mar_val)

        # these are not necessary
        # metrics.map_per_class = map_per_class_values
        # metrics[f"mar_{self.max_detection_thresholds[-1]}_per_class"] = mar_max_dets_per_class_values
        metrics['ious'] = ious
        metrics['precisions'] = precision_dict
        metrics['recalls'] = recall_dict

        return metrics


    def _summarize_results(self, precisions: Tensor, recalls: Tensor) -> Tuple[MAPMetricResults, MARMetricResults]:
        
        # map_metrics, mar_metrics = super()._summarize_results(precisions, recalls)
        map_metrics = MAPMetricResults()
        results = dict(precision=precisions, recall=recalls)
        last_max_det_thr = self.max_detection_thresholds[-1]

        for th in self.iou_thresholds:
            map_metrics[f'map_{int(np.around(th*100))}'] = self._summarize(
                                                                    results,
                                                                    True,
                                                                    iou_threshold=th,
                                                                    max_dets=last_max_det_thr)
        
        # map_metrics.map_small = self._summarize(results, True, area_range="small", max_dets=last_max_det_thr)
        # map_metrics.map_medium = self._summarize(results, True, area_range="medium", max_dets=last_max_det_thr)
        # map_metrics.map_large = self._summarize(results, True, area_range="large", max_dets=last_max_det_thr)
        mar_metrics = MARMetricResults()
        for max_det in self.max_detection_thresholds:
            for th in self.iou_thresholds:
                mar_metrics[f'mar_{int(np.around(th*100))}_det{max_det}'] = \
                                self._summarize(results, False, iou_threshold=th, max_dets=max_det)
                
            # for th in self.iou_thresholds:
            #     mar_metrics[f'mar_{int(th*100)}'] = self._summarize(
            #                         results, 
            #                         False, 
            #                         iou_threshold=th,
            #                         max_dets=last_max_det_thr)

        # for max_det in self.max_detection_thresholds:
        #     mar_metrics[f"mar_{max_det}"] = self._summarize(results, False, max_dets=max_det)
        #     for th in self.iou_thresholds:
        #         map_metrics[f'mar_{int(th*100)}_det{max_det}'] = self._summarize(
        #                     results, 
        #                     False, 
        #                     iou_threshold=th, 
        #                     max_dets=max_det)            
        
        # mar_metrics.mar_small = self._summarize(results, False, area_range="small", max_dets=last_max_det_thr)
        # mar_metrics.mar_medium = self._summarize(results, False, area_range="medium", max_dets=last_max_det_thr)
        # mar_metrics.mar_large = self._summarize(results, False, area_range="large", max_dets=last_max_det_thr)
        return map_metrics, mar_metrics


    def _summarize(
        self,
        results: Dict,
        avg_prec: bool = True,
        iou_threshold: Optional[float] = None,
        area_range: str = "all",
        max_dets: int = 100,
    ) -> Tensor:
        
        area_inds = [i for i, k in enumerate(self.bbox_area_ranges.keys()) if k == area_range]
        mdet_inds = [i for i, k in enumerate(self.max_detection_thresholds) if k == max_dets]
        if avg_prec:
            # dimension of precision: [TxRxKxAxM]
            prec = results["precision"]
            # IoU
            if iou_threshold is not None:
                thr = self.iou_thresholds.index(iou_threshold)
                prec = prec[thr, :, :, area_inds, mdet_inds]
            else:
                prec = prec[:, :, :, area_inds, mdet_inds]
        else:
            # dimension of recall: [TxKxAxM]
            prec = results["recall"]
            if iou_threshold is not None:
                thr = self.iou_thresholds.index(iou_threshold)
                # bug-fix
                # prec = prec[thr, :, :, area_inds, mdet_inds]
                prec = prec[thr, :, area_inds, mdet_inds]
            else:
                prec = prec[:, :, area_inds, mdet_inds]

        mean_prec = torch.tensor([-1.0]) if len(prec[prec > -1]) == 0 else torch.mean(prec[prec > -1])
        return mean_prec
    

    def _calculate(self, class_ids: List) -> Tuple[MAPMetricResults, MARMetricResults]:
        """Calculate the precision and recall for all supplied classes to calculate mAP/mAR.

        Args:
            class_ids:
                List of label class Ids.
        """
        img_ids = range(len(self.groundtruths))
        max_detections = self.max_detection_thresholds[-1]
        area_ranges = self.bbox_area_ranges.values()

        ious = {
            (idx, class_id): self._compute_iou(idx, class_id, max_detections)
            for idx in img_ids
            for class_id in class_ids
        }

        eval_imgs = [
            self._evaluate_image(img_id, class_id, area, max_detections, ious)
            for class_id in class_ids
            for area in area_ranges
            for img_id in img_ids
        ]

        nb_iou_thrs = len(self.iou_thresholds)
        nb_rec_thrs = len(self.rec_thresholds)
        nb_classes = len(class_ids)
        nb_bbox_areas = len(self.bbox_area_ranges)
        nb_max_det_thrs = len(self.max_detection_thresholds)
        nb_imgs = len(img_ids)
        precision = -torch.ones((nb_iou_thrs, nb_rec_thrs, nb_classes, nb_bbox_areas, nb_max_det_thrs))
        recall = -torch.ones((nb_iou_thrs, nb_classes, nb_bbox_areas, nb_max_det_thrs))
        scores = -torch.ones((nb_iou_thrs, nb_rec_thrs, nb_classes, nb_bbox_areas, nb_max_det_thrs))
        recall_raw = -torch.ones((nb_iou_thrs, nb_rec_thrs, nb_classes, nb_bbox_areas, nb_max_det_thrs))

        # move tensors if necessary
        rec_thresholds_tensor = torch.tensor(self.rec_thresholds)

        # retrieve E at each category, area range, and max number of detections
        for idx_cls, _ in enumerate(class_ids):
            for idx_bbox_area, _ in enumerate(self.bbox_area_ranges):
                for idx_max_det_thrs, max_det in enumerate(self.max_detection_thresholds):
                    recall_raw, recall, precision, scores = self.__calculate_recall_precision_scores(
                        recall_raw,
                        recall,
                        precision,
                        scores,
                        idx_cls=idx_cls,
                        idx_bbox_area=idx_bbox_area,
                        idx_max_det_thrs=idx_max_det_thrs,
                        eval_imgs=eval_imgs,
                        rec_thresholds=rec_thresholds_tensor,
                        max_det=max_det,
                        nb_imgs=nb_imgs,
                        nb_bbox_areas=nb_bbox_areas,
                    )

        return precision, recall, ious, recall_raw

    

    @staticmethod
    def __calculate_recall_precision_scores(
        recall_raw: Tensor,
        recall: Tensor,
        precision: Tensor,
        scores: Tensor,
        idx_cls: int,
        idx_bbox_area: int,
        idx_max_det_thrs: int,
        eval_imgs: list,
        rec_thresholds: Tensor,
        max_det: int,
        nb_imgs: int,
        nb_bbox_areas: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        nb_rec_thrs = len(rec_thresholds)
        idx_cls_pointer = idx_cls * nb_bbox_areas * nb_imgs
        idx_bbox_area_pointer = idx_bbox_area * nb_imgs
        # Load all image evals for current class_id and area_range
        img_eval_cls_bbox = [eval_imgs[idx_cls_pointer + idx_bbox_area_pointer + i] for i in range(nb_imgs)]
        img_eval_cls_bbox = [e for e in img_eval_cls_bbox if e is not None]
        if not img_eval_cls_bbox:
            return recall, precision, scores

        det_scores = torch.cat([e["dtScores"][:max_det] for e in img_eval_cls_bbox])

        # different sorting method generates slightly different results.
        # mergesort is used to be consistent as Matlab implementation.
        # Sort in PyTorch does not support bool types on CUDA (yet, 1.11.0)
        dtype = torch.uint8 if det_scores.is_cuda and det_scores.dtype is torch.bool else det_scores.dtype
        # Explicitly cast to uint8 to avoid error for bool inputs on CUDA to argsort
        inds = torch.argsort(det_scores.to(dtype), descending=True)
        det_scores_sorted = det_scores[inds]

        det_matches = torch.cat([e["dtMatches"][:, :max_det] for e in img_eval_cls_bbox], axis=1)[:, inds]
        det_ignore = torch.cat([e["dtIgnore"][:, :max_det] for e in img_eval_cls_bbox], axis=1)[:, inds]
        gt_ignore = torch.cat([e["gtIgnore"] for e in img_eval_cls_bbox])
        npig = torch.count_nonzero(gt_ignore == False)  # noqa: E712
        if npig == 0:
            return recall, precision, scores
        tps = torch.logical_and(det_matches, torch.logical_not(det_ignore))
        fps = torch.logical_and(torch.logical_not(det_matches), torch.logical_not(det_ignore))

        tp_sum = torch.cumsum(tps, axis=1, dtype=torch.float)
        fp_sum = torch.cumsum(fps, axis=1, dtype=torch.float)
        for idx, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
            nd = len(tp)
            rc = tp / npig
            pr = tp / (fp + tp + torch.finfo(torch.float64).eps)
            # prec = torch.zeros((nb_rec_thrs,))
            # score = torch.zeros((nb_rec_thrs,))

            recall[idx, idx_cls, idx_bbox_area, idx_max_det_thrs] = rc[-1] if nd else 0

            # Remove zigzags for AUC
            diff_zero = torch.zeros((1,), device=pr.device)
            diff = torch.ones((1,), device=pr.device)
            while not torch.all(diff == 0):

                diff = torch.clamp(torch.cat(((pr[1:] - pr[:-1]), diff_zero), 0), min=0)
                pr += diff

            inds = torch.searchsorted(rc, rec_thresholds.to(rc.device), right=False)
            num_inds = inds.argmax() if inds.max() >= nd else nb_rec_thrs
            inds = inds[:num_inds]
            
            ##### bug-fix in torch-metrics ###################################
            # never overwrite precision, scores matrices with zero values
            ##################################################################
            # prec[:num_inds] = pr[inds]
            # score[:num_inds] = det_scores_sorted[inds]
            # precision[idx, :, idx_cls, idx_bbox_area, idx_max_det_thrs] = prec
            # scores[idx, :, idx_cls, idx_bbox_area, idx_max_det_thrs] = score
            precision[idx, :num_inds, idx_cls, idx_bbox_area, idx_max_det_thrs] = pr[inds]
            recall_raw[idx, :num_inds, idx_cls, idx_bbox_area, idx_max_det_thrs] = rc[inds]
            scores[idx, :num_inds, idx_cls, idx_bbox_area, idx_max_det_thrs] = det_scores_sorted[inds]

        return recall_raw, recall, precision, scores    