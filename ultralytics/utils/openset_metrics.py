# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Open Set Detection Metrics for evaluating model's ability to detect unknown objects.

This module implements metrics for Open Set Object Detection including:
- U-Recall: Unknown class recall
- WI: Wilderness Impact  
- A-OSE: Absolute Open Set Error
"""

import numpy as np
import torch
from pathlib import Path

from ultralytics.utils import LOGGER, SimpleClass


class OpenSetDetectionMetrics:
    """
    Comprehensive metrics calculator for Open Set Object Detection.
    
    This class computes specialized metrics for evaluating a model's ability to:
    1. Detect and classify known objects correctly
    2. Identify unknown objects as "unknown"
    3. Avoid misclassifying known objects as unknown
    
    Attributes:
        known_classes (list): List of known class indices (e.g., [0, 1, 2])
        unknown_class_id (int): Class ID used to represent unknown/novel objects
        iou_threshold (float): IoU threshold for matching predictions with ground truth
        
    Metrics:
        U-Recall: Proportion of unknown objects correctly identified as unknown
        WI (Wilderness Impact): Proportion of known objects incorrectly identified as unknown
        A-OSE (Absolute Open Set Error): Overall open set detection error rate
    """
    
    def __init__(self, known_classes, unknown_class_id, iou_threshold=0.5):
        """
        Initialize OpenSetDetectionMetrics calculator.
        
        Args:
            known_classes (list): List of known class IDs
            unknown_class_id (int): ID representing unknown/novel classes
            iou_threshold (float): IoU threshold for detection matching
        """
        self.known_classes = np.array(known_classes)
        self.unknown_class_id = unknown_class_id
        self.iou_threshold = iou_threshold
        
        # Counters for computing metrics
        self.reset()
        
    def reset(self):
        """Reset all metric counters."""
        self.tp_known = 0  # True Positives for known classes
        self.fp_known = 0  # False Positives for known classes
        self.fn_known = 0  # False Negatives for known classes
        
        self.tp_unknown = 0  # Unknown objects correctly identified as unknown
        self.fn_unknown = 0  # Unknown objects not identified as unknown
        self.fp_unknown = 0  # Known objects misclassified as unknown
        
        self.total_known_gt = 0  # Total known ground truth objects
        self.total_unknown_gt = 0  # Total unknown ground truth objects
        
    def update(self, predictions, ground_truth, iou_matrix=None):
        """
        Update metrics with a batch of predictions and ground truth.
        
        Args:
            predictions (np.ndarray): Predicted boxes and classes, shape (N, 6) [x1, y1, x2, y2, conf, class]
            ground_truth (np.ndarray): Ground truth boxes and classes, shape (M, 5) [x1, y1, x2, y2, class]
            iou_matrix (np.ndarray, optional): Pre-computed IoU matrix of shape (M, N). If None, will compute.
        """
        if len(ground_truth) == 0 and len(predictions) == 0:
            return
            
        # Extract classes
        if len(predictions) > 0:
            pred_boxes = predictions[:, :4]
            pred_classes = predictions[:, 5].astype(int)
            pred_conf = predictions[:, 4]
        else:
            pred_boxes = np.zeros((0, 4))
            pred_classes = np.array([], dtype=int)
            pred_conf = np.array([])
            
        if len(ground_truth) > 0:
            gt_boxes = ground_truth[:, :4]
            gt_classes = ground_truth[:, 4].astype(int)
        else:
            gt_boxes = np.zeros((0, 4))
            gt_classes = np.array([], dtype=int)
        
        # Compute IoU if not provided
        if iou_matrix is None and len(gt_boxes) > 0 and len(pred_boxes) > 0:
            iou_matrix = self._box_iou_numpy(gt_boxes, pred_boxes)
        elif iou_matrix is None:
            iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
        
        # Separate known and unknown ground truth
        known_mask_gt = np.isin(gt_classes, self.known_classes)
        unknown_mask_gt = ~known_mask_gt
        
        # Separate known and unknown predictions
        unknown_mask_pred = pred_classes == self.unknown_class_id
        known_mask_pred = ~unknown_mask_pred
        
        # Count ground truth
        n_known_gt = known_mask_gt.sum()
        n_unknown_gt = unknown_mask_gt.sum()
        
        self.total_known_gt += n_known_gt
        self.total_unknown_gt += n_unknown_gt
        
        # Match predictions to ground truth using Hungarian matching (greedy approximation)
        matches = self._match_predictions(iou_matrix, gt_classes, pred_classes)
        
        matched_gt = set()
        matched_pred = set()
        
        # Process matches
        for gt_idx, pred_idx in matches:
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
            
            gt_cls = gt_classes[gt_idx]
            pred_cls = pred_classes[pred_idx]
            
            is_gt_known = gt_cls in self.known_classes
            is_pred_unknown = pred_cls == self.unknown_class_id
            
            if is_gt_known:
                # Ground truth is known class
                if is_pred_unknown:
                    # Known GT misclassified as unknown (counts for WI)
                    self.fp_unknown += 1
                    self.fn_known += 1  # Also a false negative for known class
                elif pred_cls == gt_cls:
                    # Correct known class prediction
                    self.tp_known += 1
                else:
                    # Wrong known class (misclassification)
                    self.fn_known += 1
                    self.fp_known += 1
            else:
                # Ground truth is unknown class
                if is_pred_unknown:
                    # Unknown GT correctly identified as unknown
                    self.tp_unknown += 1
                else:
                    # Unknown GT misclassified as known
                    self.fn_unknown += 1
                    self.fp_known += 1
        
        # Process unmatched ground truth (false negatives)
        for gt_idx in range(len(gt_classes)):
            if gt_idx not in matched_gt:
                if gt_classes[gt_idx] in self.known_classes:
                    self.fn_known += 1
                else:
                    self.fn_unknown += 1
        
        # Process unmatched predictions (false positives)
        for pred_idx in range(len(pred_classes)):
            if pred_idx not in matched_pred:
                if pred_classes[pred_idx] == self.unknown_class_id:
                    self.fp_unknown += 1
                else:
                    self.fp_known += 1
    
    def _box_iou_numpy(self, boxes1, boxes2):
        """
        Compute IoU between two sets of boxes.
        
        Args:
            boxes1 (np.ndarray): First set of boxes, shape (N, 4) [x1, y1, x2, y2]
            boxes2 (np.ndarray): Second set of boxes, shape (M, 4) [x1, y1, x2, y2]
            
        Returns:
            np.ndarray: IoU matrix of shape (N, M)
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        inter_x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0])
        inter_y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1])
        inter_x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2])
        inter_y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3])
        
        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        union_area = area1[:, None] + area2 - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        return iou
    
    def _match_predictions(self, iou_matrix, gt_classes, pred_classes):
        """
        Match predictions to ground truth using greedy matching based on IoU.
        
        Args:
            iou_matrix (np.ndarray): IoU matrix of shape (n_gt, n_pred)
            gt_classes (np.ndarray): Ground truth classes
            pred_classes (np.ndarray): Predicted classes
            
        Returns:
            list: List of (gt_idx, pred_idx) tuples representing matches
        """
        matches = []
        
        if iou_matrix.size == 0:
            return matches
        
        # Create a copy to modify
        iou_matrix = iou_matrix.copy()
        
        # Greedy matching: repeatedly find the highest IoU match above threshold
        while True:
            # Find maximum IoU
            max_iou_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[max_iou_idx]
            
            if max_iou < self.iou_threshold:
                break
            
            gt_idx, pred_idx = max_iou_idx
            matches.append((gt_idx, pred_idx))
            
            # Remove matched GT and prediction
            iou_matrix[gt_idx, :] = 0
            iou_matrix[:, pred_idx] = 0
        
        return matches
    
    def compute(self):
        """
        Compute final metrics from accumulated statistics.
        
        Returns:
            dict: Dictionary containing computed metrics:
                - u_recall: Unknown Recall
                - wi: Wilderness Impact
                - a_ose: Absolute Open Set Error
                - precision_known: Precision on known classes
                - recall_known: Recall on known classes
        """
        eps = 1e-6
        
        # U-Recall: Correctly identified unknown / Total unknown GT
        u_recall = self.tp_unknown / (self.total_unknown_gt + eps)
        
        # WI (Wilderness Impact): Known GT misclassified as unknown / Total known GT
        wi = self.fp_unknown / (self.total_known_gt + eps)
        
        # A-OSE: Total errors / Total GT
        total_errors = self.fp_known + self.fn_known + self.fp_unknown + self.fn_unknown
        total_gt = self.total_known_gt + self.total_unknown_gt
        a_ose = total_errors / (total_gt + eps)
        
        # Additional metrics for known classes
        precision_known = self.tp_known / (self.tp_known + self.fp_known + eps)
        recall_known = self.tp_known / (self.total_known_gt + eps)
        
        return {
            "U-Recall": float(u_recall),
            "WI": float(wi),
            "A-OSE": float(a_ose),
            "Precision(Known)": float(precision_known),
            "Recall(Known)": float(recall_known),
        }
    
    def get_summary(self):
        """
        Get detailed summary of metrics.
        
        Returns:
            str: Formatted string with all metrics
        """
        metrics = self.compute()
        
        summary = "Open Set Detection Metrics:\n"
        summary += "=" * 50 + "\n"
        summary += f"U-Recall (Unknown Recall):      {metrics['U-Recall']:.4f}\n"
        summary += f"WI (Wilderness Impact):         {metrics['WI']:.4f}\n"
        summary += f"A-OSE (Absolute Open Set Error): {metrics['A-OSE']:.4f}\n"
        summary += f"Precision (Known Classes):      {metrics['Precision(Known)']:.4f}\n"
        summary += f"Recall (Known Classes):         {metrics['Recall(Known)']:.4f}\n"
        summary += "=" * 50 + "\n"
        summary += f"Total Known GT:     {self.total_known_gt}\n"
        summary += f"Total Unknown GT:   {self.total_unknown_gt}\n"
        summary += f"TP Known:           {self.tp_known}\n"
        summary += f"FP Known:           {self.fp_known}\n"
        summary += f"FN Known:           {self.fn_known}\n"
        summary += f"TP Unknown:         {self.tp_unknown}\n"
        summary += f"FP Unknown:         {self.fp_unknown}\n"
        summary += f"FN Unknown:         {self.fn_unknown}\n"
        
        return summary


def evaluate_openset_detection(predictions_list, ground_truth_list, known_classes, unknown_class_id, iou_threshold=0.5):
    """
    Evaluate open set detection performance on a dataset.
    
    Args:
        predictions_list (list): List of prediction arrays, each of shape (N, 6) [x1, y1, x2, y2, conf, class]
        ground_truth_list (list): List of ground truth arrays, each of shape (M, 5) [x1, y1, x2, y2, class]
        known_classes (list): List of known class indices
        unknown_class_id (int): Class ID representing unknown objects
        iou_threshold (float): IoU threshold for matching
        
    Returns:
        dict: Dictionary with computed metrics
    """
    metrics = OpenSetDetectionMetrics(known_classes, unknown_class_id, iou_threshold)
    
    for preds, gts in zip(predictions_list, ground_truth_list):
        metrics.update(preds, gts)
    
    return metrics.compute()


# Example usage
if __name__ == "__main__":
    # Example: Evaluate open set detection
    # Assume we have 3 known classes (0, 1, 2) and unknown class ID is 999
    
    known_classes = [0, 1, 2]
    unknown_class_id = 999
    
    # Example predictions: [x1, y1, x2, y2, confidence, class]
    predictions = np.array([
        [10, 10, 50, 50, 0.9, 0],      # Known class 0
        [60, 60, 100, 100, 0.8, 999],  # Detected as unknown
        [110, 110, 150, 150, 0.7, 1],  # Known class 1
    ])
    
    # Example ground truth: [x1, y1, x2, y2, class]
    ground_truth = np.array([
        [10, 10, 50, 50, 0],      # Known class 0
        [60, 60, 100, 100, 5],    # Unknown class (not in known_classes)
        [110, 110, 150, 150, 1],  # Known class 1
    ])
    
    # Create metrics calculator
    metrics_calc = OpenSetDetectionMetrics(known_classes, unknown_class_id, iou_threshold=0.5)
    
    # Update with predictions
    metrics_calc.update(predictions, ground_truth)
    
    # Compute and print metrics
    results = metrics_calc.compute()
    print(metrics_calc.get_summary())
    
    print("\nMetrics Dictionary:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

