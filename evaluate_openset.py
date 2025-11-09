"""
Open Set Detection Evaluation Script for YOLO Models

This script evaluates a YOLO model's performance on open set detection tasks,
computing U-Recall, WI (Wilderness Impact), and A-OSE metrics.

Usage:
    python evaluate_openset.py --model path/to/model.pt --data path/to/data.yaml --known-classes 0 1 2 --unknown-id 999

Example:
    python evaluate_openset.py --model yolo11n.pt --data coco8.yaml --known-classes 0 1 2 3 4 --unknown-id -1
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.openset_metrics import OpenSetDetectionMetrics


class OpenSetValidator:
    """
    Extended validator for computing open set detection metrics.
    """
    
    def __init__(self, model, data, known_classes, unknown_class_id, conf=0.001, iou=0.6):
        """
        Initialize Open Set Validator.
        
        Args:
            model: YOLO model or path to model weights
            data: Path to dataset YAML file
            known_classes (list): List of known class indices
            unknown_class_id (int): Class ID representing unknown objects
            conf (float): Confidence threshold
            iou (float): IoU threshold for NMS and matching
        """
        if isinstance(model, (str, Path)):
            self.model = YOLO(model)
        else:
            self.model = model
            
        self.data = data
        self.known_classes = known_classes
        self.unknown_class_id = unknown_class_id
        self.conf = conf
        self.iou = iou
        
        # Initialize metrics calculator
        self.openset_metrics = OpenSetDetectionMetrics(
            known_classes=known_classes,
            unknown_class_id=unknown_class_id,
            iou_threshold=iou
        )
    
    def validate(self, split='val'):
        """
        Run validation and compute open set metrics.
        
        Args:
            split (str): Dataset split to use ('train', 'val', 'test')
            
        Returns:
            dict: Dictionary with all metrics including open set metrics
        """
        LOGGER.info("Starting Open Set Detection Evaluation...")
        LOGGER.info(f"Known classes: {self.known_classes}")
        LOGGER.info(f"Unknown class ID: {self.unknown_class_id}")
        
        # Run standard YOLO validation
        standard_metrics = self.model.val(data=self.data, split=split, conf=self.conf, iou=self.iou)
        
        # Get validation dataset
        from ultralytics.data import build_yolo_dataset, build_dataloader
        from ultralytics.cfg import get_cfg
        
        args = get_cfg(overrides={'data': self.data, 'task': 'detect'})
        
        # Build dataset
        dataset = build_yolo_dataset(
            args, 
            self.data, 
            batch=1, 
            data=self.model.trainer.data if hasattr(self.model, 'trainer') else None,
            mode='val'
        )
        
        # Reset metrics
        self.openset_metrics.reset()
        
        # Iterate through dataset and collect predictions
        LOGGER.info(f"Processing {len(dataset)} images for open set metrics...")
        
        for idx in range(len(dataset)):
            # Get image and labels
            batch = dataset[idx]
            img_path = batch.get('im_file', '')
            
            # Get ground truth
            # Labels format: [class, x_center, y_center, width, height] (normalized)
            labels = batch.get('bboxes', np.array([]))  # Shape: (N, 4) xyxy normalized
            cls = batch.get('cls', np.array([])).flatten()  # Shape: (N,)
            
            if len(labels) > 0:
                # Convert normalized xyxy to absolute coordinates
                img_shape = batch['ori_shape']  # (height, width)
                h, w = img_shape
                
                # Scale boxes
                gt_boxes = labels.copy()
                gt_boxes[:, [0, 2]] *= w  # x coordinates
                gt_boxes[:, [1, 3]] *= h  # y coordinates
                
                # Combine into ground truth array [x1, y1, x2, y2, class]
                ground_truth = np.concatenate([gt_boxes, cls.reshape(-1, 1)], axis=1)
            else:
                ground_truth = np.zeros((0, 5))
            
            # Get predictions
            results = self.model.predict(
                batch['img'], 
                conf=self.conf, 
                iou=self.iou,
                verbose=False
            )
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                pred_boxes = boxes.xyxy.cpu().numpy()  # Already in absolute coords
                pred_conf = boxes.conf.cpu().numpy()
                pred_cls = boxes.cls.cpu().numpy()
                
                # Combine into predictions array [x1, y1, x2, y2, conf, class]
                predictions = np.concatenate([
                    pred_boxes,
                    pred_conf.reshape(-1, 1),
                    pred_cls.reshape(-1, 1)
                ], axis=1)
            else:
                predictions = np.zeros((0, 6))
            
            # Update open set metrics
            self.openset_metrics.update(predictions, ground_truth)
            
            if (idx + 1) % 100 == 0:
                LOGGER.info(f"Processed {idx + 1}/{len(dataset)} images...")
        
        # Compute final metrics
        openset_results = self.openset_metrics.compute()
        
        # Print results
        LOGGER.info("\n" + "="*60)
        LOGGER.info("STANDARD YOLO METRICS")
        LOGGER.info("="*60)
        if hasattr(standard_metrics, 'results_dict'):
            for key, value in standard_metrics.results_dict.items():
                LOGGER.info(f"{key}: {value:.4f}")
        
        LOGGER.info("\n" + self.openset_metrics.get_summary())
        
        # Combine all metrics
        all_metrics = {
            'standard_metrics': standard_metrics.results_dict if hasattr(standard_metrics, 'results_dict') else {},
            'openset_metrics': openset_results
        }
        
        return all_metrics
    
    def save_results(self, results, save_path='openset_results.txt'):
        """
        Save evaluation results to file.
        
        Args:
            results (dict): Results dictionary
            save_path (str): Path to save results
        """
        save_path = Path(save_path)
        
        with open(save_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("OPEN SET DETECTION EVALUATION RESULTS\n")
            f.write("="*60 + "\n\n")
            
            f.write("Configuration:\n")
            f.write(f"Known classes: {self.known_classes}\n")
            f.write(f"Unknown class ID: {self.unknown_class_id}\n")
            f.write(f"Confidence threshold: {self.conf}\n")
            f.write(f"IoU threshold: {self.iou}\n\n")
            
            f.write("Standard YOLO Metrics:\n")
            f.write("-"*60 + "\n")
            for key, value in results.get('standard_metrics', {}).items():
                f.write(f"{key}: {value:.4f}\n")
            
            f.write("\n" + self.openset_metrics.get_summary())
        
        LOGGER.info(f"Results saved to {save_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Evaluate Open Set Detection Performance')
    
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model weights')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset YAML file')
    parser.add_argument('--known-classes', type=int, nargs='+', required=True, 
                       help='List of known class indices (e.g., 0 1 2 3)')
    parser.add_argument('--unknown-id', type=int, default=-1,
                       help='Class ID representing unknown objects (default: -1)')
    parser.add_argument('--conf', type=float, default=0.001,
                       help='Confidence threshold (default: 0.001)')
    parser.add_argument('--iou', type=float, default=0.6,
                       help='IoU threshold (default: 0.6)')
    parser.add_argument('--split', type=str, default='val',
                       help='Dataset split to use (default: val)')
    parser.add_argument('--save', type=str, default='openset_results.txt',
                       help='Path to save results (default: openset_results.txt)')
    
    args = parser.parse_args()
    
    # Create validator
    validator = OpenSetValidator(
        model=args.model,
        data=args.data,
        known_classes=args.known_classes,
        unknown_class_id=args.unknown_id,
        conf=args.conf,
        iou=args.iou
    )
    
    # Run validation
    results = validator.validate(split=args.split)
    
    # Save results
    validator.save_results(results, save_path=args.save)


if __name__ == '__main__':
    # Example usage without command line
    # Uncomment and modify the following to use directly in Python
    
    # validator = OpenSetValidator(
    #     model='yolo11n.pt',
    #     data='coco8.yaml',
    #     known_classes=[0, 1, 2, 3, 4],  # First 5 classes are "known"
    #     unknown_class_id=999,  # Use 999 to represent unknown
    #     conf=0.001,
    #     iou=0.6
    # )
    # 
    # results = validator.validate(split='val')
    # validator.save_results(results)
    
    main()

