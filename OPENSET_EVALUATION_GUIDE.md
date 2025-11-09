# Open Set Detection Evaluation Guide

This guide introduces how to use the added open set detection evaluation features to assess the model's ability to detect unknown categories.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Evaluation Metrics Explanation](#evaluation-metrics-explanation)
3. [Quick Start](#quick-start)
4. [Usage Methods](#usage-methods)
5. [Code Examples](#code-examples)
6. [FAQ](#faq)

---

## Overview

Open Set Detection evaluates model performance when facing categories not seen during training. This project implements three key metrics:

- **U-Recall (Unknown Recall)**: Unknown category recall rate
- **WI (Wilderness Impact)**: Wilderness impact
- **A-OSE (Absolute Open Set Error)**: Absolute open set error

### File Structure

```
OW-DLN/
├── ultralytics/
│   └── utils/
│       ├── metrics.py              # Extended metrics module
│       └── openset_metrics.py      # Open set detection metrics implementation
├── evaluate_openset.py             # Evaluation script
└── OPENSET_EVALUATION_GUIDE.md     # This document
```

---

## Evaluation Metrics Explanation

### 1. U-Recall (Unknown Recall) - Unknown Category Recall Rate

**Definition**: The proportion of real unknown objects correctly identified as "unknown"

**Formula**: 
```
U-Recall = TP_unknown / (TP_unknown + FN_unknown)
```
                                        
**Meaning**: 
- **High U-Recall**: The model can effectively identify unseen categories
- **Low U-Recall**: The model tends to misclassify unknown objects as known categories

**Value Range**: [0, 1], higher is better

---

### 2. WI (Wilderness Impact) - Wilderness Impact

**Definition**: The proportion of known category objects misclassified as "unknown"

**Formula**: 
```
WI = FP_unknown / Total_Known_GT
```

Where `FP_unknown` is the number of known objects misclassified as unknown

**Meaning**: 
- **Low WI**: The model does not overly affect the recognition of known categories when identifying unknown objects
- **High WI**: The model is too conservative, marking many known objects as unknown

**Value Range**: [0, 1], lower is better

---

### 3. A-OSE (Absolute Open Set Error) - Absolute Open Set Error

**Definition**: Comprehensive measure of overall error rate in open set detection

**Formula**: 
```
A-OSE = (FP_known + FN_known + FP_unknown + FN_unknown) / Total_GT
```

**Meaning**: 
- Combines all errors for both known and unknown categories
- Provides overall assessment of open set detection performance

**Value Range**: [0, ∞), lower is better

**Components**:
- `FP_known`: False positives for known categories (incorrect detection or classification errors)
- `FN_known`: False negatives for known categories (missed detections)
- `FP_unknown`: False positives for unknown categories (known objects misidentified as unknown)
- `FN_unknown`: False negatives for unknown categories (unknown objects not identified as unknown)

---

## Quick Start

### Prerequisites

Ensure required dependencies are installed:

```bash
pip install ultralytics numpy torch
```

### Basic Usage

```bash
python evaluate_openset.py \
    --model path/to/your/model.pt \
    --data path/to/your/data.yaml \
    --known-classes 0 1 2 3 4 \
    --unknown-id 999
```

---

## Usage Methods

### Method 1: Command Line Usage

#### Complete Parameter Explanation

```bash
python evaluate_openset.py \
    --model yolo11n.pt \              # Model weights path
    --data coco8.yaml \               # Dataset configuration file
    --known-classes 0 1 2 3 4 \       # Known class ID list
    --unknown-id 999 \                # Unknown class ID identifier
    --conf 0.001 \                    # Confidence threshold
    --iou 0.6 \                       # IoU threshold
    --split val \                     # Dataset split (train/val/test)
    --save openset_results.txt        # Results save path
```

#### Parameter Details

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `--model` | str | ✓ | - | YOLO model weights file path |
| `--data` | str | ✓ | - | Dataset YAML configuration file path |
| `--known-classes` | int[] | ✓ | - | ID list of known classes |
| `--unknown-id` | int | ✗ | -1 | Class ID representing unknown objects |
| `--conf` | float | ✗ | 0.001 | Detection confidence threshold |
| `--iou` | float | ✗ | 0.6 | IoU threshold for NMS |
| `--split` | str | ✗ | val | Dataset split to use |
| `--save` | str | ✗ | openset_results.txt | Results save path |

---

### Method 2: Python Code Invocation

#### Basic Usage

```python
from ultralytics.utils.openset_metrics import OpenSetDetectionMetrics
import numpy as np

# 1. Define known classes and unknown class ID
known_classes = [0, 1, 2]  # e.g.: person, bicycle, car
unknown_class_id = 999      # Used to identify unknown objects

# 2. Create metrics calculator
metrics = OpenSetDetectionMetrics(
    known_classes=known_classes,
    unknown_class_id=unknown_class_id,
    iou_threshold=0.5
)

# 3. Prepare predictions and ground truth labels
# Prediction format: [x1, y1, x2, y2, confidence, class]
predictions = np.array([
    [10, 10, 50, 50, 0.9, 0],      # Predicted as class 0 (known)
    [60, 60, 100, 100, 0.8, 999],  # Predicted as unknown
    [110, 110, 150, 150, 0.7, 1],  # Predicted as class 1 (known)
])

# Ground truth format: [x1, y1, x2, y2, class]
ground_truth = np.array([
    [10, 10, 50, 50, 0],           # True class 0 (known)
    [60, 60, 100, 100, 5],         # True class 5 (unknown)
    [110, 110, 150, 150, 1],       # True class 1 (known)
])

# 4. Update metrics
metrics.update(predictions, ground_truth)

# 5. Compute and print results
results = metrics.compute()
print(metrics.get_summary())

# 6. Access specific metrics
print(f"U-Recall: {results['U-Recall']:.4f}")
print(f"WI: {results['WI']:.4f}")
print(f"A-OSE: {results['A-OSE']:.4f}")
```

#### Advanced Usage: Complete Validation Process

```python
from evaluate_openset import OpenSetValidator

# Create validator
validator = OpenSetValidator(
    model='path/to/model.pt',       # Model path
    data='path/to/data.yaml',       # Data configuration
    known_classes=[0, 1, 2, 3, 4],  # Known classes
    unknown_class_id=999,            # Unknown identifier
    conf=0.001,                      # Confidence threshold
    iou=0.6                          # IoU threshold
)

# Run validation
results = validator.validate(split='val')

# Save results
validator.save_results(results, save_path='my_results.txt')

# Access results
print("Standard Metrics:", results['standard_metrics'])
print("OpenSet Metrics:", results['openset_metrics'])
```

---

## Code Examples

### Example 1: Single Image Evaluation

```python
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.openset_metrics import OpenSetDetectionMetrics

# Load model
model = YOLO('yolo11n.pt')

# Define known classes (assume first 10 COCO classes are known)
known_classes = list(range(10))
unknown_class_id = 999

# Create metrics calculator
metrics = OpenSetDetectionMetrics(known_classes, unknown_class_id)

# Predict single image
results = model.predict('image.jpg')

# Extract prediction results
boxes = results[0].boxes
predictions = np.concatenate([
    boxes.xyxy.cpu().numpy(),
    boxes.conf.cpu().numpy().reshape(-1, 1),
    boxes.cls.cpu().numpy().reshape(-1, 1)
], axis=1)

# Prepare ground truth (needs to be loaded from dataset)
# Format: [x1, y1, x2, y2, class]
ground_truth = np.array([
    [100, 100, 200, 200, 0],    # Known class
    [300, 300, 400, 400, 15],   # Unknown class (not in known_classes)
])

# Update and compute metrics
metrics.update(predictions, ground_truth)
results = metrics.compute()

print(f"U-Recall: {results['U-Recall']:.4f}")
print(f"WI: {results['WI']:.4f}")
print(f"A-OSE: {results['A-OSE']:.4f}")
```

### Example 2: Batch Image Evaluation

```python
from ultralytics.utils.openset_metrics import OpenSetDetectionMetrics
from pathlib import Path

# Initialize
known_classes = [0, 1, 2, 3, 4]
unknown_class_id = -1
metrics = OpenSetDetectionMetrics(known_classes, unknown_class_id)

# Iterate through dataset
image_paths = list(Path('dataset/images').glob('*.jpg'))
label_paths = [Path('dataset/labels') / f"{p.stem}.txt" for p in image_paths]

for img_path, lbl_path in zip(image_paths, label_paths):
    # Get predictions
    results = model.predict(str(img_path))
    predictions = extract_predictions(results)  # Custom function
    
    # Load labels
    ground_truth = load_labels(lbl_path)  # Custom function
    
    # Update metrics
    metrics.update(predictions, ground_truth)

# Final results
final_results = metrics.compute()
print(metrics.get_summary())
```

### Example 3: Integration with Existing Validation

```python
from ultralytics import YOLO
from ultralytics.utils.openset_metrics import OpenSetDetectionMetrics

# Standard validation
model = YOLO('model.pt')
standard_results = model.val(data='data.yaml')

# Open set validation
from evaluate_openset import OpenSetValidator

openset_validator = OpenSetValidator(
    model=model,
    data='data.yaml',
    known_classes=[0, 1, 2],
    unknown_class_id=999
)

openset_results = openset_validator.validate()

# Compare results
print("Standard mAP50:", standard_results.box.map50)
print("Open Set U-Recall:", openset_results['openset_metrics']['U-Recall'])
print("Open Set A-OSE:", openset_results['openset_metrics']['A-OSE'])
```

---

## FAQ

### Q1: How to determine the known classes list?

**A**: The known classes list should include all class IDs that the model has seen during training. For example:

```python
# If your model was trained on 5 classes
known_classes = [0, 1, 2, 3, 4]

# If you want to simulate some classes as "unknown"
# For example, only the first 3 classes are considered known
known_classes = [0, 1, 2]  # Classes 3 and 4 will be treated as unknown during testing
```

### Q2: What should the unknown class ID be set to?

**A**: The unknown class ID is what your model outputs to represent "unknown objects". Common choices:

```python
# Option 1: Use a negative number
unknown_class_id = -1

# Option 2: Use a large number not in training classes
unknown_class_id = 999

# Option 3: Use the number of classes as unknown class ID
unknown_class_id = len(known_classes)  # If there are 5 known classes, this would be 5
```

**Note**: Your model needs to be able to output this special class ID to represent unknown objects.

### Q3: How to handle unknown classes in the dataset?

**A**: In ground truth, unknown classes are those not in the `known_classes` list:

```python
known_classes = [0, 1, 2]

# Class 5 in ground truth will be automatically recognized as unknown
ground_truth = np.array([
    [10, 10, 50, 50, 0],   # Known (in known_classes)
    [60, 60, 100, 100, 5], # Unknown (not in known_classes)
])
```

### Q4: How to interpret evaluation results?

**A**: 
- **High U-Recall + Low WI**: Ideal situation, model can accurately identify unknown objects without affecting known classes
- **Low U-Recall**: Model has difficulty identifying unknown objects, may misclassify them as known classes
- **High WI**: Model is too conservative, marking many known objects as unknown
- **Low A-OSE**: Good overall open set detection performance

### Q5: How to improve open set detection performance?

**A**: Some suggestions:

1. **Use uncertainty estimation**: Add confidence thresholds or energy scores to identify unknown objects
2. **Training strategies**: Use contrastive learning, background class training, etc.
3. **Post-processing**: Mark low-confidence predictions as unknown
4. **Ensemble methods**: Combine multiple detection heads or models

```python
# Example: Simple unknown detection based on confidence
def mark_low_confidence_as_unknown(predictions, conf_threshold=0.5, unknown_id=999):
    """Mark low-confidence predictions as unknown"""
    predictions = predictions.copy()
    low_conf_mask = predictions[:, 4] < conf_threshold
    predictions[low_conf_mask, 5] = unknown_id
    return predictions
```

### Q6: Can this be used with other object detection frameworks?

**A**: Yes! The `OpenSetDetectionMetrics` class is framework-agnostic. You just need to provide predictions and labels in the correct format:

```python
# Works with any framework, as long as you convert to the correct format
# Predictions: [x1, y1, x2, y2, confidence, class]
# Ground Truth: [x1, y1, x2, y2, class]

from ultralytics.utils.openset_metrics import OpenSetDetectionMetrics

metrics = OpenSetDetectionMetrics(known_classes, unknown_class_id)
metrics.update(your_predictions, your_ground_truth)
results = metrics.compute()
```

---

## Technical Details

### Matching Algorithm

Uses greedy matching algorithm based on IoU to match predicted boxes with ground truth boxes:
1. Calculate IoU for all prediction-ground truth pairs
2. Select the pair with maximum IoU for matching
3. Remove matched boxes
4. Repeat until all IoU < threshold

### Metrics Computation Process

```
1. Collect predictions and ground truth labels
2. Distinguish known/unknown categories
3. Match predictions with ground truth boxes
4. Count TP/FP/FN
5. Calculate final metrics
```

---

## References

If you use this evaluation code, please consider citing the relevant papers:

```bibtex
@article{openset_detection,
  title={Open Set Object Detection},
  author={},
  journal={},
  year={2024}
}
```

---

## License

This code follows the AGPL-3.0 license.

---

## Contact

For questions or suggestions, please submit an Issue or contact the maintainer.

---

**Last Updated**: 2025-01-08
