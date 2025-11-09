"""
Test open set detection metrics calculation

This script is used to verify that open set detection metrics (U-Recall, WI, A-OSE) calculations are correct.
"""

import numpy as np
import sys
from pathlib import Path

# Add project path
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics.utils.openset_metrics import OpenSetDetectionMetrics


def test_basic_metrics():
    """Test basic metrics calculation"""
    print("=" * 60)
    print("Test 1: Basic Metrics Calculation")
    print("=" * 60)
    
    # Setup
    known_classes = [0, 1, 2]
    unknown_class_id = 999
    
    # Create metrics calculator
    metrics = OpenSetDetectionMetrics(known_classes, unknown_class_id, iou_threshold=0.5)
    
    # Test data
    # Scenario: 3 predictions, 3 ground truth labels, perfect match
    predictions = np.array([
        [10, 10, 50, 50, 0.9, 0],      # Known class 0, correct
        [60, 60, 100, 100, 0.8, 999],  # Predicted as unknown, correct (GT is class 5-unknown)
        [110, 110, 150, 150, 0.7, 1],  # Known class 1, correct
    ])
    
    ground_truth = np.array([
        [10, 10, 50, 50, 0],           # Known class 0
        [60, 60, 100, 100, 5],         # Unknown class 5 (not in known_classes)
        [110, 110, 150, 150, 1],       # Known class 1
    ])
    
    # Update metrics
    metrics.update(predictions, ground_truth)
    
    # Calculate results
    results = metrics.compute()
    
    print("\nPredictions:")
    print(predictions)
    print("\nGround Truth:")
    print(ground_truth)
    
    print(f"\nResults:")
    print(f"  U-Recall (expected: 1.0):     {results['U-Recall']:.4f}")
    print(f"  WI (expected: 0.0):           {results['WI']:.4f}")
    print(f"  A-OSE (expected: 0.0):        {results['A-OSE']:.4f}")
    print(f"  Precision(Known) (expected: 1.0): {results['Precision(Known)']:.4f}")
    print(f"  Recall(Known) (expected: 1.0):    {results['Recall(Known)']:.4f}")
    
    print("\nDetailed Statistics:")
    print(f"  Total Known GT: {metrics.total_known_gt}")
    print(f"  Total Unknown GT: {metrics.total_unknown_gt}")
    print(f"  TP Known: {metrics.tp_known}")
    print(f"  TP Unknown: {metrics.tp_unknown}")
    
    assert results['U-Recall'] == 1.0, "U-Recall should be 1.0"
    assert results['WI'] == 0.0, "WI should be 0.0"
    assert results['A-OSE'] == 0.0, "A-OSE should be 0.0"
    
    print("\n✅ Test 1 Passed!")
    return True


def test_unknown_misclassified():
    """Test scenario where unknown objects are misclassified"""
    print("\n" + "=" * 60)
    print("Test 2: Unknown Objects Misclassified as Known Classes")
    print("=" * 60)
    
    known_classes = [0, 1, 2]
    unknown_class_id = 999
    
    metrics = OpenSetDetectionMetrics(known_classes, unknown_class_id, iou_threshold=0.5)
    
    # Scenario: Unknown object misclassified as known class
    predictions = np.array([
        [10, 10, 50, 50, 0.9, 0],      # Known class 0, correct
        [60, 60, 100, 100, 0.8, 1],    # Misclassification: should be unknown, but predicted as class 1
        [110, 110, 150, 150, 0.7, 1],  # Known class 1, correct
    ])
    
    ground_truth = np.array([
        [10, 10, 50, 50, 0],           # Known class 0
        [60, 60, 100, 100, 5],         # Unknown class 5
        [110, 110, 150, 150, 1],       # Known class 1
    ])
    
    metrics.update(predictions, ground_truth)
    results = metrics.compute()
    
    print(f"\nResults:")
    print(f"  U-Recall (expected: 0.0):     {results['U-Recall']:.4f}")
    print(f"  WI (expected: 0.0):           {results['WI']:.4f}")
    print(f"  A-OSE (expected: > 0):        {results['A-OSE']:.4f}")
    
    # U-Recall should be 0, as unknown object was not identified as unknown
    assert results['U-Recall'] == 0.0, "U-Recall should be 0.0"
    assert results['A-OSE'] > 0, "A-OSE should be greater than 0"
    
    print("\n✅ Test 2 Passed!")
    return True


def test_known_misclassified_as_unknown():
    """Test scenario where known objects are misclassified as unknown"""
    print("\n" + "=" * 60)
    print("Test 3: Known Objects Misclassified as Unknown (WI Test)")
    print("=" * 60)
    
    known_classes = [0, 1, 2]
    unknown_class_id = 999
    
    metrics = OpenSetDetectionMetrics(known_classes, unknown_class_id, iou_threshold=0.5)
    
    # Scenario: Known object misclassified as unknown
    predictions = np.array([
        [10, 10, 50, 50, 0.9, 0],      # Known class 0, correct
        [60, 60, 100, 100, 0.8, 999],  # Misclassification: should be class 1, but predicted as unknown
        [110, 110, 150, 150, 0.7, 1],  # Known class 1, correct (different location)
    ])
    
    ground_truth = np.array([
        [10, 10, 50, 50, 0],           # Known class 0
        [60, 60, 100, 100, 1],         # Known class 1
        [110, 110, 150, 150, 1],       # Known class 1
    ])
    
    metrics.update(predictions, ground_truth)
    results = metrics.compute()
    
    print(f"\nResults:")
    print(f"  U-Recall (expected: N/A, no unknown GT): {results['U-Recall']:.4f}")
    print(f"  WI (expected: > 0):           {results['WI']:.4f}")
    print(f"  A-OSE (expected: > 0):        {results['A-OSE']:.4f}")
    
    # WI should be greater than 0, as there are known objects misclassified as unknown
    assert results['WI'] > 0, f"WI should be greater than 0, but got {results['WI']}"
    
    print("\n✅ Test 3 Passed!")
    return True


def test_mixed_scenario():
    """Test mixed scenario"""
    print("\n" + "=" * 60)
    print("Test 4: Mixed Scenario (Multiple Error Types)")
    print("=" * 60)
    
    known_classes = [0, 1, 2]
    unknown_class_id = 999
    
    metrics = OpenSetDetectionMetrics(known_classes, unknown_class_id, iou_threshold=0.5)
    
    # More complex scenario
    predictions = np.array([
        [10, 10, 50, 50, 0.9, 0],       # Correct: known class 0
        [60, 60, 100, 100, 0.8, 999],   # Correct: identified as unknown
        [110, 110, 150, 150, 0.7, 999], # Error: known class marked as unknown
        [160, 160, 200, 200, 0.6, 1],   # Error: unknown classified as known
        [210, 210, 250, 250, 0.5, 2],   # Correct: known class 2
    ])
    
    ground_truth = np.array([
        [10, 10, 50, 50, 0],            # Known class 0
        [60, 60, 100, 100, 5],          # Unknown class 5
        [110, 110, 150, 150, 1],        # Known class 1
        [160, 160, 200, 200, 6],        # Unknown class 6
        [210, 210, 250, 250, 2],        # Known class 2
    ])
    
    metrics.update(predictions, ground_truth)
    results = metrics.compute()
    
    print("\nDetailed Information:")
    print(metrics.get_summary())
    
    # Verify calculation logic
    # Total GT: 3 known + 2 unknown = 5
    # TP known: 2 (class 0 and 2 correct)
    # FN known: 1 (class 1 marked as unknown)
    # TP unknown: 1 (one unknown correctly identified)
    # FN unknown: 1 (one unknown misclassified as known)
    
    expected_u_recall = 1.0 / 2.0  # 1 correctly identified / 2 unknown GT = 0.5
    expected_wi = 1.0 / 3.0        # 1 known marked as unknown / 3 known GT ≈ 0.333
    
    print(f"\nExpected Values:")
    print(f"  U-Recall ≈ {expected_u_recall:.4f}")
    print(f"  WI ≈ {expected_wi:.4f}")
    print(f"\nActual Values:")
    print(f"  U-Recall = {results['U-Recall']:.4f}")
    print(f"  WI = {results['WI']:.4f}")
    print(f"  A-OSE = {results['A-OSE']:.4f}")
    
    assert abs(results['U-Recall'] - expected_u_recall) < 0.01, "U-Recall does not match expected value"
    assert abs(results['WI'] - expected_wi) < 0.01, "WI does not match expected value"
    
    print("\n✅ Test 4 Passed!")
    return True


def test_no_detections():
    """Test scenario with no detections"""
    print("\n" + "=" * 60)
    print("Test 5: No Detection Results")
    print("=" * 60)
    
    known_classes = [0, 1, 2]
    unknown_class_id = 999
    
    metrics = OpenSetDetectionMetrics(known_classes, unknown_class_id, iou_threshold=0.5)
    
    # Empty predictions
    predictions = np.zeros((0, 6))
    
    ground_truth = np.array([
        [10, 10, 50, 50, 0],
        [60, 60, 100, 100, 5],
    ])
    
    metrics.update(predictions, ground_truth)
    results = metrics.compute()
    
    print(f"\nResults:")
    print(f"  U-Recall: {results['U-Recall']:.4f}")
    print(f"  WI: {results['WI']:.4f}")
    print(f"  A-OSE: {results['A-OSE']:.4f}")
    
    # All should be FN
    assert results['U-Recall'] == 0.0, "U-Recall should be 0 when no detections"
    assert results['WI'] == 0.0, "WI should be 0 when no detections"
    assert results['A-OSE'] == 1.0, "A-OSE should be 1.0 when no detections (all are FN)"
    
    print("\n✅ Test 5 Passed!")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "🚀 " * 20)
    print("Starting Open Set Detection Metrics Testing")
    print("🚀 " * 20 + "\n")
    
    tests = [
        test_basic_metrics,
        test_unknown_misclassified,
        test_known_misclassified_as_unknown,
        test_mixed_scenario,
        test_no_detections,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n❌ Test failed: {test.__name__}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Open set detection metrics calculation is working correctly.")
    else:
        print("\n⚠️ Some tests failed, please check the code.")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
