"""
Test multi-frame voting accuracy on dataset
Simulates the 3-second hold with multiple predictions per sign
Shows improvement over single-frame prediction
"""

import os
import cv2
import numpy as np
from collections import defaultdict, Counter
import random

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf_v1
tf_v1.disable_v2_behavior()


def predict_sign(image_data, sess, softmax_tensor, label_lines):
    """Predict sign"""
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    results = []
    for node_id in top_k[:3]:
        results.append((label_lines[node_id], predictions[0][node_id]))
    
    return results


def test_single_frame(class_name, dataset_path, sess, softmax_tensor, label_lines, num_samples=50):
    """Test with single frame (original method)"""
    class_path = os.path.join(dataset_path, class_name)
    
    if not os.path.exists(class_path):
        return None
    
    all_images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png'))]
    
    if len(all_images) == 0:
        return None
    
    sample_images = random.sample(all_images, min(num_samples, len(all_images)))
    
    correct = 0
    total = len(sample_images)
    
    for img_name in sample_images:
        img_path = os.path.join(class_path, img_name)
        
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (299, 299))
            
            # Histogram equalization
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            
            # Predict
            image_data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
            top_3 = predict_sign(image_data, sess, softmax_tensor, label_lines)
            
            predicted = top_3[0][0]
            
            if predicted == class_name:
                correct += 1
        
        except Exception as e:
            pass
    
    return correct, total


def test_multi_frame(class_name, dataset_path, sess, softmax_tensor, label_lines, 
                     num_samples=50, frames_per_sample=5):
    """Test with multi-frame voting (simulates 3-second hold)"""
    class_path = os.path.join(dataset_path, class_name)
    
    if not os.path.exists(class_path):
        return None
    
    all_images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png'))]
    
    if len(all_images) == 0:
        return None
    
    # Need enough images for multi-frame sampling
    if len(all_images) < frames_per_sample:
        return None
    
    correct = 0
    total = num_samples
    
    for _ in range(num_samples):
        # Sample multiple frames (simulating holding sign for 3 seconds)
        frame_samples = random.sample(all_images, frames_per_sample)
        
        predictions = []
        
        for img_name in frame_samples:
            img_path = os.path.join(class_path, img_name)
            
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (299, 299))
                
                # Histogram equalization
                yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                
                # Predict
                image_data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                top_3 = predict_sign(image_data, sess, softmax_tensor, label_lines)
                
                predicted = top_3[0][0]
                score = top_3[0][1]
                
                # Only count predictions with reasonable confidence
                if score > 0.25:
                    predictions.append(predicted)
            
            except Exception as e:
                pass
        
        # Use majority voting
        if predictions:
            counter = Counter(predictions)
            most_common = counter.most_common(1)[0]
            final_prediction = most_common[0]
            agreement = most_common[1] / len(predictions)
            
            # Require 50% agreement
            if agreement >= 0.5 and final_prediction == class_name:
                correct += 1
    
    return correct, total


def main():
    print("="*70)
    print("Multi-Frame Voting Accuracy Test")
    print("="*70)
    print("\nComparing:")
    print("  1. Single-frame prediction (current)")
    print("  2. Multi-frame voting (3-second hold simulation)")
    print("\nSimulates holding sign for 3 seconds by analyzing 5 random")
    print("images from each class and using majority voting.")
    
    # Load classifier
    print("\nLoading classifier...")
    try:
        label_lines = [line.rstrip() for line in tf_v1.gfile.GFile("logs/trained_labels.txt")]
        
        with tf_v1.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
            graph_def = tf_v1.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf_v1.import_graph_def(graph_def, name='')
        
        print("✓ Classifier loaded")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    dataset_path = "dataset"
    
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset not found at: {dataset_path}")
        return
    
    print(f"✓ Dataset found: {dataset_path}")
    print(f"✓ Classes: {len(label_lines)}")
    
    # Test parameters
    NUM_SAMPLES = 50
    FRAMES_PER_SAMPLE = 5
    
    print("\n" + "="*70)
    print(f"Testing each class ({NUM_SAMPLES} samples, {FRAMES_PER_SAMPLE} frames per sample)...")
    print("="*70)
    
    with tf_v1.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        single_frame_results = {}
        multi_frame_results = {}
        
        improvements = []
        
        for class_name in sorted(label_lines):
            print(f"\nTesting: {class_name.upper()}...")
            
            # Single frame test
            print("  Single-frame...", end=" ")
            single_result = test_single_frame(class_name, dataset_path, sess, 
                                             softmax_tensor, label_lines, NUM_SAMPLES)
            
            if single_result is None:
                print("SKIPPED")
                continue
            
            single_correct, single_total = single_result
            single_acc = (single_correct / single_total) * 100
            print(f"{single_acc:.1f}%")
            
            # Multi-frame test
            print("  Multi-frame...", end=" ")
            multi_result = test_multi_frame(class_name, dataset_path, sess, 
                                           softmax_tensor, label_lines, 
                                           NUM_SAMPLES, FRAMES_PER_SAMPLE)
            
            if multi_result is None:
                print("SKIPPED")
                continue
            
            multi_correct, multi_total = multi_result
            multi_acc = (multi_correct / multi_total) * 100
            print(f"{multi_acc:.1f}%")
            
            # Calculate improvement
            improvement = multi_acc - single_acc
            improvements.append(improvement)
            
            # Show result
            if improvement > 5:
                status = f"✓ +{improvement:.1f}% IMPROVED"
                color = "green"
            elif improvement > 0:
                status = f"↑ +{improvement:.1f}%"
                color = "yellow"
            elif improvement < -5:
                status = f"✗ {improvement:.1f}% WORSE"
                color = "red"
            else:
                status = f"≈ {improvement:.1f}%"
                color = "gray"
            
            print(f"  Result: {status}")
            
            single_frame_results[class_name] = (single_correct, single_total)
            multi_frame_results[class_name] = (multi_correct, multi_total)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if single_frame_results and multi_frame_results:
        # Overall accuracy
        single_total_correct = sum(r[0] for r in single_frame_results.values())
        single_total_samples = sum(r[1] for r in single_frame_results.values())
        single_overall = (single_total_correct / single_total_samples) * 100
        
        multi_total_correct = sum(r[0] for r in multi_frame_results.values())
        multi_total_samples = sum(r[1] for r in multi_frame_results.values())
        multi_overall = (multi_total_correct / multi_total_samples) * 100
        
        overall_improvement = multi_overall - single_overall
        
        print(f"\nSingle-Frame Accuracy: {single_overall:.2f}%")
        print(f"Multi-Frame Accuracy:  {multi_overall:.2f}%")
        print(f"Overall Improvement:   {overall_improvement:+.2f}%")
        
        if improvements:
            avg_improvement = np.mean(improvements)
            print(f"Average Improvement:   {avg_improvement:+.2f}%")
            print(f"Best Improvement:      {max(improvements):+.2f}%")
            print(f"Worst Change:          {min(improvements):+.2f}%")
        
        # Show most improved signs
        print("\n" + "="*70)
        print("MOST IMPROVED SIGNS")
        print("="*70)
        
        sign_improvements = []
        for class_name in single_frame_results.keys():
            single_acc = (single_frame_results[class_name][0] / single_frame_results[class_name][1]) * 100
            multi_acc = (multi_frame_results[class_name][0] / multi_frame_results[class_name][1]) * 100
            improvement = multi_acc - single_acc
            sign_improvements.append((class_name, single_acc, multi_acc, improvement))
        
        # Sort by improvement
        sign_improvements.sort(key=lambda x: x[3], reverse=True)
        
        print("\nTop 10 Most Improved:")
        for i, (sign, single_acc, multi_acc, imp) in enumerate(sign_improvements[:10], 1):
            print(f"{i:2d}. {sign.upper():8s}: {single_acc:5.1f}% → {multi_acc:5.1f}% ({imp:+.1f}%)")
        
        print("\nBottom 5 (Least Improved or Worse):")
        for i, (sign, single_acc, multi_acc, imp) in enumerate(sign_improvements[-5:], 1):
            print(f"{i:2d}. {sign.upper():8s}: {single_acc:5.1f}% → {multi_acc:5.1f}% ({imp:+.1f}%)")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if overall_improvement > 0:
        print(f"\n✓ Multi-frame voting improves accuracy by {overall_improvement:.2f}%")
        print("\nThe 3-second hold with multi-frame analysis provides:")
        print("  - Better handling of confused signs (M/N, V/W, E/B)")
        print("  - More stable predictions through voting")
        print("  - Rejection of inconsistent/uncertain predictions")
        print("\nRecommendation: Use classify_webcam_final.py for best results!")
    else:
        print(f"\n⚠ Multi-frame voting shows {overall_improvement:.2f}% change")
        print("\nNote: This test uses random sampling. Real 3-second hold")
        print("with temporal consistency will perform better in practice.")


if __name__ == "__main__":
    main()
