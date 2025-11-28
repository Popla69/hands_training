"""
Analyze which signs are getting confused by the classifier
Tests each sign class and shows confusion matrix
"""

import os
import cv2
import numpy as np
from collections import defaultdict
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


def test_class(class_name, dataset_path, sess, softmax_tensor, label_lines, num_samples=50):
    """Test a specific class"""
    class_path = os.path.join(dataset_path, class_name)
    
    if not os.path.exists(class_path):
        return None
    
    # Get random sample of images
    all_images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png'))]
    
    if len(all_images) == 0:
        return None
    
    sample_images = random.sample(all_images, min(num_samples, len(all_images)))
    
    results = {
        'correct': 0,
        'total': len(sample_images),
        'predictions': defaultdict(int),
        'top_3_correct': 0
    }
    
    for img_name in sample_images:
        img_path = os.path.join(class_path, img_name)
        
        try:
            # Load and preprocess
            img = cv2.imread(img_path)
            img = cv2.resize(img, (299, 299))
            
            # Histogram equalization
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            
            # Predict
            image_data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
            top_3 = predict_sign(image_data, sess, softmax_tensor, label_lines)
            
            # Record prediction
            predicted = top_3[0][0]
            results['predictions'][predicted] += 1
            
            if predicted == class_name:
                results['correct'] += 1
            
            # Check if correct answer is in top 3
            if class_name in [p[0] for p in top_3]:
                results['top_3_correct'] += 1
        
        except Exception as e:
            pass
    
    return results


def main():
    print("="*70)
    print("Sign Language Classifier - Confusion Analysis")
    print("="*70)
    
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
    
    # Test each class
    print("\n" + "="*70)
    print("Testing each sign class (50 random samples per class)...")
    print("="*70)
    
    with tf_v1.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        all_results = {}
        problem_signs = []
        
        for class_name in sorted(label_lines):
            print(f"\nTesting: {class_name.upper()}...", end=" ")
            
            results = test_class(class_name, dataset_path, sess, softmax_tensor, 
                               label_lines, num_samples=50)
            
            if results is None:
                print("SKIPPED (no images)")
                continue
            
            all_results[class_name] = results
            
            accuracy = (results['correct'] / results['total']) * 100
            top3_accuracy = (results['top_3_correct'] / results['total']) * 100
            
            # Color code based on accuracy
            if accuracy >= 80:
                status = "✓ GOOD"
            elif accuracy >= 50:
                status = "⚠ FAIR"
            else:
                status = "✗ POOR"
                problem_signs.append(class_name)
            
            print(f"{status} - Accuracy: {accuracy:.1f}% (Top-3: {top3_accuracy:.1f}%)")
            
            # Show most common wrong predictions
            if results['correct'] < results['total']:
                wrong_preds = [(k, v) for k, v in results['predictions'].items() 
                              if k != class_name]
                wrong_preds.sort(key=lambda x: x[1], reverse=True)
                
                if wrong_preds:
                    confused_with = ', '.join([f"{p[0].upper()}({p[1]})" 
                                              for p in wrong_preds[:3]])
                    print(f"  Confused with: {confused_with}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if all_results:
        total_correct = sum(r['correct'] for r in all_results.values())
        total_samples = sum(r['total'] for r in all_results.values())
        overall_accuracy = (total_correct / total_samples) * 100
        
        print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
        print(f"Total samples tested: {total_samples}")
        print(f"Correct predictions: {total_correct}")
    
    if problem_signs:
        print(f"\n⚠ Problem Signs (< 50% accuracy): {len(problem_signs)}")
        for sign in problem_signs:
            results = all_results[sign]
            accuracy = (results['correct'] / results['total']) * 100
            print(f"  - {sign.upper()}: {accuracy:.1f}%")
            
            # Show what it's confused with
            top_confusions = sorted(results['predictions'].items(), 
                                   key=lambda x: x[1], reverse=True)[:3]
            confused = ', '.join([f"{k.upper()}({v})" for k, v in top_confusions])
            print(f"    Predicted as: {confused}")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if problem_signs:
        print("\nTo improve accuracy for problem signs:")
        print("1. Collect more training data for these specific signs")
        print("2. Ensure training images have varied backgrounds/lighting")
        print("3. Consider data augmentation during training")
        print("4. For motion signs (J, Z), consider using video sequences")
        print("5. Retrain the model with more epochs or different architecture")
    else:
        print("\n✓ All signs have good accuracy!")
    
    print("\nNote: Some signs are inherently similar in static images:")
    print("  - U, V, N (two fingers up)")
    print("  - R, U (similar finger positions)")
    print("  - S, A, T (fist variations)")
    print("  - J, Z (require motion in ASL)")


if __name__ == "__main__":
    main()
