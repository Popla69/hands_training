import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
import os
import glob
import cv2
import numpy as np

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile("logs/trained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def predict_image(image_path, sess, softmax_tensor):
    # Read the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    results = []
    for node_id in top_k[:5]:  # Get top 5 predictions
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        results.append((human_string, score))
    
    return results

# Get all images from Test folder
test_images = glob.glob("Test/*.jpg")
test_images.sort()

print(f"\nFound {len(test_images)} images in Test folder\n")
print("="*80)

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    for idx, image_path in enumerate(test_images, 1):
        filename = os.path.basename(image_path)
        
        print(f"\n[{idx}/{len(test_images)}] Testing: {filename}")
        print("-" * 80)
        
        results = predict_image(image_path, sess, softmax_tensor)
        
        print(f"Top prediction: {results[0][0].upper()} (confidence: {results[0][1]*100:.2f}%)")
        print("\nTop 5 predictions:")
        for i, (label, score) in enumerate(results, 1):
            print(f"  {i}. {label.upper():10s} - {score*100:6.2f}%")
        
        # Display image with prediction
        img = cv2.imread(image_path)
        if img is not None:
            # Resize for display if too large
            height, width = img.shape[:2]
            if width > 800:
                scale = 800 / width
                img = cv2.resize(img, (int(width*scale), int(height*scale)))
            
            # Add prediction text
            cv2.putText(img, f"Predicted: {results[0][0].upper()}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(img, f"Confidence: {results[0][1]*100:.1f}%", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Test Image', img)
            
            print("\nPress any key to continue, 'q' to quit, 's' to skip display...")
            key = cv2.waitKey(0)
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                cv2.destroyAllWindows()
                print("\nSkipping image display for remaining images...")
                # Continue without showing images
                for remaining_idx in range(idx + 1, len(test_images) + 1):
                    remaining_path = test_images[remaining_idx - 1]
                    remaining_filename = os.path.basename(remaining_path)
                    print(f"\n[{remaining_idx}/{len(test_images)}] Testing: {remaining_filename}")
                    print("-" * 80)
                    remaining_results = predict_image(remaining_path, sess, softmax_tensor)
                    print(f"Top prediction: {remaining_results[0][0].upper()} (confidence: {remaining_results[0][1]*100:.2f}%)")
                    print("\nTop 5 predictions:")
                    for i, (label, score) in enumerate(remaining_results, 1):
                        print(f"  {i}. {label.upper():10s} - {score*100:6.2f}%")
                break

cv2.destroyAllWindows()
print("\n" + "="*80)
print("Testing complete!")
