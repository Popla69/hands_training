import sys
import os

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def predict(image_data):

    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    max_score = 0.0
    res = ''
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        if score > max_score:
            max_score = score
            res = human_string
    return res, max_score

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("logs/trained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    c = 0

    cap = cv2.VideoCapture(0)

    res, score = '', 0.0
    i = 0
    mem = ''
    consecutive = 0
    sequence = ''
    
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        
        if ret:
            # Bigger capture box: 400x400 instead of 200x200
            x1, y1, x2, y2 = 50, 50, 450, 450
            img_cropped = img[y1:y2, x1:x2]

            c += 1
            image_data = cv2.imencode('.jpg', img_cropped)[1].tobytes()
            
            a = cv2.waitKey(1) # waits to see if `esc` is pressed
            
            if i == 4:
                res_tmp, score = predict(image_data)
                res = res_tmp
                i = 0
                if mem == res:
                    consecutive += 1
                else:
                    consecutive = 0
                if consecutive == 2 and res not in ['nothing']:
                    if res == 'space':
                        sequence += ' '
                    elif res == 'del':
                        sequence = sequence[:-1]
                    else:
                        sequence += res
                    consecutive = 0
            i += 1
            
            # Display prediction on video
            cv2.putText(img, '%s' % (res.upper()), (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
            cv2.putText(img, '(score = %.2f%%)' % (float(score)*100), (50, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mem = res
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.imshow("Sign Language Recognition", img)
            
            # Text sequence window with word wrapping
            img_sequence = np.zeros((400, 1200, 3), np.uint8)
            
            # Word wrapping logic
            max_chars_per_line = 50
            lines = []
            current_line = ""
            
            for char in sequence.upper():
                if len(current_line) >= max_chars_per_line:
                    lines.append(current_line)
                    current_line = char
                else:
                    current_line += char
            
            if current_line:
                lines.append(current_line)
            
            # Display lines
            y_offset = 50
            for line in lines[-8:]:  # Show last 8 lines
                cv2.putText(img_sequence, line, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                y_offset += 45
            
            cv2.imshow('Sequence', img_sequence)
            
            if a == 27: # when `esc` is pressed
                break

cv2.destroyAllWindows() 
cap.release()
