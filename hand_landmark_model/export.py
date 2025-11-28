"""
Export trained model to TFLite, ONNX, and optimized formats
"""

import torch
import torch.onnx
import tensorflow as tf
import numpy as np
import os

from model import create_model
from config import *


def export_to_onnx(model, output_path='models/hand_landmark.onnx'):
    """Export PyTorch model to ONNX"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model.eval()
    dummy_input = torch.randn(1, 3, INPUT_SIZE[0], INPUT_SIZE[1])
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['landmarks', 'confidence'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'landmarks': {0: 'batch_size'},
            'confidence': {0: 'batch_size'}
        }
    )
    
    print(f"Exported ONNX model to {output_path}")
    
    # Get model size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX model size: {size_mb:.2f} MB")
    
    return output_path


def export_to_tflite(onnx_path, output_path='models/hand_landmark.tflite', quantize=True):
    """Convert ONNX to TFLite with optional quantization"""
    
    try:
        import onnx
        from onnx_tf.backend import prepare
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph('models/tf_model')
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model('models/tf_model')
        
        if quantize:
            # INT8 quantization for smaller size
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
        
        tflite_model = converter.convert()
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Exported TFLite model to {output_path}")
        print(f"TFLite model size: {size_mb:.2f} MB")
        
        return output_path
        
    except Exception as e:
        print(f"TFLite export failed: {e}")
        print("Install onnx-tf: pip install onnx-tf")
        return None


def optimize_onnx_for_cpu(onnx_path, output_path='models/hand_landmark_optimized.onnx'):
    """Optimize ONNX model for CPU inference"""
    
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        # Dynamic quantization for CPU
        quantize_dynamic(
            onnx_path,
            output_path,
            weight_type=QuantType.QUInt8
        )
        
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Optimized ONNX model saved to {output_path}")
        print(f"Optimized model size: {size_mb:.2f} MB")
        
        return output_path
        
    except Exception as e:
        print(f"ONNX optimization failed: {e}")
        return onnx_path


def export_all_formats(checkpoint_path='checkpoints/best_model.pth'):
    """Export model to all formats"""
    
    print("Loading trained model...")
    model = create_model(pretrained=False)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded. Val accuracy: {checkpoint.get('val_accuracy', 0)*100:.2f}%")
    
    # Export to ONNX
    print("\n1. Exporting to ONNX...")
    onnx_path = export_to_onnx(model)
    
    # Optimize ONNX for CPU
    print("\n2. Optimizing ONNX for CPU...")
    optimized_onnx_path = optimize_onnx_for_cpu(onnx_path)
    
    # Export to TFLite
    print("\n3. Exporting to TFLite...")
    tflite_path = export_to_tflite(onnx_path, quantize=TFLITE_QUANTIZE)
    
    # Save PyTorch model
    print("\n4. Saving PyTorch model...")
    torch_path = 'models/hand_landmark.pth'
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), torch_path)
    size_mb = os.path.getsize(torch_path) / (1024 * 1024)
    print(f"PyTorch model saved to {torch_path}")
    print(f"PyTorch model size: {size_mb:.2f} MB")
    
    print("\n" + "="*50)
    print("Export Summary:")
    print("="*50)
    print(f"✓ ONNX: {onnx_path}")
    print(f"✓ ONNX Optimized: {optimized_onnx_path}")
    if tflite_path:
        print(f"✓ TFLite: {tflite_path}")
    print(f"✓ PyTorch: {torch_path}")
    print("="*50)


if __name__ == "__main__":
    export_all_formats()
