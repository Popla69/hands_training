"""
Model export utilities for ONNX and TensorFlow Lite
"""

import torch
import numpy as np
from model import create_model
from config import INPUT_SIZE


def export_to_onnx(model_path, output_path, opset_version=12):
    """
    Export PyTorch model to ONNX format
    
    Args:
        model_path: Path to PyTorch model (.pth)
        output_path: Path to save ONNX model (.onnx)
        opset_version: ONNX opset version
    """
    print(f"Exporting model to ONNX...")
    print(f"  Input: {model_path}")
    print(f"  Output: {output_path}")
    
    # Load model
    model = create_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, INPUT_SIZE[0], INPUT_SIZE[1])
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['landmarks', 'confidence'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'landmarks': {0: 'batch_size'},
            'confidence': {0: 'batch_size'}
        }
    )
    
    print(f"✓ ONNX export successful")
    
    # Validate export
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model validation passed")
    except ImportError:
        print("  Warning: onnx package not installed, skipping validation")
    except Exception as e:
        print(f"  Warning: ONNX validation failed: {e}")


def export_to_tflite(model_path, output_path, quantize=False):
    """
    Export PyTorch model to TensorFlow Lite format
    
    Args:
        model_path: Path to PyTorch model (.pth)
        output_path: Path to save TFLite model (.tflite)
        quantize: Whether to apply INT8 quantization
    """
    print(f"Exporting model to TensorFlow Lite...")
    print(f"  Input: {model_path}")
    print(f"  Output: {output_path}")
    print(f"  Quantization: {'INT8' if quantize else 'FP32'}")
    
    try:
        import tensorflow as tf
        import onnx
        from onnx_tf.backend import prepare
    except ImportError as e:
        print(f"✗ Error: Required packages not installed")
        print(f"  Install with: pip install tensorflow onnx onnx-tf")
        return
    
    # First export to ONNX
    onnx_path = output_path.replace('.tflite', '.onnx')
    export_to_onnx(model_path, onnx_path)
    
    # Convert ONNX to TensorFlow
    print("Converting ONNX to TensorFlow...")
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    
    # Export to SavedModel
    saved_model_path = output_path.replace('.tflite', '_saved_model')
    tf_rep.export_graph(saved_model_path)
    print(f"✓ TensorFlow SavedModel created")
    
    # Convert to TFLite
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    
    if quantize:
        # INT8 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
        
        # Representative dataset for quantization
        def representative_dataset():
            for _ in range(100):
                data = np.random.randn(1, 3, INPUT_SIZE[0], INPUT_SIZE[1]).astype(np.float32)
                yield [data]
        
        converter.representative_dataset = representative_dataset
    
    tflite_model = converter.convert()
    
    # Save TFLite model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✓ TFLite export successful")
    
    # Print model size
    import os
    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"  Model size: {size_mb:.2f} MB")


def validate_export(pytorch_path, onnx_path=None, tflite_path=None, num_samples=10):
    """
    Validate exported models against PyTorch model
    
    Args:
        pytorch_path: Path to PyTorch model
        onnx_path: Path to ONNX model (optional)
        tflite_path: Path to TFLite model (optional)
        num_samples: Number of test samples
    """
    print("\n" + "="*70)
    print("Validating Exported Models")
    print("="*70)
    
    # Load PyTorch model
    print("Loading PyTorch model...")
    model = create_model(pretrained=False)
    model.load_state_dict(torch.load(pytorch_path, map_location='cpu'))
    model.eval()
    
    # Generate test data
    test_inputs = [torch.randn(1, 3, INPUT_SIZE[0], INPUT_SIZE[1]) for _ in range(num_samples)]
    
    # Get PyTorch predictions
    print("Running PyTorch inference...")
    pytorch_outputs = []
    with torch.no_grad():
        for inp in test_inputs:
            landmarks, confidence = model(inp)
            pytorch_outputs.append((landmarks.numpy(), confidence.numpy()))
    
    # Validate ONNX
    if onnx_path:
        try:
            import onnxruntime as ort
            print("\nValidating ONNX model...")
            
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            
            errors = []
            for i, inp in enumerate(test_inputs):
                outputs = session.run(None, {'input': inp.numpy()})
                onnx_landmarks, onnx_confidence = outputs
                
                pt_landmarks, pt_confidence = pytorch_outputs[i]
                
                landmark_error = np.mean(np.abs(onnx_landmarks - pt_landmarks))
                confidence_error = np.mean(np.abs(onnx_confidence - pt_confidence))
                
                errors.append((landmark_error, confidence_error))
            
            avg_landmark_error = np.mean([e[0] for e in errors])
            avg_confidence_error = np.mean([e[1] for e in errors])
            
            print(f"  Average landmark error: {avg_landmark_error:.6f}")
            print(f"  Average confidence error: {avg_confidence_error:.6f}")
            
            if avg_landmark_error < 1e-5 and avg_confidence_error < 1e-5:
                print(f"  ✓ ONNX validation passed")
            else:
                print(f"  ⚠ ONNX has numerical differences (may be acceptable)")
                
        except ImportError:
            print("  Skipping ONNX validation (onnxruntime not installed)")
        except Exception as e:
            print(f"  ✗ ONNX validation failed: {e}")
    
    # Validate TFLite
    if tflite_path:
        try:
            import tensorflow as tf
            print("\nValidating TFLite model...")
            
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            errors = []
            for i, inp in enumerate(test_inputs):
                interpreter.set_tensor(input_details[0]['index'], inp.numpy())
                interpreter.invoke()
                
                tflite_landmarks = interpreter.get_tensor(output_details[0]['index'])
                tflite_confidence = interpreter.get_tensor(output_details[1]['index'])
                
                pt_landmarks, pt_confidence = pytorch_outputs[i]
                
                landmark_error = np.mean(np.abs(tflite_landmarks - pt_landmarks))
                confidence_error = np.mean(np.abs(tflite_confidence - pt_confidence))
                
                errors.append((landmark_error, confidence_error))
            
            avg_landmark_error = np.mean([e[0] for e in errors])
            avg_confidence_error = np.mean([e[1] for e in errors])
            
            print(f"  Average landmark error: {avg_landmark_error:.6f}")
            print(f"  Average confidence error: {avg_confidence_error:.6f}")
            
            if avg_landmark_error < 0.01 and avg_confidence_error < 0.01:
                print(f"  ✓ TFLite validation passed")
            else:
                print(f"  ⚠ TFLite has numerical differences (expected with quantization)")
                
        except ImportError:
            print("  Skipping TFLite validation (tensorflow not installed)")
        except Exception as e:
            print(f"  ✗ TFLite validation failed: {e}")
    
    print("="*70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python export.py <model.pth> <output_format>")
        print("  output_format: onnx, tflite, or both")
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_format = sys.argv[2].lower()
    
    if output_format in ['onnx', 'both']:
        onnx_path = model_path.replace('.pth', '.onnx')
        export_to_onnx(model_path, onnx_path)
    
    if output_format in ['tflite', 'both']:
        tflite_path = model_path.replace('.pth', '.tflite')
        export_to_tflite(model_path, tflite_path, quantize=False)
        
        tflite_quant_path = model_path.replace('.pth', '_int8.tflite')
        export_to_tflite(model_path, tflite_quant_path, quantize=True)
