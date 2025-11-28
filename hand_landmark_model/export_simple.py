"""
Simple export to PyTorch and ONNX (no TensorFlow dependency)
"""

import torch
import torch.onnx
import os

from model import create_model
from config import *


def export_pytorch(checkpoint_path='hand_landmark_model/checkpoints/best_model.pth', 
                  output_path='hand_landmark_model/models/hand_landmark.pth'):
    """Export PyTorch model"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model = create_model(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    torch.save(model.state_dict(), output_path)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ PyTorch model: {output_path} ({size_mb:.2f} MB)")
    print(f"  Val accuracy: {checkpoint.get('val_accuracy', 0)*100:.2f}%")
    
    return output_path


def export_onnx(checkpoint_path='hand_landmark_model/checkpoints/best_model.pth',
               output_path='hand_landmark_model/models/hand_landmark.onnx'):
    """Export to ONNX"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model = create_model(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
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
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ ONNX model: {output_path} ({size_mb:.2f} MB)")
    
    return output_path


def main():
    print("="*60)
    print("Exporting Hand Landmark Model")
    print("="*60)
    
    # Export PyTorch
    print("\n1. Exporting PyTorch model...")
    pytorch_path = export_pytorch()
    
    # Export ONNX
    print("\n2. Exporting ONNX model...")
    onnx_path = export_onnx()
    
    print("\n" + "="*60)
    print("Export Complete!")
    print("="*60)
    print(f"\nModel files:")
    print(f"  - {pytorch_path}")
    print(f"  - {onnx_path}")
    print("\nUsage:")
    print("  python hand_landmark_model/inference.py")
    print("="*60)


if __name__ == "__main__":
    main()
