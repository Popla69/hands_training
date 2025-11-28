"""
Hand Landmark Detection V2
High-precision hand landmark tracking optimized for real-time inference
"""

__version__ = "2.0.0"
__author__ = "Sign Language Recognition Team"

from .model import HandLandmarkModel, create_model
from .inference import HandLandmarkInference
from .kalman_filter import LandmarkKalmanFilter, LandmarkOneEuroFilter

__all__ = [
    'HandLandmarkModel',
    'create_model',
    'HandLandmarkInference',
    'LandmarkKalmanFilter',
    'LandmarkOneEuroFilter',
]
