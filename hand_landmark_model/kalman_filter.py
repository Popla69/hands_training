"""
Kalman filter for landmark smoothing and stabilization
"""

import numpy as np
from config import *


class KalmanFilter1D:
    """1D Kalman filter for single coordinate"""
    
    def __init__(self, process_noise=KALMAN_PROCESS_NOISE, 
                 measurement_noise=KALMAN_MEASUREMENT_NOISE):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.estimate = None
        self.error_covariance = KALMAN_INITIAL_COVARIANCE
        
    def update(self, measurement):
        if self.estimate is None:
            self.estimate = measurement
            return measurement
        
        # Prediction
        predicted_estimate = self.estimate
        predicted_error_covariance = self.error_covariance + self.process_noise
        
        # Update
        kalman_gain = predicted_error_covariance / (predicted_error_covariance + self.measurement_noise)
        self.estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
        self.error_covariance = (1 - kalman_gain) * predicted_error_covariance
        
        return self.estimate
    
    def reset(self):
        self.estimate = None
        self.error_covariance = KALMAN_INITIAL_COVARIANCE


class LandmarkKalmanFilter:
    """Kalman filter for all hand landmarks"""
    
    def __init__(self, num_landmarks=21):
        self.num_landmarks = num_landmarks
        self.filters = []
        
        # Create 3 filters (x, y, z) for each landmark
        for _ in range(num_landmarks * 3):
            self.filters.append(KalmanFilter1D())
    
    def update(self, landmarks):
        """
        Update with new landmarks
        landmarks: numpy array of shape (21, 3) or (21*3,)
        """
        landmarks_flat = landmarks.flatten()
        smoothed = np.zeros_like(landmarks_flat)
        
        for i, (measurement, kf) in enumerate(zip(landmarks_flat, self.filters)):
            smoothed[i] = kf.update(measurement)
        
        return smoothed.reshape(landmarks.shape)
    
    def reset(self):
        """Reset all filters"""
        for kf in self.filters:
            kf.reset()


class OneEuroFilter:
    """
    One Euro Filter - alternative to Kalman for low-latency smoothing
    Better for real-time applications
    """
    
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None
        
    def smoothing_factor(self, t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)
    
    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev
    
    def update(self, x, t=None):
        if t is None:
            import time
            t = time.time()
        
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        
        t_e = t - self.t_prev
        
        # Derivative
        dx = (x - self.x_prev) / t_e if t_e > 0 else 0.0
        
        # Smooth derivative
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        
        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # Smooth value
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        
        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat


class LandmarkOneEuroFilter:
    """One Euro Filter for all landmarks"""
    
    def __init__(self, num_landmarks=21):
        self.num_landmarks = num_landmarks
        self.filters = []
        
        for _ in range(num_landmarks * 3):
            self.filters.append(OneEuroFilter())
    
    def update(self, landmarks, t=None):
        landmarks_flat = landmarks.flatten()
        smoothed = np.zeros_like(landmarks_flat)
        
        for i, (measurement, f) in enumerate(zip(landmarks_flat, self.filters)):
            smoothed[i] = f.update(measurement, t)
        
        return smoothed.reshape(landmarks.shape)


if __name__ == "__main__":
    # Test Kalman filter
    kf = LandmarkKalmanFilter(num_landmarks=21)
    
    # Simulate noisy measurements
    true_landmarks = np.random.rand(21, 3)
    
    for i in range(10):
        noisy = true_landmarks + np.random.randn(21, 3) * 0.01
        smoothed = kf.update(noisy)
        error = np.mean(np.abs(smoothed - true_landmarks))
        print(f"Step {i}: Error = {error:.6f}")
