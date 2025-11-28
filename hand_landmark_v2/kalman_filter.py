"""
Kalman filtering for landmark smoothing and stabilization
"""

import numpy as np
from config import NUM_LANDMARKS, LANDMARK_DIM
from config import KALMAN_PROCESS_NOISE, KALMAN_MEASUREMENT_NOISE
from config import ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF


class LandmarkKalmanFilter:
    """
    Standard Kalman filter for landmark smoothing
    
    State vector: [x, y, z, vx, vy, vz] per landmark
    Measurement: [x, y, z] per landmark
    
    Reduces jitter and stabilizes landmark positions across frames
    """
    
    def __init__(self, num_landmarks=NUM_LANDMARKS, 
                 process_noise=KALMAN_PROCESS_NOISE,
                 measurement_noise=KALMAN_MEASUREMENT_NOISE):
        """
        Initialize Kalman filter
        
        Args:
            num_landmarks: Number of landmarks to track
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
        """
        self.num_landmarks = num_landmarks
        self.state_dim = 6  # [x, y, z, vx, vy, vz]
        self.meas_dim = 3   # [x, y, z]
        
        # State vectors for each landmark
        self.states = np.zeros((num_landmarks, self.state_dim))
        
        # State covariance matrices
        self.P = np.array([np.eye(self.state_dim) for _ in range(num_landmarks)])
        
        # Process noise covariance
        self.Q = np.eye(self.state_dim) * process_noise
        
        # Measurement noise covariance
        self.R = np.eye(self.meas_dim) * measurement_noise
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 0, 1, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 1, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 1],  # z = z + vz
            [0, 0, 0, 1, 0, 0],  # vx = vx
            [0, 0, 0, 0, 1, 0],  # vy = vy
            [0, 0, 0, 0, 0, 1],  # vz = vz
        ])
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ])
        
        self.initialized = False
    
    def predict(self):
        """
        Prediction step: predict next state based on motion model
        """
        for i in range(self.num_landmarks):
            # Predict state
            self.states[i] = self.F @ self.states[i]
            
            # Predict covariance
            self.P[i] = self.F @ self.P[i] @ self.F.T + self.Q
    
    def update(self, landmarks):
        """
        Update step: correct prediction with measurement
        
        Args:
            landmarks: (num_landmarks, 3) array of measured landmarks
            
        Returns:
            filtered_landmarks: (num_landmarks, 3) filtered landmarks
        """
        if not self.initialized:
            # Initialize state with first measurement
            for i in range(self.num_landmarks):
                self.states[i][:3] = landmarks[i]
                self.states[i][3:] = 0  # Zero velocity
            self.initialized = True
            return landmarks
        
        # Predict
        self.predict()
        
        # Update each landmark
        filtered_landmarks = np.zeros_like(landmarks)
        
        for i in range(self.num_landmarks):
            # Innovation (measurement residual)
            z = landmarks[i]
            y = z - (self.H @ self.states[i])
            
            # Innovation covariance
            S = self.H @ self.P[i] @ self.H.T + self.R
            
            # Kalman gain
            K = self.P[i] @ self.H.T @ np.linalg.inv(S)
            
            # Update state
            self.states[i] = self.states[i] + K @ y
            
            # Update covariance
            I = np.eye(self.state_dim)
            self.P[i] = (I - K @ self.H) @ self.P[i]
            
            # Extract position
            filtered_landmarks[i] = self.states[i][:3]
        
        return filtered_landmarks
    
    def reset(self):
        """Reset filter state"""
        self.states = np.zeros((self.num_landmarks, self.state_dim))
        self.P = np.array([np.eye(self.state_dim) for _ in range(self.num_landmarks)])
        self.initialized = False


class OneEuroFilter:
    """
    One Euro Filter for a single value
    
    Adaptive low-pass filter that reduces jitter while maintaining responsiveness
    """
    
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        """
        Initialize One Euro Filter
        
        Args:
            min_cutoff: Minimum cutoff frequency
            beta: Speed coefficient
            d_cutoff: Cutoff frequency for derivative
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None
    
    def __call__(self, x, t):
        """
        Filter value
        
        Args:
            x: Current value
            t: Current timestamp
            
        Returns:
            Filtered value
        """
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        
        # Time delta
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev
        
        # Derivative
        dx = (x - self.x_prev) / dt
        
        # Filter derivative
        alpha_d = self._alpha(dt, self.d_cutoff)
        dx_filtered = self._exponential_smoothing(alpha_d, dx, self.dx_prev)
        
        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_filtered)
        
        # Filter value
        alpha = self._alpha(dt, cutoff)
        x_filtered = self._exponential_smoothing(alpha, x, self.x_prev)
        
        # Update state
        self.x_prev = x_filtered
        self.dx_prev = dx_filtered
        self.t_prev = t
        
        return x_filtered
    
    def _alpha(self, dt, cutoff):
        """Calculate smoothing factor"""
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
    
    def _exponential_smoothing(self, alpha, x, x_prev):
        """Apply exponential smoothing"""
        return alpha * x + (1 - alpha) * x_prev


class LandmarkOneEuroFilter:
    """
    One Euro Filter for all landmarks
    
    Applies One Euro filtering independently to each landmark coordinate
    """
    
    def __init__(self, num_landmarks=NUM_LANDMARKS,
                 min_cutoff=ONE_EURO_MIN_CUTOFF,
                 beta=ONE_EURO_BETA,
                 d_cutoff=ONE_EURO_D_CUTOFF):
        """
        Initialize One Euro Filter for landmarks
        
        Args:
            num_landmarks: Number of landmarks
            min_cutoff: Minimum cutoff frequency
            beta: Speed coefficient
            d_cutoff: Cutoff frequency for derivative
        """
        self.num_landmarks = num_landmarks
        
        # Create filter for each coordinate of each landmark
        self.filters = []
        for _ in range(num_landmarks * LANDMARK_DIM):
            self.filters.append(OneEuroFilter(min_cutoff, beta, d_cutoff))
        
        self.t = 0.0
    
    def update(self, landmarks):
        """
        Filter landmarks
        
        Args:
            landmarks: (num_landmarks, 3) array of landmarks
            
        Returns:
            filtered_landmarks: (num_landmarks, 3) filtered landmarks
        """
        self.t += 1.0  # Increment timestamp
        
        # Flatten landmarks
        landmarks_flat = landmarks.flatten()
        
        # Filter each coordinate
        filtered_flat = np.zeros_like(landmarks_flat)
        for i, (value, filt) in enumerate(zip(landmarks_flat, self.filters)):
            filtered_flat[i] = filt(value, self.t)
        
        # Reshape back
        filtered_landmarks = filtered_flat.reshape(self.num_landmarks, LANDMARK_DIM)
        
        return filtered_landmarks
    
    def reset(self):
        """Reset all filters"""
        for filt in self.filters:
            filt.x_prev = None
            filt.dx_prev = 0.0
            filt.t_prev = None
        self.t = 0.0


def measure_jitter(landmarks_sequence):
    """
    Measure jitter in landmark sequence
    
    Args:
        landmarks_sequence: List of (num_landmarks, 3) landmark arrays
        
    Returns:
        jitter: Average jitter across all landmarks
    """
    if len(landmarks_sequence) < 3:
        return 0.0
    
    # Calculate second derivative (acceleration)
    accelerations = []
    for i in range(1, len(landmarks_sequence) - 1):
        prev = landmarks_sequence[i - 1]
        curr = landmarks_sequence[i]
        next_lm = landmarks_sequence[i + 1]
        
        # Second derivative
        accel = next_lm - 2 * curr + prev
        accelerations.append(np.abs(accel))
    
    # Average jitter
    jitter = np.mean(accelerations)
    return jitter


def compare_filters(landmarks_sequence):
    """
    Compare different filtering approaches
    
    Args:
        landmarks_sequence: List of (num_landmarks, 3) landmark arrays
        
    Returns:
        results: Dictionary with jitter measurements
    """
    print("Comparing filters...")
    print(f"  Sequence length: {len(landmarks_sequence)} frames")
    
    # Measure raw jitter
    raw_jitter = measure_jitter(landmarks_sequence)
    print(f"  Raw jitter: {raw_jitter:.6f}")
    
    # Apply Kalman filter
    kalman = LandmarkKalmanFilter()
    kalman_filtered = []
    for landmarks in landmarks_sequence:
        filtered = kalman.update(landmarks)
        kalman_filtered.append(filtered)
    kalman_jitter = measure_jitter(kalman_filtered)
    kalman_reduction = (1 - kalman_jitter / raw_jitter) * 100
    print(f"  Kalman jitter: {kalman_jitter:.6f} ({kalman_reduction:.1f}% reduction)")
    
    # Apply One Euro filter
    one_euro = LandmarkOneEuroFilter()
    one_euro_filtered = []
    for landmarks in landmarks_sequence:
        filtered = one_euro.update(landmarks)
        one_euro_filtered.append(filtered)
    one_euro_jitter = measure_jitter(one_euro_filtered)
    one_euro_reduction = (1 - one_euro_jitter / raw_jitter) * 100
    print(f"  One Euro jitter: {one_euro_jitter:.6f} ({one_euro_reduction:.1f}% reduction)")
    
    return {
        'raw': raw_jitter,
        'kalman': kalman_jitter,
        'one_euro': one_euro_jitter,
        'kalman_reduction': kalman_reduction,
        'one_euro_reduction': one_euro_reduction,
    }


if __name__ == "__main__":
    # Test filters
    print("Testing Kalman filters...")
    
    # Generate synthetic landmark sequence with noise
    num_frames = 100
    landmarks_sequence = []
    
    for i in range(num_frames):
        # Smooth trajectory
        t = i / num_frames
        base_landmarks = np.array([[
            0.5 + 0.1 * np.sin(2 * np.pi * t),
            0.5 + 0.1 * np.cos(2 * np.pi * t),
            0.0
        ] for _ in range(NUM_LANDMARKS)])
        
        # Add noise
        noise = np.random.randn(NUM_LANDMARKS, LANDMARK_DIM) * 0.01
        noisy_landmarks = base_landmarks + noise
        
        landmarks_sequence.append(noisy_landmarks)
    
    # Compare filters
    results = compare_filters(landmarks_sequence)
    
    print("\n✓ Filter test complete")
