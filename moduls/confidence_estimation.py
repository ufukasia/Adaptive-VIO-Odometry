import numpy as np
import cv2

def quadratic_unit_step(x):
    return np.minimum(x**2, 1)

def cubic_unit_step(x):
    return np.minimum(x**3, 1)

def quartic_unit_step(x):
    return np.minimum(x**4, 1)

def relu(x):
    return np.maximum(0, x)

def double_exponential_sigmoid(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def triple_exponential_sigmoid(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x) + np.exp(-2*x))

def quadruple_exponential_sigmoid(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x) + np.exp(-2*x) + np.exp(-3*x))

def step(x):
    return np.where(x >= 0, 1, 0)

class AdaptivePositiveNormalizer:
    def __init__(self, initial_value=0, alpha=0.01):
        self.mean = max(initial_value, 0)
        self.max_value = max(initial_value, 1e-6)  # Avoid division by zero
        self.alpha = alpha  # Learning rate for the running statistics

    def update(self, value):
        value = max(value, 0)  # Ensure non-negative value
        self.mean = (1 - self.alpha) * self.mean + self.alpha * value
        self.max_value = max(self.max_value, value)

    def normalize(self, value):
        value = max(value, 0)  # Ensure non-negative value
        if self.max_value == 0:
            return 0
        return value / self.max_value

def normalize_delta_intensity(delta):
    """
    Delta intensity değerlerini 0-256 aralığında normalize eder.
    """
    return min(delta, 256) / 256

def estimate_confidence(image, prev_intensity, config):
    intensity = calculate_intensity(image)
    entropy = calculate_entropy(image)
    motion_blur = calculate_motion_blur(image)
    
    intensity_normalizer = AdaptivePositiveNormalizer()
    motion_blur_normalizer = AdaptivePositiveNormalizer()
    
    intensity_normalizer.update(intensity)
    motion_blur_normalizer.update(motion_blur)
    
    normalized_intensity = intensity_normalizer.normalize(intensity)
    normalized_entropy = entropy / 8  # Entropi değerini 8'e bölerek normalize ediyoruz
    normalized_motion_blur = motion_blur_normalizer.normalize(motion_blur)
    
    scaled_entropy = (1 - normalized_entropy) * config['beta']
    scaled_motion_blur = normalized_motion_blur * config['gamma'] * 0.2
    
    if prev_intensity is not None:
        delta_intensity = abs(intensity - prev_intensity)
        normalized_delta_intensity = normalize_delta_intensity(delta_intensity * 255)
        scaled_delta_intensity = normalized_delta_intensity * config['alpha'] * 10
        
        combined_value = max(scaled_delta_intensity, scaled_entropy, scaled_motion_blur)
        
        if combined_value > config['theta_threshold']:
            theta = config['activation_func'](combined_value - config['theta_threshold'])
        else:
            theta = 0
    else:
        theta = 0
    
    return intensity, entropy, theta

def calculate_intensity(image):
    return np.mean(image) / 255.0

def calculate_entropy(image):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    histogram = histogram.flatten() / np.sum(histogram)
    non_zero = histogram[histogram > 0]
    return -np.sum(non_zero * np.log2(non_zero))

def calculate_motion_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()