import yaml
import cv2
import torch
import numpy as np
from models.matching import Matching
from models.utils import frame2tensor

def load_camera_params(yaml_file):
    with open(yaml_file, 'r') as file:
        params = yaml.safe_load(file)
    
    intrinsics = params['intrinsics']
    camera_matrix = np.array([
        [intrinsics[0], 0, intrinsics[2]],
        [0, intrinsics[1], intrinsics[3]],
        [0, 0, 1]
    ])
    dist_coeffs = np.array(params['distortion_coefficients'])
    
    return camera_matrix, dist_coeffs

def load_superglue_model(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    superglue_config = {
        'superpoint': {
            'nms_radius': config['superglue']['nms_radius'],
            'keypoint_threshold': config['superglue']['keypoint_threshold'],
            'max_keypoints': config['superglue']['max_keypoints']
        },
        'superglue': {
            'weights': config['superglue']['weights'],
            'sinkhorn_iterations': config['superglue']['sinkhorn_iterations'],
            'match_threshold': config['superglue']['match_threshold'],
        }
    }
    
    try:
        matching = Matching(superglue_config).eval().to(device)
        print("SuperGlue model loaded successfully")
        
        # Test the model with dummy input
        dummy_input = torch.randn(1, 1, 480, 752).to(device)
        try:
            with torch.no_grad():
                _ = matching({'image0': dummy_input, 'image1': dummy_input})
            print("SuperGlue model test passed")
        except Exception as e:
            print(f"Error during SuperGlue model test: {e}")
        
        return matching, device
    except Exception as e:
        print(f"Error loading SuperGlue model: {e}")
        print(f"SuperGlue config: {superglue_config}")
        raise

def report_outlier_method(method, threshold):
    if method == 'ransac':
        print(f"Using RANSAC with threshold: {threshold}")
    elif method == 'distance':
        print(f"Using distance-based filtering with threshold: {threshold}")
    elif method == 'confidence':
        print(f"Using confidence-based filtering with threshold: {threshold}")

def match_features_superglue(img1, img2, matching, device, outlier_method='ransac', ransac_threshold=3.0, distance_threshold=10.0, confidence_threshold=0.5):
    # Convert images to tensors
    frame_tensor1 = frame2tensor(img1, device)
    frame_tensor2 = frame2tensor(img2, device)
    
    # Perform the matching
    with torch.no_grad():
        pred = matching({'image0': frame_tensor1, 'image1': frame_tensor2})
    
    # Detach tensors and convert to numpy arrays
    pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']
    
    # Keep the matching points
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]
    
    initial_matches = len(mkpts0)
    
    if outlier_method == 'ransac':
        if len(mkpts0) >= 8 and len(mkpts1) >= 8:
            _, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, ransac_threshold)
            if mask is not None:
                mask = mask.ravel() != 0
                mkpts0 = mkpts0[mask]
                mkpts1 = mkpts1[mask]
                mconf = mconf[mask]
    elif outlier_method == 'distance':
        distances = np.linalg.norm(mkpts0 - mkpts1, axis=1)
        mask = distances < distance_threshold
        mkpts0 = mkpts0[mask]
        mkpts1 = mkpts1[mask]
        mconf = mconf[mask]
    elif outlier_method == 'confidence':
        mask = mconf > confidence_threshold
        mkpts0 = mkpts0[mask]
        mkpts1 = mkpts1[mask]
        mconf = mconf[mask]
    
    filtered_matches = len(mkpts0)
    
    return mkpts0, mkpts1, initial_matches, filtered_matches

def detect_features(image_path, prev_image_path, matching, device, outlier_method='ransac', ransac_threshold=0.5, distance_threshold=5.0, confidence_threshold=0.9, verbose=False):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if prev_image_path is not None:
        prev_image = cv2.imread(prev_image_path, cv2.IMREAD_GRAYSCALE)
        src_pts, dst_pts, initial_matches, filtered_matches = match_features_superglue(prev_image, image, matching, device, 
                                                    outlier_method, ransac_threshold, 
                                                    distance_threshold, confidence_threshold)
        return src_pts, dst_pts, initial_matches, filtered_matches
    else:
        return None, None, 0, 0

if __name__ == "__main__":
    print("This module is not meant to be run directly. Please use main-vio-file.py instead.")