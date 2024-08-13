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
        dummy_input = torch.randn(1, 1, 480, 640).to(device)
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

def match_features_superglue(img1, img2, matching, device):
    # Convert images to tensors
    frame_tensor1 = frame2tensor(img1, device)
    frame_tensor2 = frame2tensor(img2, device)
    
    # Perform the matching
    with torch.no_grad():  # Disable gradient computation
        pred = matching({'image0': frame_tensor1, 'image1': frame_tensor2})
    
    # Detach tensors and convert to numpy arrays
    pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']
    
    # Keep the matching points
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    
    return mkpts0, mkpts1

def detect_features(image_path, prev_image_path, matching, device):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if prev_image_path is not None:
        prev_image = cv2.imread(prev_image_path, cv2.IMREAD_GRAYSCALE)
        src_pts, dst_pts = match_features_superglue(prev_image, image, matching, device)
        return src_pts, dst_pts, image
    else:
        return None, None, image