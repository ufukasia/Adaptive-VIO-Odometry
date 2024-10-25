import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import zipfile
import io
import shutil
import csv
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
from moduls.camera import load_camera_params, load_superglue_model
from moduls.data_processing import process_imu_data, preprocess_imu_data
from moduls.visualization import visualize_results, visualize_error, visualize_superglue
from moduls.confidence_estimation import quadratic_unit_step, cubic_unit_step, quartic_unit_step, relu, double_exponential_sigmoid, triple_exponential_sigmoid, quadruple_exponential_sigmoid, step

DEFAULT_DATASET = "EurocMav"
DEFAULT_SEQUENCE = "MH_03_medium"

BASE_URLS = {
    "machine_hall": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/",
    "vicon_room1": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/",
    "vicon_room2": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/"
}

SEQUENCE_PREFIXES = {
    "MH": "machine_hall",
    "V1": "vicon_room1",
    "V2": "vicon_room2"
}

def get_base_url(sequence_name):
    prefix = sequence_name[:2]
    return BASE_URLS.get(SEQUENCE_PREFIXES.get(prefix))

def validate_sequence_name(sequence_name):
    valid_prefixes = ["MH_", "V1_", "V2_"]
    return any(sequence_name.startswith(prefix) for prefix in valid_prefixes)

def download_dataset(sequence_name, dataset_path):
    if not validate_sequence_name(sequence_name):
        print(f"Error: Invalid sequence name {sequence_name}. Must start with MH_, V1_, or V2_")
        return False

    base_url = get_base_url(sequence_name)
    if not base_url:
        print(f"Error: Could not determine base URL for sequence {sequence_name}")
        return False

    download_url = f"{base_url}{sequence_name}/{sequence_name}.zip"
    zip_path = Path(dataset_path) / f"{sequence_name}.zip"
    extract_path = Path(dataset_path) / sequence_name
    
    print(f"Attempting to download from: {download_url}")
    
    try:
        with urlopen(download_url) as response:
            total_size = int(response.info().get('Content-Length', -1))
            
            if total_size < 0:
                print("Unable to determine file size. Downloading without progress indication.")
                data = response.read()
            else:
                data = io.BytesIO()
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        data.write(chunk)
                        pbar.update(len(chunk))
                data = data.getvalue()

            print("\nDownload complete. Saving and verifying file...")
            
            with open(zip_path, 'wb') as f:
                f.write(data)
            
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    print("File verified as a valid zip file. Extracting...")
                    zf.extractall(extract_path)
                print("Extraction complete.")
                return True
            except zipfile.BadZipFile:
                print("Error: Downloaded file is not a valid zip file.")
                return False
    
    except HTTPError as e:
        print(f"HTTP Error during download: {e.code} {e.reason}")
        return False
    except URLError as e:
        print(f"URL Error during download: {e.reason}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def process_dataset(base_path, config, sequence_name):
    imu_file_path = base_path / 'imu0' / 'data.csv'
    camera_file_path = base_path / 'cam0' / 'data.csv'
    camera_data_path = base_path / 'cam0' / 'data'
    camera_yaml_path = base_path / 'cam0' / 'sensor.yaml'

    # Check if all required files exist
    required_files = [imu_file_path, camera_file_path, camera_yaml_path]
    for file in required_files:
        if not file.exists():
            print(f"Required file not found: {file}")
            return None

    if not camera_data_path.exists():
        print(f"Camera data directory not found: {camera_data_path}")
        return None

    print("Preprocessing IMU data...")
    preprocess_imu_data(base_path)

    imu_with_groundtruth_path = base_path / 'imu0' / 'imu_with_interpolated_groundtruth.csv'
    
    try:
        return process_imu_data(
            str(imu_with_groundtruth_path),
            str(camera_file_path),
            str(camera_data_path),
            str(camera_yaml_path),
            config,
            sequence_name,
            verbose=True
        )
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

def main(args):
    if not validate_sequence_name(args.sequence):
        print("Error: Invalid sequence name format. Must start with MH_, V1_, or V2_")
        return

    dataset_path = Path(args.dataset_path)
    sequence_path = dataset_path / args.sequence
    
    if args.download or not sequence_path.exists():
        success = download_dataset(args.sequence, dataset_path)
        if not success:
            print("Failed to download or extract the dataset. Exiting.")
            return

    base_path = sequence_path / 'mav0'
    
    if not base_path.exists():
        print(f"Dataset structure not found at {base_path}. Please check the dataset path.")
        return

    activation_functions = {
        'quadratic_unit_step': quadratic_unit_step,
        'cubic_unit_step': cubic_unit_step,
        'quartic_unit_step': quartic_unit_step,
        'relu': relu,
        'double_exponential_sigmoid': double_exponential_sigmoid,
        'triple_exponential_sigmoid': triple_exponential_sigmoid,
        'quadruple_exponential_sigmoid': quadruple_exponential_sigmoid,
        'step': step
    }

    config = {
        'alpha': args.alpha,
        'beta': args.beta,
        'gamma': args.gamma,
        'theta_threshold': args.theta_threshold,
        'activation_func': activation_functions[args.activation_function],
        'generate_superglue_visualizations': args.generate_superglue_visualizations,
        'superglue': {
            'weights': 'indoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1000,
            'nms_radius': 4
        },
        'superglue_visualization': {
            'frame_interval': 10,
            'max_pairs': 5000
        }
    }

    result = process_dataset(base_path, config, args.sequence)
    if result is None:
        return

    (data, aligned_quaternions, aligned_euler_angles, true_quaternions, 
     true_euler_angles, rmse_quaternions, rmse_euler_angles, thetas, timestamps) = result

    print("Quaternion RMSE:", rmse_quaternions)
    print("Euler Angle RMSE:", rmse_euler_angles)
    
    # Save results to CSV
    output_file = f'{args.sequence}_estimated_values.csv'
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Timestamp', 'q_w', 'q_x', 'q_y', 'q_z', 'roll', 'pitch', 'yaw', 'theta'])
        for t, q, e, theta in zip(timestamps, aligned_quaternions, aligned_euler_angles, thetas):
            writer.writerow([t] + list(q) + list(e) + [theta])
    print(f"Estimated values saved to {output_file}")
    
    print("Generating visualizations...")
    visualize_results(data, aligned_quaternions, aligned_euler_angles, 
                     true_quaternions, true_euler_angles, rmse_quaternions, 
                     rmse_euler_angles, thetas, args.sequence)
    
    visualize_error(data, aligned_quaternions, aligned_euler_angles, 
                   true_quaternions, true_euler_angles, thetas, args.sequence)
    
    if config['generate_superglue_visualizations']:
        print("Generating SuperGlue visualizations...")
        superglue_output_dir = f'super-out_{args.sequence}'
        visualize_superglue(base_path, superglue_output_dir, config)
    
    print("Processing completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Inertial Odometry Processing")
    parser.add_argument("--dataset_path", type=str, default=".", 
                       help="Path to save the dataset (default: current directory)")
    parser.add_argument("--sequence", type=str, default=DEFAULT_SEQUENCE,
                       help="Dataset sequence to use (e.g., MH_03_medium, V1_01_easy, V2_02_medium)")
    parser.add_argument("--download", action="store_true", 
                       help="Force download the dataset even if it exists")
    parser.add_argument("--alpha", type=float, default=1, 
                       help="Alpha parameter (default: 1)")
    parser.add_argument("--beta", type=float, default=1, 
                       help="Beta parameter (default: 1)")
    parser.add_argument("--gamma", type=float, default=1, 
                       help="Gamma parameter (default: 1)")
    parser.add_argument("--theta_threshold", type=float, default=0.30, 
                       help="Theta threshold (default: 0.30)")
    parser.add_argument("--activation_function", type=str, 
                       choices=['quadratic_unit_step', 'cubic_unit_step', 'quartic_unit_step',
                               'relu', 'double_exponential_sigmoid', 'triple_exponential_sigmoid',
                               'quadruple_exponential_sigmoid', 'step'], 
                       default='double_exponential_sigmoid', 
                       help="Activation function to use (default: double_exponential_sigmoid)")
    parser.add_argument("--generate_superglue_visualizations", action="store_true",
                       help="Generate SuperGlue visualizations")
    
    args = parser.parse_args()
    main(args)
