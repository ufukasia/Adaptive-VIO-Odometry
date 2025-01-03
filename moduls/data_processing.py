import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import cv2
import torch
from pathlib import Path
from .ekf import ExtendedKalmanFilter, quaternion_to_euler, ensure_quaternion_continuity, align_quaternions, ensure_angle_continuity
from .camera import load_camera_params, load_superglue_model, match_features_superglue, detect_features
from .confidence_estimation import estimate_confidence
from models.utils import frame2tensor

def process_imu_data(imu_file_path, camera_file_path, camera_data_path, camera_yaml_path, config, sequence_name, verbose=False):
    try:
        camera_matrix, dist_coeffs = load_camera_params(camera_yaml_path)

        calibration = {
            'w_RS_S_x [rad s^-1]': -0.003342,
            'w_RS_S_y [rad s^-1]': 0.020582,
            'w_RS_S_z [rad s^-1]': 0.079360,
            'a_RS_S_x [m s^-2]': 0.045,
            'a_RS_S_y [m s^-2]': 0.124,
            'a_RS_S_z [m s^-2]': 0.0628
        }

        imu_data = pd.read_csv(imu_file_path)
        camera_data = pd.read_csv(camera_file_path)

        initial_quaternion = imu_data[[' q_RS_w []', ' q_RS_x []', ' q_RS_y []', ' q_RS_z []']].iloc[0].values

        imu_data['timestamp'] = pd.to_datetime(imu_data['#timestamp [ns]'], unit='ns')
        imu_data['dt'] = imu_data['timestamp'].diff().dt.total_seconds()
        imu_data.loc[0, 'dt'] = 0

        for key in calibration:
            imu_data[key] = imu_data[key] - calibration[key]

        ekf = ExtendedKalmanFilter(initial_quaternion=initial_quaternion, 
                                   camera_matrix=camera_matrix, 
                                   dist_coeffs=dist_coeffs,
                                   theta_threshold=config['theta_threshold'])

        estimated_quaternions = []
        estimated_euler_angles = []
        thetas = []
        timestamps = []
        q_prev = initial_quaternion

        prev_image_path = None
        prev_intensity = None

        try:
            matching, device = load_superglue_model(config)
            if verbose:
                print("SuperGlue model loaded successfully")
        except Exception as e:
            if verbose:
                print(f"Error loading SuperGlue model: {e}")
            matching, device = None, None

        progress_bar = tqdm(total=len(imu_data), desc="Processing IMU data")

        match_data = []  # New list to store match data

        for _, row in imu_data.iterrows():
            gyro = np.array([row['w_RS_S_x [rad s^-1]'], row['w_RS_S_y [rad s^-1]'], row['w_RS_S_z [rad s^-1]']])
            accel = np.array([row['a_RS_S_x [m s^-2]'], row['a_RS_S_y [m s^-2]'], row['a_RS_S_z [m s^-2]']])
            
            camera_row = camera_data[camera_data['#timestamp [ns]'] == row['#timestamp [ns]']]
            
            if not camera_row.empty:
                image_path = os.path.join(camera_data_path, camera_row['filename'].values[0])
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                intensity, entropy, theta = estimate_confidence(image, prev_intensity, config)
                thetas.append(theta)
                
                if matching is not None and device is not None:
                    try:
                        src_pts, dst_pts, initial_matches, filtered_matches = detect_features(
                            image_path=image_path,
                            prev_image_path=prev_image_path,
                            matching=matching,
                            device=device,
                            outlier_method='distance',
                            ransac_threshold=0.5,
                            distance_threshold=5.0,
                            confidence_threshold=0.9,
                            verbose=verbose
                        )
                        match_data.append({
                            'timestamp': row['#timestamp [ns]'],
                            'initial_matches': initial_matches,
                            'filtered_matches': filtered_matches
                        })
                    except Exception as e:
                        if verbose:
                            print(f"Error detecting features: {e}")
                        src_pts, dst_pts = None, None
                else:
                    src_pts, dst_pts = None, None
                
                prev_image_path = image_path
                prev_intensity = intensity
                
                if src_pts is not None and dst_pts is not None:
                    world_points = np.hstack((src_pts.reshape(-1, 2), np.zeros((src_pts.shape[0], 1))))
                    image_points = dst_pts.reshape(-1, 2)
                else:
                    world_points, image_points = None, None
            else:
                world_points, image_points = None, None
                theta = thetas[-1] if thetas else 0
                thetas.append(theta)
            
            q = ekf.update(gyro, accel, image_points, world_points, row['dt'], theta)
            q = ensure_quaternion_continuity(q, q_prev)
            estimated_quaternions.append(q)
            estimated_euler_angles.append(quaternion_to_euler(q))
            timestamps.append(row['#timestamp [ns]'])
            q_prev = q

            progress_bar.update(1)

        progress_bar.close()

        # Save match data to CSV with sequence name
        match_df = pd.DataFrame(match_data)
        match_output_file = f'{sequence_name}_match_data.csv'
        match_df.to_csv(match_output_file, index=False)
        if verbose:
            print(f"Match data saved to {match_output_file}")

        estimated_quaternions = np.array(estimated_quaternions)
        estimated_euler_angles = np.array(estimated_euler_angles)
        thetas = np.array(thetas)

        true_quaternions = imu_data[[' q_RS_w []', ' q_RS_x []', ' q_RS_y []', ' q_RS_z []']].values
        true_euler_angles = np.array([quaternion_to_euler(q) for q in true_quaternions])

        aligned_quaternions = align_quaternions(estimated_quaternions, true_quaternions)
        aligned_euler_angles = np.array([quaternion_to_euler(q) for q in aligned_quaternions])

        aligned_euler_angles = ensure_angle_continuity(aligned_euler_angles)
        true_euler_angles = ensure_angle_continuity(true_euler_angles[:len(aligned_euler_angles)])

        rmse_quaternions = np.sqrt(((aligned_quaternions - true_quaternions[:len(aligned_quaternions)]) ** 2).mean(axis=0))
        rmse_euler_angles = calculate_angle_rmse(aligned_euler_angles, true_euler_angles)

        return imu_data, aligned_quaternions, aligned_euler_angles, true_quaternions[:len(aligned_quaternions)], true_euler_angles, rmse_quaternions, rmse_euler_angles, thetas, timestamps

    except Exception as e:
        if verbose:
            print(f"An error occurred in process_imu_data: {e}")
        raise

def calculate_angle_rmse(predictions, targets):
    diff = np.array([angle_difference(p, t) for p, t in zip(predictions, targets)])
    return np.sqrt((diff ** 2).mean(axis=0))

def angle_difference(angle1, angle2):
    diff = angle1 - angle2
    return (diff + 180) % 360 - 180

def preprocess_imu_data(base_path, cam_to_imu_timeshift=5.63799926987e-05):
    try:
        # Verileri okuma
        imu_df = pd.read_csv(base_path / 'imu0/data.csv')
        groundtruth_df = pd.read_csv(base_path / 'state_groundtruth_estimate0/data.csv')
        
        # Nanosaniye cinsinden timeshift hesaplama
        timeshift_ns = int(cam_to_imu_timeshift * 1e9)
        
        # Groundtruth verilerindeki timestamp'leri düzeltme
        # t_imu = t_cam + shift formülüne göre
        groundtruth_df['#timestamp'] = groundtruth_df['#timestamp'].apply(
            lambda x: x + timeshift_ns
        )
        
        groundtruth_df.set_index('#timestamp', inplace=True)
        groundtruth_df.sort_index(inplace=True)
        
        # Tüm sayısal sütunları seç
        numeric_cols = groundtruth_df.select_dtypes(include=[np.number]).columns.tolist()
        # Timestamp sütununu çıkart (eğer varsa)
        numeric_cols = [col for col in numeric_cols if col != '#timestamp']
        
        # Zaman kayması düzeltilmiş timestamp'leri kullanarak interpolasyon
        for col in numeric_cols:
            if col in groundtruth_df.columns:
                imu_df[col] = np.interp(
                    imu_df['#timestamp [ns]'],
                    groundtruth_df.index.values,
                    groundtruth_df[col].values
                )
        
        output_file = base_path / 'imu0/imu_with_interpolated_groundtruth.csv'
        imu_df.to_csv(output_file, index=False)
        print(f"Preprocessed IMU data saved to {output_file}")
        
        # Düzeltme miktarını kontrol etmek için bilgi yazdırma
        print(f"Applied timeshift correction: {timeshift_ns} ns ({cam_to_imu_timeshift} s)")
        
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        raise


if __name__ == "__main__":
    print("This module is not meant to be run directly. Please use main.py instead.")
