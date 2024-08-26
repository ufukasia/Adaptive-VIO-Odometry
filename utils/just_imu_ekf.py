import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

class ExtendedKalmanFilter:
    def __init__(self, initial_quaternion):
        
        
        self.q = initial_quaternion

        self.P = np.diag([1e-10, 1e-10, 1e-10, 1e-10])  # Increased initial uncertainty
        self.Q = np.diag([1e-10, 1e-10, 1e-10, 1e-8])  # Increased process noise for roll (last component)
        self.R = np.diag([0.001, 0.001, 0.001])  # Increased noise for roll (last component)






    def update(self, gyroscope, accelerometer, dt):
        q = self.q
        self.P = self.P + self.Q
        if np.all(accelerometer == 0):
            return self.q
        accelerometer = accelerometer / np.linalg.norm(accelerometer)

        # Prediction step
        qdot = 0.5 * np.array([
            -q[1]*gyroscope[0] - q[2]*gyroscope[1] - q[3]*gyroscope[2],
            q[0]*gyroscope[0] + q[2]*gyroscope[2] - q[3]*gyroscope[1],
            q[0]*gyroscope[1] - q[1]*gyroscope[2] + q[3]*gyroscope[0],
            q[0]*gyroscope[2] + q[1]*gyroscope[1] - q[2]*gyroscope[0]
        ])

        q_pred = q + qdot * dt
        q_pred = q_pred / np.linalg.norm(q_pred)

        # Compute objective function and Jacobian
        F = np.array([
            2*(q_pred[1]*q_pred[3] - q_pred[0]*q_pred[2]) - accelerometer[0],
            2*(q_pred[0]*q_pred[1] + q_pred[2]*q_pred[3]) - accelerometer[1],
            2*(0.5 - q_pred[1]**2 - q_pred[2]**2) - accelerometer[2]
        ])
        J = np.array([
            [-2*q_pred[2], 2*q_pred[3], -2*q_pred[0], 2*q_pred[1]],
            [2*q_pred[1], 2*q_pred[0], 2*q_pred[3], 2*q_pred[2]],
            [0, -4*q_pred[1], -4*q_pred[2], 0]
        ])

        # EKF update step
        H = J
        y = -F
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        q_update = K @ y
        self.q = q_pred + q_update
        self.q = self.q / np.linalg.norm(self.q)

        self.P = (np.eye(4) - K @ H) @ self.P

        return self.q

def quaternion_to_euler(q):
    r = R.from_quat(q)
    return r.as_euler('xyz', degrees=True)

def ensure_quaternion_continuity(q, q_prev):
    if np.dot(q, q_prev) < 0:
        return -q
    return q

def align_quaternions(estimated_quaternions, true_quaternions):
    min_length = min(len(estimated_quaternions), len(true_quaternions))
    aligned_quaternions = np.copy(estimated_quaternions[:min_length])
    for i in range(min_length):
        if np.dot(estimated_quaternions[i], true_quaternions[i]) < 0:
            aligned_quaternions[i] = -aligned_quaternions[i]
    return aligned_quaternions

def ensure_angle_continuity(angles, threshold=180):
    return np.unwrap(angles, axis=0, period=2*threshold)

def calculate_angle_rmse(predictions, targets):
    diff = np.array([angle_difference(p, t) for p, t in zip(predictions, targets)])
    return np.sqrt((diff ** 2).mean(axis=0))

def angle_difference(angle1, angle2):
    diff = angle1 - angle2
    return (diff + 180) % 360 - 180

def process_imu_data(file_path):
    # Calibration values
    calibration = {
        'w_RS_S_x [rad s^-1]': -0.00339,
        'w_RS_S_y [rad s^-1]': 0.020582,
        'w_RS_S_z [rad s^-1]': 0.079360,#roll değeri 
        'a_RS_S_x [m s^-2]': 0.049,
        'a_RS_S_y [m s^-2]': 0.084,
        'a_RS_S_z [m s^-2]': 0.0428
    }

    # Read CSV file
    data = pd.read_csv(file_path)

    # Get initial quaternion from the first row of the CSV
    initial_quaternion = data[[' q_RS_w []', ' q_RS_x []', ' q_RS_y []', ' q_RS_z []']].iloc[0].values

    # Calculate time difference
    data['timestamp'] = pd.to_datetime(data['#timestamp [ns]'], unit='ns')
    data['dt'] = data['timestamp'].diff().dt.total_seconds()
    data.loc[0, 'dt'] = 0

    # Apply calibration
    for key in calibration:
        data[key] = data[key] - calibration[key]

    # Create EKF
    ekf = ExtendedKalmanFilter(initial_quaternion=initial_quaternion)

    # Calculate quaternions and Euler angles
    estimated_quaternions = []
    estimated_euler_angles = []
    q_prev = initial_quaternion

    for _, row in data.iterrows():
        gyro = np.array([row['w_RS_S_x [rad s^-1]'], row['w_RS_S_y [rad s^-1]'], row['w_RS_S_z [rad s^-1]']])
        accel = np.array([row['a_RS_S_x [m s^-2]'], row['a_RS_S_y [m s^-2]'], row['a_RS_S_z [m s^-2]']])
        
        q = ekf.update(gyro, accel, row['dt'])
        q = ensure_quaternion_continuity(q, q_prev)
        estimated_quaternions.append(q)
        estimated_euler_angles.append(quaternion_to_euler(q))
        q_prev = q

    estimated_quaternions = np.array(estimated_quaternions)
    estimated_euler_angles = np.array(estimated_euler_angles)

    # Calculate true quaternions and Euler angles
    true_quaternions = data[[' q_RS_w []', ' q_RS_x []', ' q_RS_y []', ' q_RS_z []']].values
    true_euler_angles = np.array([quaternion_to_euler(q) for q in true_quaternions])

    # Align estimated quaternions with true quaternions
    aligned_quaternions = align_quaternions(estimated_quaternions, true_quaternions)
    aligned_euler_angles = np.array([quaternion_to_euler(q) for q in aligned_quaternions])

    # Ensure continuity in Euler angles
    aligned_euler_angles = ensure_angle_continuity(aligned_euler_angles)
    true_euler_angles = ensure_angle_continuity(true_euler_angles[:len(aligned_euler_angles)])

    # Calculate RMSE
    rmse_quaternions = np.sqrt(((aligned_quaternions - true_quaternions[:len(aligned_quaternions)]) ** 2).mean(axis=0))
    rmse_euler_angles = calculate_angle_rmse(aligned_euler_angles, true_euler_angles)

    return data, aligned_quaternions, aligned_euler_angles, true_quaternions[:len(aligned_quaternions)], true_euler_angles, rmse_quaternions, rmse_euler_angles

def visualize_results(data, aligned_quaternions, aligned_euler_angles, true_quaternions, true_euler_angles, rmse_quaternions, rmse_euler_angles):
    fig, axs = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle('Quaternion and Euler Angle Comparison (EKF)', fontsize=16)

    # Quaternion plots
    q_labels = ['w', 'x', 'y', 'z']
    for i, label in enumerate(q_labels):
        axs[i, 0].plot(data['timestamp'][:len(aligned_quaternions)], aligned_quaternions[:, i], label='Estimated', color='blue')
        axs[i, 0].plot(data['timestamp'][:len(true_quaternions)], true_quaternions[:, i], label='True', color='red', linestyle='--')
        axs[i, 0].set_title(f'Quaternion {label} (RMSE: {rmse_quaternions[i]:.4f})')
        axs[i, 0].set_xlabel('Time')
        axs[i, 0].set_ylabel(f'q_{label}')
        axs[i, 0].legend()

    # Euler angle plots
    euler_labels = ['Roll', 'Pitch', 'Yaw']
    for i, label in enumerate(euler_labels):
        axs[i, 1].plot(data['timestamp'][:len(aligned_euler_angles)], aligned_euler_angles[:, i], label='Estimated', color='blue')
        axs[i, 1].plot(data['timestamp'][:len(true_euler_angles)], true_euler_angles[:, i], label='True', color='red', linestyle='--')
        axs[i, 1].set_title(f'{label} (RMSE: {rmse_euler_angles[i]:.4f})')
        axs[i, 1].set_xlabel('Time')
        axs[i, 1].set_ylabel('Angle (degrees)')
        axs[i, 1].legend()

    # Remove empty subplot
    fig.delaxes(axs[3, 1])

    plt.tight_layout()
    plt.show()

def visualize_error(data, aligned_quaternions, aligned_euler_angles, true_quaternions, true_euler_angles):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle('Quaternion and Euler Angle Error Over Time', fontsize=16)

    # Quaternion error plot
    q_error = np.abs(aligned_quaternions - true_quaternions)
    ax1.plot(data['timestamp'][:len(q_error)], q_error[:, 0], label='w', color='red')
    ax1.plot(data['timestamp'][:len(q_error)], q_error[:, 1], label='x', color='green')
    ax1.plot(data['timestamp'][:len(q_error)], q_error[:, 2], label='y', color='blue')
    ax1.plot(data['timestamp'][:len(q_error)], q_error[:, 3], label='z', color='purple')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Quaternion Error')
    ax1.set_title('Quaternion Error Over Time')
    ax1.legend()
    ax1.grid(True)

    # Euler angle error plot
    euler_error = np.abs(aligned_euler_angles - true_euler_angles)
    ax2.plot(data['timestamp'][:len(euler_error)], euler_error[:, 0], label='Roll', color='red')
    ax2.plot(data['timestamp'][:len(euler_error)], euler_error[:, 1], label='Pitch', color='green')
    ax2.plot(data['timestamp'][:len(euler_error)], euler_error[:, 2], label='Yaw', color='blue')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Euler Angle Error (degrees)')
    ax2.set_title('Euler Angle Error Over Time')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = 'MH_01_easy\mav0\imu0\imu_with_interpolated_groundtruth.csv'

    data, aligned_quaternions, aligned_euler_angles, true_quaternions, true_euler_angles, rmse_quaternions, rmse_euler_angles = process_imu_data(file_path)

    print("Quaternion RMSE:", rmse_quaternions)
    print("Euler Angle RMSE:", rmse_euler_angles)

    visualize_results(data, aligned_quaternions, aligned_euler_angles, true_quaternions, true_euler_angles, rmse_quaternions, rmse_euler_angles)
    visualize_error(data, aligned_quaternions, aligned_euler_angles, true_quaternions, true_euler_angles)