import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# --------------------------------------------------------------------------------
# IMPORTANT: QUATERNION ORDER CORRECTION
# --------------------------------------------------------------------------------
# Scipy's Rotation.from_quat() expects quaternions in [x, y, z, w] order.
# However, our data provides quaternions in [w, x, y, z] order.
# To ensure correct rotation matrices and subsequent calculations,
# we must reorder the quaternions before using them with Scipy.

def convert_quaternion_order(q_wxyz):
    q_wxyz = np.array(q_wxyz)
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])

class ExtendedKalmanFilter:
    def __init__(self, initial_quaternion, initial_velocity, gravity_vector):
        initial_quaternion_xyzw = convert_quaternion_order(initial_quaternion)
        
        self.q = initial_quaternion  # [w, x, y, z] order
        self.v = initial_velocity
        self.state = np.concatenate([self.q, self.v])
        self.P = np.diag([1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10])  # Initial uncertainty
        self.Q = np.diag([1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10])   # Process noise
        self.R = np.diag([2000, 2000, 2000])  # Measurement noise for accelerometer only
        self.g = gravity_vector

    def update(self, gyroscope, accelerometer, dt):
        q = self.state[:4]  # [w, x, y, z]
        v = self.state[4:]
        self.P = self.P + self.Q

        # Prediction step
        qdot = 0.5 * np.array([
            -q[1]*gyroscope[0] - q[2]*gyroscope[1] - q[3]*gyroscope[2],
             q[0]*gyroscope[0] + q[2]*gyroscope[2] - q[3]*gyroscope[1],
             q[0]*gyroscope[1] - q[1]*gyroscope[2] + q[3]*gyroscope[0],
             q[0]*gyroscope[2] + q[1]*gyroscope[1] - q[2]*gyroscope[0]
        ])

        q_pred = q + qdot * dt
        q_pred = q_pred / np.linalg.norm(q_pred)

        # Convert predicted quaternion to [x, y, z, w] order for Scipy
        q_pred_xyzw = convert_quaternion_order(q_pred)
        R_pred = R.from_quat(q_pred_xyzw).as_matrix()

        a_global = R_pred @ accelerometer + self.g
        v_pred = v + a_global * dt

        state_pred = np.concatenate([q_pred, v_pred])

        # Measurement prediction (specific force: f_s = R_pred.T @ g)
        a_pred = R_pred.T @ self.g
        z_pred = a_pred

        # Compute residual
        z = accelerometer  # Measured specific force
        y = z - z_pred

        # Compute Jacobian H for accelerometer only (orientation)
        H = self.compute_jacobian(state_pred, measurement_dim=3)
        H[:,4:7] = 0  # Zero out velocity components to prevent velocity update

        # EKF update step
        S = H @ self.P @ H.T + self.R  # Measurement noise for accelerometer
        K = self.P @ H.T @ np.linalg.inv(S)

        state_update = K @ y
        self.state = state_pred + state_update
        self.state[:4] = self.state[:4] / np.linalg.norm(self.state[:4])

        self.P = (np.eye(7) - K @ H) @ self.P

        return self.state       

    def compute_jacobian(self, state, measurement_dim=3):
        H = np.zeros((measurement_dim, 7))
        delta = 1e-6
        for i in range(7):
            if i >= 4:
                H[:, i] = 0  # Zero out velocity components in Jacobian
                continue

            dstate = np.zeros(7)
            dstate[i] = delta

            state_plus = state + dstate
            state_plus[:4] /= np.linalg.norm(state_plus[:4])
            q_plus_xyzw = convert_quaternion_order(state_plus[:4])
            R_plus = R.from_quat(q_plus_xyzw).as_matrix()
            a_pred_plus = R_plus.T @ self.g
            z_plus = a_pred_plus

            state_minus = state - dstate
            state_minus[:4] /= np.linalg.norm(state_minus[:4])
            q_minus_xyzw = convert_quaternion_order(state_minus[:4])
            R_minus = R.from_quat(q_minus_xyzw).as_matrix()
            a_pred_minus = R_minus.T @ self.g
            z_minus = a_pred_minus

            H[:, i] = (z_plus - z_minus) / (2 * delta)
        return H

def quaternion_to_euler(q):
    # Convert quaternion from [w, x, y, z] to [x, y, z, w] order for Scipy
    q_xyzw = [q[1], q[2], q[3], q[0]]
    r = R.from_quat(q_xyzw)
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
        'w_RS_S_z [rad s^-1]': 0.079360,  # Roll value
        'a_RS_S_x [m s^-2]': 0.049,
        'a_RS_S_y [m s^-2]': 0.084,
        'a_RS_S_z [m s^-2]': 0.0428
    }

    # Read CSV file
    data = pd.read_csv(file_path)

    # Get initial quaternion and velocity from the first row of the CSV
    initial_quaternion = data[[' q_RS_w []', ' q_RS_x []', ' q_RS_y []', ' q_RS_z []']].iloc[0].values
    initial_velocity = data[[' v_RS_R_x [m s^-1]', ' v_RS_R_y [m s^-1]', ' v_RS_R_z [m s^-1]']].iloc[0].values

    # Calculate time difference
    data['timestamp'] = pd.to_datetime(data['#timestamp [ns]'], unit='ns')
    data['dt'] = data['timestamp'].diff().dt.total_seconds()
    data.loc[0, 'dt'] = 0

    # Apply calibration
    for key in calibration:
        data[key] = data[key] - calibration[key]

    # Known gravity vector (global frame)
    gravity_vector = np.array([0, 0, -9.81])  # Standart yerçekimi vektörü

    # Create EKF
    ekf = ExtendedKalmanFilter(initial_quaternion=initial_quaternion, initial_velocity=initial_velocity, gravity_vector=gravity_vector)

    # Initialize lists
    estimated_states = []
    estimated_euler_angles = []
    estimated_velocities = []
    q_prev = initial_quaternion

    # Process data
    for _, row in data.iterrows():
        gyro = np.array([
            row['w_RS_S_x [rad s^-1]'],
            row['w_RS_S_y [rad s^-1]'],
            row['w_RS_S_z [rad s^-1]']
        ])
        accel = np.array([
            row['a_RS_S_x [m s^-2]'],
            row['a_RS_S_y [m s^-2]'],
            row['a_RS_S_z [m s^-2]']
        ])

        state = ekf.update(gyro, accel, row['dt'])
        q = state[:4]
        v = state[4:]
        q = ensure_quaternion_continuity(q, q_prev)
        estimated_states.append(state)
        estimated_euler_angles.append(quaternion_to_euler(q))
        estimated_velocities.append(v)
        q_prev = q

    estimated_states = np.array(estimated_states)
    estimated_quaternions = estimated_states[:, :4]
    estimated_velocities = estimated_states[:, 4:]
    estimated_euler_angles = np.array(estimated_euler_angles)

    # Calculate true quaternions, Euler angles, and velocities
    true_quaternions = data[[' q_RS_w []', ' q_RS_x []', ' q_RS_y []', ' q_RS_z []']].values
    true_euler_angles = np.array([quaternion_to_euler(q) for q in true_quaternions])
    true_velocities = data[[' v_RS_R_x [m s^-1]', ' v_RS_R_y [m s^-1]', ' v_RS_R_z [m s^-1]']].values

    # Align estimated quaternions with true quaternions
    aligned_quaternions = align_quaternions(estimated_quaternions, true_quaternions)
    aligned_euler_angles = np.array([quaternion_to_euler(q) for q in aligned_quaternions])

    # Ensure continuity in Euler angles
    aligned_euler_angles = ensure_angle_continuity(aligned_euler_angles)
    true_euler_angles = ensure_angle_continuity(true_euler_angles[:len(aligned_euler_angles)])

    # Calculate RMSE
    rmse_quaternions = np.sqrt(((aligned_quaternions - true_quaternions[:len(aligned_quaternions)]) ** 2).mean(axis=0))
    rmse_euler_angles = calculate_angle_rmse(aligned_euler_angles, true_euler_angles)
    rmse_velocities = np.sqrt(((estimated_velocities - true_velocities[:len(estimated_velocities)]) ** 2).mean(axis=0))

    return data, aligned_quaternions, aligned_euler_angles, estimated_velocities, true_quaternions[:len(aligned_quaternions)], true_euler_angles, true_velocities[:len(estimated_velocities)], rmse_quaternions, rmse_euler_angles, rmse_velocities

def visualize_results(data, aligned_quaternions, aligned_euler_angles, estimated_velocities, true_quaternions, true_euler_angles, true_velocities, rmse_quaternions, rmse_euler_angles, rmse_velocities):
    # Toplamda 10 subplot oluşturuyoruz: 4 quaternion, 3 euler açıları, 3 velocity
    fig, axs = plt.subplots(10, 1, figsize=(15, 35))
    fig.suptitle('Quaternion, Euler Angle, and Velocity Comparison (EKF with Gravity Vector)', fontsize=16)

    # Quaternion plots (ilk 4 subplot)
    q_labels = ['w', 'x', 'y', 'z']
    for i, label in enumerate(q_labels):
        axs[i].plot(data['timestamp'][:len(aligned_quaternions)], aligned_quaternions[:, i], label='Estimated', color='blue')
        axs[i].plot(data['timestamp'][:len(true_quaternions)], true_quaternions[:, i], label='True', color='red', linestyle='--')
        axs[i].set_title(f'Quaternion {label} (RMSE: {rmse_quaternions[i]:.4f})')
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel(f'q_{label}')
        axs[i].legend()
        axs[i].grid(True)

    # Euler angle plots (sonraki 3 subplot)
    euler_labels = ['Roll', 'Pitch', 'Yaw']
    for i, label in enumerate(euler_labels):
        axs[i + 4].plot(data['timestamp'][:len(aligned_euler_angles)], aligned_euler_angles[:, i], label='Estimated', color='blue')
        axs[i + 4].plot(data['timestamp'][:len(true_euler_angles)], true_euler_angles[:, i], label='True', color='red', linestyle='--')
        axs[i + 4].set_title(f'{label} (RMSE: {rmse_euler_angles[i]:.4f})')
        axs[i + 4].set_xlabel('Time')
        axs[i + 4].set_ylabel('Angle (degrees)')
        axs[i + 4].legend()
        axs[i + 4].grid(True)

    # Velocity plots (son 3 subplot)
    v_labels = ['x', 'y', 'z']
    for i, label in enumerate(v_labels):
        axs[i + 7].plot(data['timestamp'][:len(estimated_velocities)], estimated_velocities[:, i], label='Estimated', color='blue')
        axs[i + 7].plot(data['timestamp'][:len(true_velocities)], true_velocities[:, i], label='True', color='red', linestyle='--')
        axs[i + 7].set_title(f'Velocity {label} (RMSE: {rmse_velocities[i]:.4f})')
        axs[i + 7].set_xlabel('Time')
        axs[i + 7].set_ylabel(f'v_{label} (m/s)')
        axs[i + 7].legend()
        axs[i + 7].grid(True)
        
        # Y-eksenini -5 ile 5 arasında sınırlama
        axs[i + 7].set_ylim(-2, 2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Ana başlık için yer bırak
    plt.show()


def visualize_error(data, aligned_quaternions, aligned_euler_angles, estimated_velocities, true_quaternions, true_euler_angles, true_velocities):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18))
    fig.suptitle('Quaternion, Euler Angle, and Velocity Error Over Time', fontsize=16)

    # Quaternion error plot
    q_error = np.abs(aligned_quaternions - true_quaternions[:len(aligned_quaternions)])
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
    euler_error = np.abs(aligned_euler_angles - true_euler_angles[:len(aligned_euler_angles)])
    ax2.plot(data['timestamp'][:len(euler_error)], euler_error[:, 0], label='Roll', color='red')
    ax2.plot(data['timestamp'][:len(euler_error)], euler_error[:, 1], label='Pitch', color='green')
    ax2.plot(data['timestamp'][:len(euler_error)], euler_error[:, 2], label='Yaw', color='blue')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Euler Angle Error (degrees)')
    ax2.set_title('Euler Angle Error Over Time')
    ax2.legend()
    ax2.grid(True)

    # Velocity error plot
    v_error = np.abs(estimated_velocities - true_velocities[:len(estimated_velocities)])
    ax3.plot(data['timestamp'][:len(v_error)], v_error[:, 0], label='x', color='red')
    ax3.plot(data['timestamp'][:len(v_error)], v_error[:, 1], label='y', color='green')
    ax3.plot(data['timestamp'][:len(v_error)], v_error[:, 2], label='z', color='blue')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Velocity Error (m/s)')
    ax3.set_title('Velocity Error Over Time')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make space for the main title
    plt.show()

if __name__ == "__main__":
    file_path = 'MH_03_medium/mav0/imu0/imu_with_interpolated_groundtruth.csv'

    data, aligned_quaternions, aligned_euler_angles, estimated_velocities, true_quaternions, true_euler_angles, true_velocities, rmse_quaternions, rmse_euler_angles, rmse_velocities = process_imu_data(file_path)

    print("Quaternion RMSE:", rmse_quaternions)
    print("Euler Angle RMSE:", rmse_euler_angles)
    print("Velocity RMSE:", rmse_velocities)

    visualize_results(data, aligned_quaternions, aligned_euler_angles, estimated_velocities, true_quaternions, true_euler_angles, true_velocities, rmse_quaternions, rmse_euler_angles, rmse_velocities)
    visualize_error(data, aligned_quaternions, aligned_euler_angles, estimated_velocities, true_quaternions, true_euler_angles, true_velocities)
