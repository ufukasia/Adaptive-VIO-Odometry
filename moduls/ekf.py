import numpy as np
from scipy.spatial.transform import Rotation
import cv2

class ExtendedKalmanFilter:
    def __init__(self, initial_quaternion, camera_matrix, dist_coeffs, theta_threshold, 
                 process_noise_scale=1e-10, measurement_noise_scale=1e-3):
        self.q = initial_quaternion
        self.P = np.diag([1e-10, 1e-10, 1e-10, 1e-10])
        self.Q = np.diag([process_noise_scale] * 4)
        self.R_camera = np.diag([measurement_noise_scale, measurement_noise_scale])
        self.R_imu = np.diag([0.0001, 0.001, 0.001, 0.00001, 0.00001, 0.00001])
        self.theta_threshold = theta_threshold
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def update(self, gyroscope, accelerometer, image_points, world_points, dt, theta, 
               camera_weight=0.5, imu_weight=0.5):
        q = self.q
        if np.all(accelerometer == 0) and np.all(gyroscope == 0):
            return self.q
        if np.linalg.norm(accelerometer) != 0:
            accelerometer = accelerometer / np.linalg.norm(accelerometer)
        q_pred = self.predict(q, gyroscope, dt)
        
        deadline = 0.50

        if theta > deadline:
            self.q = self.update_imu_only(q_pred, gyroscope, accelerometer)
        elif image_points is not None and world_points is not None and len(image_points) >= 4:
            if theta <= self.theta_threshold:
                q_camera = self.update_with_camera(q_pred, image_points, world_points, gyroscope, accelerometer)
                q_imu = self.update_imu_only(q_pred, gyroscope, accelerometer)
                self.q = self.weighted_quaternion_average([q_camera, q_imu], [camera_weight, imu_weight])
            else:
                camera_reliability = max(0, (deadline - theta) / (deadline - self.theta_threshold))
                q_camera = self.update_with_camera(q_pred, image_points, world_points, gyroscope, accelerometer)
                q_imu = self.update_imu_only(q_pred, gyroscope, accelerometer)
                self.q = self.weighted_quaternion_average([q_camera, q_imu], [camera_reliability * camera_weight, (1 - camera_reliability) * imu_weight])
        else:
            self.q = self.update_imu_only(q_pred, gyroscope, accelerometer)
        return self.q

    def predict(self, q, gyroscope, dt):
        omega = np.array([0, *gyroscope])
        q_dot = 0.5 * self.quaternion_multiply(q, omega)
        q_pred = q + q_dot * dt
        q_pred = q_pred / np.linalg.norm(q_pred)
        
        F = self.calculate_state_transition_matrix(q, gyroscope, dt)
        self.P = F @ self.P @ F.T + self.Q
        
        return q_pred

    def calculate_state_transition_matrix(self, q, gyroscope, dt):
        F = np.eye(4) + dt * 0.5 * np.array([
            [0, -gyroscope[0], -gyroscope[1], -gyroscope[2]],
            [gyroscope[0], 0, gyroscope[2], -gyroscope[1]],
            [gyroscope[1], -gyroscope[2], 0, gyroscope[0]],
            [gyroscope[2], gyroscope[1], -gyroscope[0], 0]
        ])
        return F

    def update_imu_only(self, q_pred, gyroscope, accelerometer):
        F_imu = np.array([
            2*(q_pred[1]*q_pred[3] - q_pred[0]*q_pred[2]) - accelerometer[0],
            2*(q_pred[0]*q_pred[1] + q_pred[2]*q_pred[3]) - accelerometer[1],
            2*(0.5 - q_pred[1]**2 - q_pred[2]**2) - accelerometer[2],
            *gyroscope
        ])
        J_imu = np.array([
            [-2*q_pred[2], 2*q_pred[3], -2*q_pred[0], 2*q_pred[1]],
            [2*q_pred[1], 2*q_pred[0], 2*q_pred[3], 2*q_pred[2]],
            [0, -4*q_pred[1], -4*q_pred[2], 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        H = J_imu
        y = -F_imu
        R = self.R_imu

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        x_update = K @ y
        q_update = x_update[:4]

        q = q_pred + q_update
        q = q / np.linalg.norm(q)

        self.P = (np.eye(4) - K @ H) @ self.P

        return q

    def update_with_camera(self, q_pred, image_points, world_points, gyroscope, accelerometer):
        success, rvec, tvec = self.pnp_solve(world_points, image_points, q_pred)
        
        if not success:
            return self.update_imu_only(q_pred, gyroscope, accelerometer)
        
        r_mat, _ = cv2.Rodrigues(rvec)
        q_camera = Rotation.from_matrix(r_mat).as_quat()
        
        y_camera = q_camera - q_pred
        
        H_camera = np.eye(4)
        
        F_imu = np.array([
            2*(q_pred[1]*q_pred[3] - q_pred[0]*q_pred[2]) - accelerometer[0],
            2*(q_pred[0]*q_pred[1] + q_pred[2]*q_pred[3]) - accelerometer[1],
            2*(0.5 - q_pred[1]**2 - q_pred[2]**2) - accelerometer[2],
            *gyroscope
        ])
        J_imu = np.array([
            [-2*q_pred[2], 2*q_pred[3], -2*q_pred[0], 2*q_pred[1]],
            [2*q_pred[1], 2*q_pred[0], 2*q_pred[3], 2*q_pred[2]],
            [0, -4*q_pred[1], -4*q_pred[2], 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        
        H = np.vstack((J_imu, H_camera))
        y = np.concatenate((-F_imu, y_camera))
        R = np.block([
            [self.R_imu, np.zeros((6, 4))],
            [np.zeros((4, 6)), np.eye(4) * self.R_camera[0, 0]]
        ])

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        x_update = K @ y
        q_update = x_update[:4]

        q = q_pred + q_update
        q = q / np.linalg.norm(q)

        self.P = (np.eye(4) - K @ H) @ self.P

        return q

    def quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        return np.array([w, x, y, z])

    def pnp_solve(self, world_points, image_points, q_pred):
        if len(world_points) < 4:
            return False, None, None
        
        rvec_init = Rotation.from_quat(q_pred).as_rotvec()
        tvec_init = np.zeros(3)
        
        try:
            success, rvec, tvec = cv2.solvePnP(world_points, image_points, self.camera_matrix, self.dist_coeffs,
                                               rvec=rvec_init, tvec=tvec_init, useExtrinsicGuess=True)
        except cv2.error:
            return False, None, None
        
        return success, rvec, tvec

    def weighted_quaternion_average(self, quaternions, weights):
        avg = np.zeros(4)
        for q, w in zip(quaternions, weights):
            avg += q * w
        avg = avg / np.linalg.norm(avg)
        return avg

# Yardımcı fonksiyonlar
def quaternion_to_euler(q):
    r = Rotation.from_quat(q)
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