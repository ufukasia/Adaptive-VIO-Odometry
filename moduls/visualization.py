import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from pathlib import Path
import matplotlib.cm as cm
from models.matching import Matching
from models.utils import frame2tensor

# Try setting a different backend for matplotlib
plt.switch_backend('TkAgg')

def visualize_results(data, aligned_quaternions, aligned_euler_angles, true_quaternions, true_euler_angles, rmse_quaternions, rmse_euler_angles, thetas, sequence_name):
    fig, axs = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle('Pose Estimation Errors', fontsize=16, fontweight='bold', color='darkblue', fontfamily='serif')

    q_labels = ['w', 'x', 'y', 'z']
    for i, label in enumerate(q_labels):
        time = data['timestamp'][:len(aligned_quaternions)]
        
        ax1 = axs[i, 0]
        ax1.plot(time[::50], aligned_quaternions[::50, i], label='Estimated', color='blue')
        ax1.plot(time[::50], true_quaternions[::50, i], label='True', color='red', linestyle='--')
        
        ax1.set_title(f'Quaternion {label} (RMSE: {rmse_quaternions[i]:.4f})')
        ax1.set_xlabel('Time')
        ax1.set_ylabel(f'q_{label}')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)

        ax2 = ax1.twinx()
        ax2.plot(time[::50], thetas[::50], label='Theta', color='green', alpha=0.5)
        ax2.set_ylabel('Theta', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

    euler_labels = ['Roll', 'Pitch', 'Yaw']
    for i, label in enumerate(euler_labels):
        ax1 = axs[i, 1]
        ax1.plot(data['timestamp'][:len(aligned_euler_angles):50], aligned_euler_angles[::50, i], label='Estimated', color='blue')
        ax1.plot(data['timestamp'][:len(true_euler_angles):50], true_euler_angles[::50, i], label='True', color='red', linestyle='--')
        ax1.set_title(f'{label} (RMSE: {rmse_euler_angles[i]:.4f})')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Angle (degrees)')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)

        ax2 = ax1.twinx()
        ax2.plot(data['timestamp'][:len(thetas):50], thetas[::50], label='Theta', color='green', alpha=0.5)
        ax2.set_ylabel('Theta', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

    non_zero_thetas = thetas[thetas != 0]
    axs[3, 1].hist(non_zero_thetas, bins=50, color='green', alpha=0.7)
    axs[3, 1].set_title('Histogram of Non-Zero Theta Values')
    axs[3, 1].set_xlabel('Theta')
    axs[3, 1].set_ylabel('Frequency')
    axs[3, 1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
    print(f"Results plot displayed for {sequence_name}")

def visualize_error(data, aligned_quaternions, aligned_euler_angles, true_quaternions, true_euler_angles, thetas, sequence_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle('Quaternion and Euler Angle Error Over Time', fontsize=16)

    q_error = np.abs(aligned_quaternions - true_quaternions)
    ax1.plot(data['timestamp'][:len(q_error):50], q_error[::50, 0], label='w', color='red')
    ax1.plot(data['timestamp'][:len(q_error):50], q_error[::50, 1], label='x', color='green')
    ax1.plot(data['timestamp'][:len(q_error):50], q_error[::50, 2], label='y', color='blue')
    ax1.plot(data['timestamp'][:len(q_error):50], q_error[::50, 3], label='z', color='purple')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Quaternion Error')
    ax1.legend()
    ax1.grid(True)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(data['timestamp'][:len(thetas):50], thetas[::50], label='Theta', color='orange', alpha=0.5)
    ax1_twin.set_ylabel('Theta', color='orange')
    ax1_twin.tick_params(axis='y', labelcolor='orange')

    euler_error = np.abs(aligned_euler_angles - true_euler_angles)
    ax2.plot(data['timestamp'][:len(euler_error):50], euler_error[::50, 0], label='Roll', color='red')
    ax2.plot(data['timestamp'][:len(euler_error):50], euler_error[::50, 1], label='Pitch', color='green')
    ax2.plot(data['timestamp'][:len(euler_error):50], euler_error[::50, 2], label='Yaw', color='blue')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Euler Angle Error\n(degrees)')
    ax2.legend()
    ax2.grid(True)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(data['timestamp'][:len(thetas):50], thetas[::50], label='Theta', color='orange', alpha=0.5)
    ax2_twin.set_ylabel('Theta', color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')

    plt.tight_layout()
    plt.show()
    print(f"Error plot displayed for {sequence_name}")

def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1, color, text, path=None, show_keypoints=False, margin=10, keypoint_size=5):
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = np.zeros((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    out = np.stack([out]*3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), keypoint_size, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), keypoint_size-1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), keypoint_size, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), keypoint_size-1, white, -1, lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1), color=c, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x0, y0), keypoint_size, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), keypoint_size, c, -1, lineType=cv2.LINE_AA)

    sc = min(H / 640., 2.0)
    Ht = int(30 * sc)
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    return out

def process_image_pair(image1_path, image2_path, matching, device):
    image1 = cv2.imread(str(image1_path), cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(str(image2_path), cv2.IMREAD_GRAYSCALE)
    
    tensor1 = frame2tensor(image1, device=device)
    tensor2 = frame2tensor(image2, device=device)

    with torch.no_grad():
        pred = matching({'image0': tensor1, 'image1': tensor2})
    
    pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    
    color = cm.jet(conf[valid])
    text = [
        'SuperGlue',
        f'Matches: {len(mkpts0)}'
    ]
    
    plot = make_matching_plot_fast(
        image1, image2, kpts0, kpts1, mkpts0, mkpts1, color, text,
        path=None, show_keypoints=True, keypoint_size=5)
    
    return plot

def visualize_superglue(base_path, output_path, config):
    camera_data_path = base_path / 'cam0/data'
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(camera_data_path.glob("*.png"))
    matching, device = load_superglue_model(config)

    frame_interval = config['superglue_visualization']['frame_interval']
    max_pairs = config['superglue_visualization']['max_pairs']

    for i in range(0, min(len(image_paths) - frame_interval, max_pairs * frame_interval), frame_interval):
        plot = process_image_pair(image_paths[i], image_paths[i + frame_interval], matching, device)
        
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.imshow(plot)
        ax.axis('off')
        
        plt.savefig(output_path / f'superglue_match_{i:04d}.png', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close(fig)

    print(f"SuperGlue visualizations saved to {output_path}")

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
    matching = Matching(superglue_config).eval().to(device)
    return matching, device