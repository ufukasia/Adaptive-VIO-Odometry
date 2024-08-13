# Adaptive Visual-Inertial Odometry Using SuperGlue and Dynamic EKF

This project implements an Adaptive Visual-Inertial Odometry (VIO) system using SuperGlue for feature matching and a Dynamic Extended Kalman Filter (EKF) with Information-Based Confidence Estimation. The system is designed to provide robust and accurate pose estimation in challenging environments.

## Features

- Adaptive fusion of visual and inertial data
- SuperGlue-based feature matching for improved visual odometry
- Dynamic EKF with confidence-based weighting
- Support for various activation functions for confidence estimation
- Automatic dataset download and preprocessing
- Comprehensive visualization of results

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/adaptive-vio-superglue.git
   cd adaptive-vio-superglue
   ```





2. Install PyTorch:
   
   For CPU-only:
   ```
   pip3 install torch torchvision torchaudio
   ```
   
   For GPU (CUDA) support:
   ```
   pip3 install torch torchvision torchaudio --index-url --index-url https://download.pytorch.org/whl/cu118
   ```
   Note: Replace `cu118` with your CUDA version if different.   Please visit https://pytorch.org/get-started/locally/ for different installations

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```






## Usage

The main script `main.py` can be run with various command-line arguments to customize the VIO processing:

```
python main.py [OPTIONS]
```

### Options:

- `--dataset_path`: Path to save the dataset (default: current directory)
- `--sequence`: Dataset sequence to use (default: MH_04_difficult)
- `--download`: Force download the dataset even if it exists
- `--alpha`: Alpha parameter for confidence estimation (default: 1)
- `--beta`: Beta parameter for confidence estimation (default: 1)
- `--gamma`: Gamma parameter for confidence estimation (default: 1)
- `--theta_threshold`: Theta threshold for adaptive fusion (default: 0.3)
- `--activation_function`: Activation function to use for confidence estimation (default: double_exponential_sigmoid)
- `--generate_superglue_visualizations`: Generate SuperGlue visualizations

### Example Usage:

1. Run with default settings:
   ```
   python main.py
   ```

2. Use a specific dataset sequence and download it:
   ```
   python main.py --sequence MH_01_easy --download
   ```

3. Customize confidence estimation parameters:
   ```
   python main.py --alpha 2.5 --beta 1.2 --gamma 0.8 --theta_threshold 0.25
   ```

4. Use a different activation function and generate SuperGlue visualizations:
   ```
   python main.py --activation_function relu --generate_superglue_visualizations
   ```

5. Process a locally stored dataset:
   ```
   python main.py --dataset_path /path/to/your/dataset --sequence custom_sequence
   ```

## Output

The script will generate the following outputs:

- Preprocessed IMU data
- Estimated pose values in CSV format
- Visualization plots for quaternions, Euler angles, and errors
- SuperGlue visualizations (if enabled)

## Contributing

Contributions to improve the system or add new features are welcome. Please feel free to submit pull requests or open issues for any bugs or suggestions.

## License

MIT Licence

## Citation

If you use this code in your research, please cite our paper:

```
[Your paper citation here]
```

## Acknowledgements

This project uses the EuRoC MAV Dataset:

M. Burri, J. Nikolic, P. Gohl, T. Schneider, J. Rehder, S. Omari, M. Achtelik and R. Siegwart, The EuRoC micro aerial vehicle datasets, International Journal of Robotic Research, DOI: 10.1177/0278364915620033, 2016.