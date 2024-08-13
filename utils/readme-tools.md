# Repository Tools

This repository contains a collection of Python scripts for various tasks related to image processing, CUDA testing, and Extended Kalman Filter implementation. Below is a detailed description of each tool and how to use them.

## Table of Contents

1. [CUDA Test](#cuda-test)
2. [Just IMU EKF](#just-imu-ekf)
3. [Multiprocessing Information Test](#multiprocessing-information-test)
4. [Synthetic Data Production](#synthetic-data-production)
5. [Activation Function Graphs](#activation-function-graphs)

## CUDA Test

This script checks the availability of CUDA and cuDNN on your system and reports their versions.

### Usage

```bash
python cuda_test.py
```

### Output

The script will print information about:
- PyTorch version
- CUDA availability and version
- CUDA capability (if available)
- cuDNN version (if available)

## Just IMU EKF

This script implements an Extended Kalman Filter (EKF) for processing IMU data.

### Usage

```bash
python just_imu_ekf.py
```

### Features

- Processes IMU data from a CSV file
- Implements an Extended Kalman Filter for quaternion estimation
- Visualizes results including quaternion and Euler angle comparisons
- Calculates and displays RMSE for quaternions and Euler angles

## Multiprocessing Information Test

This script processes multiple image datasets in parallel, calculating various information metrics.

### Usage

```bash
python multiprocessing_information_test.py [--alpha ALPHA] [--beta BETA] [--gamma GAMMA] [--output_dir OUTPUT_DIR]
```

### Arguments

- `--alpha`: Delta intensity scaling factor (default: 1)
- `--beta`: Entropy scaling factor (default: 1)
- `--gamma`: Motion blur scaling factor (default: 1)
- `--output_dir`: Directory to save highest value images (default: "highest_value_images")

### Example

```bash
python multiprocessing_information_test.py --alpha 1.5 --beta 0.8 --gamma 1.2 --output_dir "my_results"
```

### Features

- Processes multiple datasets in parallel
- Calculates delta intensity, entropy, and motion blur for each image
- Plots scaled information metrics for each dataset
- Saves images with the highest values for each metric

## Synthetic Data Production

This script downloads and processes EuRoC MAV datasets to create synthetic data with various effects.

### Usage

```bash
python synthetic_data_production.py [--dataset DATASET] [--interval INTERVAL] [--num_black NUM_BLACK] [--light_interval LIGHT_INTERVAL] [--light_change LIGHT_CHANGE] [--blur_interval BLUR_INTERVAL] [--blur_size BLUR_SIZE]
```

### Arguments

- `--dataset`: Name of the dataset to use (default: "MH_01_easy")
- `--interval`: Interval for applying darkening effect (default: 50)
- `--num_black`: Number of images to darken in each interval (default: 6)
- `--light_interval`: Interval for applying light changes (default: 100)
- `--light_change`: Amount of light change (-255 to 255) (default: 50)
- `--blur_interval`: Interval for applying motion blur (default: 75)
- `--blur_size`: Size of the motion blur kernel (default: 15)

### Example

```bash
python synthetic_data_production.py --dataset "MH_02_easy" --interval 40 --num_black 5 --light_interval 80 --light_change 60 --blur_interval 60 --blur_size 20
```

### Features

- Downloads specified EuRoC MAV dataset if not present
- Applies darkening effect to images at specified intervals
- Changes brightness of images at specified intervals
- Applies motion blur to images at specified intervals
- Saves processed images in a new directory

## Activation Function Graphs

This script generates and plots various activation functions used in neural networks.

### Usage

```bash
python act_fonk_graphs.py
```

### Features

- Plots the following activation functions:
  - Quadratic Unit Step
  - Cubic Unit Step
  - Quartic Unit Step
  - ReLU
  - Exponential Sigmoid
  - Double Exponential Sigmoid
  - Step Function
- Saves the plot as a high-resolution image file

## Requirements

To run these scripts, you'll need the following Python libraries:

- numpy
- opencv-python (cv2)
- matplotlib
- scipy
- pandas
- tqdm
- requests

You can install these dependencies using pip:

```bash
pip install numpy opencv-python matplotlib scipy pandas tqdm requests
```

For CUDA support, make sure you have the appropriate CUDA toolkit and cuDNN installed on your system.

## Contributing

Feel free to open issues or submit pull requests if you have any improvements or bug fixes for these tools.

## License

This project is open-source and available under the [MIT License](LICENSE).
