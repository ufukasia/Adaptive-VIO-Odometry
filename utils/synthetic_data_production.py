import os
import cv2
import numpy as np
import argparse
import requests
import zipfile
from tqdm import tqdm
from shutil import copy2

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            size = file.write(data)
            progress_bar.update(size)
    progress_bar.close()

def extract_zip(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def change_brightness(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    v = v.astype(np.int16)
    v = np.clip(v + value, 0, 255)
    v = v.astype(np.uint8)
    
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def apply_motion_blur(image, size):
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel = kernel / size
    return cv2.filter2D(image, -1, kernel)

def process_images(source_dir, target_dir, interval, num_black, light_interval, light_change, blur_interval, blur_size):
    os.makedirs(target_dir, exist_ok=True)
    image_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]
    image_files.sort()

    progress_bar = tqdm(total=len(image_files), desc="Processing Images")

    for i, filename in enumerate(image_files):
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)

        if (i % interval) < num_black:
            img = np.zeros((480, 752, 3), dtype=np.uint8)
            cv2.imwrite(target_path, img)
        else:
            img = cv2.imread(source_path)
            if img is None:
                print(f"Uyarı: {source_path} okunamadı. Atlanıyor.")
                continue
            
            if i % light_interval == 0:
                img = change_brightness(img, light_change)
            
            if i % blur_interval == 0:
                img = apply_motion_blur(img, blur_size)
            
            cv2.imwrite(target_path, img)
        
        progress_bar.update(1)

    progress_bar.close()
    print(f"İşlem tamamlandı. Sonuçlar {target_dir} dizinine kaydedildi.")

def main(args):
    datasets = {
        'MH_01_easy': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip',
        'MH_02_easy': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_02_easy/MH_02_easy.zip',
        'MH_03_medium': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_03_medium/MH_03_medium.zip',
        'MH_04_difficult': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_04_difficult/MH_04_difficult.zip',
        'MH_05_difficult': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_05_difficult/MH_05_difficult.zip',
        'V1_01_easy': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.zip',
        'V1_02_medium': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_02_medium/V1_02_medium.zip',
        'V1_03_difficult': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_03_difficult/V1_03_difficult.zip',
        'V2_01_easy': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_01_easy/V2_01_easy.zip',
        'V2_02_medium': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_02_medium/V2_02_medium.zip',
        'V2_03_difficult': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_03_difficult/V2_03_difficult.zip'
    }

    if args.dataset not in datasets:
        print(f"Hata: Geçersiz dataset adı. Mevcut datasetler: {', '.join(datasets.keys())}")
        return

    dataset_url = datasets[args.dataset]
    zip_filename = f"{args.dataset}.zip"
    
    if not os.path.exists(args.dataset):
        print(f"Dataset indiriliyor: {args.dataset}")
        download_file(dataset_url, zip_filename)
        print("Dataset çıkartılıyor...")
        extract_zip(zip_filename, '.')
        os.remove(zip_filename)
    else:
        print(f"Dataset zaten mevcut: {args.dataset}")

    source_directory = os.path.join(args.dataset, 'mav0', 'cam0', 'data')
    target_directory = os.path.join(args.dataset, 'mav0', 'cam0', 'data-synthetic')

    process_images(source_directory, target_directory, args.interval, args.num_black, args.light_interval, args.light_change, args.blur_interval, args.blur_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resim karartma, ışık değiştirme ve motion blur işlemi")
    parser.add_argument("--dataset", type=str, default="MH_01_easy", help="Kullanılacak dataset adı")
    parser.add_argument("--interval", type=int, default=50, help="Kaç resimde bir karartma işlemi yapılacağı")
    parser.add_argument("--num_black", type=int, default=6, help="Her intervalda kaç resim karartılacağı")
    parser.add_argument("--light_interval", type=int, default=100, help="Kaç resimde bir ışık değişikliği yapılacağı")
    parser.add_argument("--light_change", type=int, default=50, help="Işık değişikliği miktarı (-255 ile 255 arası)")
    parser.add_argument("--blur_interval", type=int, default=75, help="Kaç resimde bir motion blur uygulanacağı")
    parser.add_argument("--blur_size", type=int, default=15, help="Motion blur kernel boyutu")
    
    args = parser.parse_args()
    main(args)