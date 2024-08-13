import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import deque
from tqdm import tqdm
import argparse
import shutil
from multiprocessing import Pool, cpu_count

# Color scheme
COLORS = ['#0b5394', '#c90076', '#8fce00']

# Yeni normalizasyon sınıfı
class AdaptivePositiveNormalizer:
    def __init__(self, initial_value=0, alpha=0.01):
        self.mean = max(initial_value, 0)
        self.max_value = max(initial_value, 1e-6)  # Avoid division by zero
        self.alpha = alpha  # Learning rate for the running statistics

    def update(self, value):
        value = max(value, 0)  # Ensure non-negative value
        self.mean = (1 - self.alpha) * self.mean + self.alpha * value
        self.max_value = max(self.max_value, value)

    def normalize(self, value):
        value = max(value, 0)  # Ensure non-negative value
        if self.max_value == 0:
            return 0
        return value / self.max_value

# Yeni hesaplama fonksiyonları
def calculate_intensity(image):
    return np.mean(image) / 255.0

def calculate_entropy(image):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    histogram = histogram.flatten() / np.sum(histogram)
    non_zero = histogram[histogram > 0]
    return -np.sum(non_zero * np.log2(non_zero))

def calculate_motion_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def normalize_delta_intensity(delta):
    """
    Delta intensity değerlerini 0-256 aralığında normalize eder.
    """
    return min(delta, 256) / 256

def create_half_and_half_image(img1_path, img2_path, value, filename):
    """
    İlk resmin sol yarısını ve ikinci resmin sağ yarısını birleştirerek yeni bir resim oluşturur.
    Resmin üzerine delta intensity değerini ve dosya adını yazar.
    """
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # Resimlerin 480x752 olduğunu biliyoruz, ancak yine de kontrol edelim
    height, width = img1.shape[:2]
    assert img1.shape == img2.shape, "Resimler aynı boyutta olmalı"
    
    # Her resmin yarısını al
    half_width = width // 2
    left_half = img1[:, :half_width]
    right_half = img2[:, half_width:]
    
    # İki yarım resmi birleştir
    combined_img = np.hstack((left_half, right_half))
    
    # Resmin üzerine beyaz arka planlı metin ekle
    return add_text_to_image(combined_img, value, "Normalized Delta Intensity", filename, 0)

def add_text_to_image(image, value, metric, filename, color_index):
    """
    Resmin üzerine metrik değerini ve dosya adını beyaz arka planla yazar.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_color = tuple(int(COLORS[color_index][i:i+2], 16) for i in (5, 3, 1))  # HEX to BGR
    bg_color = (255, 255, 255)  # Beyaz arka plan
    thickness = 2
    
    # Metrik ismini düzelt
    if metric == "entropy":
        metric = "Entropy"
    elif metric == "motion_blur":
        metric = "Motion Blur"
    elif metric == "delta_intensity":
        metric = "Delta Intensity"
    
    # Metin boyutlarını hesapla
    text1 = f"{metric}: {value:.6f}"
    text2 = f"File: {filename}"
    (text_width1, text_height1), _ = cv2.getTextSize(text1, font, font_scale, thickness)
    (text_width2, text_height2), _ = cv2.getTextSize(text2, font, font_scale, thickness)
    
    # Arka plan boyutlarını hesapla
    padding = 10
    bg_width = max(text_width1, text_width2) + 2 * padding
    bg_height = (text_height1 + text_height2) + 3 * padding
    
    # Arka plan dikdörtgenini çiz
    cv2.rectangle(image, (20, 20), (20 + bg_width, 20 + bg_height), bg_color, -1)
    
    # Metni yaz
    cv2.putText(image, text1, (20 + padding, 20 + text_height1 + padding), font, font_scale, font_color, thickness, cv2.LINE_AA)
    cv2.putText(image, text2, (20 + padding, 20 + text_height1 + 2 * padding + text_height2), font, font_scale, font_color, thickness, cv2.LINE_AA)
    
    return image

def process_images_real_time(directory, alpha, beta, gamma):
    images = sorted([img for img in os.listdir(directory) if img.endswith(".png")])
    scaled_delta_intensities = []
    scaled_entropies = []
    scaled_motion_blurs = []
    prev_intensity = None
    prev_image = None

    intensity_normalizer = AdaptivePositiveNormalizer()
    motion_blur_normalizer = AdaptivePositiveNormalizer()

    max_delta_intensity = {"value": 0, "original_value": 0, "image": "", "prev_image": ""}
    max_entropy = {"value": 0, "original_value": 0, "image": ""}
    max_motion_blur = {"value": 0, "original_value": 0, "image": ""}

    for img_name in tqdm(images, desc=f"Processing {os.path.basename(directory)}"):
        img_path = os.path.join(directory, img_name)
        image = cv2.imread(img_path)
        
        intensity = calculate_intensity(image)
        entropy = calculate_entropy(image)
        motion_blur = calculate_motion_blur(image)
        
        intensity_normalizer.update(intensity)
        motion_blur_normalizer.update(motion_blur)
        
        normalized_intensity = intensity_normalizer.normalize(intensity)
        normalized_entropy = entropy / 8  # Entropi değerini 8'e bölerek normalize ediyoruz
        normalized_motion_blur = motion_blur_normalizer.normalize(motion_blur)
        
        scaled_entropy = (1 - normalized_entropy) * beta
        scaled_motion_blur = normalized_motion_blur * gamma * 0.2
        
        if prev_intensity is not None:
            delta_intensity = abs(intensity - prev_intensity)
            normalized_delta_intensity = normalize_delta_intensity(delta_intensity * 255)
            scaled_delta_intensity = normalized_delta_intensity * alpha * 10
            
            # En yüksek değerleri güncelle
            if scaled_delta_intensity > max_delta_intensity["value"]:
                max_delta_intensity["value"] = scaled_delta_intensity
                max_delta_intensity["original_value"] = delta_intensity
                max_delta_intensity["image"] = img_name
                max_delta_intensity["prev_image"] = prev_image
            
            if scaled_entropy > max_entropy["value"]:
                max_entropy["value"] = scaled_entropy
                max_entropy["original_value"] = entropy
                max_entropy["image"] = img_name
            
            if scaled_motion_blur > max_motion_blur["value"]:
                max_motion_blur["value"] = scaled_motion_blur
                max_motion_blur["original_value"] = motion_blur
                max_motion_blur["image"] = img_name
        else:
            scaled_delta_intensity = 0
        
        scaled_delta_intensities.append(scaled_delta_intensity)
        scaled_entropies.append(scaled_entropy)
        scaled_motion_blurs.append(scaled_motion_blur)
        
        prev_intensity = intensity
        prev_image = img_name

    return scaled_delta_intensities, scaled_entropies, scaled_motion_blurs, max_delta_intensity, max_entropy, max_motion_blur

def plot_results(all_scaled_data):
    fig = plt.figure(figsize=(12, 15), constrained_layout=True)
    gs = GridSpec(5, 1, figure=fig)

    for i, (dataset, scaled_data) in enumerate(all_scaled_data.items()):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(scaled_data['delta_intensities'], label='Δ Intensity' if i == 0 else "", color=COLORS[0], linewidth=1.2)
        ax.plot(scaled_data['entropies'], label='Entropy' if i == 0 else "", color=COLORS[1], linewidth=1.2)
        ax.plot(scaled_data['motion_blurs'], label='Motion Blur' if i == 0 else "", color=COLORS[2], linewidth=1.2)
        
        # Dataset adını sol üst köşeye yerleştir
        ax.text(0.02, 0.98, dataset, transform=ax.transAxes, fontsize=10, fontweight='bold', 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        # Y ekseni etiketini sadece en soldaki grafik için göster
        if i == 2:  # Orta grafik için
            ax.set_ylabel('Scaled Value', fontsize=12)

        # X ekseni etiketini sadece en alttaki grafik için göster
        if i == 4:  # En alttaki grafik için
            ax.set_xlabel('Image Index', fontsize=12)
        else:
            ax.set_xticklabels([])  # Diğer grafikler için x ekseni etiketlerini kaldır

        ax.set_ylim(0, 1)  # Y eksenini 0 ile 1 arasında sınırla
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=10)

        # Y eksenindeki etiketlerin üst üste binmesini önlemek için
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, prune='both'))

        # Sadece ilk grafik için legend göster
        if i == 0:
            ax.legend(fontsize=10, frameon=True, fancybox=True, framealpha=0.7, loc='upper right')

    plt.suptitle('Scaled Information Metrics', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()

def process_dataset(args):
    dataset, alpha, beta, gamma = args
    main_directory = f"{dataset}\\mav0\\cam0\\data"
    
    scaled_delta_intensities, scaled_entropies, scaled_motion_blurs, max_delta_intensity, max_entropy, max_motion_blur = process_images_real_time(
        main_directory, alpha, beta, gamma
    )
    
    return dataset, {
        'delta_intensities': scaled_delta_intensities,
        'entropies': scaled_entropies,
        'motion_blurs': scaled_motion_blurs
    }, {
        'delta_intensity': max_delta_intensity,
        'entropy': max_entropy,
        'motion_blur': max_motion_blur
    }

def main():
    parser = argparse.ArgumentParser(description="Parallel image processing script for multiple datasets with scaled information metrics.")
    parser.add_argument("--alpha", type=float, default=1, help="Delta intensity scaling factor")
    parser.add_argument("--beta", type=float, default=1, help="Entropy scaling factor")
    parser.add_argument("--gamma", type=float, default=1, help="Motion blur scaling factor")
    parser.add_argument("--output_dir", type=str, default="highest_value_images", help="Directory to save highest value images")

    args = parser.parse_args()

    datasets = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    output_directory = args.output_dir

    # Paralel işleme için argümanları hazırla
    process_args = [(dataset, alpha, beta, gamma) for dataset in datasets]

    # Kullanılabilir CPU sayısını al (en fazla dataset sayısı kadar)
    num_processes = min(len(datasets), cpu_count())

    # Paralel işleme başlat
    with Pool(num_processes) as pool:
        results = pool.map(process_dataset, process_args)

    # Sonuçları işle
    all_scaled_data = {}
    all_max_values = {}
    for dataset, scaled_data, max_values in results:
        all_scaled_data[dataset] = scaled_data
        all_max_values[dataset] = max_values

    plot_results(all_scaled_data)

    # En yüksek değerleri terminale yazdır ve resimleri kaydet
    print("\nEn yüksek değerler:")
    
    # Çıktı klasörünü oluştur
    os.makedirs(output_directory, exist_ok=True)
    
    for i, (dataset, max_values) in enumerate(all_max_values.items()):
        print(f"\n{dataset}:")
        for j, (metric, value) in enumerate(max_values.items()):
            print(f"  {metric.capitalize()}: {value['original_value']:.6f} (Image: {value['image']})")
            
            if metric == 'delta_intensity':
                # Delta intensity için yarım-yarım resim oluştur
                img1_path = os.path.join(f"{dataset}\\mav0\\cam0\\data", value['prev_image'])
                img2_path = os.path.join(f"{dataset}\\mav0\\cam0\\data", value['image'])
                combined_img = create_half_and_half_image(img1_path, img2_path, value['original_value'], value['image'])
                
                target_filename = f"{i+1}{chr(97+j)}.png"
                target_path = os.path.join(output_directory, target_filename)
                
                cv2.imwrite(target_path, combined_img)
                print(f"  Half-and-half comparison image saved: {target_path}")
            else:
                # Diğer metrikler için normal kopyalama işlemi ve metin ekleme
                source_path = os.path.join(f"{dataset}\\mav0\\cam0\\data", value['image'])
                image = cv2.imread(source_path)
                color_index = 1 if metric == 'entropy' else 2  # entropy için 1, motion blur için 2
                annotated_image = add_text_to_image(image, value['original_value'], metric, value['image'], color_index)
                
                target_filename = f"{i+1}{chr(97+j)}.png"
                target_path = os.path.join(output_directory, target_filename)
                
                cv2.imwrite(target_path, annotated_image)
                print(f"  Annotated image saved: {target_path}")

if __name__ == "__main__":
    main()