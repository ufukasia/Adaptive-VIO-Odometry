import cv2
import numpy as np
import os

# Resim boyutları ve boşluk
img_width, img_height = 752, 480
gap = 15  # Boşluk miktarını 15 piksele çıkardık

# Grid boyutları
rows, cols = 5, 3

# Sonuç resmi boyutları
result_width = cols * img_width + (cols - 1) * gap
result_height = rows * img_height + (rows - 1) * gap

# Sonuç resmini beyaz arka planla oluştur
result = np.full((result_height, result_width, 3), 255, dtype=np.uint8)  # Beyaz arka plan

# Resimleri yükle ve yerleştir
for i in range(rows):
    for j in range(cols):
        img_path = f'highest_value_images\\{i+1}{chr(97+j)}.png'
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                if img.shape[:2] != (img_height, img_width):
                    img = cv2.resize(img, (img_width, img_height))
                y = i * (img_height + gap)
                x = j * (img_width + gap)
                result[y:y+img_height, x:x+img_width] = img
        
        # Her satırın ilk resmine MH1, MH2, ... yaz
        if j == 0:
            text = f'MH{i+1}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            font_thickness = 3
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = x + 10
            text_y = y + img_height - 20
            
            # Neon mavi efekti için önce biraz bulanık bir arka plan çizelim
            for offset in range(3):
                cv2.putText(result, text, (text_x-offset, text_y), font, font_scale, (200, 100, 0), font_thickness*2)
            
            # Ana metni çizelim
            cv2.putText(result, text, (text_x, text_y), font, font_scale, (255, 255, 0), font_thickness)

# Sonuç resmini kaydet
cv2.imwrite('highest_value_images/information_fotos.png', result)

print("İşlem tamamlandı. Sonuç 'result_grid.png' olarak kaydedildi.")