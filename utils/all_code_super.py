import os

# Klasör ve dosya yollarını tanımla
klasor_yolu = './moduls/'  
models_klasor_yolu = './models/'  # models klasörü
cikis_dosyasi = 'all_code_super.txt'
ana_dosya = 'main.py'

# Eklemek istediğin models dosyaları
eklenecek_models_dosyalari = [
    'matching.py',
    'superglue.py',
    'superpoint.py',
    'utils.py'
]

# Çıkış dosyasını aç ve içeriği temizle (varsa)
with open(cikis_dosyasi, 'w', encoding='utf-8') as dosya:
    dosya.write('')

# moduls klasöründeki tüm .py dosyalarını sırayla yaz
def dosyalari_yaz(klasor, cikis):
    """Verilen klasördeki tüm .py dosyalarını cikis dosyasına yazar."""
    for dosya_adi in os.listdir(klasor):
        if dosya_adi.endswith('.py'):  # Sadece .py dosyalarını seç
            dosya_yolu = os.path.join(klasor, dosya_adi)
            with open(dosya_yolu, 'r', encoding='utf-8') as modul:
                cikis.write(f'# {dosya_adi} içeriği:\n\n')  # Başlık ekle
                cikis.write(modul.read())  # Dosya içeriğini yaz
                cikis.write('\n\n' + '#' * 40 + '\n\n')  # Ayraç ekle

# moduls klasöründeki dosyaları ekle
with open(cikis_dosyasi, 'a', encoding='utf-8') as cikis:
    dosyalari_yaz(klasor_yolu, cikis)

    # Belirtilen models dosyalarını ekle
    for dosya_adi in eklenecek_models_dosyalari:
        dosya_yolu = os.path.join(models_klasor_yolu, dosya_adi)
        if os.path.exists(dosya_yolu):
            cikis.write(f'# {dosya_adi} içeriği:\n\n')  # Başlık ekle
            with open(dosya_yolu, 'r', encoding='utf-8') as modul:
                cikis.write(modul.read())  # Dosya içeriğini yaz
                cikis.write('\n\n' + '#' * 40 + '\n\n')  # Ayraç ekle
        else:
            print(f"'{dosya_adi}' dosyası bulunamadı!")

    # Ana dizindeki main.py dosyasını sona ekle
    if os.path.exists(ana_dosya):
        cikis.write(f'# {ana_dosya} içeriği:\n\n')  # Başlık ekle
        with open(ana_dosya, 'r', encoding='utf-8') as main:
            cikis.write(main.read())  # main.py içeriğini yaz
            cikis.write('\n\n' + '#' * 40 + '\n\n')  # Ayraç ekle
    else:
        print(f"'{ana_dosya}' dosyası bulunamadı!")

print(f"İçerikler '{cikis_dosyasi}' dosyasına başarıyla yazıldı.")