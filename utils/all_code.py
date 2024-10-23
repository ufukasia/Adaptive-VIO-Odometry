import os

# Klasör ve dosya yollarını tanımla
klasor_yolu = './moduls/'  
cikis_dosyasi = 'all_code.txt'
ana_dosya = 'main.py'

# Çıkış dosyasını aç ve içeriği temizle (varsa)
with open(cikis_dosyasi, 'w', encoding='utf-8') as dosya:
    dosya.write('')

# moduls klasöründeki tüm .py dosyalarını sırayla yaz
with open(cikis_dosyasi, 'a', encoding='utf-8') as cikis:
    # moduls klasöründeki tüm dosyaları gez
    for dosya_adi in os.listdir(klasor_yolu):
        if dosya_adi.endswith('.py'):  # Sadece .py dosyalarını seç
            dosya_yolu = os.path.join(klasor_yolu, dosya_adi)
            with open(dosya_yolu, 'r', encoding='utf-8') as modul:
                cikis.write(f'# {dosya_adi} içeriği:\n\n')  # Başlık ekle
                cikis.write(modul.read())  # Dosya içeriğini yaz
                cikis.write('\n\n' + '#' * 40 + '\n\n')  # Ayraç ekle

    # Ana dizindeki main.py dosyasını sona ekle
    if os.path.exists(ana_dosya):
        cikis.write(f'# {ana_dosya} içeriği:\n\n')  # Başlık ekle
        with open(ana_dosya, 'r', encoding='utf-8') as main:
            cikis.write(main.read())  # main.py içeriğini yaz
            cikis.write('\n\n' + '#' * 40 + '\n\n')  # Ayraç ekle
    else:
        print(f"'{ana_dosya}' dosyası bulunamadı!")

print(f"İçerikler '{cikis_dosyasi}' dosyasına başarıyla yazıldı.")