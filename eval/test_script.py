import os, json

json_path = 'celeb_test.json'
frames_dir = 'Celeb-DF_Test_Frames'  # kendi klasörünüze göre düzenleyin

with open(json_path, 'r') as f:
    data = json.load(f)

print(f"JSON içinde {len(data)} adet kayıt var.")

# İlk 3 kaydı kontrol edelim
for i in range(min(3, len(data))):
    json_yolu = data[i]['img']
    dosya_adi = os.path.basename(json_yolu)
    tam_yol = os.path.join(frames_dir, dosya_adi)

    var_mi = os.path.exists(tam_yol)
    print(f"\nKayıt {i + 1}:")
    print(f"JSON'daki isim: {dosya_adi}")
    print(f"Klasörde var mı?: {'EVET' if var_mi else 'HAYIR'}")

# Klasördeki ilk 3 dosyayı görelim
gercek_dosyalar = os.listdir(frames_dir)
print(f"\nKlasördeki gerçek dosyalardan bazıları: {gercek_dosyalar[:3]}")