import os, cv2, json
from tqdm import tqdm

ROOT = r'C:\Users\Tugba Gul\Desktop\celebdf' # Videoların ana klasörü
SAVE = r'C:\Celeb-DF_Pro_Frames'
os.makedirs(SAVE, exist_ok=True)

dataset = []
folders = {'Celeb-real': 0, 'Celeb-synthesis': 1}

print("Sıfırdan kare çıkarımı başlıyor...")

for folder, label in folders.items():
    path = os.path.join(ROOT, folder)
    vids = os.listdir(path)[:300] # Her klasörden 300 video (Toplam 600 video)

    for v_name in tqdm(vids, desc=f"İşleniyor {folder}"):
        cap = cv2.VideoCapture(os.path.join(path, v_name))
        # Her videonun farklı yerlerinden 10 kare alarak çeşitliliği artırıyoruz
        for i in range(10):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * 25)
            ret, frame = cap.read()
            if ret:
                f_name = f"{v_name}_frame{i}.jpg"
                f_path = os.path.join(SAVE, f_name)
                cv2.imwrite(f_path, frame)
                dataset.append({'img': f_path, 'label': label})
        cap.release()

with open('celeb_pro_6000.json', 'w') as f:
    json.dump(dataset, f)

print(f"\nİşlem Tamam! {len(dataset)} yeni kare C:\Celeb-DF_Pro_Frames adresine kaydedildi.")