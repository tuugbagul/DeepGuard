# Bu kodu extractor_celeb_test.py olarak çalıştır
import os, cv2, json
from tqdm import tqdm

ROOT = r'C:\Users\Tugba Gul\Desktop\celebdf'  # Videoların olduğu yer
SAVE = r'C:\Celeb-DF_Test_Frames'
os.makedirs(SAVE, exist_ok=True)

dataset = []
folders = {'Celeb-real': 0, 'Celeb-synthesis': 1}

for folder, label in folders.items():
    path = os.path.join(ROOT, folder)
    vids = os.listdir(path)[:50]  # Her klasörden sadece ilk 50 videoyu al (Hızlı test)

    for v_name in tqdm(vids, desc=f"İşleniyor {folder}"):
        cap = cv2.VideoCapture(os.path.join(path, v_name))
        for i in range(5):  # Her videodan 5 kare
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * 20)
            ret, frame = cap.read()
            if ret:
                f_path = os.path.join(SAVE, f"{v_name}_{i}.jpg")
                cv2.imwrite(f_path, frame)
                dataset.append({'img': f_path, 'label': label})
        cap.release()

with open('celeb_test.json', 'w') as f:
    json.dump(dataset, f)