"""
Roop deepfake'lerini hızlı üretir.
Model bir kez yüklenir, tüm resimler tek session'da işlenir.

Çalıştır (roop venv'inden):
  /path/to/roop/venv/Scripts/python.exe roop_taha_fast.py
"""
import os, sys, json, random, shutil
import cv2

# ── YOL YAPILANDIRMASI ───────────────────────────────────────────────────────
# Bu yolları kendi ortamınıza göre düzenleyin
ROOP_DIR     = 'roop'                      # Roop kurulum klasörü
REAL_DIR     = 'roop_dataset/real'
FAKE_DIR     = 'roop_dataset/fake'
SOURCE_FACE  = 'source_faces/person.jpg'   # Deepfake kaynak yüzü
DATASET_JSON = 'roop_dataset.json'
TARGET_COUNT = 50
# ─────────────────────────────────────────────────────────────────────────────

# Roop modüllerine erişim
sys.path.insert(0, ROOP_DIR)

import roop.globals
roop.globals.execution_providers = ['CUDAExecutionProvider']
roop.globals.many_faces = True   # tüm yüzleri işle
roop.globals.reference_face_position = 0

from roop.processors.frame.face_swapper import get_face_swapper, swap_face
from roop.face_analyser import get_one_face, get_many_faces


random.seed(99)
all_real = sorted(
    [f for f in os.listdir(REAL_DIR) if f.endswith('.jpg')],
    key=lambda x: int(x.replace('real_', '').replace('.jpg', ''))
)
targets = random.sample(all_real, TARGET_COUNT)

print(f"Kaynak: {SOURCE_FACE}")
print(f"Hedef sayısı: {len(targets)}")

# Kaynak yüzü bir kez al
source_img  = cv2.imread(SOURCE_FACE)
source_face = get_one_face(source_img)
if source_face is None:
    print("HATA: Kaynak fotoğrafında yüz bulunamadı!")
    sys.exit(1)

# Model bir kez yüklenir
print("Model yükleniyor...")
swapper = get_face_swapper()
print("Model hazır. İşlem başlıyor...\n")

source_name = os.path.splitext(os.path.basename(SOURCE_FACE))[0]
ok, err, skip = 0, 0, 0
for i, real_name in enumerate(targets, 1):
    idx      = real_name.replace('real_', '').replace('.jpg', '')
    fake_out = os.path.join(FAKE_DIR, f'fake_{source_name}_{idx}.jpg')

    if os.path.exists(fake_out):
        print(f"[{i}/{len(targets)}] {os.path.basename(fake_out)} zaten var")
        skip += 1
        continue

    target_path = os.path.join(REAL_DIR, real_name)
    target_frame = cv2.imread(target_path)

    target_faces = get_many_faces(target_frame)
    if not target_faces:
        print(f"[{i}/{len(targets)}] {os.path.basename(fake_out)} — yüz yok, atlandı")
        err += 1
        continue

    result = target_frame.copy()
    for target_face in target_faces:
        result = swapper.get(result, target_face, source_face, paste_back=True)

    cv2.imwrite(fake_out, result)
    print(f"[{i}/{len(targets)}] {os.path.basename(fake_out)} OK")
    ok += 1

print(f"\nTamamlandı — OK: {ok}, Hata: {err}, Atlandı: {skip}")

# --- roop_dataset.json güncelle ---
with open(DATASET_JSON) as f:
    data = json.load(f)

existing = {s['img'] for s in data}
added = 0
for real_name in targets:
    idx      = real_name.replace('real_', '').replace('.jpg', '')
    fake_out = os.path.join(FAKE_DIR, f'fake_{source_name}_{idx}.jpg')
    if os.path.exists(fake_out) and fake_out not in existing:
        data.append({'img': fake_out, 'label': 1})
        added += 1

with open(DATASET_JSON, 'w') as f:
    json.dump(data, f, indent=2)

print(f"\nroop_dataset.json güncellendi — {added} yeni sahte eklendi")
print(f"Toplam: {len(data)}  |  Gerçek: {sum(1 for s in data if s['label']==0)}  |  Sahte: {sum(1 for s in data if s['label']==1)}")
