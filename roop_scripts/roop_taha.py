"""
Taha için roop fake'leri üretir, ardından roop_dataset.json'u günceller.

Çalıştır:
  python roop_taha.py
"""
import os, json, shutil, subprocess

# ── YOL YAPILANDIRMASI ───────────────────────────────────────────────────────
# Bu yolları kendi ortamınıza göre düzenleyin
REAL_DIR     = r'roop_dataset/real'
FAKE_DIR     = r'roop_dataset/fake'
ROOP_DIR     = 'roop'
SOURCE_FACE  = 'source_faces/person.jpg'   # Deepfake kaynak yüzü
DATASET_JSON = 'roop_dataset.json'
FFMPEG_PATH  = ''                           # FFmpeg bin klasörü (PATH'te ise boş bırakın)
# ─────────────────────────────────────────────────────────────────────────────

python_exe  = os.path.join(ROOP_DIR, 'venv', 'Scripts', 'python.exe')
roop_script = os.path.join(ROOP_DIR, 'run.py')

env = os.environ.copy()
if FFMPEG_PATH:
    env['PATH'] = FFMPEG_PATH + ';' + env['PATH']

import random
random.seed(99)

# Mevcut gerçek görsellerden 50 rastgele seç
all_real = sorted(
    [f for f in os.listdir(REAL_DIR) if f.endswith('.jpg')],
    key=lambda x: int(x.replace('real_', '').replace('.jpg', ''))
)
real_images = random.sample(all_real, 50)
print(f"Hedef sayısı: {len(real_images)}")
print(f"Kaynak: {SOURCE_FACE}")
print(f"Üretilecek: {len(real_images)} sahte taha")
print("Roop işlemi başlıyor...\n")

ok, err = 0, 0
for i, real_name in enumerate(real_images, 1):
    target = os.path.join(REAL_DIR, real_name)
    idx    = real_name.replace('real_', '').replace('.jpg', '')
    source_name = os.path.splitext(os.path.basename(SOURCE_FACE))[0]
    fake_out = os.path.join(FAKE_DIR, f'fake_{source_name}_{idx}.jpg')

    if os.path.exists(fake_out):
        print(f"[{i}/{len(real_images)}] {os.path.basename(fake_out)} zaten var, atlandı")
        ok += 1
        continue

    cmd = [python_exe, roop_script,
           '--source', SOURCE_FACE,
           '--target', target,
           '--output', fake_out,
           '--execution-provider', 'cuda']

    subprocess.run(cmd, capture_output=True, text=True, env=env)

    if os.path.exists(fake_out):
        print(f"[{i}/{len(real_images)}] {os.path.basename(fake_out)} OK")
        ok += 1
    else:
        print(f"[{i}/{len(real_images)}] {os.path.basename(fake_out)} HATA")
        err += 1

print(f"\nTamamlandı — OK: {ok}, Hata: {err}")

# --- roop_dataset.json güncelle ---
with open(DATASET_JSON) as f:
    data = json.load(f)

# Zaten eklenmiş taha sahteleri varsa tekrar ekleme
existing = {s['img'] for s in data}
added = 0
source_name = os.path.splitext(os.path.basename(SOURCE_FACE))[0]
for i, real_name in enumerate(real_images, 1):
    idx      = real_name.replace('real_', '').replace('.jpg', '')
    fake_out = os.path.join(FAKE_DIR, f'fake_{source_name}_{idx}.jpg')
    if os.path.exists(fake_out) and fake_out not in existing:
        data.append({'img': fake_out, 'label': 1})
        added += 1

with open(DATASET_JSON, 'w') as f:
    json.dump(data, f, indent=2)

print(f"\nroop_dataset.json güncellendi — {added} yeni sahte eklendi")
print(f"Toplam: {len(data)} örnek  |  Gerçek: {sum(1 for s in data if s['label']==0)}  |  Sahte: {sum(1 for s in data if s['label']==1)}")
