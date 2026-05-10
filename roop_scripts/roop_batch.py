import os, random, subprocess

# ── YOL YAPILANDIRMASI ───────────────────────────────────────────────────────
# Bu yolları kendi ortamınıza göre düzenleyin
LFW_DIR     = r'lfw-deepfunneled'          # LFW veri seti klasörü
SOURCES     = [                             # Kaynak yüz fotoğrafları (deepfake için)
    'source_faces/person1.jpg',
    'source_faces/person2.jpg',
]
OUTPUT_DIR  = 'roop_dataset'
ROOP_DIR    = 'roop'                        # Roop kurulum klasörü
FFMPEG_PATH = ''                            # FFmpeg bin klasörü (PATH'te ise boş bırakın)
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'real'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'fake'), exist_ok=True)

# Tüm fotoğrafları topla
all_images = []
for person in os.listdir(LFW_DIR):
    person_dir = os.path.join(LFW_DIR, person)
    for img in os.listdir(person_dir):
        if img.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_images.append(os.path.join(person_dir, img))

# Her kaynak için 50 hedef seç → toplam 150 sahte
random.seed(42)
selected = random.sample(all_images, 200)

print(f"Seçilen hedef: {len(selected)}, Kaynak: {len(SOURCES)}")
print(f"Toplam üretilecek sahte: {len(selected) * len(SOURCES)}")
print(f"Toplam gerçek: {len(selected)}")
print("Roop işlemi başlıyor...")

env = os.environ.copy()
env['PATH'] = FFMPEG_PATH + ';' + env['PATH']

python_exe = os.path.join(ROOP_DIR, 'venv', 'Scripts', 'python.exe')
roop_script = os.path.join(ROOP_DIR, 'run.py')

import shutil
counter = 1

for i, target in enumerate(selected, 1):
    # Gerçek fotoğrafı bir kez kopyala
    real_out = os.path.join(OUTPUT_DIR, 'real', f'real_{i}.jpg')
    shutil.copy(target, real_out)

    # Her kaynak için sahte üret
    for source in SOURCES:
        source_name = os.path.splitext(os.path.basename(source))[0]
        fake_out = os.path.join(OUTPUT_DIR, 'fake', f'fake_{source_name}_{i}.jpg')

        cmd = [python_exe, roop_script,
               '--source', source,
               '--target', target,
               '--output', fake_out,
               '--execution-provider', 'cuda']

        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        if os.path.exists(fake_out):
            print(f"[{counter}/150] {source_name} → OK")
        else:
            print(f"[{counter}/150] {source_name} → HATA")
        counter += 1

print("Tamamlandı!")
print(f"Gerçek: {len(os.listdir(os.path.join(OUTPUT_DIR, 'real')))}")
print(f"Sahte: {len(os.listdir(os.path.join(OUTPUT_DIR, 'fake')))}")
