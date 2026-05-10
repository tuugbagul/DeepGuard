# Test setinden 5 sahte + 5 gercek secer, notu olan klasore kopyalar.
# Dosya isimleri foto_01..10.jpg — hangi sinif oldugu belli degil.
# Cevaplar ayri bir not dosyasina yazilir.
# python pick_demo_10.py

import os, sys, json, random, shutil
from PIL import Image

# ── YOL YAPILANDIRMASI ───────────────────────────────────────────────────────
# Bu yolları kendi ortamınıza göre düzenleyin
ROOP_JSON  = 'roop_dataset.json'
CELEB_JSON = 'celeb_pro_6000.json'
OUT_DIR    = 'demo_10_output'
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

with open(ROOP_JSON)  as f: roop  = json.load(f)
with open(CELEB_JSON) as f: celeb = json.load(f)

random.seed(42)
random.shuffle(roop)
roop_test  = roop[int(len(roop) * 0.8):]
celeb_test = celeb[1500:]

samples = [s for s in roop_test + celeb_test if os.path.exists(s['img'])]

random.seed(99)
random.shuffle(samples)

fakes = [s for s in samples if s['label'] == 1][:5]
reals = [s for s in samples if s['label'] == 0][:5]

selected = fakes + reals
random.shuffle(selected)   # karistir, sirayı gizle

notes = []
for i, s in enumerate(selected, 1):
    ext = os.path.splitext(s['img'])[1].lower() or '.jpg'
    dst = os.path.join(OUT_DIR, f"foto_{i:02d}.jpg")
    # PIL ile aç ve JPEG olarak kaydet (uzantı farklılığı olmasın)
    with Image.open(s['img']) as pil:
        pil.convert('RGB').save(dst, 'JPEG', quality=95)
    label_str = "SAHTE" if s['label'] == 1 else "GERCEK"
    notes.append(f"foto_{i:02d}.jpg  ->  {label_str}   ({os.path.basename(s['img'])})")
    print(f"  foto_{i:02d}.jpg  [{label_str}]")

note_path = os.path.join(OUT_DIR, "CEVAPLAR_GIZLI.txt")
with open(note_path, 'w', encoding='utf-8') as f:
    f.write("Demo cevaplari (onceden bakma!)\n")
    f.write("=" * 45 + "\n")
    for line in notes:
        f.write(line + "\n")

print(f"\n10 fotograf kopyalandi -> {OUT_DIR}")
print(f"Cevaplar: CEVAPLAR_GIZLI.txt (demo bitmeden bakma!)")
