"""
sahte_veriseti klasöründeki tüm görselleri FusionModel ile test eder.
python test_folder.py
"""
import os, sys
import numpy as np
import torch
import cv2
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from fusion_model import FusionModel

# ── YOL YAPILANDIRMASI ───────────────────────────────────────────────────────
# Bu yolları kendi ortamınıza göre düzenleyin
FUSION_WEIGHTS = 'fusion_finetuned.pth'
FOLDER         = 'test_images'    # Test edilecek görsellerin klasörü
THRESHOLD      = 0.6
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FusionModel().to(device)
model.load_state_dict(torch.load(FUSION_WEIGHTS, map_location=device))
model.eval()
print(f"Model hazır ({device})\n")

files = sorted(os.listdir(FOLDER))
images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

# Gerçek (label=0) olarak bilinen dosya adları — kendi dosyalarınıza göre düzenleyin
REAL_NAMES = set()  # örn: {'real_face1.jpg', 'real_face2.png'}

print(f"{'Dosya':<22} {'Skor':>7}  {'Tahmin':<8}  {'Gerçek Etiket':<14}  {'Sonuç'}")
print("-" * 75)

correct = 0
for fname in images:
    path = os.path.join(FOLDER, fname)
    try:
        with Image.open(path) as pil:
            img = np.array(pil.convert('RGB'))
    except Exception:
        print(f"{fname:<22}  OKUNAMADI")
        continue

    img_299 = cv2.resize(img, (299, 299))
    tensor = torch.from_numpy(
        img_299.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor).squeeze()
        score = torch.sigmoid(logit).item()

    pred   = "SAHTE" if score > THRESHOLD else "GERÇEK"
    is_real = fname in REAL_NAMES
    true_label = "GERÇEK" if is_real else "SAHTE"
    ok = pred == true_label
    correct += ok
    marker = "✓" if ok else "✗"

    print(f"{fname:<22} {score*100:>6.1f}%  {pred:<8}  {true_label:<14}  {marker}")

print("-" * 75)
print(f"\nDoğru: {correct}/{len(images)}  ({correct/len(images)*100:.1f}%)")
