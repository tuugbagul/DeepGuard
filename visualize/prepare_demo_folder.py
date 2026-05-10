# Test setinden temsili ornekler secer, tahmin sonucuyla isimlendirir ve
# tek bir klasore kopyalar.
# Cikti: Desktop/demo_gorseller/
#   DOGRU_SAHTE__fake_xxx__99.jpg    <- TP
#   DOGRU_GERCEK__real_xxx__4.jpg   <- TN
#   YANLIS_GERCEK__fake_xxx__41.jpg <- FN (kacirilan sahte)
#   YANLIS_SAHTE__real_xxx__72.jpg  <- FP (yanlis alarm)
# python prepare_demo_folder.py
import os, sys, json, random, shutil
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
ROOP_JSON      = 'roop_dataset.json'
CELEB_JSON     = 'celeb_pro_6000.json'
OUT_DIR        = 'demo_output'
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLD      = 0.6
N_EACH         = 5   # her kategori icin kac ornek (TP, TN, FN, FP)

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FusionModel().to(device)
model.load_state_dict(torch.load(FUSION_WEIGHTS, map_location=device))
model.eval()
print(f"Model hazir ({device})")


def infer(path):
    try:
        with Image.open(path) as pil:
            img = np.array(pil.convert('RGB'))
    except Exception:
        return None
    img = cv2.resize(img, (299, 299))
    tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        score = torch.sigmoid(model(tensor).squeeze()).item()
    return score


def build_test_set():
    with open(ROOP_JSON) as f:
        roop = json.load(f)
    with open(CELEB_JSON) as f:
        celeb = json.load(f)
    random.seed(42)
    random.shuffle(roop)
    roop_test = roop[int(len(roop) * 0.8):]
    celeb_test = celeb[1500:]
    samples = [s for s in roop_test + celeb_test if os.path.exists(s['img'])]
    print(f"Test seti: {len(samples)} ornek")
    return samples


samples = build_test_set()
random.seed(0)
random.shuffle(samples)

tp_list, tn_list, fn_list, fp_list = [], [], [], []

print("Tahminler hesaplaniyor (ilk 800 ornek taranacak)...")
for s in samples[:800]:
    score = infer(s['img'])
    if score is None:
        continue
    pred  = 1 if score > THRESHOLD else 0
    label = s['label']
    entry = (s['img'], score, label)
    if   label == 1 and pred == 1: tp_list.append(entry)
    elif label == 0 and pred == 0: tn_list.append(entry)
    elif label == 1 and pred == 0: fn_list.append(entry)
    elif label == 0 and pred == 1: fp_list.append(entry)
    if all(len(x) >= N_EACH for x in [tp_list, tn_list, fn_list, fp_list]):
        break

print(f"  TP:{len(tp_list)}  TN:{len(tn_list)}  FN:{len(fn_list)}  FP:{len(fp_list)}")


def copy_samples(lst, prefix, n):
    for i, (path, score, label) in enumerate(lst[:n]):
        ext  = os.path.splitext(path)[1].lower() or '.jpg'
        base = os.path.splitext(os.path.basename(path))[0]
        dst  = os.path.join(OUT_DIR, f"{prefix}__{base}__{int(score*100)}{ext}")
        shutil.copy2(path, dst)


copy_samples(tp_list, "DOGRU_SAHTE",   N_EACH)
copy_samples(tn_list, "DOGRU_GERCEK",  N_EACH)
copy_samples(fn_list, "YANLIS_GERCEK", N_EACH)   # kacirilan sahteler
copy_samples(fp_list, "YANLIS_SAHTE",  N_EACH)   # yanlis alarm

total = min(N_EACH, len(tp_list)) + min(N_EACH, len(tn_list)) + \
        min(N_EACH, len(fn_list)) + min(N_EACH, len(fp_list))
print(f"\nKopyalandi: {total} gorsel -> {OUT_DIR}")
print("Klasor icerigi:")
for f in sorted(os.listdir(OUT_DIR)):
    print(" ", f)
