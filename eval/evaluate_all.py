"""
EfficientNet-B4 / PixelGuard (Xception+FFT) / FusionModel karşılaştırması.
Aynı test seti üzerinde Accuracy, Precision, Recall, F1, AUC-ROC hesaplar.

Çalıştır: python evaluate_all.py
"""
import os, sys, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from fusion_model import FusionModel
from pixelguard_model import PixelGuardHybridModel
import timm

# ── YOL YAPILANDIRMASI ───────────────────────────────────────────────────────
# Bu yolları kendi ortamınıza göre düzenleyin
FUSION_WEIGHTS = 'fusion_finetuned.pth'
EFFICIENTNET_W = 'efficientnet_roop_best.pth'
PIXELGUARD_W   = 'pixelguard_best.pth'
ROOP_JSON      = 'roop_dataset.json'
CELEB_JSON     = 'celeb_pro_6000.json'
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZE = 16
IMG_SIZE   = 299


# ---------------------------------------------------------------
# Test seti: roop %20 + celeb[1500:]
# ---------------------------------------------------------------
def build_test_set():
    with open(ROOP_JSON)  as f: roop  = json.load(f)
    with open(CELEB_JSON) as f: celeb = json.load(f)

    random.seed(42)
    random.shuffle(roop)
    roop_test  = roop[int(len(roop) * 0.8):]   # %20 test
    celeb_test = celeb[1500:]                   # finetune'da hiç kullanılmadı

    samples = roop_test + celeb_test
    # Sadece var olan dosyalar
    samples = [s for s in samples if os.path.exists(s['img'])]
    print(f"Test seti: {len(samples)} örnek  "
          f"(Gerçek: {sum(1 for s in samples if s['label']==0)}, "
          f"Sahte: {sum(1 for s in samples if s['label']==1)})")
    return samples


class TestDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            with Image.open(s['img']) as pil:
                img = np.array(pil.convert('RGB'))
        except Exception:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        import cv2
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        return tensor, torch.tensor(s['label'], dtype=torch.float32)


# ---------------------------------------------------------------
# Model sarmalayıcıları
# ---------------------------------------------------------------
class EfficientNetOnly(nn.Module):
    """Standalone EfficientNet-B4 — roop_best.pth ağırlıkları."""
    def __init__(self, weights_path):
        super().__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1)
        state = torch.load(weights_path, map_location='cpu')
        # train.py Detector sınıfı içinde self.model olarak saklıyor
        if any(k.startswith('model.') for k in state):
            state = {k[len('model.'):]: v for k, v in state.items()}
        self.model.load_state_dict(state, strict=False)

        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)


class PixelGuardWrapper(nn.Module):
    def __init__(self, weights_path):
        super().__init__()
        self.model = PixelGuardHybridModel(pretrained=False)
        self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))

        self.register_buffer(
            'mean', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',  torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)


# ---------------------------------------------------------------
def evaluate(model, loader, device, name):
    model.eval()
    probs_all, labels_all = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"{name:<22}", ncols=70):
            imgs = imgs.to(device)
            logits = model(imgs).squeeze(1).cpu()
            probs_all.extend(torch.sigmoid(logits).numpy())
            labels_all.extend(labels.numpy())

    y      = np.array(labels_all)
    probs  = np.array(probs_all)
    preds  = (probs > 0.5).astype(int)

    acc  = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec  = recall_score(y, preds, zero_division=0)
    f1   = f1_score(y, preds, zero_division=0)
    try:
        auc = roc_auc_score(y, probs)
    except ValueError:
        auc = 0.0

    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()

    return dict(name=name, acc=acc, prec=prec, rec=rec, f1=f1, auc=auc,
                tp=tp, tn=tn, fp=fp, fn=fn)


# ---------------------------------------------------------------
def print_table(results):
    header = f"{'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC-ROC':>9}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in results:
        print(f"{r['name']:<25} "
              f"{r['acc']*100:>8.2f}% "
              f"{r['prec']*100:>9.2f}% "
              f"{r['rec']*100:>7.2f}% "
              f"{r['f1']*100:>7.2f}% "
              f"{r['auc']*100:>8.2f}%")
    print(sep)

    print("\nKarışıklık Matrisi (TP / TN / FP / FN):")
    print(f"{'Model':<25} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6}")
    print("-" * 50)
    for r in results:
        print(f"{r['name']:<25} {r['tp']:>6} {r['tn']:>6} {r['fp']:>6} {r['fn']:>6}")
    print("-" * 50)


# ---------------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}\n")

    samples = build_test_set()
    loader  = DataLoader(TestDataset(samples), batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=0)

    fusion_state = torch.load(FUSION_WEIGHTS, map_location='cpu')

    models = [
        ("EfficientNet-B4",           EfficientNetOnly(EFFICIENTNET_W).to(device)),
        ("Xception+FFT",              PixelGuardWrapper(PIXELGUARD_W).to(device)),
        ("FusionModel",               FusionModel().to(device)),
    ]
    models[2][1].load_state_dict(fusion_state)

    results = []
    for name, model in models:
        r = evaluate(model, loader, device, name)
        results.append(r)

    print_table(results)


if __name__ == '__main__':
    main()
