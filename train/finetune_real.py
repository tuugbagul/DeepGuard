"""
Gerçek kişi fotoğraflarını (tugba, ilayda, mert) ekleyerek
fusion modelini kısa fine-tune ile günceller.

Çalıştır:
  python finetune_real.py
"""
import os, json, random, cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from PIL import Image

import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from fusion_model import FusionModel

# ── YOL YAPILANDIRMASI ───────────────────────────────────────────────────────
# Bu yolları kendi ortamınıza göre düzenleyin
FUSION_WEIGHTS = 'fusion_finetuned.pth'
ROOP_JSON      = 'roop_dataset.json'
CELEB_JSON     = 'celeb_pro_6000.json'
OUTPUT_DIR     = 'outputs_fusion'

# Modeli kalibre etmek için kullanılacak gerçek yüz fotoğrafları (label=0)
# Kendi fotoğraflarınızı buraya ekleyin
REAL_PHOTOS = [
    'real_faces/person1.jpg',
    'real_faces/person2.jpg',
]
# ─────────────────────────────────────────────────────────────────────────────

EPOCHS     = 5
LR         = 1e-4   # düşük LR — mevcut bilgiyi koruyarak ince ayar
BATCH_SIZE = 16
AUG_COUNT  = 40     # her gerçek fotoğraftan kaç augmented örnek üret


# ---------------------------------------------------------------
def augment(img):
    """Tek görüntüden AUG_COUNT farklı varyant üretir."""
    variants = []
    h, w = img.shape[:2]

    for i in range(AUG_COUNT):
        out = img.copy()

        # Yatay çevirme
        if random.random() < 0.5:
            out = out[:, ::-1].copy()

        # Parlaklık/kontrast
        alpha = random.uniform(0.75, 1.25)
        beta  = random.randint(-20, 20)
        out   = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Hafif rotasyon (±15°)
        angle = random.uniform(-15, 15)
        M     = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        out   = cv2.warpAffine(out, M, (w, h))

        # Crop + resize (%80–100 rastgele crop)
        scale = random.uniform(0.80, 1.0)
        ch, cw = int(h * scale), int(w * scale)
        y0 = random.randint(0, h - ch)
        x0 = random.randint(0, w - cw)
        out = cv2.resize(out[y0:y0+ch, x0:x0+cw], (w, h))

        variants.append(out)

    return variants


def build_real_samples():
    """Gerçek fotoğrafları augment eder, geçici dict listesi döner."""
    samples = []
    for path in REAL_PHOTOS:
        with Image.open(path) as pil:
            img = np.array(pil.convert('RGB'))
        for aug_img in augment(img):
            samples.append({'_arr': aug_img, 'label': 0})
    print(f"Gerçek kişi örnekleri (augmented): {len(samples)}")
    return samples


# ---------------------------------------------------------------
class MixedDataset(Dataset):
    def __init__(self, json_samples, real_samples, augment_json=True):
        self.json_samples  = json_samples
        self.real_samples  = real_samples
        self.augment_json  = augment_json
        self.all = json_samples + real_samples  # tip karışık, __getitem__ handle eder

    def __len__(self):
        return len(self.all)

    def __getitem__(self, idx):
        s = self.all[idx]

        if '_arr' in s:
            img = cv2.resize(s['_arr'], (299, 299))
        else:
            try:
                with Image.open(s['img']) as pil:
                    img = np.array(pil.convert('RGB'))
                img = cv2.resize(img, (299, 299))
            except Exception:
                img = np.zeros((299, 299, 3), dtype=np.uint8)

            if self.augment_json and random.random() < 0.5:
                img = img[:, ::-1].copy()

        img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        return torch.from_numpy(img), torch.tensor(s['label'], dtype=torch.float32)


# ---------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = (torch.sigmoid(model(imgs).squeeze(1)) > 0.5).float()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return 100 * correct / total


def evaluate_photos(model, device):
    """Gerçek fotoğrafları tek tek değerlendirir."""
    model.eval()
    names = [os.path.splitext(os.path.basename(p))[0] for p in REAL_PHOTOS]
    for name, path in zip(names, REAL_PHOTOS):
        with Image.open(path) as pil:
            img = np.array(pil.convert('RGB'))
        img = cv2.resize(img, (299, 299))
        t = torch.from_numpy(
            (img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            score = torch.sigmoid(model(t).squeeze()).item()
        pred = "SAHTE" if score > 0.5 else "GERCEK"
        ok   = "OK" if pred == "GERCEK" else "YANLIS"
        print(f"  {name:<8} skor={score:.3f}  tahmin={pred}  {ok}")


# ---------------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")

    model = FusionModel().to(device)
    model.load_state_dict(torch.load(FUSION_WEIGHTS, map_location=device))

    # Backbone'ları dondur, sadece classifier head
    for p in model.efficientnet.parameters(): p.requires_grad = False
    for p in model.pixelguard.parameters():   p.requires_grad = False

    # --- Veri ---
    with open(ROOP_JSON) as f: roop = json.load(f)
    with open(CELEB_JSON) as f: celeb = json.load(f)
    random.seed(42)

    # Eğitim: roop %80 + celeb[:1000]
    random.shuffle(roop);  roop_train = roop[:int(len(roop)*0.8)]
    random.shuffle(celeb); celeb_train = celeb[:1000]
    train_json = roop_train + celeb_train

    # Val: celeb[1000:1500]
    val_json = celeb[1000:1500]

    real_samples = build_real_samples()

    # Gerçek örnekleri train'e ekle (3x ağırlık — az olduğu için)
    train_data = MixedDataset(train_json, real_samples * 3)
    val_data   = MixedDataset(val_json, [])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    print("\nBaşlangıç durumu (fine-tune öncesi):")
    evaluate_photos(model, device)

    best_acc, best_path = 0, os.path.join(OUTPUT_DIR, 'fusion_finetuned.pth')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=70)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device.type):
                loss = criterion(model(imgs).squeeze(1), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        acc = evaluate(model, val_loader, device)
        tag = ''
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)
            tag = ' ★'
        print(f"Epoch {epoch} — val genel: %{acc:.2f}{tag}")

    print(f"\nEn iyi val: %{best_acc:.2f}")
    print(f"Model kaydedildi: {best_path}")

    print("\nFine-tune sonrası gerçek kişi kontrolü:")
    model.load_state_dict(torch.load(best_path, map_location=device))
    evaluate_photos(model, device)


if __name__ == '__main__':
    main()
