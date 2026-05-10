import os, sys, json, random, cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from fusion_model import FusionModel

# ── YOL YAPILANDIRMASI ───────────────────────────────────────────────────────
# Bu yolları kendi ortamınıza göre düzenleyin
ROOP_JSON         = 'roop_dataset.json'
CELEB_JSON        = 'celeb_pro_6000.json'
EFFICIENTNET_PATH = 'efficientnet_finetuned.pth'
PIXELGUARD_PATH   = 'pixelguard_best.pth'
OUTPUT_DIR        = 'outputs_fusion'
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Hiperparametreler ----
BATCH_SIZE = 16
LR         = 5e-4   # Yalnızca fusion başı eğitiliyor → daha yüksek LR ok
EPOCHS     = 15


# ---------------------------------------------------------------
class FusionDataset(Dataset):
    """
    Görüntüleri [0,1] aralığında 299x299 tensor olarak döner.
    Normalizasyon FusionModel içinde yapılıyor.
    """
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            with Image.open(s['img']) as pil:
                img = np.array(pil.convert('RGB'))
            img = cv2.resize(img, (299, 299))
        except Exception:
            img = np.zeros((299, 299, 3), dtype=np.uint8)

        if self.augment and random.random() < 0.5:
            img = img[:, ::-1].copy()

        img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        return torch.from_numpy(img), torch.tensor(s['label'], dtype=torch.float32)


# ---------------------------------------------------------------
def load_weights(model, device):
    """Her iki backbone'un ağırlıklarını yükle, fusion başını rastgele bırak."""

    # --- EfficientNet ---
    # Kaydedilen model Detector sınıfından: anahtarlar 'model.*' şeklinde
    eff_state = torch.load(EFFICIENTNET_PATH, map_location=device)
    eff_clean = {}
    for k, v in eff_state.items():
        if k.startswith('model.'):
            new_k = k[len('model.'):]
            if not new_k.startswith('classifier'):
                eff_clean[new_k] = v
    missing, unexpected = model.efficientnet.load_state_dict(eff_clean, strict=False)
    print(f"[EfficientNet] yüklendi — eksik: {len(missing)}, fazla: {len(unexpected)}")

    # --- PixelGuard ---
    pg_state = torch.load(PIXELGUARD_PATH, map_location=device)
    missing, unexpected = model.pixelguard.load_state_dict(pg_state, strict=False)
    print(f"[PixelGuard]   yüklendi — eksik: {len(missing)}, fazla: {len(unexpected)}")


def freeze_backbones(model):
    for p in model.efficientnet.parameters():
        p.requires_grad = False
    for p in model.pixelguard.parameters():
        p.requires_grad = False
    trainable = sum(p.numel() for p in model.classifier.parameters())
    print(f"Backbone'lar donduruldu. Eğitilecek parametre: {trainable:,}")


# ---------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    real_c, real_t, fake_c, fake_t = 0, 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = (torch.sigmoid(model(imgs).squeeze(1)) > 0.5).float()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            for p, l in zip(preds, labels):
                if l == 1:
                    fake_t += 1
                    if p == l: fake_c += 1
                else:
                    real_t += 1
                    if p == l: real_c += 1
    return (100 * correct / total,
            100 * real_c / real_t if real_t else 0,
            100 * fake_c / fake_t if fake_t else 0)


# ---------------------------------------------------------------
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")

    with open(ROOP_JSON) as f:
        roop_data = json.load(f)
    with open(CELEB_JSON) as f:
        celeb_data = json.load(f)

    # Celeb'den 1000 örnek al (train için), geri kalanı val
    random.seed(42)
    random.shuffle(celeb_data)
    celeb_train = celeb_data[:1000]
    celeb_val   = celeb_data[1000:1500]   # 500 bağımsız val

    # Roop %80 train, %20 val
    random.shuffle(roop_data)
    split = int(len(roop_data) * 0.8)
    roop_train = roop_data[:split]

    train_data = roop_train + celeb_train
    val_data   = celeb_val   # val tamamen celeb (fusion'ın görmediği domain)
    random.shuffle(train_data)

    # Gerçek sınıfını dengele (roop'ta az gerçek var)
    real_s = [s for s in train_data if s['label'] == 0]
    fake_s = [s for s in train_data if s['label'] == 1]
    balanced = real_s * 2 + fake_s
    random.shuffle(balanced)

    print(f"Train: {len(balanced)}  (gerçek×2: {len(real_s)*2}, sahte: {len(fake_s)})")
    print(f"Val:   {len(val_data)}  (celeb — tamamen bagimsiz)")

    train_loader = DataLoader(FusionDataset(balanced, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(FusionDataset(val_data),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = FusionModel().to(device)
    load_weights(model, device)
    freeze_backbones(model)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    best_acc, best_epoch = 0, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device.type):
                loss = criterion(model(imgs).squeeze(1), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        scheduler.step()

        acc, real_acc, fake_acc = evaluate(model, val_loader, device)
        tag = ''
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'fusion_best.pth'))
            tag = ' ★ EN İYİ'
        print(f"Epoch {epoch} — val: %{acc:.2f}  (gerçek %{real_acc:.2f} / sahte %{fake_acc:.2f}){tag}")

    print(f"\nEn iyi epoch: {best_epoch} (%{best_acc:.2f})")
    print(f"Model kaydedildi: {OUTPUT_DIR}/fusion_best.pth")

    # --- Final test ---
    print("\nFinal test (en iyi model)...")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'fusion_best.pth'),
                                     map_location=device))
    acc, real_acc, fake_acc = evaluate(model, val_loader, device)
    print("\n" + "=" * 40)
    print(f"GENEL BAŞARI: %{acc:.2f}")
    print(f"GERÇEK:       %{real_acc:.2f}")
    print(f"SAHTE:        %{fake_acc:.2f}")
    print("=" * 40)


if __name__ == '__main__':
    train()
