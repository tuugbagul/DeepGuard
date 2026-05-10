"""
Hard negative mining ile FusionModel fine-tune.
Kaçırılan yüksek kaliteli Roop çıktılarını öğretir.

python finetune_hard.py
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
OUTPUT_DIR     = 'outputs_fusion'
ROOP_JSON      = 'roop_dataset.json'
FOLDER         = 'test_images'     # Sahte/gerçek test görselleri klasörü
# ─────────────────────────────────────────────────────────────────────────────

LR         = 5e-6   # çok küçük — mevcut ağırlıkları bozmamak için
EPOCHS     = 8
BATCH_SIZE = 4

# ---------------------------------------------------------------
# Hard negative örnekler: model kaçırdığı output dosyaları
# ---------------------------------------------------------------
HARD_FAKES = ['output_5.jpg', 'output_6.jpg', 'output_7.jpg',
              'output_8.jpg', 'output_10.jpg']
EASY_FAKES = ['output_1.jpg', 'output_2.jpg', 'output_3.jpg',
              'output_4.jpg', 'output_9.jpg']
# Gerçek yüz fotoğraflarınızı FOLDER altına koyup buraya ekleyin
REAL_FILES = ['person1.jpg', 'person2.jpg']  # örnek — kendi dosyalarınızla değiştirin

def build_local_samples():
    samples = []
    # Hard fake'leri 4x oversample et (model bunları görmesi gerekiyor)
    for fname in HARD_FAKES:
        p = os.path.join(FOLDER, fname)
        if os.path.exists(p):
            for _ in range(4):
                samples.append({'img': p, 'label': 1})
    # Easy fake'ler 1x
    for fname in EASY_FAKES:
        p = os.path.join(FOLDER, fname)
        if os.path.exists(p):
            samples.append({'img': p, 'label': 1})
    # Gerçek fotoğraflar 2x
    for fname in REAL_FILES:
        p = os.path.join(FOLDER, fname)
        if os.path.exists(p):
            for _ in range(2):
                samples.append({'img': p, 'label': 0})
    return samples


class FusionDataset(Dataset):
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

        if self.augment:
            if random.random() < 0.5:
                img = img[:, ::-1].copy()
            # Hafif parlaklık / kontrast değişimi
            if random.random() < 0.4:
                factor = random.uniform(0.85, 1.15)
                img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        tensor = torch.from_numpy(
            img.astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        return tensor, torch.tensor(s['label'], dtype=torch.float32)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    fake_c, fake_t = 0, 0
    real_c, real_t = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = (torch.sigmoid(model(imgs).squeeze(1)) > 0.6).float()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            for p, l in zip(preds, labels):
                if l == 1:
                    fake_t += 1
                    if p == l: fake_c += 1
                else:
                    real_t += 1
                    if p == l: real_c += 1
    return (100*correct/total,
            100*real_c/real_t if real_t else 0,
            100*fake_c/fake_t if fake_t else 0)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")

    # Model yükle
    model = FusionModel().to(device)
    model.load_state_dict(torch.load(FUSION_WEIGHTS, map_location=device))

    # PixelGuard'ı dondur, EfficientNet'in son 2 bloğunu aç
    for p in model.pixelguard.parameters():
        p.requires_grad = False

    for p in model.efficientnet.parameters():
        p.requires_grad = False

    # EfficientNet son 2 blok + head unfreeze
    for name, p in model.efficientnet.named_parameters():
        if any(x in name for x in ['blocks.6', 'blocks.5', 'conv_head', 'bn2']):
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Kısmi unfreeze. Eğitilecek parametre: {trainable:,}")

    # Veri seti
    local = build_local_samples()

    # Roop train setinden 300 örnek karıştır — eski bilgiyi unutmasın
    with open(ROOP_JSON) as f:
        roop = json.load(f)
    random.seed(42)
    random.shuffle(roop)
    roop_train = roop[:int(len(roop)*0.8)]
    roop_mix   = random.sample(roop_train, min(300, len(roop_train)))

    train_data = local + roop_mix
    random.shuffle(train_data)

    # Val: sadece local örnekler (augmentsiz)
    val_data = []
    for fname in HARD_FAKES + EASY_FAKES:
        p = os.path.join(FOLDER, fname)
        if os.path.exists(p):
            val_data.append({'img': p, 'label': 1})
    for fname in REAL_FILES:
        p = os.path.join(FOLDER, fname)
        if os.path.exists(p):
            val_data.append({'img': p, 'label': 0})

    print(f"Train: {len(train_data)}  (local: {len(local)}, roop mix: {len(roop_mix)})")
    print(f"Val:   {len(val_data)}  (10 sahte + 4 gerçek)")

    train_loader = DataLoader(FusionDataset(train_data, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader   = DataLoader(FusionDataset(val_data),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    best_fake_acc = 0
    best_epoch    = 0

    print("\nBaşlangıç performansı:")
    acc, real_acc, fake_acc = evaluate(model, val_loader, device)
    print(f"  Genel: %{acc:.1f}  Gerçek: %{real_acc:.1f}  Sahte: %{fake_acc:.1f}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        # Backbone'ları eval modunda tut (BN istatistiklerini bozma)
        model.pixelguard.eval()

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

        acc, real_acc, fake_acc = evaluate(model, val_loader, device)
        tag = ''
        if fake_acc >= best_fake_acc and real_acc >= 90:
            best_fake_acc = fake_acc
            best_epoch    = epoch
            torch.save(model.state_dict(),
                       os.path.join(OUTPUT_DIR, 'fusion_hardneg.pth'))
            tag = ' ★'
        print(f"Epoch {epoch} — Genel: %{acc:.1f}  Gerçek: %{real_acc:.1f}  Sahte: %{fake_acc:.1f}{tag}")

    print(f"\nEn iyi epoch: {best_epoch} (sahte %{best_fake_acc:.1f})")
    print(f"Kaydedildi: {OUTPUT_DIR}/fusion_hardneg.pth")


if __name__ == '__main__':
    main()
