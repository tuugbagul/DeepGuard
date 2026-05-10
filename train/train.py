import os, json, random, cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# ==========================================
# 1. AYARLAR
# ==========================================
# ── YOL YAPILANDIRMASI ───────────────────────────────────────────────────────
# Bu yolları kendi ortamınıza göre düzenleyin
LIST_PATH  = 'dataset.json'
OUTPUT_DIR = 'outputs'
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 12GB VRAM DARBOĞAZI İÇİN OPTİMİZE AYARLAR
BATCH_SIZE = 32  # 64'ten 32'ye çektik (VRAM taşmasını önlemek için)
NUM_WORKERS = 4  # CPU'yu 1.12 GHz'den kurtarmak için 4 yapıyoruz
EPOCHS = 30
LR = 1e-4
IMG_SIZE = 224


# ==========================================
# 2. HIZLI VERİ OKUMA (IO OPTİMİZE)
# ==========================================
class FFPPDataset(Dataset):
    def __init__(self, samples, augment=True):
        self.samples = samples
        self.augment = augment
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # BGR -> RGB ve Resize (CPU'da hızlı işlem)
        img = cv2.imread(s['img'])
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        if self.augment and random.random() < 0.5:
            img = img[:, ::-1].copy()

        img = ((img.astype(np.float32) / 255.0 - self.mean) / self.std).transpose(2, 0, 1)
        return torch.from_numpy(img), torch.tensor(s['label'], dtype=torch.float32)


# ==========================================
# 3. MODEL (EfficientNet-B4)
# ==========================================
class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        # B4 Laptop GPU için çok idealdir
        self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=1)

    def forward(self, x):
        return self.model(x).squeeze(1)


# ==========================================
# 4. EĞİTİM MOTORU
# ==========================================
def train():
    device = torch.device('cuda')
    print(f"--- CİHAZ: {torch.cuda.get_device_name(0)} ---")

    with open(LIST_PATH, 'r') as f:
        samples = json.load(f)

    random.shuffle(samples)
    split = int(len(samples) * 0.9)
    train_loader = DataLoader(FFPPDataset(samples[:split]), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True)  # Worker'ları canlı tutar

    model = Detector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')  # Mixed Precision için

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for imgs, labels in pbar:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # Daha fazla VRAM tasarrufu

            with autocast('cuda'):
                output = model(imgs)
                loss = criterion(output, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'checkpoint_ep{epoch}.pth'))
        print(f"Epoch {epoch} tamamlandı ve kaydedildi.")


if __name__ == "__main__":
    train()