import os, json, random, cv2, torch
import numpy as np
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from PIL import Image

# ── YOL YAPILANDIRMASI ───────────────────────────────────────────────────────
# Bu yolları kendi ortamınıza göre düzenleyin
ROOP_JSON       = 'roop_dataset.json'
PRETRAINED_PATH = 'efficientnet_finetuned.pth'
OUTPUT_DIR      = 'outputs_roop'
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 16
LR = 1e-5
EPOCHS = 20


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1)

    def forward(self, x):
        return self.model(x).squeeze(1)


class RoopDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            with Image.open(s['img']) as pil_img:
                img = np.array(pil_img.convert('RGB'))
            img = cv2.resize(img, (224, 224))
        except:
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.augment and random.random() < 0.5:
            img = img[:, ::-1].copy()

        img = ((img.astype(np.float32) / 255.0 - self.mean) / self.std).transpose(2, 0, 1)
        return torch.from_numpy(img), torch.tensor(s['label'], dtype=torch.float32)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")

    with open(ROOP_JSON, 'r') as f:
        data = json.load(f)

    # %80 train, %20 test
    random.seed(42)
    random.shuffle(data)
    split = int(len(data) * 0.8)
    train_data = data[:split]
    test_data = data[split:]

    # Gerçek ve sahteleri dengele
    real = [s for s in train_data if s['label'] == 0]
    fake = [s for s in train_data if s['label'] == 1]
    balanced = real * 3 + fake
    random.shuffle(balanced)

    print(f"Train: {len(balanced)} (gerçek x3: {len(real)*3}, sahte: {len(fake)})")
    print(f"Test:  {len(test_data)}")

    model = Detector().to(device)
    model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device))

    train_loader = DataLoader(RoopDataset(balanced, augment=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    best_acc = 0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device.type):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Her epoch sonunda test setinde değerlendir
        model.eval()
        test_loader_val = DataLoader(RoopDataset(test_data), batch_size=16, shuffle=False, num_workers=0)
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader_val:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = (torch.sigmoid(model(imgs)) > 0.5).float()
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
        acc = 100 * correct_val / total_val

        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'roop_ep{epoch}.pth'))
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'roop_best.pth'))
        print(f"Epoch {epoch} — val acc: %{acc:.2f} {'★ EN İYİ' if epoch == best_epoch else ''}")

    print(f"\nEn iyi epoch: {best_epoch} (%{best_acc:.2f})")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'roop_best.pth')))
    print("Test yapılıyor (en iyi model)...")
    model.eval()
    test_loader = DataLoader(RoopDataset(test_data), batch_size=16, shuffle=False, num_workers=0)

    correct, total = 0, 0
    fake_correct, fake_total = 0, 0
    real_correct, real_total = 0, 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = (torch.sigmoid(model(imgs)) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for p, l in zip(preds, labels):
                if l == 1:
                    fake_total += 1
                    if p == l: fake_correct += 1
                else:
                    real_total += 1
                    if p == l: real_correct += 1

    print("\n" + "=" * 35)
    print(f"TOPLAM: {total}")
    print(f"GENEL:  %{100 * correct / total:.2f}")
    if real_total > 0:
        print(f"GERÇEK: %{100 * real_correct / real_total:.2f}  ({real_correct}/{real_total})")
    if fake_total > 0:
        print(f"SAHTE:  %{100 * fake_correct / fake_total:.2f}  ({fake_correct}/{fake_total})")
    print("=" * 35)


if __name__ == "__main__":
    train()
