import os, json, random, cv2, torch
import numpy as np
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# ── YOL YAPILANDIRMASI ───────────────────────────────────────────────────────
# Bu yolları kendi ortamınıza göre düzenleyin
FFPP_LIST       = 'dataset.json'
NEW_CELEB_LIST  = 'celeb_pro_6000.json'
PRETRAINED_PATH = 'checkpoint_ep30.pth'
OUTPUT_DIR      = 'outputs_finetuned'
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 32
LR = 2e-5  # Çok hassas öğrenme (mevcut bilgiyi bozmamak için)
EPOCHS = 5


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1)

    def forward(self, x):
        return self.model(x).squeeze(1)


class FineTuneDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            # En sağlam okuma yöntemi
            file_bytes = np.fromfile(s['img'], dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
        except:
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        # Basit Augmentation (Yatay çevirme)
        if random.random() < 0.5:
            img = img[:, ::-1].copy()

        img = ((img.astype(np.float32) / 255.0 - self.mean) / self.std).transpose(2, 0, 1)
        return torch.from_numpy(img), torch.tensor(s['label'], dtype=torch.float32)


def start_training():
    device = torch.device('cuda')
    print(f"--- CANAVAR UYANIYOR: {torch.cuda.get_device_name(0)} ---")

    # Verileri Birleştir
    with open(FFPP_LIST, 'r') as f1:
        ffpp = json.load(f1)
    with open(NEW_CELEB_LIST, 'r') as f2:
        celeb = json.load(f2)

    # Celeb verisini 2 katına çıkararak (Oversampling) modele bu yeni stili zorla öğretiyoruz
    combined = ffpp + (celeb * 2)
    random.shuffle(combined)

    model = Detector().to(device)
    # Önceki 30 epochluk eğitimi temele koyuyoruz
    model.load_state_dict(torch.load(PRETRAINED_PATH))

    loader = DataLoader(FineTuneDataset(combined), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')

    print(f"Eğitim başlıyor... Toplam Kare: {len(combined)}")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                loss = criterion(model(imgs), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({'loss': f"{loss.item():.5f}"})

        # Kayıt
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'finetuned_pro_ep{epoch}.pth'))
        print(f"Epoch {epoch} tamamlandı.")


if __name__ == "__main__":
    start_training()