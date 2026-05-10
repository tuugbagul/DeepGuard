import torch
import torch.nn as nn
import timm

from pixelguard_model import PixelGuardHybridModel


class FusionModel(nn.Module):
    """
    EfficientNet-B4 (FaceForensics++ + Celeb-DF + Roop) +
    PixelGuard (Xception + FFT + Cross-Attention) feature-level fusion.

    Giriş: 299x299, [0,1] aralığında ham tensor
    Her dal kendi normalizasyonunu içsel olarak uygular.
    Çıkış: ham logit (sigmoid uygulamadan)
    """

    EFF_DIM = 1792   # EfficientNet-B4 global pool çıktısı
    PG_DIM = 2304    # Xception(2048) + CrossAttn(256)

    def __init__(self):
        super().__init__()

        # --- EfficientNet-B4 kolu ---
        # num_classes=0: classifier'ı kaldırır, 1792-dim pooled feature döner
        self.efficientnet = timm.create_model(
            'efficientnet_b4', pretrained=False, num_classes=0
        )

        # --- PixelGuard kolu ---
        self.pixelguard = PixelGuardHybridModel(pretrained=False)

        # --- Normalizasyon bufferları (device'a otomatik taşınır) ---
        self.register_buffer(
            'eff_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'eff_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'pg_mean', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'pg_std', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        )

        # --- Fusion sınıflandırıcı başı ---
        self.classifier = nn.Sequential(
            nn.Linear(self.EFF_DIM + self.PG_DIM, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        )

    # -----------------------------------------------------------------
    def _pixelguard_features(self, x):
        """
        PixelGuard'ın classifier'ından önceki 2304-dim feature'ı döner.
        x: [-1,1] normalize edilmiş, 299x299 tensor
        """
        x_img = self.pixelguard.xception(x)                     # (B, 2048)

        x_gray = x.mean(dim=1, keepdim=True)
        fft_out = torch.fft.fft2(x_gray)
        fft_mag = torch.abs(torch.fft.fftshift(fft_out))
        fft_mag = torch.log(fft_mag + 1e-8)
        fft_flat = fft_mag.view(fft_mag.size(0), -1)            # (B, 299*299)
        fft_flat = (fft_flat - fft_flat.mean()) / (fft_flat.std() + 1e-8)

        x_fft = self.pixelguard.fft_branch(fft_flat)            # (B, 512)
        attended = self.pixelguard.cross_attention(x_img, x_fft) # (B, 256)
        return torch.cat([x_img, attended], dim=1)               # (B, 2304)

    # -----------------------------------------------------------------
    def forward(self, x):
        """x: [0,1] aralığında, (B, 3, 299, 299)"""
        x_eff = (x - self.eff_mean) / self.eff_std
        eff_feats = self.efficientnet(x_eff)                     # (B, 1792)

        x_pg = (x - self.pg_mean) / self.pg_std
        pg_feats = self._pixelguard_features(x_pg)               # (B, 2304)

        combined = torch.cat([eff_feats, pg_feats], dim=1)       # (B, 4096)
        return self.classifier(combined)                          # (B, 1)
