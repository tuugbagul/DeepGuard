import torch
import torch.nn as nn
import timm


class CrossAttentionBlock(nn.Module):
    def __init__(self, xception_dim=2048, fft_dim=512, embed_dim=256, num_heads=4):
        super().__init__()
        self.query_proj = nn.Linear(xception_dim, embed_dim)
        self.key_value_proj = nn.Linear(fft_dim, embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=0.3
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, xception_feats, fft_feats):
        q = self.query_proj(xception_feats).unsqueeze(1)
        k = v = self.key_value_proj(fft_feats).unsqueeze(1)
        attn_out, _ = self.attention(q, k, v)
        return self.norm(q + self.dropout(attn_out)).squeeze(1)


class PixelGuardHybridModel(nn.Module):
    """Xception + FFT + Cross-Attention deepfake detection model."""

    def __init__(self, pretrained=True):
        super().__init__()
        self.xception = timm.create_model('legacy_xception', pretrained=pretrained)
        self.xception.fc = nn.Identity()

        self.fft_branch = nn.Sequential(
            nn.Linear(299 * 299, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.cross_attention = CrossAttentionBlock(2048, 512, 256)

        self.classifier = nn.Sequential(
            nn.Linear(2048 + 256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x_img = self.xception(x)

        x_gray = x.mean(dim=1, keepdim=True)
        fft_magnitude = torch.abs(torch.fft.fftshift(torch.fft.fft2(x_gray)))
        fft_magnitude = torch.log(fft_magnitude + 1e-8)
        fft_flat = fft_magnitude.view(fft_magnitude.size(0), -1)
        fft_flat = (fft_flat - fft_flat.mean()) / (fft_flat.std() + 1e-8)

        x_fft = self.fft_branch(fft_flat)
        attended = self.cross_attention(x_img, x_fft)
        return self.classifier(torch.cat((x_img, attended), dim=1))
