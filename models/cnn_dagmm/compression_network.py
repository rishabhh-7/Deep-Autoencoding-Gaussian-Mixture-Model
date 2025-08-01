import torch
import torch.nn.functional as F
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=4, stride=2, pad=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, pad, bias=False),
            nn.BatchNorm2d(out_ch),
            # nn.LeakyReLU(0.2, inplace=True)
            nn.LeakyReLU(0.01, inplace=True)
        )
    def forward(self, x): return self.net(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=4, stride=2, pad=1, final=False):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, pad, bias=False)]
        if not final:
            layers += [nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True)]
        else:
            layers += [nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class CompressionNetwork(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        # Encoder: [B,3,64,64] → [B,256,4,4]
        self.enc1 = ConvBlock(3,  32)   # 32×32×32
        self.enc2 = ConvBlock(32, 64)   # 64×16×16
        self.enc3 = ConvBlock(64,128)   # 128×8×8
        self.enc4 = ConvBlock(128,256)  # 256×4×4
        self.flatten    = nn.Flatten()
        self.fc_latent  = nn.Linear(256*4*4, latent_dim)

        # Decoder: reverse
        self.fc_expand  = nn.Linear(latent_dim, 256*4*4)
        self.dec4 = DeconvBlock(256,128)   # 128×8×8
        self.dec3 = DeconvBlock(128,64)    # 64×16×16
        self.dec2 = DeconvBlock(64,32)     # 32×32×32
        self.dec1 = DeconvBlock(32, 3, final=True)  # 3×64×64

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        z = self.flatten(x)
        return self.fc_latent(z)

    def decode(self, z):
        x = self.fc_expand(z).view(-1,256,4,4)
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        return self.dec1(x)

    def reconstruction_loss(self, x, x_hat):
        # MSE + edge gradient penalty for sharper reconstructions
        mse = F.mse_loss(x_hat, x)
        # gradient loss
        def grad_loss(img):
            dx = torch.abs(img[:,:,1:,:] - img[:,:,:-1,:]).mean()
            dy = torch.abs(img[:,:,:,1:] - img[:,:,:,:-1]).mean()
            return dx + dy
        return mse + 0.1*grad_loss(x_hat - x)
