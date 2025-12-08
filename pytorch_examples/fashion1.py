import os
import math
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
import matplotlib.pyplot as plt


# ----------------------------
# Config
# ----------------------------
DATA_ROOT = "./data"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = 28
channels = 1

T = 200                     # number of diffusion steps
batch_size = 128
n_epochs = 5                # bump if you want better samples
lr = 2e-4
num_workers = 2
seed = 42

torch.manual_seed(seed)


# ----------------------------
# Beta schedule and helpers
# ----------------------------

def make_beta_schedule(T: int,
                       beta_start: float = 1e-4,
                       beta_end: float = 0.02) -> torch.Tensor:
    """Linear beta schedule from beta_start to beta_end."""
    return torch.linspace(beta_start, beta_end, T)


betas = make_beta_schedule(T).to(device)                       # (T,)
alphas = 1.0 - betas                                           # (T,)
alpha_bars = torch.cumprod(alphas, dim=0)                      # (T,)


def extract_coeff(coeff: torch.Tensor, t: torch.Tensor,
                  x_shape: torch.Size) -> torch.Tensor:
    """
    Extract coefficients for a batch of time indices t and
    reshape to [B, 1, 1, 1] so they broadcast over images.
    """
    out = coeff[t].float()
    while len(out.shape) < len(x_shape):
        out = out.unsqueeze(-1)
    return out


def q_sample(x0: torch.Tensor, t: torch.Tensor,
             noise: torch.Tensor) -> torch.Tensor:
    """
    Sample from q(x_t | x_0): sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) eps
    """
    sqrt_alpha_bar = extract_coeff(alpha_bars.sqrt(), t, x0.shape)
    sqrt_one_minus_alpha_bar = extract_coeff(
        (1.0 - alpha_bars).sqrt(), t, x0.shape
    )
    return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise


# ----------------------------
# Sinusoidal time embedding
# ----------------------------

class SinusoidalPosEmb(nn.Module):
    """
    Standard 1D sinusoidal time embedding used in DDPM/Transformers.
    t: (B,) with integer timestep indices.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


# ----------------------------
# Simple conv denoiser (epsilon_theta)
# ----------------------------

class SimpleDenoiser(nn.Module):
    """
    Lightweight U-Net-ish conv model for 28x28 Fashion-MNIST.
    Predicts epsilon given x_t and timestep t.
    """
    def __init__(self, img_channels: int = 1, base_channels: int = 32,
                 time_dim: int = 128):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
        )

        # down blocks
        self.conv1 = nn.Conv2d(img_channels, base_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, 3,
                               stride=2, padding=1)  # 28->14
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 2, 3,
                               padding=1)

        # time embeddings to channel dims
        self.time_to_c1 = nn.Linear(time_dim, base_channels)
        self.time_to_c2 = nn.Linear(time_dim, base_channels * 2)

        # up blocks
        self.deconv1 = nn.ConvTranspose2d(base_channels * 2,
                                          base_channels,
                                          4, stride=2, padding=1)  # 14->28
        self.conv4 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.out_conv = nn.Conv2d(base_channels, img_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) integer indices in [0, T-1]
        t_emb = self.time_mlp(t)  # (B, time_dim)

        # Down
        h = self.conv1(x)
        h = F.relu(h)
        # inject time
        t_c1 = self.time_to_c1(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t_c1

        h = self.conv2(h)
        h = F.relu(h)
        t_c2 = self.time_to_c2(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t_c2

        h = self.conv3(h)
        h = F.relu(h)

        # Up
        h = self.deconv1(h)
        h = F.relu(h)

        h = self.conv4(h)
        h = F.relu(h)

        out = self.out_conv(h)
        return out


# ----------------------------
# Dataset and DataLoader
# ----------------------------

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    # map from [0,1] -> [-1,1] for DDPM
    transforms.Lambda(lambda x: x * 2.0 - 1.0),
])

train_dataset = datasets.FashionMNIST(
    root=DATA_ROOT,
    train=True,
    download=True,
    transform=transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)


# ----------------------------
# Training
# ----------------------------

model = SimpleDenoiser(img_channels=channels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

all_losses = []


def p_losses(x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    DDPM training objective: E_{x0, t, eps} ||eps - eps_theta(x_t, t)||^2
    """
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise)
    eps_pred = model(x_t, t)
    return F.mse_loss(eps_pred, noise)


print(f"Starting training on {len(train_dataset)} Fashion-MNIST images...")
global_step = 0
for epoch in range(1, n_epochs + 1):
    epoch_losses = []
    start = time.time()
    for x, _ in train_loader:
        x = x.to(device)
        # sample random timestep for each image in batch
        t = torch.randint(0, T, (x.shape[0],), device=device).long()

        loss = p_losses(x, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_losses.append(loss.item())
        epoch_losses.append(loss.item())
        global_step += 1

    avg_loss = sum(epoch_losses) / len(epoch_losses)
    elapsed = time.time() - start
    print(f"Epoch {epoch}/{n_epochs} | "
          f"avg loss = {avg_loss:.4f} | time = {elapsed:.1f}s")

# Save loss curve
plt.figure(figsize=(8, 4))
plt.plot(all_losses)
plt.xlabel("training step")
plt.ylabel("loss")
plt.title("DDPM training loss on Fashion-MNIST")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fashion_mnist_loss_curve.png"))
plt.close()
print(f"Saved loss curve to {RESULTS_DIR}/fashion_mnist_loss_curve.png")


# ----------------------------
# Sampling (reverse diffusion)
# ----------------------------

@torch.no_grad()
def p_sample(x_t: torch.Tensor, t: int) -> torch.Tensor:
    """
    Sample x_{t-1} given x_t using predicted epsilon.
    Implements Eq. 11 in Ho et al. (DDPM paper).
    """
    b = x_t.shape[0]
    t_batch = torch.full((b,), t, device=device, dtype=torch.long)

    eps_theta = model(x_t, t_batch)

    beta_t = betas[t]
    alpha_t = alphas[t]
    alpha_bar_t = alpha_bars[t]

    # posterior variance (beta_tilde)
    if t > 0:
        alpha_bar_prev = alpha_bars[t - 1]
    else:
        alpha_bar_prev = torch.tensor(1.0, device=device)

    beta_tilde = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)

    # mean of p(x_{t-1} | x_t)
    mean = (
        1.0 / torch.sqrt(alpha_t) * (
            x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_theta
        )
    )

    if t > 0:
        noise = torch.randn_like(x_t)
        x_prev = mean + torch.sqrt(beta_tilde) * noise
    else:
        x_prev = mean  # no noise at final step

    return x_prev


@torch.no_grad()
def p_sample_loop(n_samples: int) -> torch.Tensor:
    """
    Run the full reverse diffusion chain from T-1 ... 0.
    """
    x_t = torch.randn(n_samples, channels, image_size, image_size,
                      device=device)

    for t in reversed(range(T)):
        x_t = p_sample(x_t, t)

    return x_t


print("Sampling from the trained DDPM...")
model.eval()
num_samples = 64
samples = p_sample_loop(num_samples)

# map back from [-1,1] to [0,1] for visualization
samples = (samples.clamp(-1, 1) + 1.0) / 2.0

grid = vutils.make_grid(samples, nrow=8)
plt.figure(figsize=(6, 6))
plt.axis("off")
plt.title("DDPM samples – Fashion-MNIST")
plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, "fashion_mnist_samples_final.png")
plt.savefig(out_path, dpi=200)
plt.close()
print(f"Saved generated samples to {out_path}")